"""
Tucker Decomposition for Convolutional Layers
==============================================

Decomposes a 4D conv weight tensor W ∈ R^(C_out × C_in × H × W) into:
    W ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄

where G is the core tensor and U_i are factor matrices.

For conv layers, this replaces ONE conv layer with THREE sequential layers:
    1. Pointwise conv (1×1): reduces input channels via U₂
    2. Regular conv (H×W): operates on reduced dimensions via core G
    3. Pointwise conv (1×1): expands output channels via U₁

This is the most stable decomposition method for CNNs because:
- HOSVD provides a good initialization (no iterative optimization needed)
- The rank can be set independently for each mode (flexible)
- The reconstruction error is bounded by the sum of truncated singular values

CUDA Acceleration:
- Uses torch operations throughout (runs on GPU automatically)
- HOSVD computed via batched SVD on GPU (torch.linalg.svd)
- No CPU-GPU transfers during decomposition
- torch.compile compatible for the reconstructed layers
"""

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Use PyTorch backend for tensorly so all ops stay on GPU
tl.set_backend("pytorch")


class TuckerDecomposer:
    """
    Decomposes nn.Conv2d layers using Tucker decomposition.

    The decomposition replaces a single conv layer with a sequence of three
    smaller conv layers that approximate the original computation with fewer
    parameters and FLOPs.
    """

    def __init__(self, rank_ratio: float = 0.5, device: str = "cuda"):
        """
        Args:
            rank_ratio: Fraction of original rank to keep (0.0 to 1.0).
                        0.5 means keep 50% of each dimension → ~4x compression.
                        Lower = more compression, less accuracy.
            device: Target device for decomposition computation.
        """
        self.rank_ratio = rank_ratio
        self.device = device

    def estimate_ranks(
        self, weight: torch.Tensor, rank_ratio: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Estimate Tucker ranks for a conv weight tensor.

        For a conv weight of shape (C_out, C_in, kH, kW), we decompose
        modes 0 (output channels) and 1 (input channels). Kernel dimensions
        are typically small (3×3) so we don't decompose them.

        Args:
            weight: Conv weight tensor of shape (C_out, C_in, kH, kW).
            rank_ratio: Override instance rank_ratio if provided.

        Returns:
            (rank_out, rank_in): Ranks for output and input channel modes.
        """
        ratio = rank_ratio or self.rank_ratio
        c_out, c_in = weight.shape[0], weight.shape[1]

        # Ensure ranks are at least 1 and at most the original dimension
        rank_out = max(1, int(c_out * ratio))
        rank_in = max(1, int(c_in * ratio))

        return rank_out, rank_in

    @torch.no_grad()
    def decompose_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        rank_ratio: Optional[float] = None,
    ) -> nn.Sequential:
        """
        Decompose a single Conv2d layer into three smaller conv layers.

        Original: Conv2d(C_in, C_out, kernel_size, stride, padding)
        Becomes:
            1. Conv2d(C_in, rank_in, 1×1)        --  channel reduction (pointwise)
            2. Conv2d(rank_in, rank_out, k×k)     --  spatial conv on reduced dims
            3. Conv2d(rank_out, C_out, 1×1)       --  channel expansion (pointwise)

        Args:
            conv_layer: The original Conv2d layer to decompose.
            rank_ratio: Override the instance rank_ratio for this layer.

        Returns:
            nn.Sequential containing the three replacement layers.
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in, kh, kw = weight.shape
        rank_out, rank_in = self.estimate_ranks(weight, rank_ratio)

        logger.debug(
            f"Tucker decompose: ({c_out},{c_in},{kh},{kw}) → "
            f"ranks=({rank_out},{rank_in})"
        )

        # Skip decomposition if ranks >= original dimensions (no compression)
        if rank_out >= c_out and rank_in >= c_in:
            logger.debug("  Skipping: ranks >= original dims, no compression")
            return nn.Sequential(conv_layer)

        # Perform Tucker decomposition on the weight tensor
        # We decompose modes [0, 1] (output channels, input channels)
        # Modes [2, 3] (kernel height, width) are kept intact (usually 3×3)
        #
        # partial_tucker returns:
        #   core: tensor of shape (rank_out, rank_in, kH, kW)
        #   factors: list of [U_out (c_out, rank_out), U_in (c_in, rank_in)]
        try:
            # partial_tucker returns ((core, [factors]), errors)
            result = partial_tucker(
                weight,
                modes=[0, 1],
                rank=[rank_out, rank_in],
                init="svd",        # HOSVD initialization  --  fast, deterministic
                n_iter_max=0,      # 0 iterations = pure HOSVD (fastest, good enough)
            )
            # Unpack: result is ((core_tensor, factor_list), reconstruction_errors)
            core, factors = result[0][0], result[0][1]
        except Exception as e:
            logger.warning(f"  Tucker decomposition failed: {e}. Keeping original layer.")
            return nn.Sequential(conv_layer)

        U_out, U_in = factors[0], factors[1]

        # Build the three replacement layers:

        # Layer 1: Pointwise conv (1×1)  --  reduces input channels
        # Applies U_in^T to go from C_in → rank_in
        # No bias, no stride/padding needed for pointwise
        first_layer = nn.Conv2d(
            in_channels=c_in,
            out_channels=rank_in,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # U_in shape: (c_in, rank_in) → reshape to (rank_in, c_in, 1, 1) for conv
        first_layer.weight.data = U_in.t().unsqueeze(-1).unsqueeze(-1).contiguous()

        # Layer 2: Spatial conv on reduced dimensions
        # Uses the core tensor as weights: (rank_out, rank_in, kH, kW)
        # Preserves the original stride and padding
        core_layer = nn.Conv2d(
            in_channels=rank_in,
            out_channels=rank_out,
            kernel_size=(kh, kw),
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=False,
        )
        core_layer.weight.data = core.contiguous()

        # Layer 3: Pointwise conv (1×1)  --  expands output channels
        # Applies U_out to go from rank_out → C_out
        # Bias from original conv goes here (if any)
        last_layer = nn.Conv2d(
            in_channels=rank_out,
            out_channels=c_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=conv_layer.bias is not None,
        )
        last_layer.weight.data = U_out.unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            last_layer.bias.data = conv_layer.bias.data.clone()

        decomposed = nn.Sequential(first_layer, core_layer, last_layer)
        decomposed = decomposed.to(self.device)

        # Log compression stats for this layer
        orig_params = c_out * c_in * kh * kw
        new_params = (
            c_in * rank_in            # first layer
            + rank_in * rank_out * kh * kw  # core layer
            + rank_out * c_out         # last layer
        )
        logger.debug(
            f"  Compression: {orig_params:,} → {new_params:,} params "
            f"({orig_params / max(new_params, 1):.1f}×)"
        )

        return decomposed

    def get_compression_ratio(self, conv_layer: nn.Conv2d, rank_ratio: Optional[float] = None) -> float:
        """Calculate the theoretical compression ratio for a given layer."""
        weight = conv_layer.weight.data
        c_out, c_in, kh, kw = weight.shape
        rank_out, rank_in = self.estimate_ranks(weight, rank_ratio)

        orig = c_out * c_in * kh * kw
        compressed = c_in * rank_in + rank_in * rank_out * kh * kw + rank_out * c_out
        return orig / max(compressed, 1)
