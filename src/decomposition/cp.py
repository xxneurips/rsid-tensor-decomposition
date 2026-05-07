"""
CP (CANDECOMP/PARAFAC) Decomposition for Convolutional Layers
=============================================================

Decomposes a 4D conv weight tensor W ∈ R^(C_out × C_in × H × W) into:
    W ≈ Σᵣ λᵣ · u₁ᵣ ⊗ u₂ᵣ ⊗ u₃ᵣ ⊗ u₄ᵣ

This is a rank-R approximation where each component is a rank-1 tensor.

For conv layers, CP decomposition replaces ONE conv layer with FOUR layers:
    1. Pointwise conv (1×1): input channel mixing
    2. Depthwise vertical conv (kH×1): spatial height filtering
    3. Depthwise horizontal conv (1×kW): spatial width filtering
    4. Pointwise conv (1×1): output channel mixing

Advantages over Tucker:
- More aggressive compression (fewer parameters at same rank)
- Separates spatial dimensions (height × width)

Disadvantages:
- ALS optimization can be unstable (sensitive to initialization)
- Rank selection is less intuitive (single rank for all modes)
- Convergence is not guaranteed

CUDA Acceleration:
- tensorly uses PyTorch backend → ALS runs entirely on GPU
- n_iter_max controls quality vs speed trade-off
- We use a low iteration count (50) with SVD init for fast, stable results
"""

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac
from typing import Optional
import logging

logger = logging.getLogger(__name__)

tl.set_backend("pytorch")


class CPDecomposer:
    """
    Decomposes nn.Conv2d layers using CP decomposition.

    Replaces a single conv with four sequential layers that separate
    the channel and spatial dimensions into independent rank-1 components.
    """

    def __init__(self, rank: int = 16, device: str = "cuda"):
        """
        Args:
            rank: CP rank  --  number of rank-1 components to use.
                  Higher rank = better approximation but less compression.
                  Typical: 8-64 depending on layer size.
            device: Target device.
        """
        self.rank = rank
        self.device = device

    def estimate_rank(self, weight: torch.Tensor, rank_ratio: Optional[float] = None) -> int:
        """
        Estimate CP rank for a conv weight.

        Unlike Tucker where we have per-mode ranks, CP uses a single rank R.
        We set R as a fraction of min(C_out, C_in) since that's the bottleneck.

        IMPORTANT: CP rank must be kept small relative to the tensor dimensions.
        The SVD initialization in parafac scales as O(R * product(dims)), so
        large ranks on large tensors cause massive memory allocation. We cap
        the rank at 32 to stay within memory limits.

        Args:
            weight: Conv weight tensor.
            rank_ratio: Fraction of min dimension to use as rank.

        Returns:
            CP rank (integer).
        """
        ratio = rank_ratio or (self.rank / min(weight.shape[0], weight.shape[1]))
        ratio = max(0.05, min(1.0, ratio))
        rank = max(1, int(min(weight.shape[0], weight.shape[1]) * ratio))
        # Cap rank to prevent memory explosion during SVD initialization
        # For a layer with 512 channels, rank=256 would need ~150GB for SVD
        # rank=32 is a good trade-off for CP (it's already very compressed)
        rank = min(rank, 32)
        return rank

    @torch.no_grad()
    def decompose_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        rank: Optional[int] = None,
    ) -> nn.Sequential:
        """
        Decompose a Conv2d layer using CP decomposition into four layers.

        Original: Conv2d(C_in, C_out, (kH, kW), stride, padding)
        Becomes:
            1. Conv2d(C_in, R, 1×1)                 --  input channel mixing
            2. Conv2d(R, R, (kH,1), groups=R)        --  depthwise vertical
            3. Conv2d(R, R, (1,kW), groups=R)         --  depthwise horizontal
            4. Conv2d(R, C_out, 1×1)                 --  output channel mixing

        The groups=R makes layers 2 and 3 depthwise (each channel processed
        independently), which is what makes CP so parameter-efficient.

        Args:
            conv_layer: Original Conv2d layer.
            rank: Override CP rank. If None, uses self.rank.

        Returns:
            nn.Sequential with four replacement layers.
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in, kh, kw = weight.shape
        R = rank or self.rank

        # For 1×1 convolutions, CP decomposition doesn't help  --  skip
        if kh == 1 and kw == 1:
            logger.debug(f"  Skipping 1×1 conv (CP needs spatial dims)")
            return nn.Sequential(conv_layer)

        logger.debug(f"CP decompose: ({c_out},{c_in},{kh},{kw}) → rank={R}")

        # Run CP decomposition (ALS algorithm via tensorly)
        # SVD initialization is more stable than random
        # n_iter_max=50 balances speed and quality
        #
        # NOTE: CP decomposition uses SVD internally which can fail on GPU
        # for certain tensor sizes due to cuSOLVER limitations. We run on CPU
        # as a fallback, which is fine since decomposition is a one-time cost.
        try:
            # First try SVD init on GPU (best quality)
            cp_result = parafac(
                weight,
                rank=R,
                init="svd",
                n_iter_max=50,
                normalize_factors=False,
            )
        except Exception:
            try:
                # Fallback 1: random init on GPU (avoids SVD memory issue)
                logger.debug("  CP SVD init failed, trying random init...")
                cp_result = parafac(
                    weight,
                    rank=R,
                    init="random",
                    n_iter_max=100,  # More iterations needed with random init
                    normalize_factors=False,
                )
            except Exception:
                try:
                    # Fallback 2: random init on CPU
                    logger.debug("  CP on GPU failed, falling back to CPU...")
                    cp_result = parafac(
                        weight.cpu(),
                        rank=R,
                        init="random",
                        n_iter_max=100,
                        normalize_factors=False,
                    )
                except Exception as e:
                    logger.warning(f"  CP decomposition failed: {e}. Keeping original.")
                    return nn.Sequential(conv_layer)

        # Extract factors  --  handle both old and new tensorly API
        if hasattr(cp_result, "factors"):
            lambdas = cp_result.weights
            factors = cp_result.factors
        else:
            lambdas, factors = cp_result

        U_out, U_in, U_h, U_w = factors

        # Absorb lambda scaling into the output factor matrix
        # This avoids needing a separate scaling layer
        if lambdas is not None:
            U_out = U_out * lambdas.unsqueeze(0)

        # Build the four replacement layers:

        # Layer 1: Pointwise (1×1)  --  mixes input channels → R components
        pw_in = nn.Conv2d(c_in, R, kernel_size=1, bias=False)
        pw_in.weight.data = U_in.t().unsqueeze(-1).unsqueeze(-1).contiguous()

        # Layer 2: Depthwise vertical  --  applies kH×1 filter per component
        # groups=R means each of the R channels is filtered independently
        dw_vertical = nn.Conv2d(
            R, R, kernel_size=(kh, 1),
            stride=(conv_layer.stride[0], 1),
            padding=(conv_layer.padding[0], 0),
            groups=R,
            bias=False,
        )
        # U_h shape: (kH, R) → need (R, 1, kH, 1) for depthwise conv
        dw_vertical.weight.data = U_h.t().unsqueeze(1).unsqueeze(-1).contiguous()

        # Layer 3: Depthwise horizontal  --  applies 1×kW filter per component
        dw_horizontal = nn.Conv2d(
            R, R, kernel_size=(1, kw),
            stride=(1, conv_layer.stride[1]),
            padding=(0, conv_layer.padding[1]),
            groups=R,
            bias=False,
        )
        # U_w shape: (kW, R) → need (R, 1, 1, kW) for depthwise conv
        dw_horizontal.weight.data = U_w.t().unsqueeze(1).unsqueeze(2).contiguous()

        # Layer 4: Pointwise (1×1)  --  combines R components → output channels
        pw_out = nn.Conv2d(R, c_out, kernel_size=1, bias=conv_layer.bias is not None)
        pw_out.weight.data = U_out.unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            pw_out.bias.data = conv_layer.bias.data.clone()

        decomposed = nn.Sequential(pw_in, dw_vertical, dw_horizontal, pw_out)
        decomposed = decomposed.to(self.device)

        # Log compression stats
        orig_params = c_out * c_in * kh * kw
        new_params = c_in * R + R * kh + R * kw + R * c_out
        logger.debug(
            f"  Compression: {orig_params:,} → {new_params:,} params "
            f"({orig_params / max(new_params, 1):.1f}×)"
        )

        return decomposed

    def get_compression_ratio(self, conv_layer: nn.Conv2d, rank: Optional[int] = None) -> float:
        """Calculate theoretical compression ratio for a given layer."""
        c_out, c_in, kh, kw = conv_layer.weight.shape
        R = rank or self.rank
        orig = c_out * c_in * kh * kw
        compressed = c_in * R + R * kh + R * kw + R * c_out
        return orig / max(compressed, 1)
