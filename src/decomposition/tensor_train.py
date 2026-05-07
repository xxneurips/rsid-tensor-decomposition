"""
Tensor Train (TT) Decomposition for Convolutional Layers
=========================================================

Decomposes a 4D conv weight tensor W ∈ R^(C_out × C_in × H × W) into a
chain of 3D cores:
    W[i₁,i₂,i₃,i₄] ≈ G₁[i₁] · G₂[i₂] · G₃[i₃] · G₄[i₄]

where Gₖ are 3D tensors (TT-cores) connected by contraction over shared
"bond" dimensions (TT-ranks).

For conv layers, this produces a chain of operations:
    1. Conv2d(C_in, r₁, 1×1)        --  first core (input reduction)
    2. Conv2d(r₁, r₂, kH×kW)        --  middle core (spatial + mixing)
    3. Conv2d(r₂, C_out, 1×1)        --  last core (output expansion)

The TT-ranks [r₁, r₂] control the compression. Unlike Tucker which has
per-mode ranks, TT has "bond" ranks between adjacent modes.

Advantages:
- Good for deep/wide networks (compression scales linearly with depth)
- No ALS needed  --  TT-SVD is deterministic and fast

Disadvantages:
- Mode ordering matters (different orderings give different results)
- Less flexible than Tucker for 4D tensors

Implementation note:
- We use a simplified 3-layer TT for conv weights (merge kernel dims)
- This is equivalent to a Tucker-2 decomposition but with sequential SVDs
- All SVD computations run on GPU via torch.linalg.svd
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TTDecomposer:
    """
    Decomposes nn.Conv2d layers using Tensor Train decomposition.

    Uses sequential SVD (TT-SVD algorithm) which is fast, deterministic,
    and runs entirely on GPU via torch.linalg.svd.
    """

    def __init__(self, rank_ratio: float = 0.5, device: str = "cuda"):
        """
        Args:
            rank_ratio: Fraction of min dimension for TT-ranks.
                        Controls compression level.
            device: Target device.
        """
        self.rank_ratio = rank_ratio
        self.device = device

    def estimate_ranks(
        self, weight: torch.Tensor, rank_ratio: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Estimate TT-ranks for a conv weight tensor.

        For a 4D tensor (C_out, C_in, kH, kW), we merge (kH, kW) into one
        mode and decompose as a 3-mode TT: (C_in) - (kH*kW) - (C_out).
        This gives us two TT-ranks: r₁ (between C_in and kernel) and
        r₂ (between kernel and C_out).

        Args:
            weight: Conv weight tensor.
            rank_ratio: Override fraction.

        Returns:
            (r1, r2): TT-ranks.
        """
        ratio = rank_ratio or self.rank_ratio
        c_out, c_in, kh, kw = weight.shape

        # TT-ranks are bounded by the product of dimensions on each side
        # r1 is bounded by min(C_in, kH*kW*C_out)
        # r2 is bounded by min(C_in*kH*kW, C_out)
        # In practice, we use rank_ratio * min(C_out, C_in) as a simple heuristic
        max_rank = min(c_out, c_in)
        r1 = max(1, int(max_rank * ratio))
        r2 = max(1, int(max_rank * ratio))

        return r1, r2

    @torch.no_grad()
    def decompose_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        rank_ratio: Optional[float] = None,
    ) -> nn.Sequential:
        """
        Decompose a Conv2d using TT-SVD into three sequential conv layers.

        The TT-SVD algorithm:
            1. Reshape W to (C_in, C_out*kH*kW)
            2. SVD → get first core and residual
            3. Reshape residual to (r1*kH*kW, C_out)
            4. SVD → get second core and third core

        This produces three layers:
            1. Conv2d(C_in, r1, 1×1)        --  input channel reduction
            2. Conv2d(r1, r2, kH×kW)         --  spatial filtering at reduced dim
            3. Conv2d(r2, C_out, 1×1)        --  output channel expansion

        All SVDs run on GPU via torch.linalg.svd  --  no CPU transfer needed.

        Args:
            conv_layer: Original Conv2d layer.
            rank_ratio: Override rank ratio.

        Returns:
            nn.Sequential with three replacement layers.
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in, kh, kw = weight.shape
        r1, r2 = self.estimate_ranks(weight, rank_ratio)

        # For 1×1 convolutions, simplify to a single SVD (2 layers)
        if kh == 1 and kw == 1:
            return self._decompose_pointwise(conv_layer, r1)

        logger.debug(f"TT decompose: ({c_out},{c_in},{kh},{kw}) → ranks=({r1},{r2})")

        # Skip if ranks >= original dims
        if r1 >= c_in and r2 >= c_out:
            logger.debug("  Skipping: ranks too large, no compression")
            return nn.Sequential(conv_layer)

        # === TT-SVD Step 1: Decompose input channels ===
        # Reshape to (C_in, C_out * kH * kW) and take SVD
        W_reshaped = weight.permute(1, 0, 2, 3).reshape(c_in, -1)  # (C_in, C_out*kH*kW)
        U1, S1, Vh1 = torch.linalg.svd(W_reshaped, full_matrices=False)

        # Truncate to rank r1
        U1 = U1[:, :r1]          # (C_in, r1)
        S1 = S1[:r1]             # (r1,)
        Vh1 = Vh1[:r1, :]        # (r1, C_out*kH*kW)

        # Absorb singular values into Vh1
        residual = torch.diag(S1) @ Vh1  # (r1, C_out*kH*kW)

        # === TT-SVD Step 2: Decompose spatial + output ===
        # Reshape residual to (r1*kH*kW, C_out)
        residual = residual.reshape(r1, c_out, kh, kw)
        residual = residual.permute(0, 2, 3, 1).reshape(r1 * kh * kw, c_out)

        U2, S2, Vh2 = torch.linalg.svd(residual, full_matrices=False)

        # Truncate to rank r2
        U2 = U2[:, :r2]          # (r1*kH*kW, r2)
        S2 = S2[:r2]             # (r2,)
        Vh2 = Vh2[:r2, :]        # (r2, C_out)

        # Absorb singular values into Vh2
        core3 = torch.diag(S2) @ Vh2  # (r2, C_out)

        # === Build the three replacement layers ===

        # Layer 1: Pointwise (1×1)  --  input channel reduction
        # U1 shape: (C_in, r1) → conv weight: (r1, C_in, 1, 1)
        layer1 = nn.Conv2d(c_in, r1, kernel_size=1, bias=False)
        layer1.weight.data = U1.t().unsqueeze(-1).unsqueeze(-1).contiguous()

        # Layer 2: Spatial conv (kH×kW) on reduced dimensions
        # U2 shape: (r1*kH*kW, r2) → reshape to (r2, r1, kH, kW) for conv
        core2 = U2.reshape(r1, kh, kw, r2).permute(3, 0, 1, 2)  # (r2, r1, kH, kW)
        layer2 = nn.Conv2d(
            r1, r2,
            kernel_size=(kh, kw),
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            bias=False,
        )
        layer2.weight.data = core2.contiguous()

        # Layer 3: Pointwise (1×1)  --  output channel expansion
        # core3 shape: (r2, C_out) → conv weight: (C_out, r2, 1, 1)
        layer3 = nn.Conv2d(r2, c_out, kernel_size=1, bias=conv_layer.bias is not None)
        layer3.weight.data = core3.t().unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            layer3.bias.data = conv_layer.bias.data.clone()

        decomposed = nn.Sequential(layer1, layer2, layer3)
        decomposed = decomposed.to(self.device)

        # Log compression stats
        orig_params = c_out * c_in * kh * kw
        new_params = c_in * r1 + r1 * r2 * kh * kw + r2 * c_out
        logger.debug(
            f"  Compression: {orig_params:,} → {new_params:,} params "
            f"({orig_params / max(new_params, 1):.1f}×)"
        )

        return decomposed

    @torch.no_grad()
    def _decompose_pointwise(self, conv_layer: nn.Conv2d, rank: int) -> nn.Sequential:
        """
        Special case: decompose a 1×1 conv using a single SVD.

        1×1 conv weight is essentially a matrix (C_out, C_in), so we use
        standard matrix SVD: W ≈ U · S · V^T, truncated to given rank.

        Produces two pointwise layers:
            Conv2d(C_in, rank, 1×1) → Conv2d(rank, C_out, 1×1)
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in = weight.shape[0], weight.shape[1]

        W_2d = weight.reshape(c_out, c_in)
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        layer1 = nn.Conv2d(c_in, rank, kernel_size=1, bias=False)
        layer1.weight.data = Vh.unsqueeze(-1).unsqueeze(-1).contiguous()

        layer2 = nn.Conv2d(rank, c_out, kernel_size=1, bias=conv_layer.bias is not None)
        layer2.weight.data = (U * S.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            layer2.bias.data = conv_layer.bias.data.clone()

        return nn.Sequential(layer1, layer2).to(self.device)

    def get_compression_ratio(self, conv_layer: nn.Conv2d, rank_ratio: Optional[float] = None) -> float:
        """Calculate theoretical compression ratio."""
        c_out, c_in, kh, kw = conv_layer.weight.shape
        r1, r2 = self.estimate_ranks(conv_layer.weight.data, rank_ratio)
        orig = c_out * c_in * kh * kw
        compressed = c_in * r1 + r1 * r2 * kh * kw + r2 * c_out
        return orig / max(compressed, 1)
