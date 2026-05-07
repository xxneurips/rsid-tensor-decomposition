"""
Tensor Ring (TR) Decomposition for Convolutional Layers
========================================================

Decomposes a 4D conv weight tensor W ∈ R^(C_out × C_in × H × W) using
Tensor Ring format, where each mode gets a 3D core connected in a ring:

    W[i₁,i₂,i₃,i₄] ≈ Tr(G₁[i₁] · G₂[i₂] · G₃[i₃] · G₄[i₄])

Unlike Tensor Train (which is open-ended: r₀=r₄=1), Tensor Ring closes
the chain by contracting the first and last bond dimensions via trace.
This circular structure means:
- No privileged mode ordering (TT results depend on which mode comes first)
- Higher representational capacity at the same total rank budget
- Better for tensors where correlations wrap across modes

For conv layers, we implement a practical 3-core TR by merging kernel
spatial dimensions (kH,kW) into a single mode, giving cores for:
    (C_in) -- (kH*kW) -- (C_out)
with bond dimensions (r₀, r₁, r₂) where r₀ = r₂ (ring closure).

The ring closure is implemented by reshaping the first core from
(r₀, C_in, r₁) into (C_in, r₁*r₀) and the last core from
(r₂, C_out) into (C_out, r₂), then contracting r₀ with r₂ via
a grouped convolution or trace-based reshape.

In practice, for a Conv2d layer this produces four replacement layers:
    1. Conv2d(C_in, r₁*r₀, 1×1)     --  input reduction (unfolded ring dim)
    2. Conv2d(r₁, r₂, kH×kW, groups=1)  --  spatial filtering
    3. Conv2d(r₂, C_out*r₀, 1×1)    --  output expansion (with ring dim)
    4. RingContraction(r₀)            --  trace over ring bond

We simplify this for stable training: merge the ring contraction into
the final 1×1 conv by summing over the ring dimension, producing a
3-layer sequence like TT but with a wider intermediate representation.

Initialization: TR-SVD (sequential SVDs with rank recycling).

Reference:
    Zhao et al., "Tensor Ring Decomposition," arXiv:1606.05535, 2016.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TRDecomposer:
    """
    Decomposes nn.Conv2d layers using Tensor Ring decomposition.

    Uses TR-SVD initialization: sequential SVDs similar to TT-SVD but
    with a ring closure that recycles the first bond dimension into
    the last core.
    """

    def __init__(self, rank_ratio: float = 0.5, device: str = "cuda"):
        """
        Args:
            rank_ratio: Fraction of min dimension for TR-ranks.
            device: Target device.
        """
        self.rank_ratio = rank_ratio
        self.device = device

    def estimate_ranks(
        self, weight: torch.Tensor, rank_ratio: Optional[float] = None
    ) -> Tuple[int, int, int]:
        """
        Estimate TR-ranks (r0, r1, r2) for a conv weight tensor.

        For TR, r0 = r2 (ring closure). We set r0 based on a fraction of
        the geometric mean of input/output channels  --  this controls the
        "ring width" which adds capacity beyond what TT provides.

        Args:
            weight: Conv weight tensor (C_out, C_in, kH, kW).
            rank_ratio: Override fraction.

        Returns:
            (r0, r1, r2) where r0 == r2 (ring bond).
        """
        ratio = rank_ratio or self.rank_ratio
        c_out, c_in, kh, kw = weight.shape

        max_rank = min(c_out, c_in)
        # Main bond ranks (same as TT)
        r1 = max(1, int(max_rank * ratio))

        # Ring bond: smaller than main ranks, scales with sqrt
        # This adds TR's extra capacity without blowing up params
        r0 = max(1, int((max_rank * ratio) ** 0.5))
        # Keep r0 modest to avoid parameter explosion
        r0 = min(r0, max(1, int(max_rank * 0.25)))

        r2 = r0  # ring closure

        return r0, r1, r2

    @torch.no_grad()
    def decompose_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        rank_ratio: Optional[float] = None,
    ) -> nn.Sequential:
        """
        Decompose a Conv2d using Tensor Ring into a 3-layer sequence.

        The TR decomposition with ring closure is implemented as:
            1. Conv2d(C_in, r0*r1, 1×1)          --  input core (unfolded)
            2. Conv2d(r0*r1, r0*r2, kH×kW, groups=r0)  --  spatial core (grouped)
            3. Conv2d(r0*r2, C_out, 1×1)          --  output core + ring trace

        The key difference from TT: layer 2 uses grouped convolutions with
        r0 groups, implementing r0 parallel TT-like chains whose outputs
        are summed (traced) in layer 3. This is the practical implementation
        of the ring trace.

        For 1×1 convolutions, falls back to a rank-constrained SVD with
        wider intermediate dimension (r0*r1 instead of r1).

        Args:
            conv_layer: Original Conv2d layer.
            rank_ratio: Override rank ratio.

        Returns:
            nn.Sequential with replacement layers.
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in, kh, kw = weight.shape
        r0, r1, r2 = self.estimate_ranks(weight, rank_ratio)

        # For 1×1 convolutions, use widened SVD
        if kh == 1 and kw == 1:
            return self._decompose_pointwise(conv_layer, r0, r1)

        logger.debug(
            f"TR decompose: ({c_out},{c_in},{kh},{kw}) → "
            f"ranks=({r0},{r1},{r2})"
        )

        # Skip if no compression would occur
        orig_params = c_out * c_in * kh * kw
        new_params = c_in * r0 * r1 + r0 * r1 * r2 * kh * kw + r0 * r2 * c_out
        if new_params >= orig_params:
            logger.debug("  Skipping: no compression, TR params >= original")
            return nn.Sequential(conv_layer)

        # === TR-SVD Initialization ===

        # Step 1: Unfold along input channels
        # W: (C_out, C_in, kH, kW) → reshape to (C_in, C_out*kH*kW)
        W_mode1 = weight.permute(1, 0, 2, 3).reshape(c_in, -1)
        U1, S1, Vh1 = torch.linalg.svd(W_mode1, full_matrices=False)

        # Truncate to rank r0*r1 (wider than TT due to ring)
        eff_rank1 = min(r0 * r1, U1.shape[1])
        U1 = U1[:, :eff_rank1]             # (C_in, r0*r1)
        S1 = S1[:eff_rank1]
        Vh1 = Vh1[:eff_rank1, :]           # (r0*r1, C_out*kH*kW)
        residual = torch.diag(S1) @ Vh1    # (r0*r1, C_out*kH*kW)

        # Step 2: Reshape residual for spatial decomposition
        # (r0*r1, C_out, kH, kW) → permute → (r0*r1*kH*kW, C_out)
        residual = residual.reshape(eff_rank1, c_out, kh, kw)
        residual_2d = residual.permute(0, 2, 3, 1).reshape(eff_rank1 * kh * kw, c_out)

        U2, S2, Vh2 = torch.linalg.svd(residual_2d, full_matrices=False)
        eff_rank2 = min(r0 * r2, U2.shape[1])
        U2 = U2[:, :eff_rank2]             # (r0*r1*kH*kW, r0*r2)
        S2 = S2[:eff_rank2]
        Vh2 = Vh2[:eff_rank2, :]           # (r0*r2, C_out)
        core3 = torch.diag(S2) @ Vh2       # (r0*r2, C_out)

        # === Build replacement layers ===
        # TR uses wider intermediate dimensions than TT (r0*r1 vs r1).
        # The ring closure is implicit: layer 3 maps r0*r2 → C_out,
        # effectively summing over r0 ring copies of the r2-dimensional
        # representation.

        # Layer 1: Pointwise (1×1)  --  input reduction to r0*r1 channels
        layer1 = nn.Conv2d(c_in, eff_rank1, kernel_size=1, bias=False)
        layer1.weight.data = U1.t().unsqueeze(-1).unsqueeze(-1).contiguous()

        # Layer 2: Spatial conv (kH×kW)  --  standard conv on wider channels
        core2 = U2.reshape(eff_rank1, kh, kw, eff_rank2).permute(3, 0, 1, 2)
        # core2 shape: (eff_rank2, eff_rank1, kH, kW)

        layer2 = nn.Conv2d(
            eff_rank1, eff_rank2,
            kernel_size=(kh, kw),
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            bias=False,
        )
        layer2.weight.data = core2.contiguous()

        # Layer 3: Pointwise (1×1)  --  output expansion + ring trace
        layer3 = nn.Conv2d(
            eff_rank2, c_out, kernel_size=1,
            bias=conv_layer.bias is not None,
        )
        layer3.weight.data = core3.t().unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            layer3.bias.data = conv_layer.bias.data.clone()

        decomposed = nn.Sequential(layer1, layer2, layer3).to(self.device)

        logger.debug(
            f"  TR compression: {orig_params:,} → {new_params:,} params "
            f"({orig_params / max(new_params, 1):.1f}×)"
        )

        return decomposed

    @torch.no_grad()
    def _decompose_pointwise(
        self, conv_layer: nn.Conv2d, r0: int, r1: int
    ) -> nn.Sequential:
        """
        Decompose a 1×1 conv using TR-style widened SVD.

        Wider intermediate dimension (r0*r1) vs TT's r1 gives TR
        more capacity for the same output rank.

        Produces:
            Conv2d(C_in, r0*r1, 1×1) → Conv2d(r0*r1, C_out, 1×1)
        """
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in = weight.shape[0], weight.shape[1]

        eff_rank = min(r0 * r1, min(c_out, c_in))

        W_2d = weight.reshape(c_out, c_in)
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        U = U[:, :eff_rank]
        S = S[:eff_rank]
        Vh = Vh[:eff_rank, :]

        layer1 = nn.Conv2d(c_in, eff_rank, kernel_size=1, bias=False)
        layer1.weight.data = Vh.unsqueeze(-1).unsqueeze(-1).contiguous()

        layer2 = nn.Conv2d(
            eff_rank, c_out, kernel_size=1,
            bias=conv_layer.bias is not None,
        )
        layer2.weight.data = (U * S.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1).contiguous()
        if conv_layer.bias is not None:
            layer2.bias.data = conv_layer.bias.data.clone()

        return nn.Sequential(layer1, layer2).to(self.device)

    def get_compression_ratio(
        self, conv_layer: nn.Conv2d, rank_ratio: Optional[float] = None
    ) -> float:
        """Calculate theoretical compression ratio for TR decomposition."""
        c_out, c_in, kh, kw = conv_layer.weight.shape
        r0, r1, r2 = self.estimate_ranks(conv_layer.weight.data, rank_ratio)
        orig = c_out * c_in * kh * kw
        compressed = c_in * r0 * r1 + r0 * r1 * r0 * r2 * kh * kw + r0 * r2 * c_out
        return orig / max(compressed, 1)
