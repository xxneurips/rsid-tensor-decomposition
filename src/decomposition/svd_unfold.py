"""
SVD-Unfold Decomposition for Convolutional Layers
==================================================

Decomposes a 4D conv weight tensor W ∈ R^(C_out × C_in × kH × kW) by mode-1
unfolding to a 2D matrix W_(1) ∈ R^(C_out × (C_in·kH·kW)) and applying truncated
SVD with a SINGLE rank parameter r:

    W_(1) ≈ U_r · Σ_r · V_r^T

This is the direct analogue of the truncated-SVD compression used for transformer
Linear layers in our DeiT-Small experiments. By construction, this decomposition
has only ONE rank parameter (unlike Tucker-2's two interacting per-mode ranks),
which is the variable we need for the cross-decomposition disambiguation
experiment in Appendix E (Hypothesis 1).

The decomposition replaces a single Conv2d with two sequential layers:
    1. k×k conv:   C_in → r, weight = (Σ_r · V_r^T) reshaped to (r, C_in, kH, kW)
    2. 1×1 conv:   r → C_out, weight = U_r of shape (C_out, r)

The first layer carries the spatial conv (preserves stride/padding/dilation/groups
of the original); the second is a pointwise channel expansion.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SVDUnfoldDecomposer:
    """
    Decomposes nn.Conv2d via mode-1 unfold + truncated SVD with one rank.

    This intentionally mirrors the SVD compression used on transformer Linear
    layers, providing the apples-to-apples comparison for the cross-decomposition
    asymmetry test.
    """

    def __init__(self, rank_ratio: float = 0.3, device: str = "cuda"):
        self.rank_ratio = rank_ratio
        self.device = device

    def estimate_rank(
        self, weight: torch.Tensor, rank_ratio: Optional[float] = None
    ) -> int:
        """Single rank for SVD: ⌈ratio · min(C_out, C_in·kH·kW)⌉."""
        ratio = rank_ratio or self.rank_ratio
        c_out, c_in, kh, kw = weight.shape
        full_rank = min(c_out, c_in * kh * kw)
        return max(1, int(round(c_out * ratio)))  # use C_out scale for parity with Tucker

    @torch.no_grad()
    def decompose_conv_layer(
        self,
        conv_layer: nn.Conv2d,
        rank_ratio: Optional[float] = None,
    ) -> nn.Sequential:
        """Decompose a Conv2d via SVD unfold; returns a 2-layer Sequential."""
        weight = conv_layer.weight.data.to(self.device)
        c_out, c_in, kh, kw = weight.shape
        rank = self.estimate_rank(weight, rank_ratio)

        # No compression possible
        if rank >= min(c_out, c_in * kh * kw):
            logger.debug("  SVD-unfold: rank >= full rank, skipping")
            return nn.Sequential(conv_layer)

        # Mode-1 unfolding: (C_out, C_in*kH*kW)
        W_mat = weight.reshape(c_out, c_in * kh * kw)

        try:
            U, S, Vt = torch.linalg.svd(W_mat.float(), full_matrices=False)
        except Exception as e:
            logger.warning(f"  SVD-unfold: torch.linalg.svd failed: {e}. Keeping original layer.")
            return nn.Sequential(conv_layer)

        U_r = U[:, :rank].contiguous()                       # (C_out, r)
        S_r = S[:rank]                                        # (r,)
        Vt_r = Vt[:rank, :].contiguous()                      # (r, C_in*kH*kW)

        # Absorb singular values into V^T side (could equally split as sqrt to both)
        SVt = S_r.unsqueeze(1) * Vt_r                         # (r, C_in*kH*kW)
        spatial_weight = SVt.reshape(rank, c_in, kh, kw).contiguous()  # (r, C_in, kH, kW)

        # Layer 1: k×k conv from C_in → r, carries the spatial filter
        spatial_layer = nn.Conv2d(
            in_channels=c_in,
            out_channels=rank,
            kernel_size=(kh, kw),
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=False,
        )
        spatial_layer.weight.data = spatial_weight.to(weight.dtype)

        # Layer 2: 1×1 conv from r → C_out, channel expansion
        expand_layer = nn.Conv2d(
            in_channels=rank,
            out_channels=c_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=conv_layer.bias is not None,
        )
        expand_layer.weight.data = U_r.unsqueeze(-1).unsqueeze(-1).contiguous().to(weight.dtype)
        if conv_layer.bias is not None:
            expand_layer.bias.data = conv_layer.bias.data.clone()

        decomposed = nn.Sequential(spatial_layer, expand_layer).to(self.device)

        orig_params = c_out * c_in * kh * kw
        new_params = rank * c_in * kh * kw + c_out * rank
        logger.debug(
            f"SVD-unfold: ({c_out},{c_in},{kh},{kw}) rank={rank} "
            f"params {orig_params} -> {new_params} ({new_params/orig_params:.2%})"
        )

        return decomposed
