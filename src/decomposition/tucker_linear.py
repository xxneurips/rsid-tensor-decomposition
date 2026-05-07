"""
Tucker-3 decomposition for transformer Linear layers (the missing 2x2 cell).

For a Linear layer with weight W of shape (out_dim, in_dim), we reshape it as
a 3D tensor by exposing the multi-head structure:

    W_3D shape = (n_heads_eff, head_dim_eff, in_dim)

where n_heads_eff * head_dim_eff = out_dim. Heuristically, we choose
n_heads_eff = min(out_dim // 64, 32) and head_dim_eff = out_dim / n_heads_eff
so head_dim_eff is at least 32 for non-tiny layers.

We then apply Tucker-3 with reduced ranks on modes 0 and 1 (heads, head_dim)
and full rank on mode 2 (input dim). The reconstructed weight is materialised
back to a single Linear (out, in), which keeps the runtime overhead the same
as the original Linear (no chain of small ops).

This decomposition has TWO interacting per-mode ranks (R_h, R_d) like Tucker-2
on Conv, in contrast to single-rank truncated SVD. It is the cleanest
disambiguation experiment for "decomposition family vs architecture."
"""
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from typing import Optional
import logging

logger = logging.getLogger(__name__)
tl.set_backend("pytorch")


class TuckerLinearDecomposer:
    """Tucker-3 on transformer Linear weights via multi-head reshape."""

    def __init__(self, rank_ratio: float = 0.3, device: str = "cuda"):
        self.rank_ratio = rank_ratio
        self.device = device

    def _block_factor(self, dim: int) -> int:
        """Pick number of heads for the reshape. Try to keep head_dim ≥ 32."""
        for n in (32, 24, 16, 12, 8, 6, 4, 3, 2):
            if dim % n == 0 and dim // n >= 32:
                return n
        return 1

    @torch.no_grad()
    def decompose_linear_layer(
        self, lin: nn.Linear, rank_ratio: Optional[float] = None
    ) -> nn.Linear:
        """Decompose then re-materialise (returns a new Linear with reconstructed weight)."""
        ratio = rank_ratio or self.rank_ratio
        W = lin.weight.data.to(self.device).float()  # (out, in)
        out_dim, in_dim = W.shape

        n_heads = self._block_factor(out_dim)
        if n_heads == 1:
            # Degenerates to Tucker-2 on a 2D matrix == SVD; not the experiment we want.
            # Fall back to identity (skip decomposition for this layer).
            logger.debug(f"  TuckerLinear: skipping {lin} (no good multi-head reshape)")
            return lin
        head_dim = out_dim // n_heads
        R_h = max(1, int(round(n_heads * ratio)))
        R_d = max(1, int(round(head_dim * ratio)))

        if R_h >= n_heads and R_d >= head_dim:
            return lin

        W3 = W.reshape(n_heads, head_dim, in_dim)
        try:
            (core, factors), _ = partial_tucker(
                W3, modes=[0, 1], rank=[R_h, R_d], init="svd", n_iter_max=0)
        except Exception as e:
            logger.warning(f"  TuckerLinear: partial_tucker failed: {e}")
            return lin
        # Reconstruct: contract core (R_h, R_d, in) with factors[0] (n_heads, R_h) and factors[1] (head_dim, R_d)
        # W3_recon[h, d, i] = sum_{rh, rd} factors[0][h, rh] * factors[1][d, rd] * core[rh, rd, i]
        recon3 = torch.einsum("hr,ds,rsi->hdi", factors[0], factors[1], core)
        recon = recon3.reshape(out_dim, in_dim)

        new_lin = nn.Linear(in_dim, out_dim, bias=lin.bias is not None).to(self.device)
        new_lin.weight.data = recon.to(W.dtype)
        if lin.bias is not None:
            new_lin.bias.data = lin.bias.data.clone()

        orig_params = out_dim * in_dim
        new_params = (n_heads * R_h) + (head_dim * R_d) + (R_h * R_d * in_dim)
        logger.debug(
            f"TuckerLinear: ({out_dim},{in_dim}) heads={n_heads} ranks=({R_h},{R_d}) "
            f"params {orig_params} -> {new_params} ({new_params/orig_params:.2%})"
        )
        return new_lin


def tucker_decompose_linear_layers(model, rank_ratio=0.3, device="cuda",
                                    skip_first=True, skip_last=True, min_dim=64):
    """Walk the model and replace Linear layers with Tucker-3 decomposed versions."""
    decomposer = TuckerLinearDecomposer(rank_ratio=rank_ratio, device=device)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_first and linears:
        linears = linears[1:]
    if skip_last and linears:
        linears = linears[:-1]
    n_done = 0
    for name, mod in linears:
        if min(mod.weight.shape) < min_dim:
            continue
        new_mod = decomposer.decompose_linear_layer(mod, rank_ratio=rank_ratio)
        if new_mod is not mod:
            # Replace in parent
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_mod)
            n_done += 1
    logger.info(f"TuckerLinear decomposed {n_done}/{len(linears)} Linear layers")
    return model
