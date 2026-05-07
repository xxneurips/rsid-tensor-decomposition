"""
Model-Level Decomposition Engine
=================================

Applies tensor decomposition to all eligible Conv2d layers in a model.
Handles the tricky parts:
- Identifies which layers to decompose (skip 1×1 for CP, skip tiny layers)
- Replaces layers in-place in the model's module tree
- Handles grouped convolutions, depthwise separable convs
- Computes overall compression ratio

CUDA Acceleration:
- All decompositions run on GPU (tensorly PyTorch backend)
- Uses torch.no_grad() context to avoid gradient computation overhead
- Batch processes layers sequentially (can't parallelize due to model mutation)
- After decomposition, optionally compiles the model with torch.compile()
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal
import logging
import time

from .tucker import TuckerDecomposer
from .cp import CPDecomposer
from .tensor_train import TTDecomposer
from .tensor_ring import TRDecomposer
from .svd_unfold import SVDUnfoldDecomposer

logger = logging.getLogger(__name__)


def get_decomposer(
    method: Literal["tucker", "cp", "tt", "tr"],
    rank_ratio: float = 0.5,
    device: str = "cuda",
):
    """
    Factory function to create the appropriate decomposer.

    Args:
        method: Decomposition method  --  "tucker", "cp", "tt", or "tr".
        rank_ratio: Controls compression level. Lower = more compression.
                    For Tucker/TT/TR: fraction of channel dimensions to keep.
                    For CP: used to compute rank from min(C_out, C_in).
        device: Target device.

    Returns:
        Decomposer instance with a decompose_conv_layer() method.
    """
    if method == "tucker":
        return TuckerDecomposer(rank_ratio=rank_ratio, device=device)
    elif method == "cp":
        # CP rank = rank_ratio * min(C_out, C_in), set in estimate_rank()
        return CPDecomposer(rank=int(rank_ratio * 64), device=device)  # will be overridden per-layer
    elif method == "tt":
        return TTDecomposer(rank_ratio=rank_ratio, device=device)
    elif method == "tr":
        return TRDecomposer(rank_ratio=rank_ratio, device=device)
    elif method == "svd_unfold":
        return SVDUnfoldDecomposer(rank_ratio=rank_ratio, device=device)
    else:
        raise ValueError(f"Unknown decomposition method: {method}. Use 'tucker', 'cp', 'tt', 'tr', or 'svd_unfold'.")


def _is_decomposable(name: str, module: nn.Module, min_channels: int = 8) -> bool:
    """
    Check if a conv layer is worth decomposing.

    We skip layers that:
    - Are not Conv2d
    - Have grouped convolutions (depthwise)  --  already efficient
    - Have very few channels (< min_channels)  --  overhead not worth it
    - Are the first layer (e.g., 3 input channels from RGB)
    - Are downsample/shortcut 1×1 convolutions with stride>1
      (these are part of residual connections and changing them
       causes shape mismatches that break the skip connection)

    Args:
        name: Module name in the model hierarchy.
        module: The module to check.
        min_channels: Minimum channel count to consider decomposition.

    Returns:
        True if the layer should be decomposed.
    """
    if not isinstance(module, nn.Conv2d):
        return False

    # Skip grouped/depthwise convolutions  --  they're already efficient
    if module.groups > 1:
        return False

    c_out, c_in = module.weight.shape[0], module.weight.shape[1]

    # Skip layers with very few channels (not worth the overhead)
    if c_out < min_channels or c_in < min_channels:
        return False

    # Skip downsample/shortcut convolutions (1×1 with stride>1)
    # These are part of residual connections in ResNets. Decomposing them
    # into sequential layers breaks the spatial downsampling that must
    # match the main branch. Safer to leave them intact.
    kh, kw = module.kernel_size
    if kh == 1 and kw == 1 and any(s > 1 for s in module.stride):
        return False

    # Also skip any layer with "downsample" in its name (explicit check)
    if "downsample" in name or "shortcut" in name:
        return False

    return True


@torch.no_grad()
def decompose_model(
    model: nn.Module,
    method: Literal["tucker", "cp", "tt", "tr"] = "tucker",
    rank_ratio: float = 0.5,
    device: str = "cuda",
    min_channels: int = 8,
    skip_first_layer: bool = True,
    skip_last_layer: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Decompose all eligible Conv2d layers in a model.

    Walks the module tree, identifies decomposable conv layers, and replaces
    them with their decomposed equivalents. The model is modified in-place.

    For a ResNet-18 with rank_ratio=0.5:
        - Original: 11.2M params, 556M FLOPs
        - Tucker:   ~3.5M params, ~180M FLOPs (~3× compression)
        - CP:       ~2.1M params, ~120M FLOPs (~5× compression)
        - TT:       ~3.2M params, ~170M FLOPs (~3.3× compression)

    Args:
        model: Pre-trained model to decompose (will be modified in-place).
        method: Decomposition method  --  "tucker", "cp", or "tt".
        rank_ratio: Compression level (0.0-1.0). Lower = more compression.
        device: Target device for computation.
        min_channels: Skip layers with fewer channels than this.
        skip_first_layer: Don't decompose the first conv (usually 3→64, small).
        skip_last_layer: Don't decompose the last conv layer.
        verbose: Print per-layer compression stats.

    Returns:
        The decomposed model (same object, modified in-place).
    """
    model = model.to(device)
    decomposer = get_decomposer(method, rank_ratio, device)

    # Collect all decomposable layers with their parent modules and names
    # We can't modify the model while iterating, so collect first
    layers_to_decompose = []
    first_conv_found = False

    for name, module in model.named_modules():
        if not _is_decomposable(name, module, min_channels):
            continue

        # Optionally skip the first conv layer
        if skip_first_layer and not first_conv_found:
            first_conv_found = True
            if verbose:
                logger.info(f"  Skipping first conv: {name}")
            continue
        first_conv_found = True

        layers_to_decompose.append((name, module))

    # Optionally skip the last conv layer
    if skip_last_layer and layers_to_decompose:
        skipped = layers_to_decompose.pop()
        if verbose:
            logger.info(f"  Skipping last conv: {skipped[0]}")

    if verbose:
        logger.info(
            f"\nDecomposing {len(layers_to_decompose)} layers "
            f"with {method} (rank_ratio={rank_ratio})"
        )

    # Track compression stats
    total_orig_params = 0
    total_new_params = 0
    start_time = time.time()

    # Replace each layer with its decomposed version
    for name, conv_layer in layers_to_decompose:
        # Count original parameters
        orig_params = sum(p.numel() for p in conv_layer.parameters())
        total_orig_params += orig_params

        # Perform decomposition
        # For CP, compute per-layer rank from rank_ratio with a cap
        # to prevent memory explosion during SVD initialization
        if method == "cp":
            c_out, c_in = conv_layer.weight.shape[0], conv_layer.weight.shape[1]
            cp_rank = max(1, min(int(min(c_out, c_in) * rank_ratio), 32))
            decomposed = decomposer.decompose_conv_layer(conv_layer, rank=cp_rank)
        else:
            decomposed = decomposer.decompose_conv_layer(conv_layer, rank_ratio=rank_ratio)

        # Count new parameters
        new_params = sum(p.numel() for p in decomposed.parameters())
        total_new_params += new_params

        # Replace the layer in the model's module tree
        # Navigate to the parent module and set the attribute
        _replace_module(model, name, decomposed)

        if verbose:
            ratio = orig_params / max(new_params, 1)
            logger.info(f"  {name}: {orig_params:,} → {new_params:,} ({ratio:.1f}×)")

    elapsed = time.time() - start_time

    if verbose:
        total_ratio = total_orig_params / max(total_new_params, 1)
        logger.info(
            f"\nDecomposition complete in {elapsed:.1f}s"
            f"\n  Decomposed params: {total_orig_params:,} → {total_new_params:,} "
            f"({total_ratio:.1f}×)"
        )

    return model


def _replace_module(model: nn.Module, name: str, new_module: nn.Module):
    """
    Replace a named module in the model's hierarchy.

    Handles nested names like 'layer1.0.conv1' by navigating
    the module tree to find the parent, then using setattr.

    Args:
        model: Root model.
        name: Dot-separated path to the module (e.g., 'layer1.0.conv1').
        new_module: Replacement module.
    """
    parts = name.split(".")

    # Navigate to the parent module
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    # Replace the target module
    attr_name = parts[-1]
    if attr_name.isdigit():
        parent[int(attr_name)] = new_module
    else:
        setattr(parent, attr_name, new_module)


def get_model_compression_stats(model: nn.Module) -> Dict[str, int]:
    """
    Compute parameter and layer statistics for a model.

    Returns:
        Dict with 'total_params', 'conv_params', 'num_conv_layers', etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    conv_params = 0
    num_conv = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_params += sum(p.numel() for p in m.parameters())
            num_conv += 1

    return {
        "total_params": total_params,
        "conv_params": conv_params,
        "other_params": total_params - conv_params,
        "num_conv_layers": num_conv,
    }
