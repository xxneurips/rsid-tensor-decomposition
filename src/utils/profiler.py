"""
Model Profiling Utilities
==========================

Measures model efficiency without requiring real edge hardware:
- Parameter count and model file size
- FLOPs (multiply-accumulate operations)
- CPU inference latency (single-thread, proxy for edge devices)
- GPU inference latency (CUDA event timing for precise measurement)
- Peak memory usage during inference
- Throughput (images per second)

All measurements are automated and produce publication-ready tables.
"""

import torch
import torch.nn as nn
import os
import tempfile
import time
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cuda",
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> Dict[str, float]:
    """
    Comprehensive model profiling  --  all metrics in one call.

    Args:
        model: Model to profile.
        input_size: Input tensor shape (batch, channels, height, width).
        device: Device for profiling.
        warmup_runs: Warmup iterations before timing (JIT compilation, cache).
        timed_runs: Number of timed iterations (averaged for latency).

    Returns:
        Dict with all profiling metrics:
            'total_params': Total parameter count
            'total_params_m': Total params in millions
            'trainable_params': Trainable parameter count
            'model_size_mb': Saved model file size in MB
            'flops': Estimated FLOPs (multiply-accumulate)
            'flops_m': FLOPs in millions
            'cpu_latency_ms': CPU inference latency (ms)
            'gpu_latency_ms': GPU inference latency (ms)
            'peak_memory_mb': Peak GPU memory during inference (MB)
            'throughput_gpu': Images per second on GPU (batch=64)
    """
    model = model.to(device).eval()

    results = {}

    # === Parameter count ===
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["total_params"] = total_params
    results["total_params_m"] = total_params / 1e6
    results["trainable_params"] = trainable_params

    # === Model file size ===
    # Save to a temp file and measure size. NamedTemporaryFile with delete=True
    # holds the handle open on Windows, so torch.save fails to write to it; use
    # mkstemp + manual cleanup instead.
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp_path)
        results["model_size_mb"] = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # === FLOPs estimation ===
    results["flops"], results["flops_m"] = _count_flops(model, input_size, device)

    # === GPU latency ===
    results["gpu_latency_ms"] = profile_latency(
        model, input_size, device=device,
        warmup_runs=warmup_runs, timed_runs=timed_runs,
    )

    # === CPU latency (single thread  --  proxy for edge devices) ===
    results["cpu_latency_ms"] = profile_latency(
        model, input_size, device="cpu",
        warmup_runs=max(warmup_runs // 5, 2),
        timed_runs=max(timed_runs // 5, 10),
    )

    # === Peak memory ===
    results["peak_memory_mb"] = _measure_peak_memory(model, input_size, device)

    # === Throughput ===
    results["throughput_gpu"] = _measure_throughput(model, input_size, device)

    return results


def profile_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cuda",
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> float:
    """
    Measure inference latency with precise timing.

    For GPU: uses torch.cuda.Event for sub-millisecond precision.
             CUDA events are hardware timestamps from the GPU clock,
             independent of CPU-GPU synchronization noise.

    For CPU: uses time.perf_counter with single-thread torch.
             Sets num_threads=1 to simulate edge devices (ARM cores
             are typically single-threaded for inference).

    Args:
        model: Model to time.
        input_size: Input tensor shape.
        device: 'cuda' or 'cpu'.
        warmup_runs: Warmup iterations.
        timed_runs: Timed iterations.

    Returns:
        Average latency in milliseconds.
    """
    model = model.to(device).eval()
    dummy = torch.randn(input_size, device=device)

    if "cuda" in str(device):
        return _gpu_latency(model, dummy, warmup_runs, timed_runs)
    else:
        return _cpu_latency(model, dummy, warmup_runs, timed_runs)


@torch.no_grad()
def _gpu_latency(
    model: nn.Module, dummy: torch.Tensor,
    warmup: int, runs: int,
) -> float:
    """GPU latency using CUDA events (most accurate method)."""
    # Warmup  --  fills CUDA caches, triggers JIT compilation
    for _ in range(warmup):
        model(dummy)
    torch.cuda.synchronize()

    # Timed runs with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(runs):
        start_event.record()
        model(dummy)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))  # ms

    # Return mean, excluding outliers (top/bottom 10%)
    latencies.sort()
    trim = max(1, runs // 10)
    trimmed = latencies[trim:-trim] if trim < len(latencies) // 2 else latencies
    return sum(trimmed) / len(trimmed)


@torch.no_grad()
def _cpu_latency(
    model: nn.Module, dummy: torch.Tensor,
    warmup: int, runs: int,
) -> float:
    """CPU latency using perf_counter (single-thread)."""
    # Force single thread to simulate edge device
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    model = model.cpu().eval()
    dummy = dummy.cpu()

    # Warmup
    for _ in range(warmup):
        model(dummy)

    # Timed runs
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        model(dummy)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    torch.set_num_threads(prev_threads)

    # Trimmed mean
    latencies.sort()
    trim = max(1, runs // 10)
    trimmed = latencies[trim:-trim] if trim < len(latencies) // 2 else latencies
    return sum(trimmed) / len(trimmed)


def _count_flops(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str,
) -> Tuple[int, float]:
    """
    Count FLOPs using ptflops library.

    Falls back to manual counting if ptflops is not available.

    Returns:
        (total_flops, total_flops_millions)
    """
    try:
        from ptflops import get_model_complexity_info

        model_cpu = model.cpu().eval()
        # ptflops expects (C, H, W) without batch dimension
        spatial_size = input_size[1:]

        macs, _ = get_model_complexity_info(
            model_cpu, spatial_size,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        model.to(device)
        # MACs ≈ FLOPs/2 (one MAC = one multiply + one add)
        # Convention: report MACs as "FLOPs" (common in literature)
        return int(macs), macs / 1e6

    except Exception as e:
        logger.warning(f"ptflops failed: {e}. Using manual FLOPs estimation.")
        model.to(device)
        return _manual_flop_count(model, input_size)


def _manual_flop_count(
    model: nn.Module, input_size: Tuple[int, ...]
) -> Tuple[int, float]:
    """
    Manually count FLOPs by estimating per-layer MACs.

    For Conv2d: MACs = C_out × C_in × kH × kW × H_out × W_out / groups
    For Linear: MACs = in_features × out_features
    """
    total_macs = 0

    # Use hooks to get output sizes
    hooks = []
    layer_macs = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Output spatial dimensions
            h_out, w_out = output.shape[2], output.shape[3]
            macs = (
                module.weight.shape[0]  # C_out
                * module.weight.shape[1]  # C_in / groups
                * module.weight.shape[2]  # kH
                * module.weight.shape[3]  # kW
                * h_out * w_out
            )
            layer_macs.append(macs)
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
            layer_macs.append(macs)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    # Run one forward pass to trigger hooks
    dummy = torch.randn(input_size, device=next(model.parameters()).device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_macs = sum(layer_macs)
    return total_macs, total_macs / 1e6


@torch.no_grad()
def _measure_peak_memory(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str,
) -> float:
    """
    Measure peak GPU memory during inference.

    Returns:
        Peak memory in MB.
    """
    if "cpu" in str(device):
        return 0.0  # Can't measure CPU memory this way

    model = model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    dummy = torch.randn(input_size, device=device)
    model(dummy)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return peak_mb


@torch.no_grad()
def _measure_throughput(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str,
    duration_seconds: float = 5.0,
) -> float:
    """
    Measure throughput (images/second) on GPU.

    Runs inference continuously for `duration_seconds` and counts
    total images processed. Uses batch_size=64 for realistic throughput.

    Returns:
        Images per second.
    """
    if "cpu" in str(device):
        return 0.0

    model = model.to(device).eval()
    batch_size = 64
    batch_input = (batch_size, *input_size[1:])
    dummy = torch.randn(batch_input, device=device)

    # Warmup
    for _ in range(5):
        model(dummy)
    torch.cuda.synchronize()

    # Measure
    total_images = 0
    start = time.time()
    while time.time() - start < duration_seconds:
        model(dummy)
        total_images += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return total_images / elapsed


def format_profile_table(profiles: Dict[str, Dict[str, float]]) -> str:
    """
    Format profiling results as a markdown table for the paper.

    Args:
        profiles: Dict mapping model_name → profile_results.

    Returns:
        Markdown-formatted table string.
    """
    header = (
        "| Model | Params (M) | FLOPs (M) | Size (MB) | "
        "CPU (ms) | GPU (ms) | Memory (MB) | Imgs/s |"
    )
    sep = "|" + "|".join(["---"] * 8) + "|"

    rows = [header, sep]
    for name, p in profiles.items():
        row = (
            f"| {name} | {p['total_params_m']:.2f} | {p['flops_m']:.1f} | "
            f"{p['model_size_mb']:.1f} | {p['cpu_latency_ms']:.1f} | "
            f"{p['gpu_latency_ms']:.2f} | {p['peak_memory_mb']:.1f} | "
            f"{p['throughput_gpu']:.0f} |"
        )
        rows.append(row)

    return "\n".join(rows)
