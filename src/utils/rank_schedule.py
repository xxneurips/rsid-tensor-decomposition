"""
Rank Scheduling Strategies for RSID
=====================================

The rank schedule determines how aggressively we compress at each iteration.
Given n iterations going from start_ratio to end_ratio, the schedule
produces a list of rank_ratios [r₁, r₂, ..., rₙ] where each rᵢ is passed
to the decomposer as the rank_ratio parameter.

Three strategies:

1. Linear: Equal steps from start to end.
   - Simple, predictable
   - Example: [0.8, 0.6, 0.4, 0.2] for 4 iterations
   - Good default choice

2. Exponential: Geometric decay (compress more aggressively at the start).
   - Front-loads the compression when the model is most redundant
   - Example: [0.5, 0.25, 0.125, 0.0625] for 4 iterations with γ=0.5
   - Usually best for accuracy (our hypothesis)

3. Adaptive: Compress more only when accuracy recovers above a threshold.
   - Requires validation accuracy feedback during RSID
   - Most conservative  --  preserves accuracy best
   - Slower to converge (waits for recovery)
"""

import numpy as np
from typing import List, Literal


def get_rank_schedule(
    strategy: Literal["linear", "exponential", "adaptive"] = "exponential",
    start_ratio: float = 0.8,
    end_ratio: float = 0.2,
    num_iterations: int = 3,
    decay_factor: float = 0.5,
) -> List[float]:
    """
    Generate a rank schedule for RSID iterations.

    Args:
        strategy: Scheduling strategy  --  'linear', 'exponential', or 'adaptive'.
        start_ratio: Rank ratio for the first iteration (least compressed).
                     0.8 means keep 80% of channels.
        end_ratio: Target rank ratio for the final iteration (most compressed).
                   0.2 means keep 20% of channels → ~5× compression.
        num_iterations: Number of RSID iterations.
        decay_factor: For exponential strategy, the base of the geometric decay.
                      0.5 means each step halves the rank ratio.

    Returns:
        List of rank ratios [r₁, r₂, ..., rₙ], one per RSID iteration.
        Each value is between 0.0 and 1.0.

    Examples:
        >>> get_rank_schedule('linear', 0.8, 0.2, 4)
        [0.8, 0.6, 0.4, 0.2]

        >>> get_rank_schedule('exponential', 0.8, 0.1, 4, decay_factor=0.5)
        [0.8, 0.4, 0.2, 0.1]

        >>> get_rank_schedule('adaptive', 0.8, 0.2, 4)
        [0.8, 0.6, 0.4, 0.2]  # Same as linear; actual adaptation happens at runtime
    """
    if num_iterations < 1:
        raise ValueError("num_iterations must be >= 1")
    if num_iterations == 1:
        return [end_ratio]

    if strategy == "linear":
        return _linear_schedule(start_ratio, end_ratio, num_iterations)
    elif strategy == "exponential":
        return _exponential_schedule(start_ratio, end_ratio, num_iterations, decay_factor)
    elif strategy == "adaptive":
        # Adaptive schedule starts as linear; actual adaptation happens
        # in the RSID loop based on validation accuracy feedback
        return _linear_schedule(start_ratio, end_ratio, num_iterations)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'linear', 'exponential', or 'adaptive'.")


def _linear_schedule(start: float, end: float, n: int) -> List[float]:
    """
    Linear rank schedule: equal steps from start to end.

    rank_i = start - i * (start - end) / (n - 1)
    """
    return list(np.linspace(start, end, n))


def _exponential_schedule(
    start: float, end: float, n: int, decay: float
) -> List[float]:
    """
    Exponential rank schedule: geometric decay from start toward end.

    rank_i = start * decay^i, clamped to [end, start].

    The decay factor controls how fast we compress:
    - decay=0.5: halves each step (aggressive)
    - decay=0.7: slower decay (conservative)
    """
    schedule = []
    for i in range(n):
        ratio = start * (decay ** i)
        # Clamp to not go below the target end ratio
        ratio = max(ratio, end)
        schedule.append(ratio)

    # Ensure the last element is exactly end_ratio
    schedule[-1] = end
    return schedule


def adapt_schedule_step(
    current_schedule: List[float],
    step_index: int,
    val_acc: float,
    prev_val_acc: float,
    acc_threshold: float = 0.5,
) -> float:
    """
    Adaptive schedule adjustment for a single RSID step.

    Called during RSID to potentially adjust the next rank ratio based on
    how well the student recovered accuracy at the current step.

    Logic:
    - If accuracy dropped by more than `acc_threshold`% from previous step,
      use a less aggressive (higher) rank ratio than planned
    - If accuracy is stable, proceed with the planned schedule
    - If accuracy improved, use a more aggressive (lower) rank ratio

    Args:
        current_schedule: The planned rank schedule.
        step_index: Current RSID iteration index.
        val_acc: Validation accuracy after current step's distillation.
        prev_val_acc: Validation accuracy of the previous step's model.
        acc_threshold: Max acceptable accuracy drop (%) before backing off.

    Returns:
        Adjusted rank ratio for this step.
    """
    if step_index >= len(current_schedule):
        return current_schedule[-1]

    planned_ratio = current_schedule[step_index]
    acc_drop = prev_val_acc - val_acc

    if acc_drop > acc_threshold:
        # Accuracy dropped too much  --  back off (increase ratio by 20%)
        adjusted = min(planned_ratio * 1.2, 0.95)
        return adjusted
    elif acc_drop < -0.5:
        # Accuracy improved  --  can be more aggressive (decrease ratio by 10%)
        adjusted = max(planned_ratio * 0.9, 0.05)
        return adjusted
    else:
        # On track  --  use planned ratio
        return planned_ratio


def compute_target_compression(schedule: List[float]) -> float:
    """
    Estimate the overall compression ratio from a rank schedule.

    The final compression ratio depends on the last rank_ratio in the schedule.
    For a rank_ratio r, the approximate compression of conv params is ~1/r²
    (since both input and output channels are reduced by r).

    Args:
        schedule: List of rank ratios.

    Returns:
        Estimated compression ratio (e.g., 10.0 means 10× smaller).
    """
    final_ratio = schedule[-1]
    # Approximate: compression ≈ 1 / (final_ratio²)
    # This is a rough estimate; actual depends on layer sizes and which layers are decomposed
    return 1.0 / (final_ratio ** 2)
