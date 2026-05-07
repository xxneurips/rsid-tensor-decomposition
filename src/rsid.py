"""
RSID: Rank-Scheduled Iterative Distillation
=============================================

The core algorithm of this paper. RSID compresses a neural network through
multiple stages of progressively more aggressive tensor decomposition,
with knowledge distillation at each stage to recover accuracy.

Algorithm:
    Input: Teacher model M, rank schedule [r₁, r₂, ..., rₙ], training data

    current_model = M
    for i = 1 to n:
        1. Decompose current_model with rank_ratio = rᵢ
        2. student = decomposed copy of current_model
        3. Train student via KD from current_model (the "teacher" for this stage)
        4. current_model = student

    Output: Final compressed model

Key insight: each stage's "teacher" is the *previous stage's student*, not
the original full model. This means:
- The teacher-student gap is small at each step (easier to distill)
- Information is preserved incrementally (less catastrophic loss)
- Each student is a functional model (can stop early if needed)

CUDA Acceleration:
- torch.compile() on student model at each stage
- AMP (mixed precision) training throughout
- cuDNN benchmark mode enabled
- Efficient data loading with pinned memory
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Literal, Optional
import logging
import time

from .decomposition import decompose_model
from .distillation import DistillationTrainer, CombinedDistillationLoss
from .utils import get_rank_schedule, profile_model
from .utils.rank_schedule import adapt_schedule_step

logger = logging.getLogger(__name__)


class RSID:
    """
    Rank-Scheduled Iterative Distillation.

    Progressively compresses a model through multiple decomposition +
    distillation stages.
    """

    def __init__(
        self,
        # Decomposition config
        method: Literal["tucker", "cp", "tt"] = "tucker",
        schedule_strategy: Literal["linear", "exponential", "adaptive"] = "exponential",
        start_ratio: float = 0.8,
        end_ratio: float = 0.2,
        num_iterations: int = 3,

        # Distillation config
        kd_alpha: float = 0.5,
        kd_beta: float = 0.2,
        kd_temperature: float = 3.0,
        epochs_per_stage: int = 50,
        lr: float = 0.01,

        # Optimization config
        device: str = "cuda",
        use_amp: bool = True,
        compile_model: bool = True,
        feature_distillation: bool = False,
    ):
        """
        Args:
            method: Tensor decomposition method ('tucker', 'cp', 'tt').
            schedule_strategy: Rank scheduling ('linear', 'exponential', 'adaptive').
            start_ratio: Initial rank ratio (first iteration, least compressed).
            end_ratio: Final rank ratio (last iteration, most compressed).
            num_iterations: Number of decomposition+distillation stages.
            kd_alpha: Weight for KD soft loss.
            kd_beta: Weight for feature distillation loss.
            kd_temperature: Softmax temperature for KD.
            epochs_per_stage: Training epochs per distillation stage.
            lr: Learning rate for distillation training.
            device: CUDA device string.
            use_amp: Use automatic mixed precision.
            compile_model: Use torch.compile() for faster training.
            feature_distillation: Use feature-based KD (slower but better).
        """
        self.method = method
        self.device = device
        self.use_amp = use_amp
        self.compile_model = compile_model
        self.feature_distillation = feature_distillation
        self.epochs_per_stage = epochs_per_stage
        self.lr = lr

        # Generate rank schedule
        self.schedule = get_rank_schedule(
            strategy=schedule_strategy,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            num_iterations=num_iterations,
        )

        # Loss function config
        self.loss_fn = CombinedDistillationLoss(
            alpha=kd_alpha,
            beta=kd_beta if feature_distillation else 0.0,
            temperature=kd_temperature,
        )

        logger.info(
            f"RSID initialized: method={method}, schedule={self.schedule}, "
            f"epochs/stage={epochs_per_stage}"
        )

    def compress(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        feature_layers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run the full RSID compression pipeline.

        Args:
            model: Pre-trained model to compress.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            feature_layers: Layer names for feature distillation (if enabled).

        Returns:
            Dict with:
                'model': Final compressed model
                'history': Per-stage training history
                'stage_profiles': Per-stage profiling results
                'total_time': Total wall time in seconds
        """
        start_time = time.time()
        current_model = copy.deepcopy(model).to(self.device)
        current_model.eval()

        all_history = []
        stage_profiles = []

        # Profile original model
        logger.info("\n" + "=" * 60)
        logger.info("RSID Compression Pipeline")
        logger.info("=" * 60)
        logger.info(f"\nOriginal model profile:")
        orig_profile = profile_model(
            current_model,
            input_size=(1, 3, *self._get_input_spatial(train_loader)),
            device=self.device,
        )
        stage_profiles.append({"stage": "original", **orig_profile})
        logger.info(f"  Params: {orig_profile['total_params_m']:.2f}M")
        logger.info(f"  FLOPs: {orig_profile['flops_m']:.1f}M")

        # Get initial validation accuracy
        prev_val_acc = self._evaluate(current_model, val_loader)
        logger.info(f"  Accuracy: {prev_val_acc:.2f}%")

        # Keep original model for fresh decomposition at each stage
        # IMPORTANT: We decompose from the ORIGINAL model each time, not from
        # the already-decomposed model. Re-decomposing Sequential(Conv,Conv,Conv)
        # compounds compression and destroys accuracy. Instead, each stage:
        #   1. Freshly decomposes the original model at the current rank_ratio
        #   2. Uses the previous stage's trained student as teacher
        # This way the teacher-student gap stays small while avoiding
        # compounding decomposition artifacts.
        original_model = copy.deepcopy(model).to(self.device)
        original_model.eval()

        # === RSID Iterations ===
        for stage_idx, rank_ratio in enumerate(self.schedule):
            logger.info(f"\n{'='*60}")
            logger.info(f"RSID Stage {stage_idx + 1}/{len(self.schedule)}")
            logger.info(f"  Rank ratio: {rank_ratio:.3f}")
            logger.info(f"  Method: {self.method}")
            logger.info(f"{'='*60}")

            stage_start = time.time()

            # --- Step 1: Decompose from original model at this rank ---
            # Teacher is the previous stage's trained student (or original for stage 1)
            teacher = current_model

            # Fresh decomposition from original each time to avoid compounding
            student = copy.deepcopy(original_model)
            logger.info("\nStep 1: Decomposing model (from original)...")
            student = decompose_model(
                student,
                method=self.method,
                rank_ratio=rank_ratio,
                device=self.device,
                verbose=True,
            )

            # Profile decomposed model (before distillation)
            input_spatial = self._get_input_spatial(train_loader)
            decomp_profile = profile_model(
                student,
                input_size=(1, 3, *input_spatial),
                device=self.device,
            )
            compression = orig_profile["total_params"] / max(decomp_profile["total_params"], 1)
            logger.info(
                f"\n  After decomposition: {decomp_profile['total_params_m']:.2f}M params "
                f"({compression:.1f}× compression)"
            )

            # --- Step 2: Distillation training ---
            logger.info(f"\nStep 2: Distillation training ({self.epochs_per_stage} epochs)...")

            feat_layers = feature_layers if self.feature_distillation else None
            trainer = DistillationTrainer(
                teacher=teacher,
                student=student,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=copy.deepcopy(self.loss_fn),
                device=self.device,
                lr=self.lr,
                epochs=self.epochs_per_stage,
                use_amp=self.use_amp,
                compile_model=self.compile_model,
                feature_layers=feat_layers,
            )

            history = trainer.train()
            all_history.append(history)

            # --- Step 3: Get trained student, prepare for next stage ---
            current_model = trainer.get_student()
            current_model.eval()

            # Post-distillation accuracy
            val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0

            # For adaptive scheduling: adjust next stage's ratio
            if stage_idx + 1 < len(self.schedule):
                adjusted = adapt_schedule_step(
                    self.schedule, stage_idx + 1, val_acc, prev_val_acc,
                )
                if adjusted != self.schedule[stage_idx + 1]:
                    logger.info(
                        f"  Adaptive: adjusted next ratio "
                        f"{self.schedule[stage_idx + 1]:.3f} → {adjusted:.3f}"
                    )
                    self.schedule[stage_idx + 1] = adjusted

            prev_val_acc = val_acc

            # Profile after distillation
            stage_profile = profile_model(
                current_model,
                input_size=(1, 3, *input_spatial),
                device=self.device,
            )
            stage_profiles.append({
                "stage": f"stage_{stage_idx + 1}",
                "rank_ratio": rank_ratio,
                "val_acc": val_acc,
                **stage_profile,
            })

            stage_time = time.time() - stage_start
            logger.info(
                f"\nStage {stage_idx + 1} complete: "
                f"Acc={val_acc:.2f}%, "
                f"Params={stage_profile['total_params_m']:.2f}M, "
                f"Time={stage_time:.0f}s"
            )

        # === Summary ===
        total_time = time.time() - start_time
        final_profile = stage_profiles[-1]
        final_compression = orig_profile["total_params"] / max(final_profile["total_params"], 1)

        logger.info(f"\n{'='*60}")
        logger.info("RSID Compression Complete")
        logger.info(f"{'='*60}")
        logger.info(f"  Original:    {orig_profile['total_params_m']:.2f}M params, {prev_val_acc:.2f}% acc")
        logger.info(f"  Compressed:  {final_profile['total_params_m']:.2f}M params, {final_profile.get('val_acc', 0):.2f}% acc")
        logger.info(f"  Compression: {final_compression:.1f}×")
        logger.info(f"  Total time:  {total_time:.0f}s ({total_time/60:.1f} min)")

        return {
            "model": current_model,
            "history": all_history,
            "stage_profiles": stage_profiles,
            "total_time": total_time,
        }

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, val_loader) -> float:
        """Quick evaluation to get validation accuracy."""
        model.eval()
        correct = total = 0
        for inputs, targets in val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / max(total, 1)

    def _get_input_spatial(self, loader) -> tuple:
        """Get spatial dimensions from a data loader."""
        batch = next(iter(loader))
        return batch[0].shape[2], batch[0].shape[3]
