"""
Distillation Training Loop
============================

Handles the training loop for knowledge distillation, with optimizations:

CUDA Acceleration Techniques:
1. torch.compile()  --  JIT compiles the model for ~20-40% speedup (PyTorch 2.0+)
2. Automatic Mixed Precision (AMP)  --  uses FP16 where safe for ~2× throughput
3. torch.backends.cudnn.benchmark  --  auto-tunes conv algorithms for fixed input sizes
4. Gradient accumulation  --  simulates larger batch sizes without more VRAM
5. Pin memory + persistent workers  --  faster CPU→GPU data transfer
6. Non-blocking transfers  --  overlaps data transfer with computation
7. Gradient clipping  --  prevents training instability after decomposition

The trainer handles both logit-only KD and feature-based KD depending
on the loss function configuration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Optional, Dict, Callable, List
import time
import logging

from .losses import CombinedDistillationLoss

logger = logging.getLogger(__name__)

# Enable cuDNN auto-tuner for fixed-size inputs (significant speedup)
torch.backends.cudnn.benchmark = True


class FeatureExtractor:
    """
    Extracts intermediate features from a model using forward hooks.

    Registers hooks on specified layers to capture their outputs during
    the forward pass. Used for feature-based distillation.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model: The model to extract features from.
            layer_names: List of layer names to hook (e.g., ['layer1', 'layer2']).
        """
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks = []

        for name in layer_names:
            # Navigate to the target module
            module = model
            for part in name.split("."):
                module = getattr(module, part) if not part.isdigit() else module[int(part)]

            # Register a forward hook that captures the output
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores the output tensor."""
        def hook_fn(module, input, output):
            # Detach to avoid keeping the computation graph
            self.features[name] = output.detach()
        return hook_fn

    def get_features(self) -> List[torch.Tensor]:
        """Return collected features as a list (ordered by registration)."""
        return list(self.features.values())

    def clear(self):
        """Clear stored features to free memory."""
        self.features.clear()

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class DistillationTrainer:
    """
    Trains a student model using knowledge distillation from a teacher.

    Supports:
    - Logit-based KD (soft targets from teacher)
    - Feature-based KD (matching intermediate representations)
    - Combined loss (KD + feature + hard targets)
    - AMP (automatic mixed precision) for faster training
    - torch.compile() for optimized execution
    - Gradient accumulation for larger effective batch sizes
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[CombinedDistillationLoss] = None,
        device: str = "cuda",
        lr: float = 0.01,
        epochs: int = 50,
        use_amp: bool = True,
        compile_model: bool = True,
        grad_accum_steps: int = 1,
        feature_layers: Optional[List[str]] = None,
        grad_clip: float = 1.0,
    ):
        """
        Args:
            teacher: Pre-trained teacher model (frozen during training).
            student: Student model to train (the decomposed model).
            train_loader: Training data loader.
            val_loader: Validation data loader.
            loss_fn: Distillation loss function. Defaults to CombinedDistillationLoss.
            device: Training device ('cuda', 'cuda:0', 'cuda:1').
            lr: Initial learning rate.
            epochs: Number of training epochs.
            use_amp: Enable automatic mixed precision (recommended for 3090).
                     Uses FP16 for forward pass, FP32 for loss/gradients.
            compile_model: Use torch.compile() for ~20-40% speedup.
                          Requires PyTorch 2.0+. First epoch is slower (compilation).
            grad_accum_steps: Accumulate gradients over this many batches before
                              stepping the optimizer. Effective batch = batch_size * steps.
            feature_layers: List of layer names for feature distillation.
                           If None, only logit-based KD is used.
            grad_clip: Max gradient norm. Prevents exploding gradients after
                       decomposition (decomposed models can be unstable initially).
        """
        self.device = device
        self.epochs = epochs
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip

        # Move models to device
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)

        # Freeze teacher  --  no gradients needed, saves memory
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Optionally compile student model for faster execution
        # torch.compile uses TorchDynamo + TorchInductor to JIT-compile
        # the model into optimized kernels. ~20-40% speedup after warmup.
        # NOTE: torch.compile requires sm_70+ (Volta or newer). Older
        # pre-sm_70 GPUs fail with Triton ptxas errors, so skip compile there.
        can_compile = compile_model and hasattr(torch, "compile")
        if can_compile and torch.cuda.is_available():
            gpu_idx = torch.device(device).index or 0
            cc = torch.cuda.get_device_capability(gpu_idx)
            if cc[0] < 7:
                can_compile = False
                logger.info(f"Skipping torch.compile()  --  GPU sm_{cc[0]}{cc[1]} < sm_70")
        if can_compile:
            try:
                self.student = torch.compile(self.student, mode="reduce-overhead")
                logger.info("Student model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Using eager mode.")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        self.loss_fn = loss_fn or CombinedDistillationLoss()
        self.loss_fn = self.loss_fn.to(device)

        # Optimizer: SGD with momentum works better than Adam for KD
        # (Adam can overfit to teacher's noise)
        self.optimizer = torch.optim.SGD(
            self.student.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        # Cosine annealing scheduler  --  smoothly decays LR to near zero
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        # AMP scaler for mixed precision training
        # Scales loss to prevent FP16 underflow, then unscales gradients
        self.scaler = GradScaler(device) if use_amp else None

        # Feature extraction hooks (for feature-based KD)
        self.teacher_extractor = None
        self.student_extractor = None
        if feature_layers:
            self.teacher_extractor = FeatureExtractor(self.teacher, feature_layers)
            self.student_extractor = FeatureExtractor(self.student, feature_layers)

    def train(self) -> Dict[str, list]:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history:
                'train_loss': list of per-epoch average training losses
                'val_acc': list of per-epoch validation accuracies
                'val_loss': list of per-epoch validation losses
                'lr': list of per-epoch learning rates
        """
        history = {"train_loss": [], "val_acc": [], "val_loss": [], "lr": []}
        best_acc = 0.0

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss, val_acc = self._validate()

            # Step the LR scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # Track best model
            if val_acc > best_acc:
                best_acc = val_acc

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        logger.info(f"\nBest validation accuracy: {best_acc:.2f}%")

        # Clean up hooks
        if self.teacher_extractor:
            self.teacher_extractor.remove_hooks()
        if self.student_extractor:
            self.student_extractor.remove_hooks()

        return history

    def _train_epoch(self) -> float:
        """
        Train for one epoch with AMP and gradient accumulation.

        The training loop is optimized with:
        - AMP autocast: FP16 forward pass for ~2× throughput
        - GradScaler: prevents FP16 underflow in gradients
        - Gradient accumulation: effective batch = batch_size * grad_accum_steps
        - Non-blocking transfers: overlaps CPU→GPU data transfer with compute
        """
        self.student.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Non-blocking transfer overlaps data copy with GPU computation
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Mixed precision forward pass
            with autocast(self.device, enabled=self.use_amp):
                # Teacher forward (no gradients needed)
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)

                # Student forward
                student_logits = self.student(inputs)

                # Get features if using feature distillation
                student_feats = None
                teacher_feats = None
                if self.student_extractor and self.teacher_extractor:
                    student_feats = self.student_extractor.get_features()
                    teacher_feats = self.teacher_extractor.get_features()

                # Compute loss
                loss = self.loss_fn(
                    student_logits, teacher_logits, targets,
                    student_feats, teacher_feats,
                )

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward pass with AMP scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every grad_accum_steps batches
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # Clear feature caches
            if self.student_extractor:
                self.student_extractor.clear()
            if self.teacher_extractor:
                self.teacher_extractor.clear()

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> tuple:
        """
        Validate the student model.

        Returns:
            (val_loss, val_accuracy) tuple.
        """
        self.student.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(self.device, enabled=self.use_amp):
                student_logits = self.student(inputs)
                teacher_logits = self.teacher(inputs)
                loss = self.loss_fn(student_logits, teacher_logits, targets)

            total_loss += loss.item() * targets.size(0)
            _, predicted = student_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        val_loss = total_loss / max(total, 1)
        val_acc = 100.0 * correct / max(total, 1)

        return val_loss, val_acc

    def get_student(self) -> nn.Module:
        """Return the trained student model (unwrapped from compile if needed)."""
        model = self.student
        # Unwrap torch.compile wrapper if present
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model
