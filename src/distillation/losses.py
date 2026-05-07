"""
Knowledge Distillation Loss Functions
======================================

Three types of distillation losses used in RSID:

1. KD Loss (Hinton et al. 2015):
   - Matches soft probability distributions between teacher and student
   - Uses temperature scaling to soften the distributions
   - Higher temperature → softer distributions → more information transfer

2. Feature Distillation Loss (FitNets, Romero et al. 2015):
   - Matches intermediate feature representations
   - Student learns to mimic teacher's internal representations
   - Uses MSE between feature maps (with optional projection)

3. Combined Loss:
   - L = α · L_KD + β · L_feature + (1 - α - β) · L_CE
   - α controls soft target weight (teacher knowledge)
   - β controls feature matching weight (intermediate knowledge)
   - (1-α-β) controls hard target weight (ground truth)

All losses use torch operations and are fully GPU-compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class KDLoss(nn.Module):
    """
    Standard Knowledge Distillation loss (Hinton et al. 2015).

    Combines two terms:
    - Soft loss: KL divergence between temperature-scaled teacher/student logits
    - Hard loss: Cross-entropy with ground truth labels

    The soft loss transfers "dark knowledge"  --  the teacher's beliefs about
    which wrong classes are similar to the correct class. This information
    is lost in hard labels but preserved in soft probabilities.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Args:
            temperature: Softmax temperature. Higher = softer distributions.
                         T=1 is standard softmax. T=3-5 works well for KD.
                         T=20 makes the distribution almost uniform.
            alpha: Weight for soft loss. (1-alpha) is weight for hard loss.
                   alpha=0.5 gives equal weight. Higher alpha trusts the teacher more.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KD loss.

        Args:
            student_logits: Raw logits from student model, shape (B, C).
            teacher_logits: Raw logits from teacher model, shape (B, C).
            targets: Ground truth class indices, shape (B,).

        Returns:
            Scalar loss value.
        """
        T = self.temperature

        # Soft loss: KL divergence between soft distributions
        # Multiply by T² to maintain gradient magnitude at high temperatures
        # (the gradient of KL w.r.t. logits scales as 1/T², so T² compensates)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        # Hard loss: standard cross-entropy with true labels
        hard_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss (FitNets-style).

    Matches intermediate feature maps between teacher and student.
    When the teacher and student have different channel dimensions
    (which happens after decomposition), a 1×1 projection conv
    aligns them before computing MSE.

    This helps the student learn good intermediate representations,
    not just the final output distribution.
    """

    def __init__(self):
        super().__init__()
        # Projection layers to align dimensions (created lazily)
        self.projectors: nn.ModuleDict = nn.ModuleDict()

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute feature matching loss between corresponding layers.

        Args:
            student_features: List of feature maps from student hook points.
            teacher_features: List of feature maps from teacher hook points.
                              Must have same length as student_features.

        Returns:
            Mean MSE loss across all feature pairs.
        """
        assert len(student_features) == len(teacher_features), (
            f"Feature list length mismatch: {len(student_features)} vs {len(teacher_features)}"
        )

        total_loss = torch.tensor(0.0, device=student_features[0].device)

        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # If channel dimensions differ, project student features
            if s_feat.shape[1] != t_feat.shape[1]:
                key = f"proj_{i}"
                if key not in self.projectors:
                    # Create 1×1 conv to project student channels → teacher channels
                    proj = nn.Conv2d(
                        s_feat.shape[1], t_feat.shape[1],
                        kernel_size=1, bias=False,
                    ).to(s_feat.device)
                    self.projectors[key] = proj
                s_feat = self.projectors[key](s_feat)

            # If spatial dimensions differ, resize student to match teacher
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])

            # Normalize features before MSE to prevent scale issues
            # This makes the loss magnitude consistent across layers
            s_norm = F.normalize(s_feat.flatten(2), dim=2)
            t_norm = F.normalize(t_feat.flatten(2), dim=2)

            total_loss = total_loss + F.mse_loss(s_norm, t_norm)

        return total_loss / max(len(student_features), 1)


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss for RSID: KD + Feature Matching + Cross-Entropy.

    L = α · L_KD(soft) + β · L_feature + (1 - α - β) · L_CE(hard)

    This is the main loss function used in RSID training. Each component
    provides different information:
    - L_KD: What the teacher's output distribution looks like
    - L_feature: What the teacher's internal representations look like
    - L_CE: What the correct answer actually is
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.2,
        temperature: float = 3.0,
    ):
        """
        Args:
            alpha: Weight for KD soft loss.
            beta: Weight for feature distillation loss.
            temperature: KD temperature for soft targets.

        Note: alpha + beta should be <= 1.0.
              The hard loss weight is (1 - alpha - beta).
              If alpha + beta > 1.0, hard loss weight becomes 0.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.kd_loss = KDLoss(temperature=temperature, alpha=1.0)  # pure soft loss
        self.feature_loss = FeatureDistillationLoss()
        self.hard_weight = max(0.0, 1.0 - alpha - beta)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Student's output logits (B, C).
            teacher_logits: Teacher's output logits (B, C).
            targets: Ground truth labels (B,).
            student_features: Optional list of intermediate student features.
            teacher_features: Optional list of intermediate teacher features.

        Returns:
            Scalar combined loss.
        """
        # KD soft loss
        loss = self.alpha * self.kd_loss(student_logits, teacher_logits, targets)

        # Hard cross-entropy loss
        if self.hard_weight > 0:
            loss = loss + self.hard_weight * F.cross_entropy(student_logits, targets)

        # Feature distillation loss (if features are provided)
        if (
            self.beta > 0
            and student_features is not None
            and teacher_features is not None
            and len(student_features) > 0
        ):
            loss = loss + self.beta * self.feature_loss(student_features, teacher_features)

        return loss
