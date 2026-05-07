"""
Model Factory
==============

Loads pre-trained models and adapts them for different datasets.

Supported architectures:
- ResNet-18:       Standard residual network, skip connections, 11.2M params
- ResNet-50:       Bottleneck residual network, deeper, 25.6M params
- MobileNetV2:     Inverted residuals + depthwise separable convs, 3.4M params
- EfficientNet-B0: NAS-designed, compound scaling (depth/width/resolution), 5.3M params

For CIFAR-10/100 (32×32 images):
- We modify the first conv layer to use kernel_size=3 instead of 7
  (original was designed for 224×224 ImageNet images)
- Remove the aggressive stride and maxpool from the stem

For TinyImageNet (64×64):
- Use models closer to their original design (kernel_size=3 for stem)
- Adjust avgpool to match smaller spatial dimensions
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Registry of supported models and their feature extraction layer names
SUPPORTED_MODELS = {
    "resnet18": {
        "factory": models.resnet18,
        "feature_layers": ["layer1", "layer2", "layer3", "layer4"],
    },
    "resnet50": {
        "factory": models.resnet50,
        "feature_layers": ["layer1", "layer2", "layer3", "layer4"],
    },
    "mobilenetv2": {
        "factory": models.mobilenet_v2,
        "feature_layers": [
            "features.3",   # After first inverted residual block
            "features.6",   # Mid-network
            "features.13",  # Deep features
            "features.17",  # Final features before classifier
        ],
    },
    "efficientnet_b0": {
        "factory": models.efficientnet_b0,
        "feature_layers": [
            "features.2",   # Early features
            "features.4",   # Mid features
            "features.6",   # Deep features
            "features.8",   # Final features
        ],
    },
}


def get_model(
    name: str,
    dataset: str = "cifar10",
    pretrained: bool = True,
    num_classes: Optional[int] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    Load a model adapted for the target dataset.

    For CIFAR-10/100: modifies the stem (first conv) and classifier head.
    For TinyImageNet: modifies only the classifier head.

    When pretrained=True, loads ImageNet weights and then modifies the
    architecture. The modified layers lose their pre-training, but the
    bulk of the network retains ImageNet features (good initialization).

    Args:
        name: Model name (see SUPPORTED_MODELS).
        dataset: Target dataset ('cifar10', 'cifar100', 'tinyimagenet').
        pretrained: Load ImageNet pre-trained weights.
        num_classes: Override number of output classes.
        device: Target device.

    Returns:
        Model ready for the target dataset.
    """
    name = name.lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {name}. Supported: {list(SUPPORTED_MODELS.keys())}"
        )

    # Determine number of classes
    dataset_classes = {
        "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,
        "imagenette": 10, "imagewoof": 10,
        "imagenet": 1000, "imagenet100": 100,
    }
    n_classes = num_classes or dataset_classes.get(dataset.lower(), 10)

    # Load base model with optional ImageNet pre-training
    weights = "IMAGENET1K_V1" if pretrained else None
    model = SUPPORTED_MODELS[name]["factory"](weights=weights)

    # Adapt for target dataset
    if dataset.lower() in ("cifar10", "cifar100"):
        model = _adapt_for_cifar(model, name, n_classes)
    elif dataset.lower() == "tinyimagenet":
        model = _adapt_for_tinyimagenet(model, name, n_classes)
    elif dataset.lower() in ("imagenette", "imagewoof"):
        # Real ImageNet resolution (224x224): keep original stem and maxpool;
        # only the classifier head is replaced with a 10-class linear layer.
        model = _set_classifier(model, name, n_classes)
    elif dataset.lower() == "imagenet":
        # Full ImageNet-1k: pretrained weights are already 1000-class, no
        # changes needed. Use ImageNet-pretrained model directly as teacher.
        pass
    elif dataset.lower() == "imagenet100":
        # 100-class subset of ImageNet at 224x224. Replace classifier head only.
        model = _set_classifier(model, name, n_classes)
    else:
        # ImageNet-1k  --  just modify classifier head if num_classes != 1000
        model = _set_classifier(model, name, n_classes)

    logger.info(f"Loaded {name} for {dataset} ({n_classes} classes)")
    return model.to(device)


def get_feature_layers(name: str) -> List[str]:
    """
    Get the names of layers to use for feature distillation.

    These are intermediate layers whose outputs will be matched
    between teacher and student during feature-based KD.

    Args:
        name: Model name.

    Returns:
        List of layer name strings.
    """
    return SUPPORTED_MODELS[name.lower()]["feature_layers"]


def _adapt_for_cifar(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """
    Adapt a model designed for ImageNet (224×224) to CIFAR (32×32).

    Key changes:
    - Replace 7×7 stride-2 stem conv with 3×3 stride-1 conv
    - Remove the initial maxpool (would shrink 32×32 too aggressively)
    - Replace the classifier head for the target number of classes
    """
    if name in ("resnet18", "resnet50"):
        # Replace stem: 7×7 stride 2 → 3×3 stride 1
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove maxpool (it would reduce 32→16 too early)
        model.maxpool = nn.Identity()
        # Replace classifier
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "mobilenetv2":
        # MobileNetV2 stem: Conv2d(3, 32, 3, stride=2) → stride=1 for CIFAR
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            3, first_conv.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        # Replace classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "efficientnet_b0":
        # EfficientNet stem: Conv2d(3, 32, 3, stride=2) → stride=1 for CIFAR
        stem_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            3, stem_conv.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        # Replace classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def _adapt_for_tinyimagenet(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """
    Adapt for TinyImageNet (64×64).

    64×64 is closer to ImageNet's 224×224, so we make fewer modifications:
    - Use 3×3 stem instead of 7×7 (64 is still small)
    - Keep the maxpool for ResNets
    - Replace classifier head
    """
    if name in ("resnet18", "resnet50"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Keep maxpool for TinyImageNet (64→32 is fine)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "mobilenetv2":
        # Use stride 1 for 64×64
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            3, first_conv.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "efficientnet_b0":
        stem_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            3, stem_conv.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def _set_classifier(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Replace the classifier head only."""
    if name in ("resnet18", "resnet50"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "mobilenetv2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
