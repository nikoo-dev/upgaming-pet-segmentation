"""A pretrained backbone is requested only for a new training run."""
import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def build_model(pretrained=False):
    return lraspp_mobilenet_v3_large(weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None,
        num_classes=2)


def configure_training(model, frozen):
    for p in model.backbone.parameters():
        p.requires_grad_(not frozen)
    training_mode(model, frozen)


def training_mode(model, frozen):
    model.train()
    if frozen:
        model.backbone.eval()


def make_optimizer(model, config):
    groups = [{"params": model.classifier.parameters(), "lr": config["head_lr"]}]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": config["backbone_lr"]})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def segmentation_loss(logits, target):
    if not torch.isin(target, torch.tensor([0, 1, 255], device=target.device)).all():
        raise ValueError("Invalid training target")
    if not (target != 255).any():
        return None
    loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    return loss
