import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],           # 'resnet34'
        encoder_weights=m["encoder_weights"], # 'imagenet'
        in_channels=m["in_channels"],
        classes=m["num_classes"],            # 8
        activation=None,                     # raw logits
    )
