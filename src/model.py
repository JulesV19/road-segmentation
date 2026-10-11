import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: dict, pretrained: bool = True) -> nn.Module:
    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"] if pretrained else None,
        classes=m["num_classes"],
        activation=None,
    )


def load_model(checkpoint: str, cfg: dict, device: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {checkpoint} (epoch {ckpt['epoch'] + 1})")
    return model.to(device).eval(), ckpt
