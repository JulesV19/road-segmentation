import copy

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    return smp.Unet(
        encoder_name=m["encoder"],
        encoder_weights=m["encoder_weights"],
        in_channels=m["in_channels"],
        classes=m["num_classes"],
        activation=None,
    )


def load_model_for_inference(
    checkpoint: str, cfg: dict, device: str, encoder: str | None = None
) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = copy.deepcopy(cfg)
    if "cfg" in ckpt:
        cfg["model"] = copy.deepcopy(ckpt["cfg"]["model"])
    if encoder:
        cfg["model"]["encoder"] = encoder
    cfg["model"]["encoder_weights"] = None

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {checkpoint} — encoder {cfg['model']['encoder']}, epoch {ckpt.get('epoch', -1) + 1}")
    return model.to(device).eval(), ckpt
