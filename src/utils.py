import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# BGR-like palette matching standard Cityscapes color conventions
CLASS_COLORS = np.array([
    [0,   0,   0],    # 0  void      — black (should be absent after remapping)
    [128, 64,  128],  # 1  flat      — purple (road)
    [70,  70,  70],   # 2  construction — dark grey
    [153, 153, 153],  # 3  object    — light grey
    [107, 142, 35],   # 4  nature    — olive green
    [70,  130, 180],  # 5  sky       — steel blue
    [220, 20,  60],   # 6  human     — crimson
    [0,   0,   142],  # 7  vehicle   — dark blue
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) label mask to (H, W, 3) RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    rgb[mask == 255] = IGNORE_COLOR
    return rgb


def save_checkpoint(state: dict, path: str | Path, drive_path: str | Path | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if drive_path is not None:
        # async copy to Drive so training is not blocked
        def _copy():
            import shutil
            drive_path_ = Path(drive_path)
            drive_path_.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, drive_path_)
        threading.Thread(target=_copy, daemon=True).start()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cuda",
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


def visualize_predictions(
    images: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    n: int = 4,
    denorm_mean=(0.485, 0.456, 0.406),
    denorm_std=(0.229, 0.224, 0.225),
):
    """Display a grid of image / GT / prediction triplets."""
    n = min(n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[None]

    mean = np.array(denorm_mean)[:, None, None]
    std = np.array(denorm_std)[:, None, None]

    for i in range(n):
        img = images[i].cpu().numpy()
        img = (img * std + mean).clip(0, 1).transpose(1, 2, 0)
        gt = targets[i].cpu().numpy()
        pred = preds[i].cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(mask_to_rgb(gt))
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(mask_to_rgb(pred))
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    return fig
