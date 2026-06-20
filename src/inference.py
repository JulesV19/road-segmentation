from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.utils import CLASS_COLORS, IGNORE_COLOR

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame_bgr: np.ndarray, width: int, height: int) -> torch.Tensor:
    """BGR frame (H, W, 3) uint8 → normalised tensor (1, 3, H, W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = TF.to_tensor(rgb)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    return tensor.unsqueeze(0)


def postprocess_mask(logits: torch.Tensor, orig_w: int, orig_h: int) -> np.ndarray:
    """Logits (1, C, H, W) → RGB mask (orig_h, orig_w, 3) uint8."""
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[pred == cls_id] = color
    rgb[pred == 255] = IGNORE_COLOR
    return cv2.resize(rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def blend_overlay(frame_bgr: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(frame_bgr, 1 - alpha, mask_bgr, alpha, 0)


def side_by_side(frame_bgr: np.ndarray, blended_bgr: np.ndarray) -> np.ndarray:
    """Horizontal concat: original | segmentation overlay."""
    return np.concatenate([frame_bgr, blended_bgr], axis=1)


@torch.no_grad()
def run_frames(
    model: torch.nn.Module,
    frames_dir: str,
    output_path: str,
    cfg: dict,
    alpha: float = 0.5,
    fps: float = 17.0,
    device: str = "cpu",
    comparison: bool = True,
):
    """
    Read sorted PNG frames from frames_dir, run inference, write MP4.

    Args:
        comparison: if True, output is side-by-side (original | overlay);
                    if False, output is overlay only.
        fps:        output framerate (Cityscapes demo ≈ 17 fps)
    """
    model.eval()
    width = cfg["data"].get("infer_width", 512)
    height = cfg["data"].get("infer_height", 256)
    use_amp = cfg["training"].get("mixed_precision", False)

    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {frames_dir}")

    # Determine output size from first frame
    first = cv2.imread(str(frames[0]))
    orig_h, orig_w = first.shape[:2]
    out_w = orig_w * 2 if comparison else orig_w

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, orig_h))

    print(f"Frames : {len(frames)} PNGs from {frames_dir}")
    print(f"Output : {output_path}  ({'side-by-side' if comparison else 'overlay only'})")
    print(f"Infer  : {width}×{height} → output {out_w}×{orig_h} @ {fps}fps")

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(str(frame_path))
        tensor = preprocess_frame(frame, width, height).to(device)

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            logits = model(tensor)

        mask_rgb = postprocess_mask(logits, orig_w, orig_h)
        blended = blend_overlay(frame, mask_rgb, alpha=alpha)
        result = side_by_side(frame, blended) if comparison else blended
        writer.write(result)

        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            print(f"  {i + 1}/{len(frames)} frames", end="\r")

    writer.release()
    print(f"\nDone — {len(frames)} frames written to {output_path}")
