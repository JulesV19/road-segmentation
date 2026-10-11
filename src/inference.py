from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.utils import IMAGENET_MEAN, IMAGENET_STD, mask_to_rgb


@torch.no_grad()
def run_frames(model, frames_dir: str, output_path: str, cfg: dict, device: str,
               alpha: float = 0.5, fps: float = 17.0):
    frames = sorted(Path(frames_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {frames_dir}")

    height, width = cv2.imread(str(frames[0])).shape[:2]
    size = (cfg["data"]["infer_width"], cfg["data"]["infer_height"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * width, height))

    for frame_path in tqdm(frames):
        frame = cv2.imread(str(frame_path))  # OpenCV reads BGR
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size)
        x = TF.normalize(TF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

        pred = model(x).argmax(dim=1)[0].cpu().numpy()
        mask = cv2.resize(mask_to_rgb(pred), (width, height), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(frame, 1 - alpha, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR), alpha, 0)
        writer.write(np.concatenate([frame, overlay], axis=1))

    writer.release()
    print(f"{len(frames)} frames written to {output_path}")
