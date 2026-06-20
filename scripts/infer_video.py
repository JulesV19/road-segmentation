"""
Local inference on a Cityscapes demo sequence (folder of PNGs).

Usage:
    python scripts/infer_video.py \
        --checkpoint path/to/best.pth \
        --frames_dir path/to/stuttgart_00 \
        --output     path/to/output.mp4 \
        [--alpha 0.5] \
        [--fps 17] \
        [--no_comparison] \
        [--device cuda|mps|cpu]
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_frames
from src.model import build_model
from src.utils import load_checkpoint


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--frames_dir",    required=True, help="Folder containing sorted PNG frames")
    parser.add_argument("--output",        required=True, help="Output .mp4 path")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--alpha",         type=float, default=0.5)
    parser.add_argument("--fps",           type=float, default=17.0)
    parser.add_argument("--no_comparison", action="store_true",
                        help="Output overlay only instead of side-by-side")
    parser.add_argument("--device",        default=None)
    args = parser.parse_args()

    device = args.device or auto_device()
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if device != "cuda":
        cfg["training"]["mixed_precision"] = False

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    run_frames(
        model,
        frames_dir=args.frames_dir,
        output_path=args.output,
        cfg=cfg,
        alpha=args.alpha,
        fps=args.fps,
        device=device,
        comparison=not args.no_comparison,
    )


if __name__ == "__main__":
    main()
