import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_frames
from src.model import load_model
from src.utils import auto_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames_dir", required=True, help="Folder of PNG frames, sorted by name")
    parser.add_argument("--output",     required=True, help="Output .mp4 path")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = auto_device()
    model, _ = load_model(args.checkpoint, cfg, device)
    run_frames(model, args.frames_dir, args.output, cfg, device)


if __name__ == "__main__":
    main()
