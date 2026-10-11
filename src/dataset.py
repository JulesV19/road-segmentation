from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.utils import IMAGENET_MEAN, IMAGENET_STD


def train_transform(aug: dict) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.ColorJitter(**aug["color_jitter"], p=0.5),
    ])


class CityscapesDataset(Dataset):
    def __init__(self, root: str, csv_file: str, transform: A.Compose | None = None):
        self.root = Path(root)
        self.df = pd.read_csv(self.root / csv_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.root / row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(self.root / row["mask_path"]))

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image, torch.from_numpy(mask).long()


def build_loaders(cfg: dict):
    data = cfg["data"]
    train_ds = CityscapesDataset(data["root"], data["train_csv"], train_transform(cfg["augmentation"]))
    val_ds = CityscapesDataset(data["root"], data["val_csv"])

    kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=data["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=data["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    return train_loader, val_loader
