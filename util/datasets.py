import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(root):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files = [p for p in root_path.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    return sorted(files)


class ImageDirDataset(Dataset):
    def __init__(self, root, transform=None, mode="RGB"):
        self.root = root
        self.transform = transform
        self.mode = mode
        self.files = _list_images(root)
        if not self.files:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert(self.mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, path.name


class PairedImageDirDataset(Dataset):
    def __init__(self, sar_root, opt_root, transform=None):
        self.sar_root = sar_root
        self.opt_root = opt_root
        self.transform = transform
        self.sar_files = _list_images(sar_root)
        self.opt_files = _list_images(opt_root)
        if len(self.sar_files) != len(self.opt_files):
            raise ValueError("SAR and OPT datasets must be the same length.")
        for sar_path, opt_path in zip(self.sar_files, self.opt_files):
            if sar_path.name != opt_path.name:
                raise ValueError(f"Mismatched filenames: {sar_path.name} vs {opt_path.name}")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]
        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")
        if self.transform is not None:
            if _is_paired_transform(self.transform):
                sar_img, opt_img = self.transform(sar_img, opt_img)
            else:
                sar_img = self.transform(sar_img)
                opt_img = self.transform(opt_img)
        return sar_img, opt_img


def _is_paired_transform(transform):
    return bool(getattr(transform, "is_paired", False))
