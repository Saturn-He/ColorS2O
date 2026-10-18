from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


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
    def __init__(
        self,
        sar_root,
        opt_root,
        hint_root=None,
        transform=None,
        random_hflip_prob=0.0,
        return_names=False,
    ):
        self.sar_root = sar_root
        self.opt_root = opt_root
        self.hint_root = hint_root
        self.transform = transform
        self.random_hflip_prob = random_hflip_prob
        self.return_names = return_names
        self.sar_files = _list_images(sar_root)
        self.opt_files = _list_images(opt_root)
        self.hint_files = _list_images(hint_root) if hint_root else None
        if len(self.sar_files) != len(self.opt_files):
            raise ValueError("SAR and OPT datasets must be the same length.")
        if self.hint_files is not None and len(self.hint_files) != len(self.sar_files):
            raise ValueError("Hint and SAR datasets must be the same length.")
        for sar_path, opt_path in zip(self.sar_files, self.opt_files):
            if sar_path.name != opt_path.name:
                raise ValueError(f"Mismatched filenames: {sar_path.name} vs {opt_path.name}")
        if self.hint_files is not None:
            for sar_path, hint_path in zip(self.sar_files, self.hint_files):
                if sar_path.name != hint_path.name:
                    raise ValueError(f"Mismatched filenames: {sar_path.name} vs {hint_path.name}")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_path = self.sar_files[idx]
        opt_path = self.opt_files[idx]
        hint_path = self.hint_files[idx] if self.hint_files is not None else None
        sar_img = Image.open(sar_path).convert("L")
        opt_img = Image.open(opt_path).convert("RGB")
        hint_img = Image.open(hint_path).convert("RGB") if hint_path is not None else None
        if self.random_hflip_prob > 0 and torch.rand(1).item() < self.random_hflip_prob:
            sar_img = TF.hflip(sar_img)
            opt_img = TF.hflip(opt_img)
            if hint_img is not None:
                hint_img = TF.hflip(hint_img)
        if self.transform is not None:
            sar_img = self.transform(sar_img)
            opt_img = self.transform(opt_img)
            if hint_img is not None:
                hint_img = self.transform(hint_img)
        if hint_img is not None:
            hint_color = hint_img
            hint_mask = (hint_color.sum(dim=0, keepdim=True) > 0).to(torch.float32)
            if self.return_names:
                return sar_img, opt_img, hint_color, hint_mask, sar_path.name
            return sar_img, opt_img, hint_color, hint_mask
        else:
            if self.return_names:
                return sar_img, opt_img, sar_path.name
            return sar_img, opt_img
