import math
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_MESHGRID_CACHE = {}


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


def _get_meshgrid(height, width, device):
    key = (device.type, device.index, height, width)
    cached = _MESHGRID_CACHE.get(key)
    if cached is None:
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing="ij",
        )
        cached = (yy, xx)
        _MESHGRID_CACHE[key] = cached
    return cached


def build_hints(
    opt_img,
    hint_dropout_prob=0.5,
    hint_max_ratio=0.05,
    hint_color_thresh=0.1,
    hint_num_regions=1,
    hint_sampling_mode="stripe",
    meshgrid=None,
):
    _, height, width = opt_img.shape
    max_pixels = max(1, int(hint_max_ratio * height * width))

    if torch.rand(1, device=opt_img.device).item() < hint_dropout_prob:
        hint_color = torch.zeros_like(opt_img, dtype=torch.float32)
        hint_mask = torch.zeros(1, height, width, dtype=torch.float32, device=opt_img.device)
        return hint_color, hint_mask

    if meshgrid is None:
        yy, xx = _get_meshgrid(height, width, opt_img.device)
    else:
        yy, xx = meshgrid

    hint_mask = torch.zeros(height, width, dtype=torch.bool, device=opt_img.device)
    attempts = 0
    max_attempts = 1000
    hint_count = 0

    if hint_sampling_mode not in {"stripe", "dot"}:
        raise ValueError("hint_sampling_mode must be 'stripe' or 'dot'.")

    while hint_count < max_pixels and attempts < max_attempts:
        attempts += 1
        seed_y = torch.randint(0, height, (1,), device=opt_img.device).item()
        seed_x = torch.randint(0, width, (1,), device=opt_img.device).item()

        if hint_sampling_mode == "stripe":
            theta = torch.empty(1, device=opt_img.device).uniform_(0.0, math.pi).item()
            thickness = torch.empty(1, device=opt_img.device).uniform_(1.0, 4.0).item()
            length = torch.empty(1, device=opt_img.device).uniform_(5.0, 30.0).item()
            radius = 0.5 * math.sqrt(length * length + thickness * thickness)
            y_min = max(0, int(seed_y - radius))
            y_max = min(height, int(seed_y + radius) + 1)
            x_min = max(0, int(seed_x - radius))
            x_max = min(width, int(seed_x + radius) + 1)
            yy_local = yy[y_min:y_max, x_min:x_max]
            xx_local = xx[y_min:y_max, x_min:x_max]
            x_rel = xx_local - seed_x
            y_rel = yy_local - seed_y
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x_rot = x_rel * cos_t + y_rel * sin_t
            y_rot = -x_rel * sin_t + y_rel * cos_t
            stripe_mask = (x_rot.abs() <= length / 2.0) & (y_rot.abs() <= thickness / 2.0)
            hint_mask[y_min:y_max, x_min:x_max] |= stripe_mask
        else:
            radius = torch.empty(1, device=opt_img.device).uniform_(1.0, 6.0).item()
            y_min = max(0, int(seed_y - radius))
            y_max = min(height, int(seed_y + radius) + 1)
            x_min = max(0, int(seed_x - radius))
            x_max = min(width, int(seed_x + radius) + 1)
            yy_local = yy[y_min:y_max, x_min:x_max]
            xx_local = xx[y_min:y_max, x_min:x_max]
            circle_mask = (xx_local - seed_x).pow(2) + (yy_local - seed_y).pow(2) <= radius**2
            hint_mask[y_min:y_max, x_min:x_max] |= circle_mask

        hint_count = int(hint_mask.sum().item())

    if hint_count == 0:
        seed_y = torch.randint(0, height, (1,), device=opt_img.device).item()
        seed_x = torch.randint(0, width, (1,), device=opt_img.device).item()
        hint_mask[seed_y, seed_x] = True
        hint_count = 1

    if hint_count > max_pixels:
        flat_mask = hint_mask.flatten()
        keep = torch.multinomial(flat_mask.float(), max_pixels, replacement=False)
        hint_mask = torch.zeros_like(flat_mask, dtype=torch.bool)
        hint_mask[keep] = True
        hint_mask = hint_mask.view(height, width)

    hint_color = torch.zeros_like(opt_img, dtype=torch.float32)
    hint_color[:, hint_mask] = opt_img[:, hint_mask].to(torch.float32)
    hint_mask = hint_mask.to(torch.float32).unsqueeze(0)
    return hint_color, hint_mask


class PairedImageDirDataset(Dataset):
    def __init__(
        self,
        sar_root,
        opt_root,
        transform=None,
        random_hflip_prob=0.0,
        hint_dropout_prob=0.5,
        hint_max_ratio=0.05,
        hint_color_thresh=0.1,
        hint_num_regions=1,
        hint_sampling_mode="stripe",
        return_names=False,
        build_hints=True,
    ):
        self.sar_root = sar_root
        self.opt_root = opt_root
        self.transform = transform
        self.random_hflip_prob = random_hflip_prob
        self.hint_dropout_prob = hint_dropout_prob
        self.hint_max_ratio = hint_max_ratio
        self.hint_color_thresh = hint_color_thresh
        self.hint_num_regions = hint_num_regions
        self.hint_sampling_mode = hint_sampling_mode
        self.return_names = return_names
        self.build_hints = build_hints
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
        if self.random_hflip_prob > 0 and torch.rand(1).item() < self.random_hflip_prob:
            sar_img = TF.hflip(sar_img)
            opt_img = TF.hflip(opt_img)
        if self.transform is not None:
            sar_img = self.transform(sar_img)
            opt_img = self.transform(opt_img)
        if self.build_hints:
            meshgrid = _get_meshgrid(opt_img.shape[1], opt_img.shape[2], opt_img.device)
            hint_color, hint_mask = build_hints(
                opt_img,
                hint_dropout_prob=self.hint_dropout_prob,
                hint_max_ratio=self.hint_max_ratio,
                hint_color_thresh=self.hint_color_thresh,
                hint_num_regions=self.hint_num_regions,
                hint_sampling_mode=self.hint_sampling_mode,
                meshgrid=meshgrid,
            )
            if self.return_names:
                return sar_img, opt_img, hint_color, hint_mask, sar_path.name
            return sar_img, opt_img, hint_color, hint_mask
        else:
            if self.return_names:
                return sar_img, opt_img, sar_path.name
            return sar_img, opt_img
