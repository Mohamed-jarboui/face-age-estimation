"""
UTKFace data: download, label parsing, train/val/test split and in-memory dataset.

UTKFace file names encode the labels: [age]_[gender]_[race]_[date&time].jpg(.chip.jpg)
gender: 0 = male, 1 = female; race: 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Others.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Original "aligned & cropped" UTKFace archive, mirrored on Hugging Face (pinned revision)
UTKFACE_URL = (
    "https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/"
    "e80f21fb21fa3631380c3cb634b79be3ac2f4cba/UTKFace.tar.gz"
)

GENDERS = {0: "Male", 1: "Female"}
RACES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
AGE_BINS = [0, 13, 20, 30, 40, 50, 60, 70, 120]
AGE_LABELS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class Sample:
    path: Path
    age: int
    gender: int
    race: int


def download_utkface(data_dir: str | Path) -> Path:
    """Download and extract UTKFace into data_dir. Returns the folder with the images."""
    data_dir = Path(data_dir)
    image_dir = data_dir / "UTKFace"
    if image_dir.exists() and any(image_dir.iterdir()):
        return image_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "UTKFace.tar.gz"
    if not archive.exists():
        print(f"Downloading UTKFace to {archive} ...")
        urllib.request.urlretrieve(UTKFACE_URL, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(data_dir, filter="data")
        except TypeError:  # Python versions without extraction filters
            tar.extractall(data_dir)
    return image_dir


def parse_filename(name: str) -> tuple[int, int, int] | None:
    """Return (age, gender, race) from a UTKFace file name, or None if it is malformed."""
    parts = name.split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (0 < age <= 116 and gender in GENDERS and race in RACES):
        return None
    return age, gender, race


def load_samples(image_dir: str | Path) -> list[Sample]:
    samples = []
    for path in sorted(Path(image_dir).glob("*.jpg*")):
        labels = parse_filename(path.name)
        if labels:
            samples.append(Sample(path, *labels))
    return samples


def age_group(ages: np.ndarray) -> np.ndarray:
    return np.digitize(ages, AGE_BINS[1:-1])


def split_samples(samples: list[Sample], val_frac=0.15, test_frac=0.15, seed=42):
    """Split stratified by age group so every split has the same age distribution."""
    rng = np.random.default_rng(seed)
    groups = age_group(np.array([s.age for s in samples]))
    train, val, test = [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        test += [samples[i] for i in idx[:n_test]]
        val += [samples[i] for i in idx[n_test:n_test + n_val]]
        train += [samples[i] for i in idx[n_test + n_val:]]
    return train, val, test


def load_images(samples: list[Sample], size: int) -> np.ndarray:
    """Decode and resize all images once into a uint8 array (N, size, size, 3)."""
    out = np.empty((len(samples), size, size, 3), dtype=np.uint8)
    for i, s in enumerate(samples):
        with Image.open(s.path) as img:
            out[i] = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR))
    return out


class FaceAgeDataset(Dataset):
    """Images kept in memory; light augmentation for training."""

    def __init__(self, images: np.ndarray, ages: np.ndarray, train: bool = False):
        self.images = images
        self.ages = torch.as_tensor(ages, dtype=torch.float32)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        if self.train:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            # small random brightness/contrast change
            x = (x - 0.5) * (0.8 + 0.4 * torch.rand(1)) + 0.5 + 0.1 * (torch.rand(1) - 0.5)
            # random shift of up to 8 pixels
            dx, dy = torch.randint(-8, 9, (2,)).tolist()
            x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
            x = x.clamp(0, 1)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, self.ages[i]


def preprocess_image(path: str | Path, size: int) -> torch.Tensor:
    """Prepare a single image for prediction: (1, 3, size, size)."""
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB").resize((size, size), Image.BILINEAR))
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
