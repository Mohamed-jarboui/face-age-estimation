"""
Predict the age of the person in one or more face images.

    python -m age_estimation.predict photo1.jpg photo2.jpg --model results/model.pt

Works best on cropped, roughly frontal faces like the UTKFace training images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import preprocess_image
from .model import AgeRegressor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+")
    p.add_argument("--model", default="results/model.pt")
    p.add_argument("--size", type=int, default=None, help="defaults to the training size")
    args = p.parse_args(argv)

    size = args.size
    if size is None:
        cfg = Path(args.model).with_name("train_config.json")
        size = json.loads(cfg.read_text())["size"] if cfg.exists() else 128

    model = AgeRegressor(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        for path in args.images:
            age = model(preprocess_image(path, size)).item()
            print(f"{path}: {age:.0f} years")


if __name__ == "__main__":
    main()
