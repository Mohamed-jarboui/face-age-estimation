"""
Fine-tune EfficientNet-B0 to predict age from a face image.

    python -m age_estimation.train --data-dir data --out-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FaceAgeDataset, download_utkface, load_images, load_samples, split_samples
from .model import AgeRegressor, count_parameters


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return torch.cat([model(x) for x, _ in loader]).numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--size", type=int, default=128, help="input image size")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=1, help="epochs training only the head")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="use only N images (quick tests)")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    args = p.parse_args(argv)

    torch.set_num_threads(os.cpu_count() or 1)
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----- data -----
    samples = load_samples(download_utkface(args.data_dir))
    if args.limit:
        samples = samples[: args.limit]
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f"Images: {len(samples)} (train {len(train_s)}, val {len(val_s)}, test {len(test_s)})")

    t0 = time.time()
    train_ages = np.array([s.age for s in train_s])
    train_ds = FaceAgeDataset(load_images(train_s, args.size), train_ages, train=True)
    val_ds = FaceAgeDataset(load_images(val_s, args.size), np.array([s.age for s in val_s]))
    print(f"Loaded images in {time.time() - t0:.0f}s")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=128)

    # ----- model -----
    model = AgeRegressor(pretrained=not args.no_pretrained, init_age=float(train_ages.mean()))
    print(f"Parameters: {count_parameters(model) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.lr * 0.3},
        {"params": model.head.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr * 0.3, args.lr],
        total_steps=args.epochs * len(train_dl), pct_start=0.15,
    )
    loss_fn = nn.SmoothL1Loss(beta=2.0)  # L1 for large errors, smooth near zero

    history, best_mae, bad_epochs = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        model.freeze_backbone(frozen)
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in train_dl:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += (pred.detach() - y).abs().sum().item()
            seen += len(y)

        val_mae = float(np.abs(predict(model, val_dl) - val_ds.ages.numpy()).mean())
        history.append({"epoch": epoch, "train_mae": total / seen, "val_mae": val_mae,
                        "backbone_frozen": frozen, "seconds": round(time.time() - t0)})
        print(f"epoch {epoch:2d}  train MAE {total / seen:5.2f}  val MAE {val_mae:5.2f}  "
              f"({time.time() - t0:.0f}s{', backbone frozen' if frozen else ''})", flush=True)

        if val_mae < best_mae:
            best_mae, bad_epochs = val_mae, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping")
                break

    config = vars(args) | {"best_val_mae": best_mae, "n_train": len(train_s),
                           "n_val": len(val_s), "n_test": len(test_s)}
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "train_config.json").write_text(json.dumps(config, indent=2))
    print(f"Best val MAE: {best_mae:.2f} years -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
