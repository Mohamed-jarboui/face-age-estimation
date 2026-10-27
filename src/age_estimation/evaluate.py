"""
Evaluate the trained model on the held-out test split and draw the figures.

    python -m age_estimation.evaluate --data-dir data --out-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (AGE_LABELS, GENDERS, RACES, FaceAgeDataset, age_group, download_utkface,
                   load_images, load_samples, split_samples)
from .model import AgeRegressor
from .train import predict

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e8e7e3"


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_abs_error": float(np.median(np.abs(err))),
        "r2": 1 - ss_res / ss_tot,
        "within_5_years": float((np.abs(err) <= 5).mean()),
        "within_10_years": float((np.abs(err) <= 10).mean()),
    }


def mae_by(keys: np.ndarray, y: np.ndarray, pred: np.ndarray, names: dict) -> dict:
    return {names[k]: {"n": int((keys == k).sum()), "mae": float(np.abs(pred - y)[keys == k].mean())}
            for k in sorted(names) if (keys == k).any()}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def plot_figures(y, pred, by_age, history, baseline_mae, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Predicted vs true age
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, s=6, alpha=0.25, color=BLUE, linewidths=0)
    lim = [0, max(y.max(), pred.max()) + 5]
    ax.plot(lim, lim, color=INK, lw=1, ls="--", label="perfect prediction")
    ax.set(xlim=lim, ylim=lim, xlabel="True age (years)", ylabel="Predicted age (years)")
    ax.set_title("Predicted vs true age (test set)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "predicted_vs_true.png", dpi=160)
    plt.close(fig)

    # MAE by age group
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(by_age)
    vals = [by_age[k]["mae"] for k in labels]
    bars = ax.bar(labels, vals, color=BLUE, width=0.65)
    for b, k in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{b.get_height():.1f}\n(n={by_age[k]['n']})", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    ax.set(xlabel="True age group", ylabel="Mean absolute error (years)", ylim=(0, max(vals) * 1.3))
    ax.set_title("Error grows with age, where training images are scarcer", loc="left", fontweight="bold")
    style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "mae_by_age_group.png", dpi=160)
    plt.close(fig)

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_mae"] for h in history], color=BLUE, lw=2, marker="o", label="train")
    ax.plot(ep, [h["val_mae"] for h in history], color=ORANGE, lw=2, marker="o", label="validation")
    ax.axhline(baseline_mae, color=MUTED, ls="--", lw=1)
    ax.text(ep[-1], baseline_mae + 0.3, f"baseline (always predict median age): {baseline_mae:.1f}",
            ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set(xlabel="Epoch", ylabel="MAE (years)", ylim=(0, None))
    ax.set_title("Training curve", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curve.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    config = json.loads((out / "train_config.json").read_text())
    history = json.loads((out / "history.json").read_text())

    samples = load_samples(download_utkface(args.data_dir))
    if config["limit"]:
        samples = samples[: config["limit"]]
    train_s, _, test_s = split_samples(samples, seed=config["seed"])

    y = np.array([s.age for s in test_s], dtype=float)
    test_dl = DataLoader(FaceAgeDataset(load_images(test_s, config["size"]), y), batch_size=128)

    model = AgeRegressor(pretrained=False)
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    pred = predict(model, test_dl)

    baseline_pred = np.full_like(y, np.median([s.age for s in train_s]))
    by_age = mae_by(age_group(y), y, pred, dict(enumerate(AGE_LABELS)))
    metrics = {
        "test_images": len(test_s),
        "model": regression_metrics(y, pred),
        "baseline_median_age": regression_metrics(y, baseline_pred),
        "mae_by_age_group": by_age,
        "mae_by_gender": mae_by(np.array([s.gender for s in test_s]), y, pred, GENDERS),
        "mae_by_race": mae_by(np.array([s.race for s in test_s]), y, pred, RACES),
        "epochs_trained": len(history),
        "best_val_mae": config["best_val_mae"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(out / "test_predictions.csv", "w") as f:
        f.write("file,true_age,predicted_age\n")
        for s, t, pr in zip(test_s, y, pred):
            f.write(f"{s.path.name},{int(t)},{pr:.1f}\n")

    plot_figures(y, pred, by_age, history, metrics["baseline_median_age"]["mae"], out / "figures")

    m = metrics["model"]
    print(f"Test MAE {m['mae']:.2f} years | RMSE {m['rmse']:.2f} | R2 {m['r2']:.3f} | "
          f"within 5 years {m['within_5_years']:.0%} | baseline MAE "
          f"{metrics['baseline_median_age']['mae']:.2f}")


if __name__ == "__main__":
    main()
