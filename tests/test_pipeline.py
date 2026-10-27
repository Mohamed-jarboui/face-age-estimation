"""
Fast tests on synthetic images named like UTKFace files (no download, no pretrained weights).
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from age_estimation import evaluate, predict, train
from age_estimation.data import (FaceAgeDataset, age_group, load_images, load_samples,
                                 parse_filename, split_samples)
from age_estimation.model import AgeRegressor


@pytest.fixture(scope="module")
def fake_utkface(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    image_dir = root / "UTKFace"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(160):
        age = int(rng.integers(1, 90))
        name = f"{age}_{i % 2}_{i % 5}_2017011{i:07d}.jpg.chip.jpg"
        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(image_dir / name, "JPEG")
    (image_dir / "badname.jpg").write_bytes(b"")  # must be ignored
    return root


# ---------- labels and splits ----------

@pytest.mark.parametrize("name, expected", [
    ("26_1_2_20170116174525125.jpg.chip.jpg", (26, 1, 2)),
    ("1_0_0_20161219140623097.jpg", (1, 0, 0)),
    ("39_1_20170116174525125.jpg.chip.jpg", None),   # missing race
    ("abc_0_0_20170116.jpg", None),
    ("200_0_0_20170116.jpg", None),                   # impossible age
])
def test_parse_filename(name, expected):
    assert parse_filename(name) == expected


def test_load_samples_skips_bad_files(fake_utkface):
    assert len(load_samples(fake_utkface / "UTKFace")) == 160


def test_split_is_disjoint_complete_and_reproducible(fake_utkface):
    samples = load_samples(fake_utkface / "UTKFace")
    train_s, val_s, test_s = split_samples(samples, seed=1)
    names = [{s.path.name for s in part} for part in (train_s, val_s, test_s)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
    assert sum(map(len, names)) == len(samples)
    assert [s.path for s in split_samples(samples, seed=1)[2]] == [s.path for s in test_s]


def test_split_keeps_age_distribution():
    from age_estimation.data import Sample
    samples = [Sample(path=f"{i}", age=a, gender=0, race=0)
               for i, a in enumerate(np.repeat([5, 25, 45, 75], 100))]
    _, _, test_s = split_samples(samples, test_frac=0.2, seed=0)
    groups, counts = np.unique(age_group(np.array([s.age for s in test_s])), return_counts=True)
    assert len(groups) == 4 and (counts == 20).all()


# ---------- dataset and model ----------

def test_dataset_output(fake_utkface):
    samples = load_samples(fake_utkface / "UTKFace")[:4]
    images = load_images(samples, 32)
    assert images.shape == (4, 32, 32, 3) and images.dtype == np.uint8
    x, y = FaceAgeDataset(images, np.array([s.age for s in samples]), train=True)[0]
    assert x.shape == (3, 32, 32) and y.item() == samples[0].age


def test_model_predicts_one_age_per_image():
    model = AgeRegressor(pretrained=False, init_age=30.0).eval()
    with torch.no_grad():
        out = model(torch.zeros(3, 3, 64, 64))
    assert out.shape == (3,)


def test_freeze_backbone():
    model = AgeRegressor(pretrained=False)
    model.freeze_backbone(True)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


# ---------- end to end ----------

def test_train_evaluate_predict(fake_utkface, tmp_path, capsys):
    out = tmp_path / "results"
    common = ["--data-dir", str(fake_utkface), "--out-dir", str(out)]
    train.main(common + ["--size", "32", "--epochs", "2", "--batch-size", "16", "--no-pretrained"])
    assert (out / "model.pt").exists()
    assert len(json.loads((out / "history.json").read_text())) == 2

    evaluate.main(common)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["model"]["mae"] >= 0 and metrics["test_images"] > 0
    assert set(metrics["mae_by_gender"]) == {"Male", "Female"}
    for fig in ["predicted_vs_true.png", "mae_by_age_group.png", "training_curve.png"]:
        assert (out / "figures" / fig).exists()

    image = next((fake_utkface / "UTKFace").glob("*.chip.jpg"))
    predict.main([str(image), "--model", str(out / "model.pt")])
    assert "years" in capsys.readouterr().out
