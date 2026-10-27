# Face Age Estimation – Transfer Learning with EfficientNet-B0

[![Tests](https://github.com/Mohamed-jarboui/face-age-estimation/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-jarboui/face-age-estimation/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)

Predicts a person's age from a face image. An ImageNet-pretrained **EfficientNet-B0** is fine-tuned as a regressor on the **UTKFace** dataset (23,700 faces aged 1 to 116). Training and evaluation run end to end on GitHub Actions, and results are reported on a held-out test set together with a naive baseline and error breakdowns by age, gender and ethnicity.

## Results

_Training is running on GitHub Actions; the test results will be added here._

## Approach

| Step | Choice | Why |
|---|---|---|
| Data | UTKFace aligned & cropped faces; age, gender and ethnicity parsed from file names | Standard public benchmark for age estimation |
| Split | 70 / 15 / 15 train / validation / test, **stratified by age group**, fixed seed | Every split has the same age distribution; the test set is never used for training decisions |
| Input | 128×128 RGB, ImageNet normalization | Keeps CPU training under an hour |
| Augmentation | Horizontal flip, brightness/contrast jitter, small shifts | Faces are roughly symmetric and centered |
| Model | EfficientNet-B0 backbone + 2-layer regression head (4.3M parameters) | Strong accuracy for its size; fast enough to train on CPU |
| Training | 1 epoch with the backbone frozen, then full fine-tuning; AdamW, one-cycle LR (backbone at 0.3× the head LR), Smooth L1 loss, early stopping on validation MAE | Warming up the new head first avoids destroying the pretrained features |
| Metrics | MAE (years), RMSE, R², share within 5 and 10 years; MAE by age group, gender and ethnicity | MAE is the standard age-estimation metric; the breakdowns show where the model is weaker |

## Project structure

```
├── src/age_estimation/
│   ├── data.py        # download, label parsing, stratified split, in-memory dataset
│   ├── model.py       # EfficientNet-B0 regressor
│   ├── train.py       # fine-tuning with early stopping
│   ├── evaluate.py    # test metrics, baseline, breakdowns, figures
│   └── predict.py     # predict the age of new images
├── tests/             # pipeline tests on synthetic images (no download needed)
├── results/           # metrics, training history, figures
└── .github/workflows/
    ├── ci.yml         # tests on every push
    └── train.yml      # full training + evaluation (manual run)
```

## Run it

```bash
git clone https://github.com/Mohamed-jarboui/face-age-estimation.git
cd face-age-estimation
pip install -r requirements.txt && pip install -e .

python -m age_estimation.train        # downloads UTKFace (~100 MB) and trains
python -m age_estimation.evaluate     # test metrics and figures in results/
python -m age_estimation.predict my_photo.jpg
```

The trained weights are published as a [release](https://github.com/Mohamed-jarboui/face-age-estimation/releases); put `model.pt` in `results/` to use `predict` without training. To retrain on GitHub's servers instead: **Actions → Train and evaluate → Run workflow**.

## Data and responsible use

- **Dataset:** [UTKFace](https://susanqq.github.io/UTKFace/) (Zhang, Song and Qi, CVPR 2017), downloaded from a [Hugging Face mirror](https://huggingface.co/datasets/py97/UTKFace-Cropped). It is released for non-commercial research. Face images are not stored in this repository.
- **Bias:** UTKFace is unbalanced: most images are of people aged 20 to 40, and some ethnic groups are under-represented. The per-group errors above show how this affects accuracy.
- Apparent age from a photo is an estimate, not a fact about a person. This model is a learning project and should not be used for decisions about individuals.

## Author

Mohamed Jarboui – Data Science and AI engineering, École Polytechnique de Sousse
