"""
Two diagnostics on the cached MLP validation/test features:
  1. Alarm rate (prediction != interictal) by dataset x true class.
  2. Dataset-identification probe: can a plain logistic regression tell
     CHB-MIT windows from Siena windows using the same 108 features, on
     patients it never saw? If yes, dataset identity can act as a shortcut.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for sub in ("training", "models", "features", "evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from train_mlp import MLPClassifier, N_FEATURES, predict_proba  # noqa: E402
from cache_provenance import validate_current_artifact  # noqa: E402

CLASSES = ["interictal", "preictal", "ictal"]
DATASETS = ["chbmit", "siena"]

ckpt = ROOT / "checkpoints"
validate_current_artifact(ckpt / "mlp_eval_data.pt", "mlp_eval_data", root=ROOT,
                          expected_details={"seed": 42})
validate_current_artifact(ckpt / "mlp_classifier.pt", "mlp_classifier", root=ROOT,
                          expected_details={"seed": 42})
data = torch.load(ckpt / "mlp_eval_data.pt")
model = MLPClassifier(N_FEATURES)
model.load_state_dict(torch.load(ckpt / "mlp_classifier.pt", map_location="cpu"))
model.eval()


def dataset_ids(d):
    is_siena = np.array([n.startswith("siena/") for n in d["group_names"]])
    return is_siena[d["groups"].numpy()].astype(np.int64)


print("=== 1. Alarm rate by dataset x true class (alarm = argmax != interictal) ===")
for split in ("val", "test"):
    d = data[split]
    probs = predict_proba(model, d["X"], torch.device("cpu"))
    alarm = probs.argmax(axis=1) != 0
    ds, y = dataset_ids(d), d["y"].numpy()
    print(f"\n[{split}]")
    print(f"{'dataset':<8}{'class':<12}{'windows':>9}{'alarm rate':>12}")
    for di, dname in enumerate(DATASETS):
        for c, cname in enumerate(CLASSES):
            m = (ds == di) & (y == c)
            rate = f"{alarm[m].mean():.3f}" if m.sum() else "-"
            print(f"{dname:<8}{cname:<12}{int(m.sum()):>9}{rate:>12}")


def robust_scale(train_X, X):
    q25, med, q75 = np.percentile(train_X, [25, 50, 75], axis=0)
    scale = np.where((q75 - q25) > 1e-8, (q75 - q25) / 1.349, 1.0)
    return np.clip((X - med) / scale, -10, 10).astype(np.float32)


def fit_logreg(X, y, steps=400, lr=0.05):
    X, y = torch.from_numpy(X), torch.from_numpy(y).float()
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos_weight = ((y == 0).sum().float() / (y == 1).sum().clamp_min(1).float()).reshape(1)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(steps):
        opt.zero_grad()
        lossf(X @ w + b, y).backward()
        opt.step()
    return w.detach(), b.detach()


def balanced_acc(w, b, X, y):
    pred = ((torch.from_numpy(X) @ w + b) > 0).numpy().astype(int)
    return np.mean([(pred[y == k] == k).mean() for k in (0, 1)])


print("\n\n=== 2. Dataset-identification probe (features only, unseen patients) ===")
Xv, yv = data["val"]["X"].numpy(), dataset_ids(data["val"])
Xt, yt = data["test"]["X"].numpy(), dataset_ids(data["test"])
for name, (Xa, ya, Xb, yb) in (("train on val -> test on test", (Xv, yv, Xt, yt)),
                               ("train on test -> test on val", (Xt, yt, Xv, yv))):
    w, b = fit_logreg(robust_scale(Xa, Xa), ya)
    acc = balanced_acc(w, b, robust_scale(Xa, Xb), yb)
    print(f"{name}: balanced accuracy = {acc:.3f}  (0.5 = cannot tell the datasets apart)")
