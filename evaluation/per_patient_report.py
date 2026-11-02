"""Per-patient breakdown of the MLP classifier on the cached val/test features."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for sub in ("training", "models", "features", "evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from train_mlp import MLPClassifier, N_FEATURES, predict_proba  # noqa: E402
from metrics import roc_auc_binary, confusion_matrix, per_class_metrics  # noqa: E402
from cache_provenance import validate_current_artifact  # noqa: E402

ckpt = ROOT / "checkpoints"
validate_current_artifact(ckpt / "mlp_eval_data.pt", "mlp_eval_data", root=ROOT,
                          expected_details={"seed": 42})
validate_current_artifact(ckpt / "mlp_classifier.pt", "mlp_classifier", root=ROOT,
                          expected_details={"seed": 42})
data = torch.load(ckpt / "mlp_eval_data.pt")
model = MLPClassifier(N_FEATURES)
model.load_state_dict(torch.load(ckpt / "mlp_classifier.pt", map_location="cpu"))
model.eval()


def f(v, ok=True):
    return f"{v:.2f}" if ok and not np.isnan(v) else "-"


for split in ("val", "test"):
    d = data[split]
    probs = predict_proba(model, d["X"], torch.device("cpu"))
    y, g = d["y"].numpy(), d["groups"].numpy()
    print(f"\n=== {split}: per patient ===")
    print(f"{'patient':<16}{'inter':>8}{'pre':>8}{'ictal':>7}{'alarmAUC':>10}"
          f"{'rec_int':>9}{'rec_pre':>9}{'rec_ict':>9}")
    for k, name in enumerate(d["group_names"]):
        m = g == k
        yk, pk = y[m], probs[m]
        rec, _, _, sup = per_class_metrics(confusion_matrix(yk, pk.argmax(1)))
        auc = roc_auc_binary(1.0 - pk[:, 0], yk != 0)
        print(f"{name:<16}{int(sup[0]):>8}{int(sup[1]):>8}{int(sup[2]):>7}{f(auc):>10}"
              f"{f(rec[0], sup[0] > 0):>9}{f(rec[1], sup[1] > 0):>9}{f(rec[2], sup[2] > 0):>9}")
