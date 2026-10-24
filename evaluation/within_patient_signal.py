"""
Within-patient diagnostic: is there ANY preictal-vs-interictal signal in the
108 hand-crafted features, ignoring the cross-patient problem?

Per patient (interictal vs preictal windows only), computes the AUC of every
feature. Since windows overlap and are strongly autocorrelated, the best of
108 features exceeds 0.5 by chance. The null therefore circularly shifts the
preictal labels (keeping their block structure) and records the best-feature
deviation under each shift; its 95th percentile is the chance level.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "features"))
from handcrafted_features import FEATURE_NAMES, N_FEATURES_PER_CHANNEL  # noqa: E402
from cache_provenance import validate_current_artifact  # noqa: E402

MIN_WINDOWS = 500
N_SHIFTS = 100
rng = np.random.default_rng(0)

validate_current_artifact(ROOT / "checkpoints" / "mlp_eval_data.pt", "mlp_eval_data", root=ROOT,
                          expected_details={"seed": 42})
data = torch.load(ROOT / "checkpoints" / "mlp_eval_data.pt")


def feature_name(j):
    return f"ch{j // N_FEATURES_PER_CHANNEL}:{FEATURE_NAMES[j % N_FEATURES_PER_CHANNEL]}"


def auc_from_ranks(ranks, pos):
    n_pos = int(pos.sum())
    n_neg = len(pos) - n_pos
    return (ranks[pos].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


results = []
print(f"{'patient':<14}{'inter':>7}{'pre':>6}{'median|dev|':>13}{'best|dev|':>11}{'null95':>8}  best feature (signed AUC)")
for split in ("val", "test"):
    d = data[split]
    X = d["X"].numpy().astype(np.float64)
    y = d["y"].numpy()
    g = d["groups"].numpy()
    for k, name in enumerate(d["group_names"]):
        keep = (g == k) & (y != 2)                  # interictal vs preictal only
        Xp, pos = X[keep], (y[keep] == 1)
        n_pre, n_int = int(pos.sum()), int((~pos).sum())
        if n_pre < MIN_WINDOWS or n_int < MIN_WINDOWS:
            print(f"{name:<14}{n_int:>7}{n_pre:>6}   skipped (needs >= {MIN_WINDOWS} windows of each class)")
            continue
        ranks = rankdata(Xp, axis=0)
        auc = auc_from_ranks(ranks, pos)
        dev = np.abs(auc - 0.5)
        n = len(pos)
        null_best = [
            np.abs(auc_from_ranks(ranks, np.roll(pos, int(rng.integers(int(0.1 * n), int(0.9 * n))))) - 0.5).max()
            for _ in range(N_SHIFTS)
        ]
        null95 = float(np.percentile(null_best, 95))
        j = int(dev.argmax())
        print(f"{name:<14}{n_int:>7}{n_pre:>6}{np.median(dev):>13.3f}{dev[j]:>11.3f}{null95:>8.3f}  "
              f"{feature_name(j)} ({auc[j]:.2f})")
        results.append(auc)

if len(results) >= 3:
    A = np.array(results)                           # patients x 108 signed AUCs
    mean_auc = A.mean(axis=0)
    agree = np.where(mean_auc >= 0.5, (A > 0.55).sum(axis=0), (A < 0.45).sum(axis=0))
    oppose = np.where(mean_auc >= 0.5, (A < 0.45).sum(axis=0), (A > 0.55).sum(axis=0))
    order = np.lexsort((-np.abs(mean_auc - 0.5), -(agree - oppose)))
    print(f"\nMost direction-consistent features across {len(results)} patients "
          f"(agree/oppose = patients with AUC beyond 0.55/0.45):")
    for j in order[:8]:
        print(f"  {feature_name(j):<26} mean AUC={mean_auc[j]:.2f}  agree={agree[j]}  oppose={oppose[j]}")
