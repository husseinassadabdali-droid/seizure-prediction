"""
Controls on the saved detection scores (no training; seconds).
1. Per-patient AUC: ictal vs preictal only (same recordings) and ictal vs interictal only.
2. Primary setting overall, without chb12, and at patient level (should reproduce the
   headline numbers in the 'all' row).
3. Detections where the alarm was already ON at the first ictal window.
Usage: python3 detection_controls.py [hgb|logreg]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("evaluation", "features"):
    sys.path.insert(0, str(ROOT / sub))

from loso_cache_analysis import load_cache  # noqa: E402
from detection_loso import QUANTILES, PRIMARY, run_setting, short  # noqa: E402
from metrics import roc_auc_binary  # noqa: E402
from cache_provenance import validate_current_artifact  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "hgb"
scores_path = ROOT / "checkpoints" / f"detect_oof_chbmit_{model}.npz"
validate_current_artifact(scores_path, "detection_oof", root=ROOT,
                          expected_details={"datasets": ["chbmit"], "model": model,
                                            "seed": 0, "train_cap": 120000})
X, y, t, f, pc, grp, blocks, gnames = load_cache(["chbmit"])
del X
d = np.load(scores_path)
scores, thr = d["scores"], d["thr"]
assert len(scores) == len(y) and np.array_equal(d["y"], y) and np.array_equal(d["t"], t)


def auc(sc, pos, neg):
    m = pos | neg
    if pos.sum() < 20 or neg.sum() < 20:
        return float("nan")
    return roc_auc_binary(sc[m], pos[m])


print(f"model: {model}\n\n--- 1. per-patient AUC (ictal vs ...) ---")
print(f"{'patient':<9}{'vs preictal':>13}{'vs interictal':>15}")
rows = []
for a, b, name in blocks:
    yb, sb = y[a:b], scores[a:b]
    ict = yb == 2
    if not ict.any():
        continue
    r = (auc(sb, ict, yb == 1), auc(sb, ict, yb == 0))
    rows.append(r)
    fmt = lambda v: "-" if v != v else f"{v:.2f}"
    print(f"{short(name):<9}{fmt(r[0]):>13}{fmt(r[1]):>15}")
R = np.array(rows)
print(f"median   ictal vs preictal (same files): {np.nanmedian(R[:, 0]):.3f} | "
      f"ictal vs interictal: {np.nanmedian(R[:, 1]):.3f}")

qp, kp = QUANTILES.index(PRIMARY[0]), PRIMARY[1]
per = run_setting(scores, y, t, blocks, grp, thr, qp, kp)   # (name,a,b,runs,alarm,det,n,lat,fa,hours)


def pooled(sub):
    det, n = sum(p[5] for p in sub), sum(p[6] for p in sub)
    fa, hrs = sum(p[8] for p in sub), sum(p[9] for p in sub)
    return det, n, fa / hrs


print(f"\n--- 2. primary setting q={PRIMARY[0]}, k={kp} ---")
for label, sub in (("all patients", per),
                   ("without chb12", [p for p in per if not p[0].endswith("/chb12")])):
    det, n, fah = pooled(sub)
    print(f"  {label:<14}: detected {det}/{n} ({det / n:.0%}) | FA/h {fah:.2f}")
withsz = [p for p in per if p[6] > 0]
sens = np.array([p[5] / p[6] for p in withsz])
fah_p = np.array([p[8] / p[9] for p in withsz])
print(f"  patient level ({len(withsz)} patients): mean sensitivity {sens.mean():.2f}, "
      f"median {np.median(sens):.2f}; all seizures detected in {(sens == 1).sum()}, none in {(sens == 0).sum()}")
print(f"  FA/h per patient: median {np.median(fah_p):.2f}, <=1 in {(fah_p <= 1).sum()}, "
      f">5 in {(fah_p > 5).sum()}")

on_at_start = detected = 0
for (name, a, b, runs, alarm, *_) in per:
    for s, e in runs:
        if alarm[s:e + 1].any():
            detected += 1
            on_at_start += bool(alarm[s])
print(f"\n--- 3. detections with the alarm already ON at the first ictal window: "
      f"{on_at_start}/{detected} ---")
