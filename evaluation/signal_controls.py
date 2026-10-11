"""
Controls on the cached MLP val/test features (no feature extraction; minutes on CPU).

A. Event-level chance baseline. The alarm sequence is circularly shifted inside each
   patient (keeps the alarm duty cycle and its burstiness, breaks the link to
   seizures) and the event metrics are recomputed. Shows how many seizures a
   same-shaped alarm process "predicts in advance" by chance.
B. Within-file ramp test. Per seizure: mean alarm score 0-10 min before onset minus
   mean alarm score 20-30 min before onset, preictal windows only. Both bands usually
   come from the same recording, so session identity is constant by construction.
   Compared with placebo anchors drawn from interictal time of the same patients.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("training", "models", "features", "evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from dataset import STEP_SEC  # noqa: E402
from train_mlp import MLPClassifier, N_FEATURES, predict_proba, ALARM_SETTINGS  # noqa: E402
from metrics import make_alarm, event_metrics, _runs  # noqa: E402

MAX_GAP = 60.0
N_SHIFTS = 100
N_PLACEBO = 1000
MIN_WINDOWS = 100          # per time band (100 windows = 200 s)
rng = np.random.default_rng(0)


def patient_blocks(groups):
    cuts = np.flatnonzero(np.diff(groups) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(groups)]])
    blocks = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    assert len(blocks) == len(np.unique(groups)), "windows of a patient are not contiguous"
    return blocks


def shifted(alarm, blocks):
    out = np.empty_like(alarm)
    for a, b in blocks:
        n = b - a
        shift = int(rng.integers(int(0.1 * n), int(0.9 * n) + 1)) if n >= 10 else 0
        out[a:b] = np.roll(alarm[a:b], shift)
    return out


def delta_at(anchor, label, y, times, score):
    """Mean score 0-10 min before `anchor` minus mean score 20-30 min before it,
    using only windows with the given label. None if either band is too sparse."""
    tte = anchor - times
    sel = y == label
    late = sel & (tte > 0) & (tte <= 600)
    early = sel & (tte >= 1200) & (tte <= 1800)
    if late.sum() < MIN_WINDOWS or early.sum() < MIN_WINDOWS:
        return None
    return float(score[late].mean() - score[early].mean())


def self_test():
    t = np.arange(0, 3000, 2.0)
    y = np.concatenate([np.ones(len(t) - 5, dtype=int), np.full(5, 2)])
    ramp = np.linspace(0.2, 0.8, len(t))
    flat = np.full(len(t), 0.5)
    assert abs(delta_at(t[-5], 1, y, t, ramp) - 0.24) < 0.01
    assert abs(delta_at(t[-5], 1, y, t, flat)) < 1e-9
    assert delta_at(t[-5], 0, y, t, ramp) is None
    print("helper self-test: OK")


def analyse(split, data, model):
    d = data[split]
    probs = predict_proba(model, d["X"], torch.device("cpu"))
    y = d["y"].numpy()
    times = d["times"].numpy().astype(np.float64)
    groups = d["groups"].numpy()
    names = d["group_names"]
    score = 1.0 - probs[:, 0]
    blocks = patient_blocks(groups)
    runs = _runs(y == 2, times, MAX_GAP)

    print(f"\n{'=' * 78}\n{split.upper()}: {len(y):,} windows, {len(runs)} seizures")
    print("patients: " + ", ".join(
        f"{n} ({sum(1 for s, _ in runs if names[groups[s]] == n)} sz)" for n in names))

    # ---------- A ----------
    print(f"\n--- A. event-level result vs circular-shift chance ({N_SHIFTS} shifts) ---")
    for thr, k in ALARM_SETTINGS:
        alarm = make_alarm(probs, thr, k)
        obs = event_metrics(y, alarm, times, STEP_SEC, MAX_GAP)
        ne, nf = [], []
        for _ in range(N_SHIFTS):
            ev = event_metrics(y, shifted(alarm, blocks), times, STEP_SEC, MAX_GAP)
            ne.append(ev["predicted_in_advance"])
            nf.append(ev["false_alarms_per_hour"])
        ne = np.array(ne)
        p = (1 + np.sum(ne >= obs["predicted_in_advance"])) / (1 + N_SHIFTS)
        print(f"thr={thr:.2f} consec={k}: alarm on {alarm.mean():5.1%} of windows | "
              f"early observed {obs['predicted_in_advance']}/{obs['n_seizures']} | "
              f"chance mean {ne.mean():.1f}, 95th pct {np.percentile(ne, 95):.0f}, p={p:.3f} | "
              f"FA/h observed {obs['false_alarms_per_hour']:.1f} vs chance {np.nanmean(nf):.1f}")

    # ---------- B ----------
    deltas, rows = [], []
    for s, _ in runs:
        dlt = delta_at(times[s], 1, y, times, score)
        if dlt is not None:
            deltas.append(dlt)
            rows.append((names[groups[s]], dlt))

    pats = {int(groups[s]) for s, _ in runs}
    cand = np.flatnonzero((y == 0) & np.isin(groups, list(pats)))
    placebo, attempts = [], 0
    while cand.size and len(placebo) < N_PLACEBO and attempts < 20 * N_PLACEBO:
        attempts += 1
        dlt = delta_at(times[rng.choice(cand)], 0, y, times, score)
        if dlt is not None:
            placebo.append(dlt)

    print("\n--- B. within-file ramp: mean alarm score 0-10 min before onset "
          "minus 20-30 min before ---")
    for name, dlt in sorted(rows):
        print(f"  {name:<14} {dlt:+.4f}")
    summarize(deltas, placebo, split)
    return deltas, placebo


def summarize(deltas, placebo, label):
    n = len(deltas)
    if n == 0 or not placebo:
        print(f"{label}: not enough data")
        return
    deltas, placebo = np.array(deltas), np.array(placebo)
    obs = deltas.mean()
    null_means = np.array([rng.choice(placebo, size=n, replace=True).mean() for _ in range(5000)])
    p_perm = (1 + np.sum(null_means >= obs)) / 5001
    k = int((deltas > 0).sum())
    p_sign = binomtest(k, n, 0.5, alternative="greater").pvalue
    print(f"{label}: seizures used={n} | mean change={obs:+.4f} | positive in {k}/{n} "
          f"(sign test p={p_sign:.3f}) | placebo mean={placebo.mean():+.4f} sd={placebo.std():.4f} "
          f"| permutation p={p_perm:.3f}")


if __name__ == "__main__":
    self_test()
    ckpt = ROOT / "checkpoints"
    data = torch.load(ckpt / "mlp_eval_data.pt")
    model = MLPClassifier(N_FEATURES)
    model.load_state_dict(torch.load(ckpt / "mlp_classifier.pt", map_location="cpu"))
    model.eval()

    all_d, all_p = [], []
    for split in ("val", "test"):
        d_, p_ = analyse(split, data, model)
        all_d += d_
        all_p += p_
    print("\n" + "=" * 78)
    summarize(all_d, all_p, "VAL+TEST combined")
