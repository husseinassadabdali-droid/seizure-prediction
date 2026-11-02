"""
Seizure DETECTION on the cached features: ictal (2) vs non-ictal (0 and 1), LOSO.

Per fold (held-out subject; chb21 grouped with chb01):
  * train on other subjects: all ictal windows (up to cap/5) vs 4x as many random
    non-ictal windows (interictal + preictal);
  * alarm threshold = a quantile of the model's scores on a separate sample of
    non-ictal windows of the OTHER subjects (never used for fitting) -> a fixed
    rule applied to the unseen subject.
Events: a seizure is detected if the smoothed alarm (k consecutive windows above
threshold) is on at some window of its ictal run. A false alarm is an alarm run
with no ictal window. False alarms per hour = false alarms / non-ictal hours.
Chance baseline (primary setting only): circular shifts of the alarm sequence
inside each patient.

Pre-specified PRIMARY result: hgb, CHB-MIT only, quantile 0.999, k=3.
Everything else is descriptive.

Caveats: postictal windows are excluded from the data (false-alarm hours are
slightly underestimated); latency is measured from the first ictal window start
and is approximate (window length 4 s); CHB-MIT/Siena seizures are not all
generalized tonic-clonic.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "features"))

from loso_cache_analysis import load_cache, robust_stats, transform, fit_logreg  # noqa: E402
from metrics import roc_auc_binary, _runs  # noqa: E402
from cache_provenance import build_dataset_provenance, write_artifact_manifest  # noqa: E402

STEP_SEC = 2.0
MAX_GAP = 60.0
QUANTILES = (0.99, 0.995, 0.999)
SMOOTH_K = (1, 3, 5)
PRIMARY = (0.999, 3)


# ---------------- core helpers ----------------

def alarm_block(score_blk, thr, k):
    raw = score_blk >= thr
    if k <= 1:
        return raw
    counts = np.convolve(raw.astype(np.int32), np.ones(k, np.int32), mode="full")[:len(raw)]
    return counts >= k


def block_result(alarm, yb, tb, runs):
    det, lat = 0, []
    for s, e in runs:
        idx = np.flatnonzero(alarm[s:e + 1])
        if idx.size:
            det += 1
            lat.append(float(tb[s + idx[0]] - tb[s]))
    fa = sum(1 for s, e in _runs(alarm, tb, MAX_GAP) if np.all(yb[s:e + 1] != 2))
    hours = float((yb != 2).sum()) * STEP_SEC / 3600.0
    return det, len(runs), lat, fa, hours


def self_test():
    tb = np.arange(0, 400, 2.0)
    yb = np.zeros(200, dtype=int)
    yb[100:110] = 2
    alarm = np.zeros(200, dtype=bool)
    alarm[103:108] = True
    alarm[10:12] = True
    runs = _runs(yb == 2, tb, MAX_GAP)
    det, n, lat, fa, hours = block_result(alarm, yb, tb, runs)
    assert (det, n, fa) == (1, 1, 1) and abs(lat[0] - 6.0) < 1e-9
    assert abs(hours - 190 * 2 / 3600) < 1e-9
    a = alarm_block(np.array([0, 1, 1, 1, 0, 1, 1], dtype=float), 0.5, 3)
    assert a.tolist() == [False, False, False, True, False, False, False]
    print("helper self-test: OK")


# ---------------- LOSO ----------------

def fit_predict(model_name, Xt, yt):
    if model_name == "logreg":
        w, b = fit_logreg(Xt, yt)
        return lambda Z: 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                         early_stopping=False, random_state=0)
    clf.fit(Xt, yt)
    return lambda Z: clf.predict_proba(Z)[:, 1]


def detect_loso(X, y, grp, gnames, model_name, cap, seed):
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(y), dtype=np.float64)
    thr = np.zeros((len(gnames), len(QUANTILES)))
    t0 = time.time()
    for gi, gname in enumerate(gnames):
        test = grp == gi
        pos = np.flatnonzero(~test & (y == 2))
        neg = np.flatnonzero(~test & (y != 2))
        n_pos = min(cap // 5, len(pos))
        n_neg = min(len(neg) // 2, 4 * n_pos)
        sel_pos = rng.choice(pos, n_pos, replace=False)
        sel_neg = rng.choice(neg, n_neg, replace=False)
        rest = np.setdiff1d(neg, sel_neg)
        cal = rng.choice(rest, min(60000, len(rest)), replace=False)

        sel = np.concatenate([sel_pos, sel_neg])
        yt = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
        center, scale = robust_stats(X[sel])
        predict = fit_predict(model_name, transform(X[sel], center, scale), yt)
        scores[test] = predict(transform(X[test], center, scale))
        thr[gi] = np.quantile(predict(transform(X[cal], center, scale)), QUANTILES)
        if (gi + 1) % 4 == 0 or gi + 1 == len(gnames):
            el = time.time() - t0
            print(f"  fold {gi + 1}/{len(gnames)} ({gname}) | elapsed {el / 60:.1f} min | "
                  f"ETA {el / (gi + 1) * (len(gnames) - gi - 1) / 60:.1f} min", flush=True)
    return scores, thr


# ---------------- evaluation ----------------

def run_setting(scores, y, t, blocks, grp, thr, qi, k):
    per = []
    for a, b, name in blocks:
        yb, tb = y[a:b], t[a:b]
        runs = _runs(yb == 2, tb, MAX_GAP)
        alarm = alarm_block(scores[a:b], thr[grp[a], qi], k)
        per.append((name, a, b, runs, alarm) + block_result(alarm, yb, tb, runs))
    return per        # (name, a, b, runs, alarm, det, n, lat, fa, hours)


def pooled(per):
    det, n = sum(p[5] for p in per), sum(p[6] for p in per)
    fa, hours = sum(p[8] for p in per), sum(p[9] for p in per)
    lat = [x for p in per for x in p[7]]
    return det, n, fa / hours if hours else float("nan"), (float(np.median(lat)) if lat else float("nan"))


def chance(per, y, t, rng, shifts):
    dets, fas = [], []
    for _ in range(shifts):
        d = f = 0
        h = 0.0
        for name, a, b, runs, alarm, *_ in per:
            n = b - a
            sh = int(rng.integers(int(0.1 * n), int(0.9 * n) + 1)) if n >= 10 else 0
            r = block_result(np.roll(alarm, sh), y[a:b], t[a:b], runs)
            d += r[0]
            f += r[3]
            h += r[4]
        dets.append(d)
        fas.append(f / h)
    return np.array(dets), np.array(fas)


def short(n):
    return n.replace("chbmit/", "").replace("siena/", "")


def report(model, scores, thr, y, t, blocks, grp, args, rng):
    print("\n" + "=" * 88 + f"\nMODEL: {model}\n" + "=" * 88)
    qp, kp = QUANTILES.index(PRIMARY[0]), PRIMARY[1]

    per = run_setting(scores, y, t, blocks, grp, thr, qp, kp)
    print(f"\n--- per patient at q={PRIMARY[0]}, k={kp} ---")
    print(f"{'patient':<9}{'seizures':>9}{'AUC':>7}{'detected':>10}{'FA/h':>8}{'median lat (s)':>16}")
    aucs, sens = [], []
    for (name, a, b, runs, alarm, det, n, lat, fa, hours) in per:
        pos = y[a:b] == 2
        auc = roc_auc_binary(scores[a:b], pos) if pos.any() and (~pos).any() else float("nan")
        if n:
            aucs.append(auc)
            sens.append(det / n)
        print(f"{short(name):<9}{n:>9}{('-' if auc != auc else f'{auc:.2f}'):>7}{f'{det}/{n}':>10}"
              f"{(fa / hours if hours else float('nan')):>8.2f}"
              f"{(np.median(lat) if lat else float('nan')):>16.1f}")
    print(f"patients with seizures: {len(aucs)} | median window AUC = {np.nanmedian(aucs):.3f} | "
          f"median per-patient sensitivity = {np.median(sens):.2f}")

    print("\n--- pooled over patients, all settings (quantile, k) ---")
    print("(pooled sensitivity is weighted by seizure count; one patient can carry many seizures)")
    for qi, q in enumerate(QUANTILES):
        for k in SMOOTH_K:
            d, n, fah, lat = pooled(run_setting(scores, y, t, blocks, grp, thr, qi, k))
            tag = "  <-- PRIMARY" if (q, k) == PRIMARY else ""
            print(f"  q={q:<6} k={k}: detected {d}/{n} ({d / max(n, 1):.0%}) | FA/h {fah:6.2f} | "
                  f"median latency {lat:5.1f} s{tag}")

    d, n, fah, lat = pooled(per)
    cd, cf = chance(per, y, t, rng, args.shifts)
    p = (1 + np.sum(cd >= d)) / (1 + len(cd))
    print(f"\n--- primary vs circular-shift chance ({args.shifts} shifts) ---")
    print(f"observed detected {d}/{n} | chance mean {cd.mean():.1f}, 95th pct {np.percentile(cd, 95):.0f}, "
          f"p={p:.3f} | FA/h observed {fah:.2f} vs chance {cf.mean():.2f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="chbmit")
    p.add_argument("--models", default="logreg,hgb")
    p.add_argument("--train-cap", type=int, default=120000)
    p.add_argument("--shifts", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    self_test()
    selected = [d.strip() for d in args.datasets.split(",")]
    X, y, t, f, pc, grp, blocks, gnames = load_cache(selected)
    print(f"datasets={selected} | {len(y):,} windows | {len(blocks)} patient blocks | "
          f"{len(gnames)} LOSO groups | ictal windows: {int((y == 2).sum()):,}")
    rng = np.random.default_rng(args.seed)
    for m in [x.strip() for x in args.models.split(",")]:
        scores, thr = detect_loso(X, y, grp, gnames, m, args.train_cap, args.seed)
        out = ROOT / "checkpoints" / f"detect_oof_{'_'.join(selected)}_{m}.npz"
        np.savez_compressed(out, scores=scores, thr=thr, t=t, y=y)
        dataset_fp = build_dataset_provenance()["dataset_fingerprint"]
        write_artifact_manifest(out, dataset_fp, "detection_oof",
                                {"datasets": selected, "model": m, "seed": args.seed,
                                 "train_cap": args.train_cap})
        report(m, scores, thr, y, t, blocks, grp, args, rng)


if __name__ == "__main__":
    main()
