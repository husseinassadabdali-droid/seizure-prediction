"""
Noise-robustness test for seizure DETECTION (ictal vs non-ictal), LOSO, cached features.

Question: how much does detection degrade when the test signal carries dry-electrode-like
noise, and does training with that noise recover it?

Scenarios (model trained on X features; alarm threshold = a quantile of its scores on X
windows of OTHER patients; tested on Y features of the held-out patient):
  S1 clean -> clean            baseline (must reproduce detection_loso.py hgb: 62/182, 1.28/h)
  S2 clean -> noisy            naive deployment, nothing adapted to the noise
  S3 clean -> noisy, thr recal same model, threshold recalibrated on noisy windows
  S4 noisy -> noisy            noise-aware training (matches our simulator)
  S5 noisy -> clean            does noise-aware training cost lab performance?
The same training/calibration windows are used in both versions (paired design).

Pre-specified PRIMARY result: hgb, CHB-MIT only, q=0.999, k=3, comparing S1, S2, S4 on
(a) pooled detection and false alarms/h, (b) median per-patient window AUC (threshold-free).
Everything else is descriptive.

Caveat: the noisy features come from our synthetic simulator (white noise + slow drift +
mains), i.e. the same family used for noise-aware training. S4 therefore measures how well
training matches the simulator, not real dry-electrode recordings.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("evaluation", "features"):
    sys.path.insert(0, str(ROOT / sub))

from handcrafted_features import FEATURE_NAMES, N_FEATURES_PER_CHANNEL  # noqa: E402
from loso_cache_analysis import ALIAS, robust_stats, transform  # noqa: E402
from detection_loso import (QUANTILES, SMOOTH_K, PRIMARY, MAX_GAP, STEP_SEC, run_setting,  # noqa: E402
                            pooled, chance, fit_predict, short)
from metrics import roc_auc_binary, _runs  # noqa: E402
from cache_provenance import validate_cache_directory  # noqa: E402

CLEAN = ROOT / "data/processed/feature_cache"
NOISY = ROOT / "data/processed/feature_cache_aug"
THR_COMBOS = [("clean", "clean"), ("clean", "noisy"), ("noisy", "noisy")]   # (train, calibration)
AUC_COMBOS = [("clean", "clean"), ("clean", "noisy"), ("noisy", "noisy"), ("noisy", "clean")]  # (train, test)
AUC_TAGS = ["C>C", "C>N", "N>N", "N>C"]
SCENARIOS = [  # (name, train, calibration, test)
    ("S1 clean->clean", "clean", "clean", "clean"),
    ("S2 clean->noisy", "clean", "clean", "noisy"),
    ("S3 clean->noisy, thr recal", "clean", "noisy", "noisy"),
    ("S4 noisy->noisy", "noisy", "noisy", "noisy"),
    ("S5 noisy->clean", "noisy", "noisy", "clean"),
]


# ---------------- loading ----------------

def load_pair(selected):
    infos = {
        "clean": validate_cache_directory(CLEAN, augment=False, seed=42, smoke=False, root=ROOT),
        "noisy": validate_cache_directory(NOISY, augment=True, seed=42, smoke=False, root=ROOT),
    }
    for tag, d in (("clean", CLEAN), ("noisy", NOISY)):
        info = infos[tag]
        assert info["feature_names"] == FEATURE_NAMES, f"{d}: different feature set"
        assert not info["smoke"], f"{d} is a smoke cache"
        assert bool(info["augment"]) == (tag == "noisy"), f"{d}: unexpected augment flag"
    mc, mn = np.load(CLEAN / "meta.npz"), np.load(NOISY / "meta.npz")
    for k in ("y", "time_sec", "start_sample", "index_idx", "file_id", "patient_code", "dataset_code"):
        assert np.array_equal(mc[k], mn[k]), f"clean and noisy caches differ in '{k}'"
    Xc, Xn = np.load(CLEAN / "features.npy"), np.load(NOISY / "features.npy")
    assert Xc.shape == Xn.shape

    pnames = [str(s) for s in mc["patient_names"]]
    dnames = [str(s) for s in mc["dataset_names"]]
    keep = np.isin(mc["dataset_code"], [i for i, d in enumerate(dnames) if d in selected])
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(mc["time_sec"][idx], kind="stable")]
    Xc, Xn = Xc[order], Xn[order]
    y = mc["y"][order].astype(np.int64)
    t = mc["time_sec"][order].astype(np.float64)
    pc = mc["patient_code"][order].astype(np.int64)

    cuts = np.flatnonzero(np.diff(pc) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(pc)]])
    blocks = [(int(a), int(b), pnames[pc[a]]) for a, b in zip(bounds[:-1], bounds[1:])]
    assert len(blocks) == len(np.unique(pc)), "patient windows not contiguous after sorting by time"
    gnames = sorted({ALIAS.get(n, n) for _, _, n in blocks})
    gcode = {n: i for i, n in enumerate(gnames)}
    grp = np.empty(len(y), dtype=np.int64)
    for a, b, n in blocks:
        grp[a:b] = gcode[ALIAS.get(n, n)]
    return Xc, Xn, y, t, blocks, grp, gnames


def noise_sanity(Xc, Xn):
    sub = slice(None, None, 10)
    same = float(np.all(Xc[sub] == Xn[sub], axis=1).mean())
    cols = [ch * N_FEATURES_PER_CHANNEL + FEATURE_NAMES.index("std") for ch in range(6)]
    rc, rn = Xc[sub][:, cols], Xn[sub][:, cols]
    ok = rc > 0.2
    ratio = float(np.median(rn[ok] / rc[ok]))
    print(f"noisy cache sanity: windows identical to clean = {same:.2%} | "
          f"median channel-std ratio noisy/clean = {ratio:.3f} (expected ~1.10)")
    if same > 0.02:
        sys.exit("ERROR: part of the 'noisy' cache equals the clean one (apply_prob < 1 when it was built).\n"
                 "Check that build_feature_cache.py contains 'apply_prob=1.0', then DELETE "
                 "data/processed/feature_cache_aug before rebuilding (the shard hash does not include apply_prob).")


# ---------------- LOSO with cross-condition scoring ----------------

def loso_cross(Xc, Xn, y, grp, gnames, model_name, cap, seed):
    X = {"clean": Xc, "noisy": Xn}
    rng = np.random.default_rng(seed)
    scores = {(tr, te): np.zeros(len(y)) for tr in X for te in X}
    thr = {c: np.zeros((len(gnames), len(QUANTILES))) for c in THR_COMBOS}
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

        for tr in X:
            center, scale = robust_stats(X[tr][sel])
            predict = fit_predict(model_name, transform(X[tr][sel], center, scale), yt)
            for te in X:
                scores[(tr, te)][test] = predict(transform(X[te][test], center, scale))
            for ctr, cal_cond in THR_COMBOS:
                if ctr == tr:
                    thr[(tr, cal_cond)][gi] = np.quantile(
                        predict(transform(X[cal_cond][cal], center, scale)), QUANTILES)
        if (gi + 1) % 4 == 0 or gi + 1 == len(gnames):
            el = time.time() - t0
            print(f"  fold {gi + 1}/{len(gnames)} ({gname}) | elapsed {el / 60:.1f} min | "
                  f"ETA {el / (gi + 1) * (len(gnames) - gi - 1) / 60:.1f} min", flush=True)
    return scores, thr


# ---------------- report ----------------

def report(model, scores, thr, y, t, blocks, grp, args, rng):
    print("\n" + "=" * 94 + f"\nMODEL: {model}\n" + "=" * 94)
    for key, tr, cal, te in SCENARIOS:
        print(f"  {key.split()[0]}: train={tr}, threshold calibrated on={cal}, test={te}")
    qp, kp = QUANTILES.index(PRIMARY[0]), PRIMARY[1]

    # ---- 1. threshold-free ----
    print("\n--- 1. per-patient window AUC, ictal vs all other windows (threshold-free) ---")
    print("    columns: train>test  (C = clean, N = noisy)")
    print(f"{'patient':<9}{'seiz':>5}" + "".join(f"{lab:>9}" for lab in AUC_TAGS))
    rows = []
    for a, b, name in blocks:
        pos = y[a:b] == 2
        if not pos.any() or pos.all():
            continue
        n_sz = len(_runs(pos, t[a:b], MAX_GAP))
        aucs = [roc_auc_binary(scores[c][a:b], pos) for c in AUC_COMBOS]
        rows.append(aucs)
        print(f"{short(name):<9}{n_sz:>5}" + "".join(f"{v:>9.2f}" for v in aucs))
    if not rows:
        print("  no patients with ictal windows")
        return
    R = np.array(rows)
    print(f"{'median':<14}" + "".join(f"{np.median(R[:, i]):>9.3f}" for i in range(4)))

    def paired(i, j, label):
        d = R[:, i] - R[:, j]
        n = int((d != 0).sum())
        k = int((d > 0).sum())
        p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
        print(f"  {label}: median change {np.median(d):+.3f}, higher in {k}/{n} patients (sign test p={p:.3f})")

    print("paired changes (descriptive; seizures of one patient are not independent):")
    paired(1, 0, "noise at test, no adaptation (C>N minus C>C)")
    paired(2, 1, "noise-aware training, noisy test (N>N minus C>N)")
    paired(3, 0, "noise-aware training, clean test (N>C minus C>C)")

    # ---- run all settings once ----
    results = {}
    for key, tr, cal, te in SCENARIOS:
        results[key] = {(qi, k): run_setting(scores[(tr, te)], y, t, blocks, grp, thr[(tr, cal)], qi, k)
                        for qi in range(len(QUANTILES)) for k in SMOOTH_K}

    # ---- 2. grid ----
    print("\n--- 2. pooled over patients: detected/total  false-alarms-per-hour ---")
    print(f"{'setting':<16}" + "".join(f"{key.split()[0]:>16}" for key, *_ in SCENARIOS))
    for qi, q in enumerate(QUANTILES):
        for k in SMOOTH_K:
            line = f"{'q=%.3f k=%d' % (q, k):<16}"
            for key, *_ in SCENARIOS:
                d, n, fah, _ = pooled(results[key][(qi, k)])
                line += f"{f'{d}/{n} {fah:5.2f}':>16}"
            print(line + ("   <-- primary setting" if (q, k) == PRIMARY else ""))

    # ---- 3. patient-level at the primary setting ----
    print(f"\n--- 3. primary setting (q={PRIMARY[0]}, k={kp}): patient-level view ---")
    print(f"{'scenario':<28}{'pooled':>10}{'w/o chb12':>11}{'FA/h pool':>10}{'sens mean/med':>15}"
          f"{'all/none':>10}{'FA/h med':>9}{'<=1/>5':>8}")
    for key, *_ in SCENARIOS:
        per = results[key][(qp, kp)]
        withsz = [p for p in per if p[6] > 0]
        d, n, fah, _ = pooled(per)
        sub = [p for p in per if not p[0].endswith("/chb12")]
        d2, n2, _, _ = pooled(sub)
        sens = np.array([p[5] / p[6] for p in withsz])
        fahp = np.array([p[8] / p[9] for p in withsz if p[9] > 0])
        print(f"{key:<28}{f'{d}/{n}':>10}{f'{d2}/{n2}':>11}{fah:>10.2f}"
              f"{f'{sens.mean():.2f}/{np.median(sens):.2f}':>15}"
              f"{f'{int((sens == 1).sum())}/{int((sens == 0).sum())}':>10}"
              f"{np.median(fahp):>9.2f}{f'{int((fahp <= 1).sum())}/{int((fahp > 5).sum())}':>8}")

    # ---- 4. chance ----
    print(f"\n--- 4. primary setting vs circular-shift chance ({args.shifts} shifts) ---")
    for key, *_ in SCENARIOS:
        per = results[key][(qp, kp)]
        d, n, fah, _ = pooled(per)
        cd, cf = chance(per, y, t, rng, args.shifts)
        p = (1 + np.sum(cd >= d)) / (1 + len(cd))
        print(f"  {key:<28} observed {d}/{n} | chance mean {cd.mean():.1f}, 95th pct {np.percentile(cd, 95):.0f}, "
              f"p={p:.3f} | FA/h {fah:.2f} vs chance {cf.mean():.2f}")
    print("\nNOTE: 'noisy' = our synthetic simulator, not real dry-electrode data.")


# ---------------- self-test on synthetic data (exercises the whole report path) ----------------

def self_test():
    rng = np.random.default_rng(0)
    n_pat, n = 3, 3000
    y = np.zeros(n_pat * n, dtype=np.int64)
    t = np.zeros(n_pat * n)
    grp = np.repeat(np.arange(n_pat), n)
    blocks = []
    for p in range(n_pat):
        a, b = p * n, (p + 1) * n
        t[a:b] = p * 1e8 + np.arange(n) * STEP_SEC
        y[a + 1700:a + 2000] = 1
        y[a + 2000:a + 2010] = 2
        blocks.append((a, b, f"chbmit/chb{p + 1:02d}"))
    scores = {}
    for tr in ("clean", "noisy"):
        for te in ("clean", "noisy"):
            s = rng.random(len(y)) * 0.5
            s[y == 2] += 0.5
            scores[(tr, te)] = s
    thr = {c: np.tile(np.quantile(rng.random(1000) * 0.5, QUANTILES), (n_pat, 1)) for c in THR_COMBOS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report("selftest", scores, thr, y, t, blocks, grp, argparse.Namespace(shifts=3),
               np.random.default_rng(1))
    out = buf.getvalue()
    assert "S4 noisy->noisy" in out and "median" in out and "NOTE" in out
    print("report self-test (synthetic data): OK")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="chbmit")
    p.add_argument("--models", default="hgb")
    p.add_argument("--train-cap", type=int, default=120000)
    p.add_argument("--shifts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    self_test()
    selected = [d.strip() for d in args.datasets.split(",")]
    Xc, Xn, y, t, blocks, grp, gnames = load_pair(selected)
    noise_sanity(Xc, Xn)
    print(f"datasets={selected} | {len(y):,} windows | {len(blocks)} patient blocks | "
          f"{len(gnames)} LOSO groups | ictal windows: {int((y == 2).sum()):,}")
    rng = np.random.default_rng(args.seed)
    for m in [x.strip() for x in args.models.split(",")]:
        scores, thr = loso_cross(Xc, Xn, y, grp, gnames, m, args.train_cap, args.seed)
        out = {"y": y, "t": t, "grp": grp}
        out.update({f"scores_{tr}_{te}": s for (tr, te), s in scores.items()})
        out.update({f"thr_{tr}_{cal}": th for (tr, cal), th in thr.items()})
        np.savez_compressed(ROOT / "checkpoints" / f"noise_robust_{'_'.join(selected)}_{m}.npz", **out)
        report(m, scores, thr, y, t, blocks, grp, args, rng)


if __name__ == "__main__":
    main()
