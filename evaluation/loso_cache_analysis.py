"""
LOSO analysis on the cached hand-crafted features (clean features, labeling v2).

1. Leave-one-subject-out out-of-fold scores (chb21 is grouped with chb01).
   Task: preictal (1) vs interictal (0); ictal windows are excluded from training.
   Models: logistic regression (torch) and, if scikit-learn is installed,
   HistGradientBoosting.
2. Per-patient window AUC (preictal vs interictal) plus a session-identity control:
   AUC between two random halves of the patient's INTERICTAL files.
3. B. Within-file ramp: per seizure, mean score 0-10 min before onset minus 20-30 min
   before, minus the same quantity at placebo anchors from the same patient's
   interictal time. Patient-level statistics (seizures of one patient are not
   independent). Reported for all seizures and for "same-file" seizures only.
4. A. Event-level chance baseline at fixed per-patient alarm duty cycles
   (threshold = per-patient score quantile; uses no labels) vs circular shifts.

--plant EFFECT adds a known synthetic ramp (rel_beta, 6 channels) to preictal windows
IN MEMORY ONLY, as a power check of this whole pipeline.
"""
import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("features", "evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from handcrafted_features import FEATURE_NAMES, N_FEATURES_PER_CHANNEL  # noqa: E402
from metrics import event_metrics, roc_auc_binary, _runs  # noqa: E402
from cache_provenance import (validate_cache_directory, build_dataset_provenance,
                              write_artifact_manifest, fold_normalization_manifest)  # noqa: E402

STEP_SEC = 2.0
CACHE = ROOT / "data/processed/feature_cache"
ALIAS = {"chbmit/chb21": "chbmit/chb01"}     # same physical subject
MAX_GAP = 60.0
LATE_BAND = (float(os.environ.get("LATE_START", "0")), 600.0)   # LATE_START=120 skips the last 2 min
EARLY_BAND = (1200.0, 1800.0)
MIN_WIN = 100                                # windows needed in each band


# ---------------- loading ----------------

def load_cache(selected):
    info = validate_cache_directory(CACHE, augment=False, seed=42, smoke=False, root=ROOT)
    assert info["feature_names"] == FEATURE_NAMES, "cache built with a different feature set"
    assert not info["augment"] and not info["smoke"], "need the full clean cache"
    X = np.load(CACHE / "features.npy")
    m = np.load(CACHE / "meta.npz")
    pnames = [str(s) for s in m["patient_names"]]
    dnames = [str(s) for s in m["dataset_names"]]
    keep = np.isin(m["dataset_code"], [i for i, d in enumerate(dnames) if d in selected])
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(m["time_sec"][idx], kind="stable")]
    X = X[order]
    y = m["y"][order].astype(np.int64)
    t = m["time_sec"][order].astype(np.float64)
    f = m["file_id"][order].astype(np.int64)
    pc = m["patient_code"][order].astype(np.int64)

    cuts = np.flatnonzero(np.diff(pc) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(pc)]])
    blocks = [(int(a), int(b), pnames[pc[a]]) for a, b in zip(bounds[:-1], bounds[1:])]
    assert len(blocks) == len(np.unique(pc)), "patient windows are not contiguous after sorting by time"

    gnames = sorted({ALIAS.get(n, n) for _, _, n in blocks})
    gcode = {n: i for i, n in enumerate(gnames)}
    grp = np.empty(len(y), dtype=np.int64)
    for a, b, n in blocks:
        grp[a:b] = gcode[ALIAS.get(n, n)]
    return X, y, t, f, pc, grp, blocks, gnames


# ---------------- optional synthetic signal (power check) ----------------

def plant_signal(X, y, t, blocks, effect):
    feats = [k * N_FEATURES_PER_CHANNEL + FEATURE_NAMES.index("rel_beta") for k in range(6)]
    n_planted = 0
    for a, b, _ in blocks:
        ys, ts = y[a:b], t[a:b]
        runs = _runs(ys == 2, ts, MAX_GAP)
        idx = np.flatnonzero(ys == 1)
        if not runs or idx.size == 0:
            continue
        onsets = np.array([ts[s0] for s0, _ in runs])
        pos = np.searchsorted(onsets, ts[idx], side="right")
        ok = pos < len(onsets)
        idx, pos = idx[ok], pos[ok]
        ramp = np.clip((1800.0 - (onsets[pos] - ts[idx])) / 1800.0, 0.0, 1.0)
        for j in feats:
            X[a + idx, j] += effect * ramp  # fixed synthetic units; no cohort-fitted scale
        n_planted += len(idx)
    print(f"PLANTED SIGNAL: +{effect} fixed feature units x ramp on rel_beta (6 channels), "
          f"{n_planted:,} preictal windows modified (in memory only)")


# ---------------- models / LOSO ----------------

def robust_stats(Xs):
    q25, med, q75 = np.percentile(Xs, [25, 50, 75], axis=0)
    scale = (q75 - q25) / 1.349
    std = Xs.std(axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return med.astype(np.float32), scale.astype(np.float32)


def transform(X, center, scale):
    return np.clip((X - center) / scale, -10, 10).astype(np.float32)


def fit_fold_feature_scaler(X, training_rows, training_patients):
    """Fit the robust feature scaler exclusively from this fold's train rows."""
    training_rows = np.asarray(training_rows, dtype=np.int64)
    if training_rows.size == 0:
        raise ValueError("cannot fit fold feature normalization without training rows")
    center, scale = robust_stats(X[training_rows])
    manifest = fold_normalization_manifest(training_patients, center, scale)
    return center, scale, manifest


def fit_logreg(Xt, yt, steps=300, lr=0.05, l2=1e-3):
    Xt = torch.from_numpy(Xt)
    yt = torch.from_numpy(yt.astype(np.float32))
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + l2 * (w * w).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.item())


def loso(X, y, grp, gnames, models, cap, seed):
    scores = {m: np.zeros(len(y), dtype=np.float32) for m in models}
    fold_normalization = []
    fold_calibration = {m: {} for m in models}
    rng = np.random.default_rng(seed)
    if "hgb" in models:
        from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    for gi, gname in enumerate(gnames):
        test = grp == gi
        train = ~test & (y != 2)
        pos = np.flatnonzero(train & (y == 1))
        neg = np.flatnonzero(train & (y == 0))
        n = min(cap // 2, len(pos), len(neg))
        n_neg_fit = min(n, max(1, len(neg) // 2))
        sel_pos = rng.choice(pos, n, replace=False)
        sel_neg = rng.choice(neg, n_neg_fit, replace=False)
        sel = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(sel)
        cal = np.setdiff1d(neg, sel_neg)
        if cal.size == 0:
            cal = sel_neg  # only when the training fold has no held-out calibration negatives
        Xs, ys = X[sel], (y[sel] == 1).astype(np.int64)
        fit_patients = sorted(gnames[g] for g in np.unique(grp[sel]))
        center, scale, norm_manifest = fit_fold_feature_scaler(X, sel, fit_patients)
        fold_normalization.append({"held_out_patient": gname, **norm_manifest})
        Xt, Xh = transform(Xs, center, scale), transform(X[test], center, scale)
        if "logreg" in models:
            w, b = fit_logreg(Xt, ys)
            scores["logreg"][test] = 1.0 / (1.0 + np.exp(-(Xh @ w + b)))
            cal_scores = 1.0 / (1.0 + np.exp(-(transform(X[cal], center, scale) @ w + b)))
            fold_calibration["logreg"][gname] = {
                duty: float(np.quantile(cal_scores, 1.0 - duty)) for duty in (0.02, 0.05, 0.10)}
        if "hgb" in models:
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                 max_leaf_nodes=31, early_stopping=False,
                                                 random_state=0)
            clf.fit(Xt, ys)
            scores["hgb"][test] = clf.predict_proba(Xh)[:, 1]
            cal_scores = clf.predict_proba(transform(X[cal], center, scale))[:, 1]
            fold_calibration["hgb"][gname] = {
                duty: float(np.quantile(cal_scores, 1.0 - duty)) for duty in (0.02, 0.05, 0.10)}
        if (gi + 1) % 4 == 0 or gi + 1 == len(gnames):
            el = time.time() - t0
            print(f"  LOSO fold {gi + 1}/{len(gnames)} ({gname}) | elapsed {el / 60:.1f} min | "
                  f"ETA {el / (gi + 1) * (len(gnames) - gi - 1) / 60:.1f} min", flush=True)
    return scores, fold_normalization, fold_calibration


# ---------------- helpers ----------------

def band_stat(anchor, t, y, f, s, label):
    """(mean score 0-10 min before anchor - mean score 20-30 min before, same_file) over windows
    with the given label; None if either band has < MIN_WIN windows."""
    tte = anchor - t
    sel = y == label
    late = sel & (tte > LATE_BAND[0]) & (tte <= LATE_BAND[1])
    early = sel & (tte >= EARLY_BAND[0]) & (tte <= EARLY_BAND[1])
    if late.sum() < MIN_WIN or early.sum() < MIN_WIN:
        return None
    same = np.unique(f[late | early]).size == 1
    return float(s[late].mean() - s[early].mean()), bool(same)


def interictal_control(s, y, f, rng, n_rep=20, min_per_half=500):
    """mean |AUC-0.5| between two random halves of the patient's interictal FILES."""
    inter = y == 0
    files, counts = np.unique(f[inter], return_counts=True)
    files = files[counts >= 100]
    if len(files) < 2:
        return None
    devs = []
    for _ in range(n_rep):
        perm = rng.permutation(files)
        h = len(perm) // 2
        ma = inter & np.isin(f, perm[:h])
        mb = inter & np.isin(f, perm[h:])
        if ma.sum() < min_per_half or mb.sum() < min_per_half:
            continue
        auc = roc_auc_binary(np.concatenate([s[ma], s[mb]]),
                             np.concatenate([np.ones(ma.sum(), bool), np.zeros(mb.sum(), bool)]))
        devs.append(abs(auc - 0.5))
    return float(np.mean(devs)) if devs else None


def self_test():
    t = np.arange(0, 3000, 2.0)
    y = np.ones(len(t), dtype=np.int64)
    f = np.zeros(len(t), dtype=np.int64)
    s = np.linspace(0.2, 0.8, len(t))
    delta, same = band_stat(3000.0, t, y, f, s, 1)
    # linear ramp 0.2->0.8 over 2998 s: expected delta = slope * (mean time-to-event of early band - of late band)
    expected = 0.6 / 2998.0 * (1500.0 - 0.5 * (LATE_BAND[0] + LATE_BAND[1]))
    assert abs(delta - expected) < 0.01 and same
    f2 = f.copy()
    f2[t >= 1750] = 1
    assert band_stat(3000.0, t, y, f2, s, 1)[1] is False
    assert band_stat(3000.0, t, y, f, s, 0) is None
    print("helper self-test: OK")


def short(name):
    return name.replace("chbmit/", "").replace("siena/", "")


def summarize(rows, title):
    """rows: [(patient, n_seizures, seizure_delta, placebo_delta, excess)]"""
    print(f"\n  {title}")
    if not rows:
        print("    no patients with enough data")
        return
    for label, pick in (("all", lambda p: True), ("chbmit", lambda p: p.startswith("chbmit/")),
                        ("siena", lambda p: p.startswith("siena/"))):
        sub = [r for r in rows if pick(r[0])]
        if not sub:
            continue
        ex = np.array([r[4] for r in sub])
        k, n = int((ex > 0).sum()), len(ex)
        p = binomtest(k, n, 0.5, alternative="greater").pvalue
        se = ex.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        print(f"    {label:<7} patients={n:2d} seizures={sum(r[1] for r in sub):3d} | "
              f"mean excess ramp={ex.mean():+.4f} (SE {se:.4f}) | positive in {k}/{n} patients "
              f"(sign test p={p:.3f})")


# ---------------- analyses ----------------

def analyse(model, s, y, t, f, pc, blocks, args, rng, fold_calibration):
    print("\n" + "=" * 86 + f"\nMODEL: {model}\n" + "=" * 86)

    # ---- 1. per-patient window AUC + session-identity control ----
    print("\n--- 1. per-patient AUC, preictal vs interictal (out-of-fold) ---")
    print(f"{'patient':<9}{'inter':>9}{'pre':>8}{'AUC':>7}{'|AUC-.5|':>10}{'control |dev|':>15}")
    rows = []
    for a, b, name in blocks:
        ys, ss, fs = y[a:b], s[a:b], f[a:b]
        pre, inter = ys == 1, ys == 0
        if pre.sum() < MIN_WIN or inter.sum() < MIN_WIN:
            continue
        auc = roc_auc_binary(np.concatenate([ss[pre], ss[inter]]),
                             np.concatenate([np.ones(pre.sum(), bool), np.zeros(inter.sum(), bool)]))
        ctrl = interictal_control(ss, ys, fs, rng)
        rows.append((name, auc, ctrl))
        print(f"{short(name):<9}{int(inter.sum()):>9}{int(pre.sum()):>8}{auc:>7.2f}"
              f"{abs(auc - 0.5):>10.3f}{('-' if ctrl is None else f'{ctrl:.3f}'):>15}")
    for label, pick in (("all", lambda p: True), ("chbmit", lambda p: p.startswith("chbmit/")),
                        ("siena", lambda p: p.startswith("siena/"))):
        sub = [r for r in rows if pick(r[0])]
        if not sub:
            continue
        aucs = np.array([r[1] for r in sub])
        k = int((aucs > 0.5).sum())
        p = binomtest(k, len(aucs), 0.5, alternative="greater").pvalue
        both = [(abs(r[1] - 0.5), r[2]) for r in sub if r[2] is not None]
        ctl = (f" | real |dev| {np.mean([d for d, _ in both]):.3f} vs control {np.mean([c for _, c in both]):.3f}"
               if both else "")
        print(f"  {label:<7} patients={len(aucs):2d} mean AUC={aucs.mean():.3f} median={np.median(aucs):.3f} "
              f"| AUC>0.5 in {k}/{len(aucs)} (sign test p={p:.3f}){ctl}")
    print("  (control |dev| = how far the score moves between two sets of interictal files of the SAME "
          "patient, i.e. what session identity alone can produce.)")

    # ---- 2. B: within-file ramp with per-patient placebo ----
    block_of = np.repeat(np.arange(len(blocks)), [b - a for a, b, _ in blocks])
    seizures = _runs(y == 2, t, MAX_GAP)
    sz, pl = [], []
    for si, (s0, _) in enumerate(seizures):
        bi = block_of[s0]
        a, b, name = blocks[bi]
        r = band_stat(t[s0], t[a:b], y[a:b], f[a:b], s[a:b], 1)
        if r is not None:
            sz.append((name, r[0], r[1]))
    for a, b, name in blocks:
        if not any(blocks[block_of[s0]][2] == name for s0, _ in seizures):
            continue
        inter_idx = np.flatnonzero(y[a:b] == 0)
        if inter_idx.size == 0:
            continue
        for ia in rng.choice(inter_idx, size=min(args.placebo, inter_idx.size), replace=False):
            r = band_stat(t[a + ia], t[a:b], y[a:b], f[a:b], s[a:b], 0)
            if r is not None:
                pl.append((name, r[0], r[1]))

    print(f"\n--- 3. B. within-file ramp: {len(seizures)} seizures in the data, {len(sz)} with both bands "
          f"populated ({sum(1 for _, _, sm in sz if sm)} with both bands inside ONE file) ---")
    print("    excess ramp = (mean score 0-10 min before onset - 20-30 min before) for the patient's "
          "seizures, minus the same at placebo anchors in the patient's interictal time")
    for same_only, title in ((False, "all seizures with both bands populated"),
                             (True, "same-file seizures only (session identity constant)")):
        rows_b = []
        for p in sorted({n for n, _, _ in sz}):
            sd = [d for q, d, sm in sz if q == p and (sm or not same_only)]
            pdl = [d for q, d, sm in pl if q == p and (sm or not same_only)]
            if sd and len(pdl) >= 10:
                rows_b.append((p, len(sd), float(np.mean(sd)), float(np.mean(pdl)),
                               float(np.mean(sd) - np.mean(pdl))))
        summarize(rows_b, title)
        if same_only and rows_b:
            print(f"    {'patient':<9}{'seizures':>9}{'seizure delta':>15}{'placebo delta':>15}{'excess':>9}")
            for p, n, sdm, pdm, ex in rows_b:
                print(f"    {short(p):<9}{n:>9}{sdm:>15.4f}{pdm:>15.4f}{ex:>9.4f}")

    # ---- 4. A: event-level chance baseline at fixed duty cycles ----
    print(f"\n--- 4. A. event-level vs circular-shift chance ({args.shifts} shifts) ---")
    print("    threshold is calibrated on non-ictal training-fold rows; no held-out signal or label is used")
    for duty in (0.02, 0.05, 0.10):
        alarm = np.zeros(len(y), dtype=bool)
        for a, b, patient in blocks:
            group = ALIAS.get(patient, patient)
            alarm[a:b] = s[a:b] >= fold_calibration[group][duty]
        obs = event_metrics(y, alarm, t, STEP_SEC, MAX_GAP)
        ne, nf = [], []
        for _ in range(args.shifts):
            sh = np.empty_like(alarm)
            for a, b, _ in blocks:
                n = b - a
                k = int(rng.integers(int(0.1 * n), int(0.9 * n) + 1)) if n >= 10 else 0
                sh[a:b] = np.roll(alarm[a:b], k)
            ev = event_metrics(y, sh, t, STEP_SEC, MAX_GAP)
            ne.append(ev["predicted_in_advance"])
            nf.append(ev["false_alarms_per_hour"])
        ne = np.array(ne)
        p = (1 + np.sum(ne >= obs["predicted_in_advance"])) / (1 + len(ne))
        print(f"  duty {duty:4.0%}: seizures={obs['n_seizures']} | early observed {obs['predicted_in_advance']} "
              f"(only-during {obs['detected_only_during_seizure']}, missed {obs['missed']}) | "
              f"chance mean {ne.mean():.1f}, 95th pct {np.percentile(ne, 95):.0f}, p={p:.3f} | "
              f"FA/h observed {obs['false_alarms_per_hour']:.1f} vs chance {np.nanmean(nf):.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="chbmit,siena")
    p.add_argument("--models", default="logreg,hgb")
    p.add_argument("--plant", type=float, default=0.0, help="synthetic ramp effect (power check)")
    p.add_argument("--train-cap", type=int, default=120000)
    p.add_argument("--shifts", type=int, default=40)
    p.add_argument("--placebo", type=int, default=100, help="placebo anchors per patient")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    self_test()
    selected = [d.strip() for d in args.datasets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    if "hgb" in models:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            print("scikit-learn not installed -> skipping hgb")
            models.remove("hgb")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X, y, t, f, pc, grp, blocks, gnames = load_cache(selected)
    print(f"datasets={selected} | {len(y):,} windows | {len(blocks)} patient blocks | {len(gnames)} LOSO groups | "
          f"labels: interictal={int((y == 0).sum()):,} preictal={int((y == 1).sum()):,} ictal={int((y == 2).sum()):,}")
    if args.plant:
        plant_signal(X, y, t, blocks, args.plant)

    scores, fold_normalization, fold_calibration = loso(
        X, y, grp, gnames, models, args.train_cap, args.seed)
    tag = "_".join(selected) + (f"_plant{args.plant}" if args.plant else "")
    out = ROOT / "checkpoints" / f"loso_oof_{tag}.npz"
    np.savez_compressed(out, t=t, y=y, f=f, pc=pc, **scores)
    dataset_fp = build_dataset_provenance()["dataset_fingerprint"]
    write_artifact_manifest(out, dataset_fp, "loso_oof",
                            {"datasets": selected, "seed": args.seed,
                             "models": models, "train_cap": args.train_cap,
                             "fold_normalization": fold_normalization,
                             "fold_calibration": fold_calibration})
    print(f"saved out-of-fold scores to {out}")

    for m in models:
        analyse(m, scores[m], y, t, f, pc, blocks, args, rng, fold_calibration[m])


if __name__ == "__main__":
    main()
