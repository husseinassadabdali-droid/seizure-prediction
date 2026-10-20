"""
Same-split comparison: the trained TCN (3-class) vs a tree/linear model on the cached
hand-crafted features, on the SAME patients and the SAME windows.

Inputs
  checkpoints/tcn_eval_probs.npz   TCN probabilities for the full val and test windows
  checkpoints/mlp_split.json       patient split shared by the MLP and TCN runs
  data/processed/feature_cache     features of every window (native dataset order)

Scores
  detection  (ictal vs everything else): TCN P(ictal)  | model trained on train patients
  prediction (preictal vs interictal):   TCN P(pre)/(P(pre)+P(inter)) | model likewise

Pre-specified PRIMARY result: median per-patient detection AUC over the val+test
patients that have ictal windows, TCN vs model. Everything else is descriptive.

Event-level detection uses CROSS-calibration: the alarm threshold for the test patients
is a quantile of the non-ictal scores of the val patients, and vice versa, so every
window is judged with a threshold derived from other patients.

Caveats (printed again at the end)
  * The TCN checkpoint was selected on the val patients (epoch with best 3-class
    balanced accuracy), so the val side is optimistic for the TCN only.
  * The TCN trained on noisy windows; --hgb-train noisy matches that for the model.
  * The comparison covers one split (about 10 evaluation patients), one TCN run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("evaluation", "features"):
    sys.path.insert(0, str(ROOT / _sub))

from handcrafted_features import FEATURE_NAMES  # noqa: E402
from loso_cache_analysis import ALIAS, robust_stats, transform, interictal_control  # noqa: E402
from detection_loso import (fit_predict, alarm_block, block_result, pooled, chance,  # noqa: E402
                            MAX_GAP, QUANTILES, SMOOTH_K)
from metrics import roc_auc_binary, _runs  # noqa: E402

CKPT = ROOT / "checkpoints"
CLEAN = ROOT / "data/processed/feature_cache"
NOISY = ROOT / "data/processed/feature_cache_aug"
MIN_WIN = 100
PRIMARY = (0.999, 3)


# ---------------- loading / alignment ----------------

def load_native(cache_dir, expect_augment):
    info = json.loads((cache_dir / "info.json").read_text())
    assert info["feature_names"] == FEATURE_NAMES, f"{cache_dir}: different feature set"
    assert not info["smoke"], f"{cache_dir} is a smoke cache"
    assert bool(info["augment"]) == expect_augment, f"{cache_dir}: unexpected augment flag"
    m = np.load(cache_dir / "meta.npz")
    return {"X": np.load(cache_dir / "features.npy"),
            "y": m["y"].astype(np.int64),
            "t": m["time_sec"].astype(np.float64),
            "fid": m["file_id"].astype(np.int64),
            "pcode": m["patient_code"].astype(np.int64),
            "pnames": [str(s) for s in m["patient_names"]]}


def split_masks(C, split):
    group_of_code = np.array([ALIAS.get(n, n) for n in C["pnames"]])
    group = group_of_code[C["pcode"]]
    masks = {k: np.isin(group, split[k]) for k in ("train", "val", "test")}
    total = sum(int(m.sum()) for m in masks.values())
    overlap = ((masks["train"] & masks["val"]) | (masks["train"] & masks["test"])
               | (masks["val"] & masks["test"])).any()
    assert total == len(group) and not overlap, \
        "mlp_split.json does not partition the cached windows (different patients/exclusions?)"
    return masks


def make_blocks(pc, pnames):
    cuts = np.flatnonzero(np.diff(pc) != 0) + 1
    bounds = np.concatenate([[0], cuts, [len(pc)]])
    blocks = [(int(a), int(b), pnames[pc[a]]) for a, b in zip(bounds[:-1], bounds[1:])]
    assert len(blocks) == len(np.unique(pc)), "windows of a patient are not contiguous"
    return blocks


# ---------------- models on the cached features ----------------

def train_models(Xsrc, y, train_rows, model_name, cap, seed):
    rng = np.random.default_rng(seed)

    pos = train_rows[y[train_rows] == 2]                       # detection: ictal vs rest
    neg = train_rows[y[train_rows] != 2]
    n_pos = min(cap // 5, len(pos))
    n_neg = min(len(neg) // 2, 4 * n_pos)
    sel = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])
    lab = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    c1, s1 = robust_stats(Xsrc[sel])
    f1 = fit_predict(model_name, transform(Xsrc[sel], c1, s1), lab)

    pos = train_rows[y[train_rows] == 1]                       # prediction: preictal vs interictal
    neg = train_rows[y[train_rows] == 0]
    n = min(cap // 2, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    lab = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    c2, s2 = robust_stats(Xsrc[sel])
    f2 = fit_predict(model_name, transform(Xsrc[sel], c2, s2), lab)

    return (lambda Z: f1(transform(Z, c1, s1))), (lambda Z: f2(transform(Z, c2, s2)))


# ---------------- event helpers ----------------

def events(scores, y, t, blocks, thrs, k):
    per = []
    for (a, b, name), thr in zip(blocks, thrs):
        yb, tb = y[a:b], t[a:b]
        runs = _runs(yb == 2, tb, MAX_GAP)
        alarm = alarm_block(scores[a:b], thr, k)
        per.append((name, a, b, runs, alarm) + block_result(alarm, yb, tb, runs))
    return per          # (name, a, b, runs, alarm, det, n, lat, fa, hours)


def duty(per, y):
    """Fraction of NON-ictal windows during which the smoothed alarm is on."""
    on = tot = 0
    for p in per:
        non = y[p[1]:p[2]] != 2
        on += int((p[4] & non).sum())
        tot += int(non.sum())
    return on / tot if tot else float("nan")


def self_test():
    t = np.arange(0, 400, 2.0)
    y = np.zeros(200, dtype=int)
    y[100:110] = 2
    s = np.zeros(200)
    s[50:60] = 1.0
    s[103:108] = 1.0
    per = events(s, y, t, [(0, 200, "x")], [0.5], 3)
    det, n, fah, lat = pooled(per)
    assert (det, n) == (1, 1) and abs(lat - 10.0) < 1e-9
    assert abs(fah - 1 / (190 * 2 / 3600)) < 1e-9
    assert abs(duty(per, y) - 8 / 190) < 1e-12
    dets, fas = chance(per, y, t, np.random.default_rng(0), 3)
    assert len(dets) == 3 and len(fas) == 3
    print("helper self-test: OK")


def short(n):
    return n.replace("chbmit/", "").replace("siena/", "")


# ---------------- main ----------------

def main():
    args = parse_args()
    self_test()
    label = args.model.upper()
    models = ["TCN", label]
    rng = np.random.default_rng(args.seed)

    split = json.loads((CKPT / "mlp_split.json").read_text())["patients"]
    C = load_native(CLEAN, expect_augment=False)
    masks = split_masks(C, split)
    X_train_src = C["X"]
    if args.hgb_train == "noisy":
        N = load_native(NOISY, expect_augment=True)
        for k in ("y", "t", "pcode"):
            assert np.array_equal(C[k], N[k]), f"clean and noisy caches differ in '{k}'"
        X_train_src = N["X"]

    tcn = np.load(CKPT / "tcn_eval_probs.npz")
    rows_of, Y, T, FID, blocks = {}, {}, {}, {}, {}
    for key in ("val", "test"):
        rows = np.flatnonzero(masks[key])
        n_tcn = len(tcn[f"{key}_y"])
        assert len(rows) == n_tcn, f"{key}: {len(rows):,} cached windows vs {n_tcn:,} in tcn_eval_probs.npz"
        assert np.array_equal(C["t"][rows], tcn[f"{key}_times"]) and \
            np.array_equal(C["y"][rows], tcn[f"{key}_y"]), \
            f"{key}: cached windows are NOT aligned with the TCN evaluation windows -- stop"
        rows_of[key], Y[key], T[key], FID[key] = rows, C["y"][rows], C["t"][rows], C["fid"][rows]
        blocks[key] = make_blocks(C["pcode"][rows], C["pnames"])
        print(f"{key}: {len(rows):,} windows, {len(blocks[key])} patients, aligned with the TCN windows: OK")

    print(f"\nTraining {label} on {int(masks['train'].sum()):,} windows of the TRAIN patients "
          f"(features: {args.hgb_train}) ...", flush=True)
    t0 = time.time()
    det_fn, pre_fn = train_models(X_train_src, C["y"], np.flatnonzero(masks["train"]),
                                  args.model, args.cap, args.seed)
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)

    S = {}
    for key in ("val", "test"):
        p = tcn[f"{key}_probs"].astype(np.float64)
        Xk = C["X"][rows_of[key]]
        S[key] = {"TCN": {"det": p[:, 2], "pre": p[:, 1] / (p[:, 0] + p[:, 1] + 1e-9)},
                  label: {"det": det_fn(Xk), "pre": pre_fn(Xk)}}

    # ---------- 1. prediction: preictal vs interictal, per patient ----------
    print("\n" + "=" * 96)
    print("1. PREDICTION: per-patient AUC, preictal vs interictal (ictal windows excluded)")
    print("   ctrl = mean |AUC-0.5| between two random halves of the patient's INTERICTAL files "
          "(what session identity alone produces)")
    print(f"{'patient':<9}{'split':>6}{'inter':>9}{'pre':>8}{models[0] + ' AUC':>10}{'ctrl':>7}"
          f"{models[1] + ' AUC':>11}{'ctrl':>7}")
    pred_rows, skipped = [], 0
    for key in ("val", "test"):
        for a, b, name in blocks[key]:
            yb = Y[key][a:b]
            n_i, n_p = int((yb == 0).sum()), int((yb == 1).sum())
            if n_i < MIN_WIN or n_p < MIN_WIN:
                skipped += 1
                continue
            m = yb != 2
            vals = []
            for mdl in models:
                s = S[key][mdl]["pre"][a:b]
                auc = roc_auc_binary(s[m], (yb == 1)[m])
                ctrl = interictal_control(s, yb, FID[key][a:b], rng)
                vals.append((auc, ctrl))
            pred_rows.append((vals[0][0], vals[1][0], vals[0][1], vals[1][1]))
            fc = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"{short(name):<9}{key:>6}{n_i:>9}{n_p:>8}{vals[0][0]:>10.2f}{fc(vals[0][1]):>7}"
                  f"{vals[1][0]:>11.2f}{fc(vals[1][1]):>7}")
    if pred_rows:
        A = np.array([[r[0], r[1]] for r in pred_rows])
        print(f"patients: {len(A)} (skipped {skipped} with < {MIN_WIN} windows of a class)")
        for i, mdl in enumerate(models):
            print(f"  {mdl:<7} median AUC {np.median(A[:, i]):.3f} | AUC>0.5 in {int((A[:, i] > 0.5).sum())}/{len(A)}")

    # ---------- 2. detection: per-patient AUC (primary) ----------
    print("\n" + "=" * 96)
    print("2. DETECTION (PRIMARY): per-patient AUC, ictal vs all other windows (threshold-free)")
    print(f"{'patient':<9}{'split':>6}{'seizures':>9}{models[0]:>9}{models[1]:>9}")
    det_rows = []
    for key in ("val", "test"):
        for a, b, name in blocks[key]:
            pos = Y[key][a:b] == 2
            if pos.sum() < 1 or (~pos).sum() < MIN_WIN:
                continue
            n_sz = len(_runs(pos, T[key][a:b], MAX_GAP))
            aucs = [roc_auc_binary(S[key][mdl]["det"][a:b], pos) for mdl in models]
            det_rows.append(aucs)
            print(f"{short(name):<9}{key:>6}{n_sz:>9}{aucs[0]:>9.2f}{aucs[1]:>9.2f}")
    if det_rows:
        D = np.array(det_rows)
        d = D[:, 0] - D[:, 1]
        print(f"patients with seizures: {len(D)}")
        for i, mdl in enumerate(models):
            print(f"  {mdl:<7} MEDIAN AUC {np.median(D[:, i]):.3f}")
        print(f"  {models[0]} minus {models[1]}: median {np.median(d):+.3f}, "
              f"{models[0]} higher in {int((d > 0).sum())}/{len(d)} patients (descriptive)")

    # ---------- 3. detection events, cross-calibrated thresholds ----------
    nv = len(Y["val"])
    y_all = np.concatenate([Y["val"], Y["test"]])
    t_all = np.concatenate([T["val"], T["test"]])
    blocks_all = list(blocks["val"]) + [(a + nv, b + nv, n) for a, b, n in blocks["test"]]
    side = ["val"] * len(blocks["val"]) + ["test"] * len(blocks["test"])
    score_all = {m: np.concatenate([S["val"][m]["det"], S["test"][m]["det"]]) for m in models}

    def thrs_for(mdl, q):
        cal = {"val": np.quantile(S["test"][mdl]["det"][Y["test"] != 2], q),
               "test": np.quantile(S["val"][mdl]["det"][Y["val"] != 2], q)}
        return [cal[s] for s in side]

    print("\n" + "=" * 96)
    print("3. DETECTION EVENTS (val+test patients; threshold = non-ictal score quantile from the OTHER split)")
    print("   cells: detected/seizures (rate) | false alarms per hour | alarm duty = share of non-ictal "
          "windows with the alarm on | median latency")
    print(f"{'setting':<15}" + "".join(f"{m:>50}" for m in models))
    for q in QUANTILES:
        for k in SMOOTH_K:
            line = f"{'q=%.3f k=%d' % (q, k):<15}"
            for mdl in models:
                per = events(score_all[mdl], y_all, t_all, blocks_all, thrs_for(mdl, q), k)
                dd, nn, fah, lat = pooled(per)
                line += f"{f'{dd}/{nn} ({dd / max(nn, 1):.0%}) | {fah:.2f}/h | duty {duty(per, y_all):.1%} | {lat:.0f}s':>50}"
            print(line + ("   <-- primary setting" if (q, k) == PRIMARY else ""))

    print(f"\nprimary setting (q={PRIMARY[0]}, k={PRIMARY[1]}) vs circular-shift chance ({args.shifts} shifts):")
    for mdl in models:
        per = events(score_all[mdl], y_all, t_all, blocks_all, thrs_for(mdl, PRIMARY[0]), PRIMARY[1])
        dd, nn, fah, _ = pooled(per)
        cd, cf = chance(per, y_all, t_all, rng, args.shifts)
        p = (1 + np.sum(cd >= dd)) / (1 + len(cd))
        print(f"  {mdl:<7} observed {dd}/{nn} | chance mean {cd.mean():.1f}, 95th pct {np.percentile(cd, 95):.0f}, "
              f"p={p:.3f} | FA/h {fah:.2f} vs chance {cf.mean():.2f}")

    out = {"val_y": Y["val"], "val_t": T["val"], "test_y": Y["test"], "test_t": T["test"]}
    for key in ("val", "test"):
        for mdl in models:
            for task in ("det", "pre"):
                out[f"{key}_{mdl}_{task}"] = S[key][mdl][task]
    np.savez_compressed(CKPT / "same_split_scores.npz", **out)
    print(f"\nsaved scores to {CKPT / 'same_split_scores.npz'}")
    print("\nCAVEATS: TCN checkpoint selected on the val patients (optimistic on val for the TCN only); "
          f"TCN trained on noisy windows, {label} on {args.hgb_train} features; one split, about 10 "
          "evaluation patients, one TCN run; CHB-MIT/Siena seizures are not all GTC; wet electrodes.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    p.add_argument("--hgb-train", choices=["clean", "noisy"], default="clean")
    p.add_argument("--cap", type=int, default=120000)
    p.add_argument("--shifts", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
