"""
One-time cache of hand-crafted features for EVERY window of the labeled dataset.

Why: extracting features on the fly costs ~20 min per epoch. With a cache,
cross-patient (LOSO) experiments, per-patient analyses and event-level tests
over all ~180 seizures run in minutes.

Layout (default data/processed/feature_cache/):
  shards/<seg>_<hash>.npy  per-recording feature shards (resumable)
  features.npy             (n_windows, 108) float32, in dataset index order
  meta.npz                 y, time_sec, start_sample, index_idx, file_id,
                           patient_code/names, dataset_code/names
  info.json                config used to build it

Shard names embed a hash of (path, window starts, patient mean/std, flags,
feature names), so shards built under different labeling/normalization are
never silently reused.

Features are computed from the CLEAN z-scored window by default. --augment
builds a second cache with fixed-seed dry-electrode noise (separate folder).

Usage:
  python3 build_feature_cache.py --smoke      # 4 recordings, minutes, full pipeline
  python3 build_feature_cache.py              # full build (resumable)
  python3 build_feature_cache.py --verify     # re-check an existing cache
  python3 build_feature_cache.py --augment    # noisy copy (not verifiable bitwise)
Load later with: from build_feature_cache import load_cache
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("training", "features", "preprocessing"):
    sys.path.insert(0, str(ROOT / _sub))

from dataset import (SeizureDataset, FS, WINDOW_SEC, STEP_SEC, PREICTAL_SEC,  # noqa: E402
                     POSTICTAL_SEC, BUFFER_SEC, EXCLUDE_PATIENTS)
from handcrafted_features import (extract_features_from_window, FEATURE_NAMES,  # noqa: E402
                                  N_FEATURES_PER_CHANNEL)
from noise_injection import apply_dry_electrode_augmentation  # noqa: E402

WIN = int(WINDOW_SEC * FS)
N_FEAT = 6 * N_FEATURES_PER_CHANNEL
CACHE_VERSION = 1

CONFIGS = [
    {"name": "chbmit", "processed_dir": ROOT / "data/processed/chbmit",
     "raw_dir": ROOT / "data/raw/chbmit", "notch_freq": 60.0},
    {"name": "siena", "processed_dir": ROOT / "data/processed/siena",
     "raw_dir": ROOT / "data/raw/siena", "notch_freq": 50.0},
]


def cache_dir_for(augment, smoke):
    name = "feature_cache" + ("_aug" if augment else "") + ("_smoke" if smoke else "")
    return ROOT / "data/processed" / name


def fmt_time(sec):
    if sec != sec or sec == float("inf"):
        return "?"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


# ---------------- segments / tasks ----------------

def build_segments(ds):
    """[(seg_id, lo, hi)] = contiguous index ranges, one per recording file."""
    idx = ds.index
    segs, lo = [], 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i]["npy_path"] != idx[lo]["npy_path"]:
            segs.append((len(segs), lo, i))
            lo = i
    assert len({idx[lo]["npy_path"] for _, lo, _ in segs}) == len(segs), \
        "windows of one file are not contiguous in the dataset index"
    return segs


def make_tasks(ds, segs, shard_dir, augment, seed):
    tasks = []
    for seg_id, lo, hi in segs:
        first = ds.index[lo]
        mean, std = ds.patient_stats_cache[(first["dataset"], first["patient_id"])]
        starts = np.fromiter((ds.index[i]["start_sample"] for i in range(lo, hi)),
                             dtype=np.int64, count=hi - lo)
        path = str(first["npy_path"])
        h = hashlib.md5()
        for part in (path.encode(), starts.tobytes(), mean.tobytes(), std.tobytes(),
                     f"{augment}|{seed}|{first['notch_freq']}|{CACHE_VERSION}".encode(),
                     "|".join(FEATURE_NAMES).encode()):
            h.update(part)
        out = shard_dir / f"{seg_id:05d}_{h.hexdigest()[:12]}.npy"
        tasks.append((seg_id, path, starts, mean, std, first["notch_freq"],
                      augment, seed, str(out)))
    return tasks


def work(task):
    """Compute (or reuse) the feature shard of one recording file."""
    seg_id, path, starts, mean, std, notch, augment, seed, out_path = task
    out = Path(out_path)
    n = len(starts)
    if out.exists():
        try:
            if np.load(out, mmap_mode="r").shape == (n, N_FEAT):
                return seg_id, n, True
        except Exception:
            pass

    arr = np.load(path).astype(np.float32, copy=False)
    arr = (arr - mean[:, None]) / std[:, None]      # same ops as SeizureDataset.__getitem__
    rng = np.random.default_rng([seed, seg_id]) if augment else None

    feats = np.empty((n, N_FEAT), dtype=np.float32)
    for k in range(n):
        s = int(starts[k])
        w = arr[:, s:s + WIN]
        if augment:
            w = apply_dry_electrode_augmentation(w, fs=FS, notch_freq=notch, rng=rng)
        feats[k] = extract_features_from_window(w, fs=FS)

    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, feats)
    os.replace(tmp, out)
    return seg_id, n, False


# ---------------- build ----------------

def build(args):
    ds = SeizureDataset(CONFIGS, apply_augmentation=False)
    segs_all = build_segments(ds)
    if args.smoke:
        keep = sorted({0, 1, len(segs_all) // 2, len(segs_all) - 1})
        segs = [segs_all[i] for i in keep]
        print(f"SMOKE MODE: {len(segs)} of {len(segs_all)} recordings")
    else:
        segs = segs_all

    cache_dir = args.cache_dir or cache_dir_for(args.augment, args.smoke)
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = make_tasks(ds, segs, shard_dir, args.augment, args.seed)
    total = sum(len(t[1 + 1]) for t in tasks)   # len(starts)
    print(f"{len(tasks)} recordings, {total:,} windows -> {cache_dir} "
          f"({args.workers} workers, augment={args.augment})")

    done = computed = 0
    t_start, last_print = time.time(), 0.0
    with Pool(args.workers) as pool:
        for _, n, was_cached in pool.imap_unordered(work, tasks, chunksize=1):
            done += n
            if not was_cached:
                computed += n
            now = time.time()
            if now - last_print > 60 or done == total:
                rate = computed / (now - t_start) if computed else 0.0
                eta = (total - done) / rate if rate else float("nan")
                print(f"  {done:>9,}/{total:,} windows ({done / max(total, 1):5.1%}) | "
                      f"{rate:5.0f} win/s | elapsed {fmt_time(now - t_start)} | ETA {fmt_time(eta)}",
                      flush=True)
                last_print = now

    # ---- merge in dataset-index order ----
    print("Merging shards...", flush=True)
    X = np.empty((total, N_FEAT), dtype=np.float32)
    row = 0
    for task, (_, lo, hi) in zip(tasks, segs):
        shard = np.load(task[-1])
        assert shard.shape == (hi - lo, N_FEAT), f"bad shard {task[-1]}"
        X[row:row + len(shard)] = shard
        row += len(shard)
    assert row == total
    assert np.isfinite(X).all(), "non-finite values in the merged cache"

    index_idx = np.concatenate([np.arange(lo, hi, dtype=np.int64) for _, lo, hi in segs])
    entries = [ds.index[int(i)] for i in index_idx]
    y = np.fromiter((e["label"] for e in entries), dtype=np.int8, count=total)
    time_sec = np.fromiter((e["time_sec"] for e in entries), dtype=np.float64, count=total)
    start_sample = np.fromiter((e["start_sample"] for e in entries), dtype=np.int64, count=total)
    patient_names = sorted({f"{e['dataset']}/{e['patient_id']}" for e in entries})
    pcode = {n: k for k, n in enumerate(patient_names)}
    patient_code = np.fromiter((pcode[f"{e['dataset']}/{e['patient_id']}"] for e in entries),
                               dtype=np.int32, count=total)
    dataset_names = sorted({e["dataset"] for e in entries})
    dcode = {n: k for k, n in enumerate(dataset_names)}
    dataset_code = np.fromiter((dcode[e["dataset"]] for e in entries), dtype=np.int8, count=total)
    file_id = np.repeat(np.array([s for s, _, _ in segs], dtype=np.int32),
                        [hi - lo for _, lo, hi in segs])

    tmp = cache_dir / "features.tmp.npy"
    np.save(tmp, X)
    os.replace(tmp, cache_dir / "features.npy")
    np.savez(cache_dir / "meta.npz", y=y, time_sec=time_sec, start_sample=start_sample,
             index_idx=index_idx, file_id=file_id, patient_code=patient_code,
             patient_names=np.array(patient_names), dataset_code=dataset_code,
             dataset_names=np.array(dataset_names))
    info = {
        "cache_version": CACHE_VERSION, "n_windows": int(total), "n_features": N_FEAT,
        "feature_names": FEATURE_NAMES, "augment": bool(args.augment), "seed": args.seed,
        "smoke": bool(args.smoke), "fs": FS, "window_sec": WINDOW_SEC, "step_sec": STEP_SEC,
        "preictal_sec": PREICTAL_SEC, "postictal_sec": POSTICTAL_SEC, "buffer_sec": BUFFER_SEC,
        "exclude_patients": sorted(map(list, EXCLUDE_PATIENTS)),
        "label_counts": {"interictal": int((y == 0).sum()), "preictal": int((y == 1).sum()),
                         "ictal": int((y == 2).sum())},
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (cache_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {total:,} x {N_FEAT} features ({X.nbytes / 1e6:.0f} MB), "
          f"{len(patient_names)} patient groups, labels: {info['label_counts']}")

    if args.augment:
        print("Verification skipped (noisy cache is not bitwise reproducible from the dataset).")
    else:
        verify(ds, cache_dir)


# ---------------- load / verify ----------------

def load_cache(cache_dir=None, mmap=True):
    """Returns (X, meta, info). X is memory-mapped by default (read-only)."""
    d = Path(cache_dir) if cache_dir else cache_dir_for(False, False)
    X = np.load(d / "features.npy", mmap_mode="r" if mmap else None)
    meta = {k: v for k, v in np.load(d / "meta.npz").items()}
    info = json.loads((d / "info.json").read_text())
    assert info["feature_names"] == FEATURE_NAMES, \
        "cache was built with a different feature set; rebuild it"
    assert X.shape == (info["n_windows"], N_FEAT)
    return X, meta, info


def verify(ds, cache_dir, n=100, seed=0):
    """(1) labels/times of every cached row equal the current dataset's;
    (2) n random windows recomputed through the dataset's own __getitem__."""
    X, meta, info = load_cache(cache_dir)
    assert not info["augment"], "verify only supports the clean cache"
    idx = meta["index_idx"]

    labels = np.fromiter((ds.index[int(i)]["label"] for i in idx), dtype=np.int8, count=len(idx))
    times = np.fromiter((ds.index[int(i)]["time_sec"] for i in idx), dtype=np.float64, count=len(idx))
    starts = np.fromiter((ds.index[int(i)]["start_sample"] for i in idx), dtype=np.int64, count=len(idx))
    assert np.array_equal(labels, meta["y"]), "labels differ from the current dataset"
    assert np.array_equal(times, meta["time_sec"]), "window times differ from the current dataset"
    assert np.array_equal(starts, meta["start_sample"]), "window starts differ from the current dataset"
    print("Verify 1/2: labels, times and window starts match the current dataset: OK")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    worst = 0.0
    for r in pick:
        window, _ = ds[int(idx[r])]
        f = extract_features_from_window(window.numpy(), fs=FS)
        worst = max(worst, float(np.abs(f - X[r]).max()))
        assert np.allclose(f, X[r], rtol=1e-4, atol=1e-5), f"feature mismatch at cache row {r}"
    print(f"Verify 2/2: {len(pick)} random windows recomputed via the dataset: OK "
          f"(max abs difference {worst:.2e})")


def parse_args():
    p = argparse.ArgumentParser(description="Build the hand-crafted feature cache.")
    p.add_argument("--smoke", action="store_true", help="4 recordings only (pipeline test)")
    p.add_argument("--augment", action="store_true", help="noisy copy (separate folder)")
    p.add_argument("--verify", action="store_true", help="only verify an existing cache")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        if args.augment:
            sys.exit("--verify supports only the clean cache")
        ds_ = SeizureDataset(CONFIGS, apply_augmentation=False)
        verify(ds_, args.cache_dir or cache_dir_for(False, args.smoke))
    else:
        build(args)
