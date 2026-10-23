import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN = ROOT / "data/processed/feature_cache"
DEFAULT_NOISY = ROOT / "data/processed/feature_cache_aug"
REQUIRED_META = {
    "y", "time_sec", "start_sample", "index_idx", "file_id",
    "patient_code", "patient_names", "dataset_code", "dataset_names",
}


def fail(message):
    raise ValueError(message)


def load_cache(path, expected_augment=None, chunk_rows=65536):
    path = Path(path)
    for name in ("features.npy", "meta.npz", "info.json"):
        if not (path / name).is_file():
            fail(f"{path}: missing {name}")

    with (path / "info.json").open(encoding="utf-8") as stream:
        info = json.load(stream)

    X = np.load(path / "features.npy", mmap_mode="r", allow_pickle=False)
    with np.load(path / "meta.npz", allow_pickle=False) as archive:
        missing = REQUIRED_META.difference(archive.files)
        if missing:
            fail(f"{path}: missing metadata fields: {sorted(missing)}")
        meta = {key: archive[key].copy() for key in REQUIRED_META}

    if X.ndim != 2:
        fail(f"{path}: features must be 2D, got shape {X.shape}")
    n = X.shape[0]
    if n == 0:
        fail(f"{path}: cache contains no rows")
    if info.get("n_windows") != n:
        fail(f"{path}: info n_windows={info.get('n_windows')} but features has {n} rows")
    if info.get("n_features") != X.shape[1]:
        fail(f"{path}: info n_features does not match feature matrix shape")
    if len(info.get("feature_names", [])) != X.shape[1]:
        fail(f"{path}: feature_names length does not match feature matrix width")
    if expected_augment is not None and bool(info.get("augment")) != expected_augment:
        fail(f"{path}: augment={info.get('augment')}, expected {expected_augment}")

    for key in ("y", "time_sec", "start_sample", "index_idx", "file_id",
                "patient_code", "dataset_code"):
        if meta[key].ndim != 1 or len(meta[key]) != n:
            fail(f"{path}: {key} must be a vector with {n} elements")

    allowed_labels = np.array([0, 1, 2])
    if not np.isin(meta["y"], allowed_labels).all():
        fail(f"{path}: labels contain values outside 0, 1, 2")
    if not np.isfinite(meta["time_sec"]).all():
        fail(f"{path}: time_sec contains non-finite values")
    if (meta["start_sample"] < 0).any() or (meta["index_idx"] < 0).any():
        fail(f"{path}: negative sample/index values")
    if len(np.unique(meta["index_idx"])) != n:
        fail(f"{path}: index_idx contains duplicate rows")
    if len(meta["patient_names"]) == 0 or len(meta["dataset_names"]) == 0:
        fail(f"{path}: patient_names or dataset_names is empty")
    if (meta["patient_code"] < 0).any() or (meta["patient_code"] >= len(meta["patient_names"])).any():
        fail(f"{path}: patient_code references a missing patient name")
    if (meta["dataset_code"] < 0).any() or (meta["dataset_code"] >= len(meta["dataset_names"])).any():
        fail(f"{path}: dataset_code references a missing dataset name")

    for start in range(0, n, chunk_rows):
        block = np.asarray(X[start:start + chunk_rows])
        if not np.isfinite(block).all():
            fail(f"{path}: non-finite feature value in rows {start}:{start + len(block)}")

    counts = {str(label): int(np.count_nonzero(meta["y"] == label)) for label in (0, 1, 2)}
    recorded = info.get("label_counts")
    if recorded is not None:
        expected_counts = {
            "interictal": counts["0"],
            "preictal": counts["1"],
            "ictal": counts["2"],
        }
        if recorded != expected_counts:
            fail(f"{path}: label_counts in info.json do not match meta.npz")

    return X, meta, info, counts


def compare_caches(clean_path, noisy_path):
    _, clean, clean_info, _ = load_cache(clean_path, expected_augment=False)
    _, noisy, noisy_info, _ = load_cache(noisy_path, expected_augment=True)

    for key in ("y", "time_sec", "start_sample", "index_idx", "file_id",
                "patient_code", "patient_names", "dataset_code", "dataset_names"):
        if not np.array_equal(clean[key], noisy[key]):
            fail(f"clean/noisy cache metadata differs in {key}")

    if clean_info.get("feature_names") != noisy_info.get("feature_names"):
        fail("clean/noisy caches use different feature definitions")
    if clean_info.get("n_windows") != noisy_info.get("n_windows"):
        fail("clean/noisy caches have different window counts")

    print("Clean/noisy metadata alignment: OK")
    print(f"Windows: {clean_info['n_windows']:,}; features: {clean_info['n_features']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate feature-cache shape, metadata, label counts and finite values."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--compare-noisy", type=Path, default=None)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_rows < 1:
        raise SystemExit("--chunk-rows must be a positive integer")

    _, _, info, counts = load_cache(args.cache_dir, expected_augment=False if args.compare_noisy else None,
                                    chunk_rows=args.chunk_rows)
    print(f"Cache integrity: OK ({args.cache_dir})")
    print(f"Shape: {info['n_windows']:,} x {info['n_features']}")
    print(f"Labels: interictal={counts['0']:,}, preictal={counts['1']:,}, ictal={counts['2']:,}")
    print(f"Patients: {len(np.load(args.cache_dir / 'meta.npz', allow_pickle=False)['patient_names'])}")

    if args.compare_noisy:
        compare_caches(args.cache_dir, args.compare_noisy)


if __name__ == "__main__":
    main()
