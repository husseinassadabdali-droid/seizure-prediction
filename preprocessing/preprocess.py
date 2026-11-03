"""
Preprocessing pipeline for EEG seizure prediction.
Reads raw EDF files from multiple datasets (CHB-MIT, Siena, ...), applies
bandpass/notch filtering, standardizes channel order per dataset, and
rejects recordings with missing or ambiguous target channels before saving.
Channel lookup is case-insensitive to handle naming inconsistencies across
files within the same dataset (e.g. "EEG Fp2" vs "EEG FP2").
Records with missing or ambiguous target channels are withheld and not saved as usable arrays.

Note: Z-score normalization is intentionally not applied here. The MLP and TCN
training paths fit frozen statistics from training patients after splitting.
Legacy/demo callers may still select per-patient normalization in dataset.py.
"""

import mne
import numpy as np
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

# ---- Config ----
TARGET_SFREQ = 250  # Hz, unified sampling rate across all datasets
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0

DATASETS = {
    "chbmit": {
        "raw_dir": "chbmit",
        "notch_freq": 60.0,
        # CHB-MIT is recorded as pre-defined bipolar pairs. These 6 channels
        # are the closest available proxies to our target frontal/temporal
        # positions (Fp1, Fp2, F7, F8, T3=T7, T4=T8).
        "target_channels": [
            "FP1-F7",
            "FP2-F8",
            "F7-T7",
            "F8-T8",
            "FP1-F3",
            "FP2-F4",
        ],
    },
    "siena": {
        "raw_dir": "siena",
        "notch_freq": 50.0,  # Italy -- 50Hz mains
        "target_channels": [
            "EEG Fp1",
            "EEG Fp2",
            "EEG F7",
            "EEG F8",
            "EEG T3",
            "EEG T4",
        ],
    },
}


def match_target_channels(available_channels: list, target_channels: list):
    """Match exact channel labels case-insensitively; never infer substitutes.

    Returns (indices, missing, ambiguous). Whitespace around EDF labels is
    ignored. A target with duplicate matching labels is ambiguous and is not
    selected, avoiding the old dict-overwrite/first-index behavior.
    """
    matches = {}
    for index, name in enumerate(available_channels):
        key = str(name).strip().casefold()
        matches.setdefault(key, []).append(index)

    indices, missing, ambiguous = [], [], []
    for target in target_channels:
        key = str(target).strip().casefold()
        found = matches.get(key, [])
        if len(found) == 1:
            indices.append(found[0])
        elif not found:
            indices.append(None)
            missing.append(target)
        else:
            indices.append(None)
            ambiguous.append(target)
    return indices, missing, ambiguous


def load_and_filter(edf_path: str, notch_freq: float) -> mne.io.Raw:
    """Load an EDF file and apply bandpass + notch filtering."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Do not silently drop duplicate labels. The target matcher treats an
    # ambiguous exact label as unavailable, and process_file refuses to save
    # the recording. MNE may normalize duplicate EDF labels; matching uses the
    # channel names available after reading.

    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _edf_header_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.sha256(handle.read(256)).hexdigest()


def _canonical_fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _processing_config_fingerprint(dataset_name: str) -> str:
    config = DATASETS[dataset_name]
    return _canonical_fingerprint({
        "dataset": dataset_name, "target_sfreq": TARGET_SFREQ,
        "bandpass_low": BANDPASS_LOW, "bandpass_high": BANDPASS_HIGH,
        "notch_freq": config.get("notch_freq"),
        "channel_order": list(config.get("target_channels", [])), "units": "V",
        "preprocess_code_sha256": _sha256_file(Path(__file__).resolve()),
        "mne_version": getattr(mne, "__version__", None), "numpy_version": np.__version__,
    })


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def process_file(edf_path: Path, output_dir: Path, target_channels: list,
                  notch_freq: float, dataset_name: str = None) -> list:
    """
    Extract target channels via case-insensitive exact-name lookup.
    Missing or ambiguous targets raise before an output NPY is written.
    Returns an empty list on success. No Z-score is applied.
    """
    if dataset_name is not None and dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset for processing provenance: {dataset_name}")

    raw = load_and_filter(str(edf_path), notch_freq)
    available_ch_names = raw.info["ch_names"]
    n_samples = raw.n_times
    if n_samples <= 0:
        del raw
        gc.collect()
        raise ValueError("refusing to save recording with zero samples")

    channel_indices, missing, ambiguous = match_target_channels(
        available_ch_names, target_channels)

    missing_in_this_file = list(missing) + [f"{name} (ambiguous duplicate label)"
                                            for name in ambiguous]
    if missing_in_this_file:
        del raw
        gc.collect()
        raise ValueError(
            "refusing to save recording with missing/ambiguous target channels: "
            + ", ".join(missing_in_this_file))

    final_data = np.empty((len(target_channels), n_samples), dtype=np.float32)
    for i, ch_idx in enumerate(channel_indices):
        final_data[i, :] = raw.get_data(picks=[ch_idx])[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{edf_path.stem}.npy"
    fd, temp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, final_data, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        provenance_path = output_path.with_name(output_path.name + ".provenance.json")
        if provenance_path.exists():
            provenance_path.unlink()
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    # Production dataset runs must identify the dataset so the evidence can be
    # checked against the current channel/filter policy. Legacy utility callers
    # may omit it, in which case no verified sidecar is issued.
    if dataset_name is not None:
        evidence = {
            "schema_version": 1, "status": "VERIFIED",
            "source_file": Path(edf_path).name,
            "source_fingerprint": _edf_header_sha256(edf_path),
            "source_size": Path(edf_path).stat().st_size,
            "preprocessing_config_fingerprint": _processing_config_fingerprint(dataset_name),
            "channel_order": list(target_channels), "sample_rate_hz": TARGET_SFREQ,
            "units": "V", "dataset": dataset_name,
            "output_file": output_path.name, "output_size": output_path.stat().st_size,
            "output_sha256": _sha256_file(output_path),
        }
        _atomic_json(output_path.with_name(output_path.name + ".provenance.json"), evidence)

    del raw, final_data
    gc.collect()

    return []


def process_dataset(dataset_name: str, project_root: Path, config: dict):
    raw_dir = project_root / "data" / "raw" / config["raw_dir"]
    processed_dir = project_root / "data" / "processed" / config["raw_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_channels = config["target_channels"]
    notch_freq = config["notch_freq"]

    edf_files = sorted(raw_dir.rglob("*.edf"))
    total_files = len(edf_files)
    print(f"\n{'#'*50}")
    print(f"# Dataset: {dataset_name}")
    print(f"{'#'*50}")
    print(f"Found {total_files} EDF files in {raw_dir}")

    if total_files == 0:
        print(f"[!] No EDF files found for '{dataset_name}' -- skipping.")
        return

    files_with_missing_channels = 0
    missing_stats = {ch: 0 for ch in target_channels}

    for edf_path in edf_files:
        try:
            print(f"Processing: {edf_path.name} ... ", end="")
            missing = process_file(edf_path, processed_dir, target_channels, notch_freq, dataset_name=dataset_name)

            if missing:
                files_with_missing_channels += 1
                for ch in missing:
                    missing_stats[ch] += 1
                print(f"[!] Missing: {missing}")
            else:
                print("[OK]")

        except Exception as e:
            print(f"[ERROR] Failed: {e}")

    print("\n" + "=" * 40)
    print(f" DIAGNOSTIC REPORT -- {dataset_name}")
    print("=" * 40)
    print(f"Total files processed : {total_files}")
    pct = (files_with_missing_channels / total_files) * 100
    print(f"Files missing channels: {files_with_missing_channels} ({pct:.2f}%)")

    if files_with_missing_channels > 0:
        print("\nMissing frequency per channel:")
        for ch, count in missing_stats.items():
            if count > 0:
                print(f"  - {ch}: {count} files")
    print("=" * 40)


if __name__ == "__main__":
    # Dynamic paths based on this script's own location -- works regardless
    # of the current working directory or username.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Runs every dataset defined in DATASETS above. To process only one
    # dataset, call: process_dataset("chbmit", project_root, DATASETS["chbmit"])
    for name, cfg in DATASETS.items():
        process_dataset(name, project_root, cfg)
