"""
Preprocessing pipeline for EEG seizure prediction.
Reads raw EDF files from multiple datasets (CHB-MIT, Siena, ...), applies
bandpass/notch filtering, standardizes channel order per dataset, and
handles missing channels via zero-padding (preserving raw amplitudes).
Channel lookup is case-insensitive to handle naming inconsistencies across
files within the same dataset (e.g. "EEG Fp2" vs "EEG FP2").
Includes a diagnostic report for missing channels to inform future
masking decisions.

Note: Z-score normalization is intentionally NOT applied here. It is applied
later in dataset.py, computed per-patient across all of that patient's
recordings combined -- not per single file -- to preserve real inter-file
baseline differences and keep raw values available for future recalibration
against the dry-electrode device.
"""

import mne
import numpy as np
import gc
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


def load_and_filter(edf_path: str, notch_freq: float) -> mne.io.Raw:
    """Load an EDF file and apply bandpass + notch filtering."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Some files have duplicate channel names; drop duplicates
    ch_names = raw.info["ch_names"]
    duplicates = [ch for ch in ch_names if ch_names.count(ch) > 1]
    if duplicates:
        raw.drop_channels(duplicates[1::2])

    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def process_file(edf_path: Path, output_dir: Path, target_channels: list,
                  notch_freq: float) -> list:
    """
    Extract target channels via case-insensitive name lookup (handles
    naming inconsistencies like "EEG Fp2" vs "EEG FP2" across files of
    the same dataset). Missing channels are zero-padded so every output
    array has an identical shape (len(target_channels), n_samples).
    Returns list of missing channels for diagnostic tracking. No Z-score
    applied -- raw filtered amplitudes are saved as-is.
    """
    raw = load_and_filter(str(edf_path), notch_freq)
    available_ch_names = raw.info["ch_names"]
    n_samples = raw.n_times

    # Case-insensitive lookup table: lowercase name -> actual channel name
    lower_to_actual = {ch.lower(): ch for ch in available_ch_names}

    final_data = np.zeros((len(target_channels), n_samples), dtype=np.float32)
    missing_in_this_file = []

    for i, target_ch in enumerate(target_channels):
        actual_ch = lower_to_actual.get(target_ch.lower())
        if actual_ch is not None:
            ch_idx = available_ch_names.index(actual_ch)
            final_data[i, :] = raw.get_data(picks=[ch_idx])[0]
        else:
            missing_in_this_file.append(target_ch)
            # final_data[i, :] stays zero (zero-padding)

    output_path = output_dir / f"{edf_path.stem}.npy"
    np.save(output_path, final_data)

    del raw, final_data
    gc.collect()

    return missing_in_this_file


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
            missing = process_file(edf_path, processed_dir, target_channels, notch_freq)

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