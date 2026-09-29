"""
Preprocessing pipeline for EEG seizure prediction.
Reads raw EDF files, applies filtering, and extracts a fixed set of
6 target channels in strict, consistent order (frontal/temporal proxies).
Missing channels are zero-padded so every output array has an identical
shape (6, n_samples), which the TCN/MLP branches require.

Note: Z-score normalization is intentionally NOT applied here. It is
applied later in dataset.py, computed per-patient across all of that
patient's recordings combined — not per single file — to preserve
real inter-file baseline differences and keep the raw values available
for future recalibration (e.g. against the dry-electrode device).
"""

import mne
import numpy as np
import gc
from pathlib import Path

# ---- Config ----
TARGET_SFREQ = 250  # Hz, unified sampling rate across all datasets
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 60.0  # use 50.0 for TUH/Siena depending on recording region

# CHB-MIT is recorded as pre-defined bipolar pairs. These 6 channels are
# the closest available proxies to our target frontal/temporal positions
# (Fp1, Fp2, F7, F8, T3=T7, T4=T8). Order here is the fixed channel order
# used in every output array — must stay consistent across all files.
CHBMIT_TARGET_CHANNELS = [
    "FP1-F7",  # proxy for Fp1 / F7 (left frontal-temporal)
    "FP2-F8",  # proxy for Fp2 / F8 (right frontal-temporal)
    "F7-T7",   # proxy for F7 / T3 (left temporal)
    "F8-T8",   # proxy for F8 / T4 (right temporal)
    "FP1-F3",  # secondary left frontal proxy
    "FP2-F4",  # secondary right frontal proxy
]


def load_and_filter(edf_path: str) -> mne.io.Raw:
    """Load an EDF file and apply bandpass + notch filtering."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Some CHB-MIT files have duplicate channel names; drop duplicates
    ch_names = raw.info["ch_names"]
    duplicates = [ch for ch in ch_names if ch_names.count(ch) > 1]
    if duplicates:
        raw.drop_channels(duplicates[1::2])

    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def extract_fixed_channels(raw: mne.io.Raw, target_channels: list) -> np.ndarray:
    """
    Extract target channels by explicit name-index lookup (not raw.pick(),
    which does not guarantee output order). Missing channels are
    zero-padded so the output shape is always (len(target_channels), n_samples).
    """
    available_ch_names = raw.info["ch_names"]
    n_samples = raw.n_times

    final_data = np.zeros((len(target_channels), n_samples), dtype=np.float32)
    missing_channels = []

    for i, target_ch in enumerate(target_channels):
        if target_ch in available_ch_names:
            ch_idx = available_ch_names.index(target_ch)
            final_data[i, :] = raw.get_data(picks=[ch_idx])[0].astype(np.float32)
        else:
            missing_channels.append(target_ch)
            # final_data[i, :] stays zero (zero-padding)

    if missing_channels:
        print(f"  [!] Missing channels replaced with zeros: {missing_channels}")

    return final_data, missing_channels


def process_file(edf_path: Path, output_dir: Path, target_channels: list):
    print(f"Processing: {edf_path.name}")

    raw = load_and_filter(str(edf_path))
    final_data, missing_channels = extract_fixed_channels(raw, target_channels)

    output_path = output_dir / f"{edf_path.stem}.npy"
    np.save(output_path, final_data)
    print(f"  -> saved {final_data.shape} to {output_path}"
          f"{' (with zero-padded channels)' if missing_channels else ''}")

    # Explicit memory cleanup -- important when looping over tens of GB
    del raw, final_data
    gc.collect()


def process_directory(raw_dir: str, processed_dir: str, target_channels: list):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(raw_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in {raw_dir}")

    n_missing_files = 0
    for edf_path in edf_files:
        try:
            process_file(edf_path, processed_dir, target_channels)
        except Exception as e:
            print(f"  [ERROR] Failed on {edf_path.name}: {e}")

    print(f"\nDone. Processed {len(edf_files)} files from {raw_dir}.")


if __name__ == "__main__":
    process_directory(
        raw_dir="../data/raw/chbmit",
        processed_dir="../data/processed/chbmit",
        target_channels=CHBMIT_TARGET_CHANNELS,
    )