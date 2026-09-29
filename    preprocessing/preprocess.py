"""
Preprocessing pipeline for EEG seizure prediction.
Reads raw EDF files, applies filtering, selects the 6 target channels
(closest available proxies to Fp1/Fp2/F7/F8/T3/T4), and saves the
processed signal for downstream use by both the TCN and MLP branches.
"""

import mne
import numpy as np
from pathlib import Path

# ---- Config ----
TARGET_SFREQ = 250  # Hz, unified sampling rate across all datasets
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 45.0
NOTCH_FREQ = 60.0  # use 50.0 for TUH/Siena depending on recording region

# CHB-MIT is recorded as pre-defined bipolar pairs. These 6 channels are
# the closest available proxies to our target frontal/temporal positions
# (Fp1, Fp2, F7, F8, T3=T7, T4=T8).
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
    raw.drop_channels([ch for ch in raw.info["ch_names"]
                        if raw.info["ch_names"].count(ch) > 1][1::2])

    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    if raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def select_target_channels(raw: mne.io.Raw, target_channels: list) -> mne.io.Raw:
    """Keep only the channels available in this recording that match
    our target list. Warns if some target channels are missing."""
    available = [ch for ch in target_channels if ch in raw.info["ch_names"]]
    missing = set(target_channels) - set(available)
    if missing:
        print(f"  [!] Missing channels in this file: {missing}")
    return raw.pick(available)


def process_file(edf_path: Path, output_dir: Path, target_channels: list):
    print(f"Processing: {edf_path.name}")
    raw = load_and_filter(str(edf_path))
    raw = select_target_channels(raw, target_channels)

    data = raw.get_data()  # shape: (n_channels, n_samples)

    output_path = output_dir / f"{edf_path.stem}.npy"
    np.save(output_path, data)
    print(f"  -> saved {data.shape} to {output_path}")


def process_directory(raw_dir: str, processed_dir: str, target_channels: list):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(raw_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in {raw_dir}")

    for edf_path in edf_files:
        try:
            process_file(edf_path, processed_dir, target_channels)
        except Exception as e:
            print(f"  [ERROR] Failed on {edf_path.name}: {e}")


if __name__ == "__main__":
    process_directory(
        raw_dir="../data/raw/chbmit",
        processed_dir="../data/processed/chbmit",
        target_channels=CHBMIT_TARGET_CHANNELS,
    )