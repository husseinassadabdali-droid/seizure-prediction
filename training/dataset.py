"""
PyTorch Dataset for EEG seizure prediction + detection (3-class).

Pipeline:
  1. Parse seizure annotation files (CHB-MIT summary.txt, Siena
     Seizures-list-*.txt) into per-file seizure intervals (seconds).
  2. Slide a window (default 4s, 2s step = 2s overlap) across each
     processed .npy recording and label each window as one of:
       0 = interictal (clean, far from any seizure)
       1 = preictal   (within PREICTAL_SEC before a seizure onset)
       2 = ictal      (during a seizure)
     Windows in the postictal period or in the buffer zone between
     preictal and the 4-hour interictal cutoff are EXCLUDED entirely
     (ambiguous, not used for training).
  3. Per-patient Z-score normalization: mean/std computed across ALL
     of that patient's recordings combined (not per-file, not per-
     window), as decided earlier -- preserves real baseline
     differences and keeps raw values recoverable.
  4. Optional dry-electrode noise augmentation (noise_injection.py),
     applied identically to the raw window before it's handed to
     both the TCN branch (raw) and the MLP branch (features computed
     from this same augmented window) -- enforced by applying noise
     once here, in the Dataset, not separately in each branch.

Known simplification: preictal/buffer windows are computed WITHIN
each file independently. For patients with back-to-back continuous
recordings, a seizure very close to a file's start could in reality
have preictal signal in the previous file; this version does not
stitch files together to capture that. Flagged as a future
refinement, not handled here.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from noise_injection import apply_dry_electrode_augmentation  # noqa: E402

# ---- Label constants ----
INTERICTAL = 0
PREICTAL = 1
ICTAL = 2

# ---- Windowing config ----
FS = 250  # Hz, matches preprocess.py TARGET_SFREQ
WINDOW_SEC = 4
STEP_SEC = 2  # 2s step on a 4s window = 2s overlap
PREICTAL_SEC = 30 * 60      # 30 minutes before seizure onset
POSTICTAL_SEC = 30 * 60     # 30 minutes after seizure end, excluded
BUFFER_SEC = 4 * 60 * 60    # 4 hours: interictal must be this far from any seizure


# ============================================================
# Annotation parsing
# ============================================================

def parse_chbmit_summary(summary_path: Path) -> dict:
    """
    Parse a CHB-MIT *-summary.txt file.
    Returns {edf_filename_stem: [(start_sec, end_sec), ...]}.
    Times are already seconds-from-file-start in this format.
    """
    text = summary_path.read_text(errors="ignore")
    seizures_by_file = {}

    # Split into per-file blocks on "File Name:"
    blocks = re.split(r"(?=File Name:)", text)
    for block in blocks:
        name_match = re.search(r"File Name:\s*(\S+\.edf)", block)
        if not name_match:
            continue
        stem = Path(name_match.group(1)).stem

        starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", block)]
        ends = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", block)]

        if starts and ends and len(starts) == len(ends):
            seizures_by_file[stem] = list(zip(starts, ends))
        else:
            seizures_by_file.setdefault(stem, [])

    return seizures_by_file


def _extract_time_seconds(text_fragment: str):
    """
    Best-effort extraction of a HH.MM.SS (or HH:MM:SS) time from a messy
    text fragment -- Siena's annotation files contain human typos
    (extra words, missing spaces, duplicate values on one line). Takes
    the FIRST valid-looking time match and ignores trailing garbage.
    Returns total seconds since midnight, or None if nothing matched.
    """
    match = re.search(r"(\d{1,2})\s*[.:]\s*(\d{2})\s*[.:]\s*(\d{2})", text_fragment)
    if not match:
        return None
    h, m, s = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def parse_siena_seizure_list(list_path: Path) -> dict:
    """
    Parse a Siena Seizures-list-PNxx.txt file.
    Returns {edf_filename_stem: [(start_sec, end_sec), ...]} where
    seconds are relative to that recording's own start (Registration
    start time), matching the convention used for CHB-MIT.

    Malformed lines (typos, missing fields) are skipped with a
    printed warning rather than crashing the whole parse -- review
    these warnings manually for the handful of affected entries.
    """
    text = list_path.read_text(errors="ignore")
    blocks = re.split(r"(?=Seizure n \d+)", text)

    seizures_by_file = defaultdict(list)

    for block in blocks:
        file_match = re.search(r"File name:\s*(\S+\.edf)", block)
        reg_start_match = re.search(r"Registration start time:\s*([^\n]+)", block)
        sz_start_match = re.search(r"Seizure start time:\s*([^\n]+)", block)
        sz_end_match = re.search(r"Seizure end time:\s*([^\n]+)", block)

        if not (file_match and reg_start_match and sz_start_match and sz_end_match):
            continue

        stem = Path(file_match.group(1)).stem
        reg_start = _extract_time_seconds(reg_start_match.group(1))
        sz_start = _extract_time_seconds(sz_start_match.group(1))
        sz_end = _extract_time_seconds(sz_end_match.group(1))

        if reg_start is None or sz_start is None or sz_end is None:
            print(f"  [!] Could not parse times in block for {stem} in {list_path.name} -- skipped")
            continue

        # Handle recordings that cross midnight (seizure time < registration
        # start implies it happened after a midnight rollover)
        def relative(t):
            rel = t - reg_start
            if rel < 0:
                rel += 24 * 3600
            return rel

        start_sec = relative(sz_start)
        end_sec = relative(sz_end)

        if end_sec <= start_sec:
            print(f"  [!] Seizure end <= start for {stem} in {list_path.name} "
                  f"(start={start_sec}s, end={end_sec}s) -- skipped, check source file manually")
            continue

        seizures_by_file[stem].append((start_sec, end_sec))

    return dict(seizures_by_file)


def build_seizure_index(dataset_name: str, raw_dir: Path) -> dict:
    """
    Scan a raw dataset directory for annotation files and build a
    combined {edf_filename_stem: [(start_sec, end_sec), ...]} index.
    Files with no seizures (or not mentioned at all) simply won't
    appear as keys -- treated as fully interictal elsewhere.
    """
    index = {}

    if dataset_name == "chbmit":
        for summary_file in sorted(raw_dir.rglob("*-summary.txt")):
            index.update(parse_chbmit_summary(summary_file))

    elif dataset_name == "siena":
        for list_file in sorted(raw_dir.rglob("Seizures-list-*.txt")):
            file_seizures = parse_siena_seizure_list(list_file)
            for stem, seizures in file_seizures.items():
                index.setdefault(stem, []).extend(seizures)

    else:
        raise ValueError(f"Unknown dataset_name for labeling: {dataset_name}")

    return index


# ============================================================
# Windowing + labeling
# ============================================================

def label_windows_for_file(seizures: list, n_samples: int, fs: int = FS,
                            window_sec: float = WINDOW_SEC, step_sec: float = STEP_SEC,
                            preictal_sec: float = PREICTAL_SEC,
                            postictal_sec: float = POSTICTAL_SEC,
                            buffer_sec: float = BUFFER_SEC) -> list:
    """
    Slide a window across one file's samples and label each window.
    Returns a list of (start_sample, label) for windows that are
    KEPT (postictal and buffer-zone windows are dropped entirely).
    """
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    kept = []
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        ws = start / fs
        we = end / fs

        label = INTERICTAL
        excluded = False

        for (ss, se) in seizures:
            if we > ss and ws < se:
                label = ICTAL
                break
            if ss - preictal_sec <= ws < ss:
                label = PREICTAL
                break
            if ss - buffer_sec <= ws < ss - preictal_sec:
                excluded = True
                break
            if se < we <= se + postictal_sec:
                excluded = True
                break
            if se + postictal_sec < we <= se + buffer_sec:
                excluded = True
                break

        if not excluded:
            kept.append((start, label))

        start += step_samples

    return kept


# ============================================================
# Per-patient normalization stats
# ============================================================

def get_patient_id(npy_stem: str, dataset_name: str) -> str:
    """Extract a patient identifier from a processed filename stem."""
    if dataset_name == "chbmit":
        return npy_stem.split("_")[0]          # chb19_18 -> chb19
    elif dataset_name == "siena":
        return npy_stem.split("-")[0]          # PN10-7.8.9 -> PN10
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def compute_patient_stats(patient_files: list) -> tuple:
    """
    Compute per-channel mean and std across ALL recordings belonging
    to one patient combined (not per-file). Returns (mean, std), each
    shape (n_channels,).
    """
    all_data = [np.load(f) for f in patient_files]
    concatenated = np.concatenate(all_data, axis=1)  # (channels, total_samples)
    mean = concatenated.mean(axis=1)
    std = concatenated.std(axis=1)
    std[std == 0] = 1.0  # avoid division by zero for a dead/flat channel
    return mean.astype(np.float32), std.astype(np.float32)


# ============================================================
# PyTorch Dataset
# ============================================================

class SeizureDataset(Dataset):
    """
    dataset_configs: list of dicts, one per data source, e.g.:
        [
          {"name": "chbmit", "processed_dir": Path(...), "raw_dir": Path(...), "notch_freq": 60.0},
          {"name": "siena",  "processed_dir": Path(...), "raw_dir": Path(...), "notch_freq": 50.0},
        ]
    apply_augmentation: whether to apply dry-electrode noise injection
        (should be True for training splits, False for val/test splits).
    """

    def __init__(self, dataset_configs: list, apply_augmentation: bool = False,
                 augmentation_kwargs: dict = None, seed: int = 42):
        self.apply_augmentation = apply_augmentation
        self.augmentation_kwargs = augmentation_kwargs or {}
        self.rng = np.random.default_rng(seed)

        self.index = []            # list of dicts: file path, start_sample, label, patient stats, notch_freq
        self.patient_stats_cache = {}  # (dataset_name, patient_id) -> (mean, std)

        for cfg in dataset_configs:
            self._index_dataset(cfg)

        print(f"SeizureDataset built: {len(self.index)} windows total "
              f"({sum(1 for e in self.index if e['label'] == INTERICTAL)} interictal, "
              f"{sum(1 for e in self.index if e['label'] == PREICTAL)} preictal, "
              f"{sum(1 for e in self.index if e['label'] == ICTAL)} ictal)")

    def _index_dataset(self, cfg: dict):
        name = cfg["name"]
        processed_dir = Path(cfg["processed_dir"])
        raw_dir = Path(cfg["raw_dir"])
        notch_freq = cfg["notch_freq"]

        seizure_index = build_seizure_index(name, raw_dir)

        files_by_patient = defaultdict(list)
        for npy_file in sorted(processed_dir.glob("*.npy")):
            patient_id = get_patient_id(npy_file.stem, name)
            files_by_patient[patient_id].append(npy_file)

        for patient_id, files in files_by_patient.items():
            mean, std = compute_patient_stats(files)
            self.patient_stats_cache[(name, patient_id)] = (mean, std)

            for npy_file in files:
                n_samples = np.load(npy_file, mmap_mode="r").shape[1]
                seizures = seizure_index.get(npy_file.stem, [])
                windows = label_windows_for_file(seizures, n_samples)

                for (start_sample, label) in windows:
                    self.index.append({
                        "dataset": name,
                        "patient_id": patient_id,
                        "npy_path": npy_file,
                        "start_sample": start_sample,
                        "label": label,
                        "notch_freq": notch_freq,
                    })

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        window_samples = int(WINDOW_SEC * FS)

        arr = np.load(entry["npy_path"], mmap_mode="r")
        window = np.array(
            arr[:, entry["start_sample"]: entry["start_sample"] + window_samples],
            dtype=np.float32,
        )

        mean, std = self.patient_stats_cache[(entry["dataset"], entry["patient_id"])]
        window = (window - mean[:, None]) / std[:, None]

        if self.apply_augmentation:
            window = apply_dry_electrode_augmentation(
                window, fs=FS, notch_freq=entry["notch_freq"],
                rng=self.rng, **self.augmentation_kwargs,
            )

        return torch.from_numpy(window), torch.tensor(entry["label"], dtype=torch.long)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    configs = [
        {
            "name": "chbmit",
            "processed_dir": project_root / "data/processed/chbmit",
            "raw_dir": project_root / "data/raw/chbmit",
            "notch_freq": 60.0,
        },
        {
            "name": "siena",
            "processed_dir": project_root / "data/processed/siena",
            "raw_dir": project_root / "data/raw/siena",
            "notch_freq": 50.0,
        },
    ]

    ds = SeizureDataset(configs, apply_augmentation=True)

    print(f"\nTotal dataset size: {len(ds)} windows")
    sample_x, sample_y = ds[0]
    print(f"Sample window shape: {sample_x.shape}, label: {sample_y.item()}")
