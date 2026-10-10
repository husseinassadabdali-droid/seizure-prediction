"""
PyTorch Dataset for EEG seizure prediction + detection (3-class).

Labels: 0 = interictal, 1 = preictal, 2 = ictal. Windows in the postictal
period or in the buffer zones are EXCLUDED (ambiguous, never used).

Pipeline
  1. Parse seizure annotations (CHB-MIT summary.txt, Siena Seizures-list-*.txt).
  2. Slide a window (4 s, 2 s step) over each processed .npy recording and
     label it from its distance to seizures.
  3. Per-patient Z-score: mean/std over ALL recordings of that patient
     (not per file, not per window).
  4. Optional dry-electrode noise augmentation (noise_injection.py), applied
     once here so every branch sees the same noisy window.

Labeling rules (v2)
  * Priority, later wins: buffer-excluded < preictal < postictal-excluded < ictal.
    (v1 stopped at the first seizure whose rule matched, which dropped the
    ictal/preictal windows of any seizure that followed another one closely.)
    A window that is postictal for one seizure and preictal for the next is
    EXCLUDED (conservative).
  * CHB-MIT: file start times from the summaries put every file of a patient
    on one clock-time line, so distances to seizures are measured ACROSS
    files. Files without a timeline entry fall back to within-file labels.
  * Siena: within-file labels only (the lists only carry times for files that
    contain seizures).
  * chb17a/b/c are one subject (patient id "chb17").
  * EXCLUDE_PATIENTS: chb24's summary has no file start times, so its files
    cannot be placed on a timeline and its interictal windows cannot be
    verified as clean. Remove it from the set to include it again.

Known assumptions
  * A clock going backwards between consecutive files means midnight passed
    (+24 h). A real gap longer than a day would be underestimated.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from noise_injection import apply_dry_electrode_augmentation  # noqa: E402

# ---- Label constants ----
INTERICTAL = 0
PREICTAL = 1
ICTAL = 2
EXCLUDED = -1

# ---- Windowing config ----
FS = 250  # Hz, matches preprocess.py TARGET_SFREQ
WINDOW_SEC = 4
STEP_SEC = 2  # 2 s step on a 4 s window = 2 s overlap
PREICTAL_SEC = 30 * 60
POSTICTAL_SEC = 30 * 60
BUFFER_SEC = 4 * 60 * 60

# Patients left out entirely (dataset_name, patient_id)
EXCLUDE_PATIENTS = {("chbmit", "chb24")}


# ============================================================
# Annotation parsing
# ============================================================

def parse_chbmit_summary(summary_path: Path) -> dict:
    """
    Parse a CHB-MIT *-summary.txt file.
    Returns {edf_filename_stem: [(start_sec, end_sec), ...]}, with times in
    seconds from the start of THAT file.
    """
    text = summary_path.read_text(errors="ignore")
    seizures_by_file = {}

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


def _hms_to_sec(match) -> int:
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def parse_chbmit_file_times(summary_path: Path) -> list:
    """[(file_stem, start_time_seconds_or_None)] in summary order."""
    text = summary_path.read_text(errors="ignore")
    entries = []
    for block in re.split(r"(?=File Name:)", text):
        name = re.search(r"File Name:\s*(\S+)\.edf", block)
        if not name:
            continue
        t0 = re.search(r"File Start Time:\s*(\d+):(\d+):(\d+)", block)
        entries.append((name.group(1), _hms_to_sec(t0) if t0 else None))
    return entries


def place_on_timeline(entries: list) -> dict:
    """
    Absolute start (seconds) per file stem. A clock going backwards means
    midnight passed (+24 h). Entries without a time are skipped.
    """
    starts, day, prev = {}, 0, None
    for stem, t0 in entries:
        if t0 is None:
            continue
        a = t0 + day
        while prev is not None and a < prev:
            day += 86400
            a = t0 + day
        starts[stem] = a
        prev = a
    return starts


def build_chbmit_timeline(raw_dir: Path):
    """
    Returns (file_start, patient_seizures):
      file_start[stem]        = (patient_id, absolute_start_sec)
      patient_seizures[pid]   = [(abs_start, abs_end), ...] on that patient's timeline
    Only files with a parsable start time are included.
    """
    file_start, patient_seizures = {}, {}
    for summary in sorted(raw_dir.rglob("*-summary.txt")):
        pid = summary.stem.replace("-summary", "")
        abs_start = place_on_timeline(parse_chbmit_file_times(summary))
        per_file = parse_chbmit_summary(summary)
        seizures = []
        for stem, a in abs_start.items():
            file_start[stem] = (pid, a)
            for (s, e) in per_file.get(stem, []):
                seizures.append((a + s, a + e))
        patient_seizures[pid] = seizures
    return file_start, patient_seizures


def _extract_time_seconds(text_fragment: str):
    """
    Best-effort extraction of a HH.MM.SS (or HH:MM:SS) time from a messy
    text fragment -- Siena's annotation files contain human typos. Takes the
    FIRST valid-looking time and ignores trailing garbage. Returns seconds
    since midnight, or None.
    """
    match = re.search(r"(\d{1,2})\s*[.:]\s*(\d{2})\s*[.:]\s*(\d{2})", text_fragment)
    if not match:
        return None
    h, m, s = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def parse_siena_seizure_list(list_path: Path) -> dict:
    """
    Parse a Siena Seizures-list-PNxx.txt file.
    Returns {edf_filename_stem: [(start_sec, end_sec), ...]}, seconds relative
    to that recording's own start (Registration start time).
    Malformed entries are skipped with a printed warning.
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

        def relative(t):
            rel = t - reg_start
            if rel < 0:  # recording crossed midnight
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
    Within-file seizure index {edf_filename_stem: [(start_sec, end_sec), ...]}.
    Files not mentioned are treated as seizure-free.
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
                            buffer_sec: float = BUFFER_SEC,
                            offset_sec: float = 0.0) -> list:
    """
    Slide a window across one file and label each window.

    seizures:   [(start_sec, end_sec)] in the SAME time frame as the file,
                where the file starts at `offset_sec` (0 for within-file
                labeling; the file's absolute start for timeline labeling).
    Returns [(start_sample, label)] for KEPT windows only.

    Priority (later wins): buffer-excluded < preictal < postictal-excluded < ictal.
    """
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    if n_samples < window_samples:
        return []

    n_win = (n_samples - window_samples) // step_samples + 1
    starts = np.arange(n_win, dtype=np.int64) * step_samples
    ws = offset_sec + starts / fs
    we = ws + window_sec

    labels = np.full(n_win, INTERICTAL, dtype=np.int8)
    if len(seizures):
        ss = np.array([s for s, _ in seizures], dtype=np.float64)
        se = np.array([e for _, e in seizures], dtype=np.float64)
        W, E = ws[:, None], we[:, None]

        ictal = ((E > ss) & (W < se)).any(axis=1)
        post = ((E > se) & (E <= se + postictal_sec)).any(axis=1)
        pre = ((W >= ss - preictal_sec) & (W < ss)).any(axis=1)
        buf = (((W >= ss - buffer_sec) & (W < ss - preictal_sec))
               | ((E > se + postictal_sec) & (E <= se + buffer_sec))).any(axis=1)

        labels[buf] = EXCLUDED
        labels[pre] = PREICTAL
        labels[post] = EXCLUDED
        labels[ictal] = ICTAL

    keep = labels != EXCLUDED
    return list(zip(starts[keep].tolist(), labels[keep].astype(int).tolist()))


# ============================================================
# Per-patient normalization stats
# ============================================================

def get_patient_id(npy_stem: str, dataset_name: str) -> str:
    """Patient identifier from a processed filename stem."""
    if dataset_name == "chbmit":
        m = re.match(r"(chb\d+)", npy_stem)       # chb17a_03 -> chb17, chb19_18 -> chb19
        return m.group(1) if m else npy_stem.split("_")[0]
    elif dataset_name == "siena":
        return npy_stem.split("-")[0]              # PN10-7.8.9 -> PN10
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def compute_patient_stats(patient_files: list) -> tuple:
    """
    Per-channel mean and std across ALL recordings of one patient combined.
    Returns (mean, std), each shape (n_channels,).
    """
    all_data = [np.load(f) for f in patient_files]
    concatenated = np.concatenate(all_data, axis=1)  # (channels, total_samples)
    mean = concatenated.mean(axis=1)
    std = concatenated.std(axis=1)
    std[std == 0] = 1.0  # dead/flat channel
    return mean.astype(np.float32), std.astype(np.float32)


# ============================================================
# PyTorch Dataset
# ============================================================

class SeizureDataset(Dataset):
    """
    dataset_configs: list of dicts, one per data source, e.g.:
        {"name": "chbmit", "processed_dir": Path(...), "raw_dir": Path(...), "notch_freq": 60.0}
    apply_augmentation: dry-electrode noise injection (True for training views,
        False for validation/test views).
    exclude_patients: set of (dataset_name, patient_id) to leave out;
        None uses EXCLUDE_PATIENTS.
    """

    def __init__(self, dataset_configs: list, apply_augmentation: bool = False,
                 augmentation_kwargs: dict = None, seed: int = 42,
                 exclude_patients: set = None):
        self.apply_augmentation = apply_augmentation
        self.augmentation_kwargs = augmentation_kwargs or {}
        self.rng = np.random.default_rng(seed)
        self.exclude_patients = set(EXCLUDE_PATIENTS if exclude_patients is None else exclude_patients)

        self.index = []                # dicts: dataset, patient_id, npy_path, start_sample, label, notch_freq
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

        within_file = build_seizure_index(name, raw_dir)
        if name == "chbmit":
            file_start, patient_seizures = build_chbmit_timeline(raw_dir)
        else:
            file_start, patient_seizures = {}, {}

        files_by_patient = defaultdict(list)
        excluded = defaultdict(int)
        for npy_file in sorted(processed_dir.glob("*.npy")):
            patient_id = get_patient_id(npy_file.stem, name)
            if (name, patient_id) in self.exclude_patients:
                excluded[patient_id] += 1
                continue
            files_by_patient[patient_id].append(npy_file)

        counts = np.zeros(3, dtype=np.int64)
        n_timeline = n_within = 0

        for patient_id, files in files_by_patient.items():
            mean, std = compute_patient_stats(files)
            self.patient_stats_cache[(name, patient_id)] = (mean, std)

            for npy_file in files:
                n_samples = np.load(npy_file, mmap_mode="r").shape[1]
                stem = npy_file.stem

                if stem in file_start:
                    timeline_pid, offset = file_start[stem]
                    windows = label_windows_for_file(
                        patient_seizures.get(timeline_pid, []), n_samples, offset_sec=offset)
                    n_timeline += 1
                else:
                    windows = label_windows_for_file(within_file.get(stem, []), n_samples)
                    n_within += 1

                for (start_sample, label) in windows:
                    counts[label] += 1
                    self.index.append({
                        "dataset": name,
                        "patient_id": patient_id,
                        "npy_path": npy_file,
                        "start_sample": start_sample,
                        "label": label,
                        "notch_freq": notch_freq,
                    })

        print(f"[{name}] windows: interictal={counts[0]:,} preictal={counts[1]:,} ictal={counts[2]:,} | "
              f"files labeled on patient timeline: {n_timeline}, within-file only: {n_within}")
        if excluded:
            print(f"[{name}] excluded patients: "
                  + ", ".join(f"{p} ({n} files)" for p, n in sorted(excluded.items())))

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
