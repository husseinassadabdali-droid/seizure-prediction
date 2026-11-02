"""
Read-only diagnostic: how wrong are the current within-file labels for CHB-MIT?

Builds one clock-time line per patient from the summary files (file start
times), places every seizure on it, and relabels every window using distances
measured ACROSS files. Compares with the labels training actually used
(label_windows_for_file in dataset.py, which only sees seizures inside the
same file).

Stitched priority (later wins): buffer-excluded < preictal < postictal-excluded < ictal.
Nothing is written; no features, no model.
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from dataset import (parse_chbmit_summary, label_windows_for_file, FS, WINDOW_SEC,  # noqa: E402
                     STEP_SEC, PREICTAL_SEC, POSTICTAL_SEC, BUFFER_SEC)

RAW = ROOT / "data/raw/chbmit"
PROC = ROOT / "data/processed/chbmit"
WIN, STEP = int(WINDOW_SEC * FS), int(STEP_SEC * FS)
GAP_THRESHOLD = 60.0  # seconds


def hms(m):
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def parse_file_times(summary_path):
    """[(stem, start_sec, end_sec)] in summary order; times may be None."""
    text = summary_path.read_text(errors="ignore")
    out = []
    for block in re.split(r"(?=File Name:)", text):
        name = re.search(r"File Name:\s*(\S+)\.edf", block)
        if not name:
            continue
        t0 = re.search(r"File Start Time:\s*(\d+):(\d+):(\d+)", block)
        t1 = re.search(r"File End Time:\s*(\d+):(\d+):(\d+)", block)
        out.append((name.group(1), hms(t0) if t0 else None, hms(t1) if t1 else None))
    return out


def place_on_timeline(entries):
    """Absolute start (s) per file stem; a clock going backwards means midnight passed (+24h)."""
    start, day, prev = {}, 0, None
    for stem, t0, _ in entries:
        if t0 is None:
            continue
        a = t0 + day
        while prev is not None and a < prev:
            day += 86400
            a = t0 + day
        start[stem] = a
        prev = a
    return start


def processed_len(stem):
    p = PROC / f"{stem}.npy"
    return np.load(p, mmap_mode="r").shape[1] if p.exists() else None


trans = np.zeros((4, 4), dtype=np.int64)  # index = label + 1  (-1 excluded, 0, 1, 2)
rows, gap_report = [], []
all_stems = set()
n_listed = n_no_time = n_placed = n_seiz = n_seiz_early = 0
dur_diff = 0.0

for sp in sorted(RAW.rglob("*-summary.txt")):
    pid = sp.stem.replace("-summary", "")
    entries = parse_file_times(sp)
    file_seizures = parse_chbmit_summary(sp)  # {stem: [(start_s, end_s)]} within each file
    starts = place_on_timeline(entries)
    n_listed += len(entries)
    n_no_time += sum(1 for _, t0, _ in entries if t0 is None)
    all_stems.update(stem for stem, _, _ in entries)

    seizures_abs = []
    for stem, _, _ in entries:
        if stem in starts:
            for (s, e) in file_seizures.get(stem, []):
                seizures_abs.append((starts[stem] + s, starts[stem] + e))
                n_seiz += 1
                n_seiz_early += s < PREICTAL_SEC

    files = []  # (stem, abs_start, n_samples) for processed files only
    for stem, t0, t1 in entries:
        ns = processed_len(stem)
        if ns is None or stem not in starts:
            continue
        files.append((stem, starts[stem], ns))
        if t0 is not None and t1 is not None:
            dur_diff = max(dur_diff, abs((t1 - t0) % 86400 - ns / FS))
    n_placed += len(files)
    if not files:
        continue

    gaps = [files[i + 1][1] - (files[i][1] + files[i][2] / FS) for i in range(len(files) - 1)]
    big = [g for g in gaps if g > GAP_THRESHOLD]
    if big:
        gap_report.append((pid, len(big), max(big) / 60.0))

    ws_parts, old_parts = [], []
    for stem, a, ns in files:
        n_win = (ns - WIN) // STEP + 1 if ns >= WIN else 0
        if n_win <= 0:
            continue
        old = np.full(n_win, -1, dtype=np.int8)
        for (st, lab) in label_windows_for_file(file_seizures.get(stem, []), ns):
            old[st // STEP] = lab
        ws_parts.append(a + np.arange(n_win) * STEP / FS)
        old_parts.append(old)
    ws = np.concatenate(ws_parts)
    old = np.concatenate(old_parts)
    we = ws + WINDOW_SEC

    new = np.zeros(len(ws), dtype=np.int8)
    if seizures_abs:
        ss = np.array([s for s, _ in seizures_abs])
        se = np.array([e for _, e in seizures_abs])
        W, E = ws[:, None], we[:, None]
        ictal = ((E > ss) & (W < se)).any(axis=1)
        post = ((E > se) & (E <= se + POSTICTAL_SEC)).any(axis=1)
        pre = ((W >= ss - PREICTAL_SEC) & (W < ss)).any(axis=1)
        buf = (((W >= ss - BUFFER_SEC) & (W < ss - PREICTAL_SEC))
               | ((E > se + POSTICTAL_SEC) & (E <= se + BUFFER_SEC))).any(axis=1)
        new[buf] = -1
        new[pre] = 1
        new[post] = -1
        new[ictal] = 2

    np.add.at(trans, (old.astype(int) + 1, new.astype(int) + 1), 1)
    rows.append((pid, len(seizures_abs), int((old == 0).sum()), int((new == 0).sum()),
                 int((old == 1).sum()), int((new == 1).sum())))

orphans = {p.stem for p in PROC.glob("*.npy")} - all_stems

print("=== Timeline sanity ===")
print(f"files listed in summaries          : {n_listed} (without parsable time: {n_no_time})")
print(f"processed files placed on timeline : {n_placed}")
print(f"processed files not in any summary : {len(orphans)}")
print(f"max |summary duration - processed duration| : {dur_diff:.1f} s (a few seconds is normal)")
gap_report.sort(key=lambda r: -r[2])
print(f"patients with gaps > {GAP_THRESHOLD:.0f}s between consecutive processed files: "
      f"{len(gap_report)} of {len(rows)}")
for pid, cnt, mx in gap_report[:8]:
    print(f"  {pid}: {cnt} gap(s), longest {mx:.1f} min")
print(f"seizures placed: {n_seiz}; onset within the first 30 min of their file: "
      f"{n_seiz_early} ({n_seiz_early / max(n_seiz, 1):.0%})")

names = ["excluded", "interictal", "preictal", "ictal"]
print("\n=== Old (within-file) -> new (stitched) labels, window counts ===")
print("old \\ new".ljust(13) + "".join(f"{n:>13}" for n in names))
for i, n in enumerate(names):
    print(n.ljust(13) + "".join(f"{trans[i, j]:>13,}" for j in range(4)))

old_inter, old_pre, new_pre = trans[1].sum(), trans[2].sum(), trans[:, 2].sum()
print(f"\nold interictal windows: {old_inter:,}")
print(f"  now excluded : {trans[1, 0]:,} ({trans[1, 0] / max(old_inter, 1):.1%})")
print(f"  now preictal : {trans[1, 2]:,} ({trans[1, 2] / max(old_inter, 1):.1%})")
print(f"preictal windows: old={old_pre:,}  new={new_pre:,}  "
      f"(old covers {old_pre / max(new_pre, 1):.1%} of new)")
print(f"check - old ictal windows that are no longer ictal (must be 0): {trans[3, :3].sum()}")

print(f"\n{'patient':<8}{'seizures':>9}{'inter old':>11}{'inter new':>11}{'pre old':>9}{'pre new':>9}")
for pid, ns_, io, inn, po, pn in rows:
    print(f"{pid:<8}{ns_:>9}{io:>11,}{inn:>11,}{po:>9,}{pn:>9,}")
