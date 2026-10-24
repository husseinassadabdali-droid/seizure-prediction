"""
PyTorch Dataset for EEG seizure prediction + detection (3-class).

Labels: 0 = interictal, 1 = preictal, 2 = ictal. Windows in the postictal
period or in the buffer zones are EXCLUDED (ambiguous, never used).

Pipeline
  1. Parse seizure annotations (CHB-MIT summary.txt, Siena Seizures-list-*.txt).
  2. Slide a window (4 s, 2 s step) over each processed .npy recording and
     label it from its distance to seizures.
  3. Z-score using either the legacy per-patient mode or frozen statistics
     fitted only on training patients after the split.
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
import csv
from collections import defaultdict
from itertools import product
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

# Data trust is independent of the model's three class labels.
VALIDATED = "VALIDATED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
NO_SEIZURE_CONFIRMED = "NO_SEIZURE_CONFIRMED"

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
    Extract one HH.MM.SS (or HH:MM:SS) token from a fragment.

    Callers handling a complete SIENA annotation field must first use
    `_siena_time_candidates` so additional values are not silently ignored.
    """
    match = re.search(r"(\d{1,2})\s*[.:]\s*(\d{2})\s*[.:]\s*(\d{2})", text_fragment)
    if not match:
        return None
    h, m, s = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def _siena_records(raw_dir: Path):
    """Read RECORDS into exact EDF stems and duration metadata (header only)."""
    records_path = raw_dir / "RECORDS"
    available, durations, starts = {}, {}, {}
    if not records_path.exists():
        return available, durations, starts
    for line in records_path.read_text(errors="ignore").splitlines():
        rel = line.strip()
        if not rel.lower().endswith(".edf"):
            continue
        path = raw_dir / rel
        stem = Path(rel).stem
        available[stem] = path
        try:
            with path.open("rb") as f:
                header = f.read(256)
            if len(header) < 256:
                continue
            n_records = int(header[236:244].decode("ascii").strip())
            record_sec = float(header[244:252].decode("ascii").strip())
            durations[stem] = n_records * record_sec
            date = header[168:176].decode("ascii").strip()
            clock = header[176:184].decode("ascii").strip()
            starts[stem] = (date, clock)
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    return available, durations, starts


def _siena_clock_seconds(value):
    """Parse source clock text, including the known PN10 `1 6.49.25` typo."""
    raw = value.strip()
    corrected = re.sub(r"^1\s+6([.:]\s*\d{2}[.:]\s*\d{2})$", r"16\1", raw)
    seconds = _extract_time_seconds(corrected)
    return seconds, (corrected if corrected != raw else None)


_SIENA_TIME_RE = re.compile(
    r"(?<![\w])(?P<hour>[A-Za-z0-9]+)\s*(?P<sep1>[.:])\s*"
    r"(?P<minute>[A-Za-z0-9]+)\s*(?P<sep2>[.:])\s*"
    r"(?P<second>[A-Za-z0-9]+)(?P<extra>(?:\s*[.:]\s*[A-Za-z0-9]+)*)(?![\w])"
)
_SIENA_CONTEXT_SPLIT_RE = re.compile(r"[;,/]|\b(?:or|and|versus|vs\.?)\b", re.I)


def _siena_time_candidates(raw_text: str, field_name: str) -> list:
    """Extract valid and invalid clock-like candidates with local context."""
    if raw_text is None:
        return []
    raw_text = raw_text.strip()
    whole_seconds, whole_correction = _siena_clock_seconds(raw_text)
    matches = list(_SIENA_TIME_RE.finditer(raw_text))

    # The split-hour typo is interpreted without replacing the original field.
    if whole_correction and len(matches) == 1:
        values = [{"raw": raw_text, "seconds": whole_seconds,
                   "correction": whole_correction, "span": (0, len(raw_text)),
                   "context": raw_text, "separators": ".", "valid": True,
                   "invalid_reason": None}]
    else:
        values = []
        for match in matches:
            hour, minute, second = (match.group(key) for key in ("hour", "minute", "second"))
            invalid_reason = None
            if not (hour.isdigit() and minute.isdigit() and second.isdigit()):
                invalid_reason = "time components are not numeric"
            elif match.group("extra"):
                invalid_reason = "unexpected extra clock component(s)"
            elif len(hour) > 2:
                invalid_reason = "hour must contain one or two digits"
            elif len(minute) != 2 or len(second) != 2:
                invalid_reason = "minute and second must contain two digits"
            else:
                h, m, s = int(hour), int(minute), int(second)
                if h > 23:
                    invalid_reason = "hour outside 00..23"
                elif m > 59:
                    invalid_reason = "minute outside 00..59"
                elif s > 59:
                    invalid_reason = "second outside 00..59"

            separators = list(_SIENA_CONTEXT_SPLIT_RE.finditer(raw_text))
            left = max((sep.end() for sep in separators if sep.end() <= match.start()), default=0)
            right = min((sep.start() for sep in separators if sep.start() >= match.end()),
                        default=len(raw_text))
            context = raw_text[left:right].strip()
            seconds = None if invalid_reason else int(hour) * 3600 + int(minute) * 60 + int(second)
            values.append({
                "raw": match.group(0), "seconds": seconds, "correction": None,
                "span": match.span(), "context": context,
                "separators": match.group("sep1") + match.group("sep2"),
                "valid": invalid_reason is None, "invalid_reason": invalid_reason,
            })

    candidates = []
    for value in values:
        raw = value["raw"]
        seconds = value["seconds"]
        correction = value["correction"]
        context = value["context"]
        separators = value["separators"]
        lowered = context.lower()
        semantic = None
        has_clinical = bool(re.search(r"\bclinical\b", lowered))
        has_electrical = bool(re.search(r"\b(electric|electrical)\b", lowered))
        semantic_conflict = has_clinical and has_electrical
        if field_name == "registration_start":
            semantic = "registration_start"
        elif field_name == "seizure_end":
            semantic = "seizure_end"
        elif has_clinical and not has_electrical:
            semantic = "clinical_onset"
        elif has_electrical and not has_clinical:
            semantic = "electrical_onset"
        candidates.append({
            "field": field_name, "raw": raw, "context": context, "span": tuple(value["span"]),
            "seconds_since_midnight": seconds,
            "normalized": (f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
                           if seconds is not None else None),
            "format_separators": separators, "semantic": semantic,
            "semantic_conflict": semantic_conflict,
            "valid": value["valid"], "invalid_reason": value["invalid_reason"],
            "correction": correction,
        })
    return candidates


def _siena_time_resolution(fields: dict, duration: float = None) -> dict:
    """Resolve only equivalent spellings; preserve all distinct interpretations."""
    candidates = {name: _siena_time_candidates(raw, name) for name, raw in fields.items()}
    missing = [name for name, values in candidates.items() if not values]
    if missing:
        return {"status": "unparseable", "reason": f"unparseable time field(s): {', '.join(missing)}",
                "candidates": candidates, "intervals": [], "invalid_intervals": [],
                "candidate_combinations": [], "invalid_candidates": [],
                "expected_candidate_combinations": 0,
                "recorded_candidate_combinations": 0,
                "combination_coverage_complete": False,
                "window_labels_differ": None}

    semantic_fields = []
    invalid_candidates = []
    alternative_connectors = []
    for name, values in candidates.items():
        if any(not v["valid"] for v in values):
            invalid_candidates.extend((name, i, v) for i, v in enumerate(values) if not v["valid"])
        semantics = {v["semantic"] for v in values}
        if any(v["semantic_conflict"] for v in values):
            semantic_fields.append(name)
        elif len(values) > 1 and (len(semantics) > 1 or None in semantics and len(semantics) > 1):
            semantic_fields.append(name)
        raw_field = fields[name] or ""
        if len(values) > 1 and re.search(r"/|\b(?:or|versus|vs\.?)\b", raw_field, re.I):
            alternative_connectors.append(name)

    intervals = set()
    combinations = []
    valid_candidates = {
        name: [(i, value) for i, value in enumerate(values) if value["valid"]]
        for name, values in candidates.items()
    }
    for registration, start, end in product(
            valid_candidates["registration_start"], valid_candidates["seizure_start"],
            valid_candidates["seizure_end"]):
        ri, registration = registration
        si, start = start
        ei, end = end
        reg = registration["seconds_since_midnight"]
        onset = start["seconds_since_midnight"]
        offset = end["seconds_since_midnight"]
        onset_rel = (onset - reg) % (24 * 3600)
        offset_rel = (offset - reg) % (24 * 3600)
        if offset_rel == onset_rel:
            interval = (onset_rel, offset_rel)
            intervals.add(interval)
            within_duration = None if duration is None else offset_rel <= duration
            combinations.append({
                "candidate_indices": {"registration_start": ri, "seizure_start": si, "seizure_end": ei},
                "interval": interval,
                "within_edf": within_duration,
                "boundary_status": "unknown" if within_duration is None else (
                    "within" if within_duration else "outside"),
                "valid_interval": False,
            })
            continue
        if offset_rel < onset_rel:
            offset_rel += 24 * 3600
        interval = (onset_rel, offset_rel)
        intervals.add(interval)
        within_duration = None if duration is None else offset_rel <= duration
        combinations.append({
            "candidate_indices": {"registration_start": ri, "seizure_start": si, "seizure_end": ei},
            "interval": interval, "within_edf": within_duration,
            "boundary_status": "unknown" if within_duration is None else (
                "within" if within_duration else "outside"),
            "valid_interval": (None if within_duration is None else
                               offset_rel > onset_rel and within_duration),
        })
    sorted_intervals = sorted(intervals)
    invalid = [(a, b) for a, b in sorted_intervals
               if b <= a or (duration is not None and b > duration)]
    signatures = None
    if duration is not None and sorted_intervals:
        signatures = [label_windows_for_file([interval], int(duration * FS))
                      for interval in sorted_intervals]
    labels_differ = (len({tuple(signature) for signature in signatures}) > 1
                     if signatures is not None else None)

    combination_keys = [tuple(combo["candidate_indices"][name]
                              for name in ("registration_start", "seizure_start", "seizure_end"))
                        for combo in combinations]
    expected_combinations = (len(valid_candidates["registration_start"])
                             * len(valid_candidates["seizure_start"])
                             * len(valid_candidates["seizure_end"]))
    combinations_complete = (
        not invalid_candidates
        and len(combination_keys) == expected_combinations
        and len(set(combination_keys)) == len(combination_keys)
    )
    common = {"candidates": candidates, "intervals": sorted_intervals,
              "invalid_intervals": invalid, "candidate_combinations": combinations,
              "invalid_candidates": [
                  {"field": name, "candidate_index": i, "raw": value["raw"],
                   "reason": value["invalid_reason"]}
                  for name, i, value in invalid_candidates],
              "expected_candidate_combinations": expected_combinations,
              "recorded_candidate_combinations": len(combinations),
              "combination_coverage_complete": combinations_complete,
              "window_labels_differ": labels_differ}
    if semantic_fields:
        return {**common, "status": "review", "reason": "distinct onset semantics require review"}
    if invalid_candidates:
        return {**common, "status": "review", "reason": "one or more clock candidates are invalid"}
    if alternative_connectors:
        return {**common, "status": "review", "reason": "alternative time interpretations require review"}
    if invalid:
        return {**common, "status": "review", "reason": "one or more time interpretations are outside EDF bounds"}
    if len(intervals) > 1:
        return {**common, "status": "review", "reason": "distinct time values require review"}
    equivalent = any(len(values) > 1 for values in candidates.values())
    for name, values in candidates.items():
        if len(values) < 2:
            continue
        if len({v["seconds_since_midnight"] for v in values}) > 1:
            return {**common, "status": "review", "reason": "distinct time values require review"}
        meanings = {v["semantic"] for v in values}
        if None in meanings and len(meanings) > 1:
            return {**common, "status": "review", "reason": "candidate meaning is unresolved"}
        if meanings == {None} and len({v["format_separators"] for v in values}) < 2:
            return {**common, "status": "review", "reason": "duplicate time candidates have unresolved meaning"}
    return {**common, "status": "equivalent" if equivalent else "resolved",
            "reason": ("multiple source forms resolve to the same event and window labels"
                       if equivalent else None)}


def parse_siena_seizure_list(list_path: Path, available_files=None,
                             edf_durations=None, diagnostics=None, edf_starts=None) -> dict:
    """
    Parse a Siena Seizures-list-PNxx.txt file.
    Returns {edf_filename_stem: [(start_sec, end_sec), ...]}, seconds relative
    to that recording's own start (Registration start time).
    Malformed entries are skipped and described in `diagnostics` when supplied.
    """
    text = list_path.read_text(errors="ignore")
    # Source lists contain both `Seizure n1` and `Seizure n 1` spellings.
    blocks = re.split(r"(?=Seizure\s+n\s*\d+\b)", text, flags=re.I)

    seizures_by_file = defaultdict(list)
    diagnostics = diagnostics if diagnostics is not None else []
    has_record_inventory = available_files is not None
    has_header_inventory = edf_starts is not None
    available_files = available_files or {}
    edf_durations = edf_durations or {}
    edf_starts = edf_starts or {}
    # Explicit source-name corrections. A correction is accepted only when the
    # target is actually present in RECORDS; never normalize arbitrary letters.
    explicit_aliases = {
        "PN01": "PN01-1", "PNO6-1": "PN06-1", "PNO6-2": "PN06-2",
        "PNO6-4": "PN06-4", "PN11-": "PN11-1",
    }
    previous_file = previous_registration = None

    for block in blocks:
        file_match = re.search(r"File name:\s*(\S+\.edf)", block, re.I)
        # PN01's source uses `Start time`/`End time`, unlike the other lists.
        sz_start_match = re.search(
            r"(?<!Registration )(?:Seizure\s+)?Start time:\s*([^\n]+)", block, re.I)
        sz_end_match = re.search(r"Seizure end time:\s*([^\n]+)", block)
        if not sz_end_match:
            sz_end_match = re.search(
                r"(?<!Registration )(?<!Seizure )End time:\s*([^\n]+)", block, re.I)
        raw_file = Path(file_match.group(1)).stem if file_match else None
        reg_start_match = re.search(r"Registration start time:\s*([^\n]+)", block, re.I)
        raw_reg = reg_start_match.group(1).strip() if reg_start_match else None

        # Only inherit context into an immediately following block with the
        # exact same annotation filename. A new filename resets the context.
        if raw_file is None:
            raw_file = previous_file
        if raw_file != previous_file:
            previous_registration = None
        if raw_reg is None and raw_file == previous_file:
            raw_reg = previous_registration
        previous_file = raw_file
        if reg_start_match:
            previous_registration = reg_start_match.group(1).strip()

        if not (raw_file and raw_reg and sz_start_match and sz_end_match):
            if sz_start_match or sz_end_match:
                diagnostics.append({"source": list_path.name, "raw_file": raw_file,
                                    "status": "review", "event_valid": False,
                                    "reason": "missing file or registration time"})
            continue

        stem = raw_file
        matched_stem = stem
        linked = (not has_record_inventory) or stem in available_files
        if not linked and stem in explicit_aliases and explicit_aliases[stem] in available_files:
            matched_stem = explicit_aliases[stem]
            linked = True
        duration = edf_durations.get(matched_stem) if linked else None
        raw_fields = {
            "registration_start": raw_reg,
            "seizure_start": sz_start_match.group(1).strip(),
            "seizure_end": sz_end_match.group(1).strip(),
        }
        resolution = _siena_time_resolution(raw_fields, duration=duration)
        intervals = resolution["intervals"]
        representative = intervals[0] if len(intervals) == 1 else (None, None)
        reg_candidates = resolution["candidates"]["registration_start"]
        reg_values = {c["seconds_since_midnight"] for c in reg_candidates}
        reg_correction = next((c["correction"] for c in reg_candidates if c["correction"]), None)
        time_corrections = [c["correction"] for key in ("seizure_start", "seizure_end")
                            for c in resolution["candidates"][key] if c["correction"]]
        entry = {
            "source": list_path.name, "raw_file": stem,
            "matched_file": matched_stem if linked else None,
            "start_sec": representative[0], "end_sec": representative[1],
            "raw_registration_start": raw_reg, "raw_time_fields": raw_fields,
            "time_candidates": resolution["candidates"],
            "candidate_intervals": intervals,
            "candidate_combinations": resolution["candidate_combinations"],
            "invalid_time_candidates": resolution["invalid_candidates"],
            "expected_candidate_combinations": resolution["expected_candidate_combinations"],
            "recorded_candidate_combinations": resolution["recorded_candidate_combinations"],
            "combination_coverage_complete": resolution["combination_coverage_complete"],
            "registration_correction": reg_correction,
            "time_corrections": time_corrections,
            "time_resolution": resolution["status"],
            "window_labels_differ": resolution["window_labels_differ"],
            "status": "linked" if linked and resolution["status"] in {"resolved", "equivalent"} else "review",
            "event_valid": True, "reason": resolution["reason"],
        }

        def add_review_reason(reason):
            entry["status"] = "review"
            entry["reason"] = f"{entry['reason']}; {reason}" if entry["reason"] else reason

        if not linked:
            entry.update(status="unlinked", reason="annotation filename not found in RECORDS")
        header_start = edf_starts.get(matched_stem) if linked else None
        if linked and duration is None:
            add_review_reason("EDF duration unavailable")
        if linked and has_header_inventory and header_start is None:
            add_review_reason("EDF start time unavailable")
        if header_start and len(reg_values) == 1:
            header_clock = _extract_time_seconds(header_start[1])
            if header_clock is not None and header_clock not in reg_values:
                mismatch = f"registration start {raw_reg!r} differs from EDF header time {header_start[1]}"
                add_review_reason(mismatch)
        if linked and duration is not None and resolution["invalid_intervals"]:
            bounds_reason = f"event outside EDF duration ({duration:g}s)"
            if bounds_reason not in (entry["reason"] or ""):
                add_review_reason(bounds_reason)
        if linked:
            # Keep in-file events only. Out-of-bounds events are surfaced for
            # review and cannot silently extend labels to the recording end.
            if entry["status"] == "linked" and representative[0] is not None:
                seizures_by_file[matched_stem].append(representative)
        diagnostics.append(entry)

    return dict(seizures_by_file)


def compare_siena_event_counts(raw_dir: Path) -> dict:
    """Compare subject_info counts with annotation headings without preferring either.

    A mismatch or missing side is explicitly unresolved. A match is only a
    completeness check; it does not establish clinical correctness or certify
    an event-free recording.
    """
    raw_dir = Path(raw_dir)
    info_path = raw_dir / "subject_info.csv"
    declared = {}
    if info_path.is_file():
        with info_path.open(newline="", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                clean = {(key or "").strip(): (value or "").strip()
                         for key, value in row.items()}
                patient = clean.get("patient_id")
                raw_count = clean.get("number_seizures")
                if patient:
                    try:
                        declared[patient] = int(raw_count)
                    except (TypeError, ValueError):
                        declared[patient] = None

    annotated = {}
    annotation_paths = {}
    for path in sorted(raw_dir.rglob("Seizures-list-*.txt")):
        match = re.search(r"Seizures-list-(PN\d+)", path.name, re.I)
        if not match:
            continue
        patient = match.group(1).upper()
        text = path.read_text(errors="ignore")
        annotated[patient] = len(re.findall(
            r"^\s*Seizure\s+n\s*\d+\b", text, re.I | re.M))
        annotation_paths[patient] = str(path)

    result = {}
    for patient in sorted(set(declared) | set(annotated)):
        declared_count = declared.get(patient)
        annotation_count = annotated.get(patient)
        matches = (declared_count is not None and annotation_count is not None
                   and declared_count == annotation_count)
        result[patient] = {
            "subject_info_count": declared_count,
            "annotation_heading_count": annotation_count,
            "annotation_file": annotation_paths.get(patient),
            "status": "match" if matches else "review",
            "reason": None if matches else
                      "event-count sources are missing or disagree; neither source was selected",
        }
    return result


def _quality_record(status: str, reason: str = None, training_eligible=None,
                    event_eval_eligible=None) -> dict:
    """Per-recording provenance with independently overridable use permissions."""
    trusted = status in {VALIDATED, NO_SEIZURE_CONFIRMED}
    return {
        "status": status,
        "reason": reason,
        "training_eligible": trusted if training_eligible is None else bool(training_eligible),
        "event_eval_eligible": trusted if event_eval_eligible is None else bool(event_eval_eligible),
    }


def _quality_for_recording(stem: str, events_by_file: dict, review_reasons=None,
                           confirmed_no_seizure_files=None) -> dict:
    """Resolve one file independently; missing event metadata is not a clean label."""
    review_reasons = review_reasons or {}
    confirmed_no_seizure_files = set(confirmed_no_seizure_files or ())
    if stem in review_reasons:
        return _quality_record(REVIEW_REQUIRED, review_reasons[stem])
    if events_by_file.get(stem):
        return _quality_record(VALIDATED)
    if stem in confirmed_no_seizure_files:
        return _quality_record(NO_SEIZURE_CONFIRMED, "explicit no-seizure confirmation")
    return _quality_record(REVIEW_REQUIRED, "no explicit seizure-free confirmation")


def _chbmit_quality_from_summary(parsed_events, declared_count, start_marker_count,
                                end_marker_count, has_other_seizure_marker=False):
    """Trust CHB-MIT quality only when the summary's explicit count agrees."""
    if (declared_count == 0 and not parsed_events
            and start_marker_count == 0 and end_marker_count == 0
            and not has_other_seizure_marker):
        return _quality_record(NO_SEIZURE_CONFIRMED,
                               "CHB-MIT summary explicitly says zero seizures")
    if (declared_count is not None and declared_count > 0
            and len(parsed_events) == declared_count
            and start_marker_count == declared_count
            and end_marker_count == declared_count):
        return _quality_record(VALIDATED)
    return _quality_record(REVIEW_REQUIRED,
                           "CHB-MIT seizure count missing or inconsistent with parsed intervals")


def _quality_allows(quality: dict, purpose: str = "both") -> bool:
    """Check one permission, or the intersection required by the shared index."""
    if purpose == "training":
        return bool(quality.get("training_eligible", False))
    if purpose == "event_eval":
        return bool(quality.get("event_eval_eligible", False))
    if purpose == "both":
        return (_quality_allows(quality, "training")
                and _quality_allows(quality, "event_eval"))
    raise ValueError(f"Unknown data-quality purpose: {purpose}")


def filter_recordings_by_quality(recordings, quality_by_file, stem_getter=lambda x: x):
    """Fail-closed file selector shared by window indexing and normalization input."""
    trusted, withheld = [], []
    for recording in recordings:
        stem = stem_getter(recording)
        quality = quality_by_file.get(
            stem, _quality_record(REVIEW_REQUIRED, "recording absent from trusted source inventory"))
        (trusted if _quality_allows(quality, "both") else withheld).append(recording)
    return trusted, withheld


def build_seizure_index(dataset_name: str, raw_dir: Path, return_quality: bool = False,
                        confirmed_no_seizure_files=None):
    """
    Within-file seizure index {edf_filename_stem: [(start_sec, end_sec), ...]}.
    With `return_quality=True`, also return per-file trust and use permissions.
    Unmentioned SIENA files require explicit no-seizure confirmation before use.
    """
    index = {}
    quality_by_file = {}

    if dataset_name == "chbmit":
        for summary_file in sorted(raw_dir.rglob("*-summary.txt")):
            index.update(parse_chbmit_summary(summary_file))
            text = summary_file.read_text(errors="ignore")
            for block in re.split(r"(?=File Name:)", text):
                name_match = re.search(r"File Name:\s*(\S+\.edf)", block)
                if not name_match:
                    continue
                stem = Path(name_match.group(1)).stem
                count_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
                parsed_events = index.get(stem, [])
                declared_count = int(count_match.group(1)) if count_match else None
                starts_count = len(re.findall(
                    r"^\s*Seizure(?:\s+\d+)?\s+Start Time:", block, re.M))
                ends_count = len(re.findall(
                    r"^\s*Seizure(?:\s+\d+)?\s+End Time:", block, re.M))
                other_seizure_marker = any(
                    re.search(r"\bSeizure\b", line, re.I)
                    and not re.search(r"Number of Seizures in File:", line, re.I)
                    for line in block.splitlines())
                quality_by_file[stem] = _chbmit_quality_from_summary(
                    parsed_events, declared_count, starts_count, ends_count,
                    other_seizure_marker)

    elif dataset_name == "siena":
        available, durations, starts = _siena_records(raw_dir)
        aggregate = []
        review_reasons = {}
        review_patient_reasons = defaultdict(set)
        event_count_audit = compare_siena_event_counts(raw_dir)
        for patient, audit in event_count_audit.items():
            if audit["status"] == "match":
                continue
            review_patient_reasons[patient].add(
                f"subject_info/annotation event count unresolved "
                f"({audit['subject_info_count']} vs {audit['annotation_heading_count']})")
            print(f"[SIENA event-count review] {patient}: "
                  f"subject_info={audit['subject_info_count']}, "
                  f"annotation_headings={audit['annotation_heading_count']}; "
                  "neither source selected as authoritative")
        for list_file in sorted(raw_dir.rglob("Seizures-list-*.txt")):
            file_diagnostics = []
            file_seizures = parse_siena_seizure_list(
                list_file, available_files=available, edf_durations=durations,
                diagnostics=file_diagnostics, edf_starts=starts)
            aggregate.extend(file_diagnostics)
            for item in file_diagnostics:
                if item.get("registration_correction") or item.get("time_corrections"):
                    print(f"[SIENA time correction] {item['source']} {item.get('raw_file')}: "
                          f"{item.get('raw_registration_start')} -> "
                          f"{item.get('registration_correction') or 'unchanged'}")
                if item["status"] in {"unlinked", "review", "rejected"}:
                    print(f"[SIENA {item['status']}] {item['source']} {item.get('raw_file')}: {item['reason']}")
                    if item.get("matched_file"):
                        review_reasons[item["matched_file"]] = item["reason"] or item["status"]
                    else:
                        patient = re.search(r"Seizures-list-(PN\d+)", item["source"])
                        if patient:
                            review_patient_reasons[patient.group(1)].add(
                                "unlinked seizure annotation for patient")
            for stem, seizures in file_seizures.items():
                index.setdefault(stem, []).extend(seizures)
        counts = defaultdict(int)
        reasons = defaultdict(int)
        for item in aggregate:
            counts[item["status"]] += 1
            if item.get("reason"):
                reasons[item["reason"]] += 1
        valid_events = [item for item in aggregate if item.get("event_valid")]
        print(f"[SIENA annotations] extracted={len(valid_events)}, "
              f"linked={sum(item['status'] == 'linked' for item in valid_events)}, "
              f"review={sum(item['status'] == 'review' for item in valid_events)}, "
              f"unlinked={sum(item['status'] == 'unlinked' for item in valid_events)}, "
              f"rejected={sum(item['status'] == 'rejected' for item in aggregate)}, "
              f"malformed={len(aggregate) - len(valid_events)}")
        for reason, count in sorted(reasons.items()):
            print(f"  {count} event(s): {reason}")

        # If an annotation cannot be linked to its intended file, quarantine
        # that patient's SIENA recordings; do not convert the orphan to a clean
        # negative through the missing-key fallback.
        for stem in available:
            patient = stem.split("-")[0]
            if patient in review_patient_reasons:
                patient_reason = "; ".join(sorted(review_patient_reasons[patient]))
                existing = review_reasons.get(stem)
                review_reasons[stem] = "; ".join(dict.fromkeys(
                    part for part in (existing, patient_reason) if part))
        confirmed = set(confirmed_no_seizure_files or ())
        for stem in available:
            quality_by_file[stem] = _quality_for_recording(
                stem, index, review_reasons, confirmed)
        print(f"[SIENA recording quality] validated={sum(q['status'] == VALIDATED for q in quality_by_file.values())}, "
              f"no_seizure_confirmed={sum(q['status'] == NO_SEIZURE_CONFIRMED for q in quality_by_file.values())}, "
              f"review_required={sum(q['status'] == REVIEW_REQUIRED for q in quality_by_file.values())}")
        quality_reasons = defaultdict(list)
        for stem, quality in quality_by_file.items():
            if quality["status"] == REVIEW_REQUIRED:
                quality_reasons[quality["reason"]].append(stem)
        for reason, stems in sorted(quality_reasons.items()):
            print(f"  withheld ({len(stems)} file(s)): {reason}: {', '.join(sorted(stems))}")

    else:
        raise ValueError(f"Unknown dataset_name for labeling: {dataset_name}")

    return (index, quality_by_file) if return_quality else index


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


def windows_for_quality(seizures: list, n_samples: int, quality: dict,
                        purpose: str = "both", **label_kwargs) -> list:
    """Return windows only when the selected use has its own quality approval."""
    if not _quality_allows(quality, purpose):
        return []
    return label_windows_for_file(seizures, n_samples, **label_kwargs)


def quality_aware_window_entries(stem: str, seizures: list, n_samples: int,
                                 quality: dict, **label_kwargs) -> list:
    """Build the same quality-tagged rows consumed by train/eval split code."""
    if not _quality_allows(quality, "both"):
        return []
    return [
        {"file_stem": stem, "start_sample": start_sample, "label": label,
         "data_quality": quality["status"],
         "training_eligible": quality["training_eligible"],
         "event_eval_eligible": quality["event_eval_eligible"]}
        for start_sample, label in label_windows_for_file(seizures, n_samples, **label_kwargs)
    ]


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
    return compute_patient_stats_from_arrays(all_data)


def compute_channel_stats_from_files(files: list, chunk_samples: int = 250_000) -> tuple:
    """Streaming channel moments over files; intended for training-only fit sets."""
    count = 0
    mean = m2 = None
    for path in files:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"expected channels x samples in {path}, got {arr.shape}")
        if mean is None:
            mean = np.zeros(arr.shape[0], dtype=np.float64)
            m2 = np.zeros(arr.shape[0], dtype=np.float64)
        elif arr.shape[0] != len(mean):
            raise ValueError("training recordings have inconsistent channel counts")
        for lo in range(0, arr.shape[1], chunk_samples):
            block = np.asarray(arr[:, lo:lo + chunk_samples], dtype=np.float64)
            n = block.shape[1]
            if not n:
                continue
            block_mean = block.mean(axis=1)
            block_m2 = np.square(block - block_mean[:, None]).sum(axis=1)
            delta = block_mean - mean
            total = count + n
            m2 += block_m2 + delta * delta * count * n / total
            mean += delta * n / total
            count = total
    if mean is None or count == 0:
        raise ValueError("cannot fit normalization from an empty training recording set")
    std = np.sqrt(np.maximum(m2 / count, 0.0))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_patient_stats_from_arrays(all_data: list) -> tuple:
    """Pure array form used to test normalization selection without opening NPY files."""
    concatenated = np.concatenate(all_data, axis=1)  # (channels, total_samples)
    mean = concatenated.mean(axis=1)
    std = concatenated.std(axis=1)
    std[std == 0] = 1.0  # dead/flat channel
    return mean.astype(np.float32), std.astype(np.float32)


def apply_channel_normalization(array, stats):
    """Apply frozen channel mean/std to a (channels, samples) array."""
    mean, std = stats
    return (np.asarray(array, dtype=np.float32) - mean[:, None]) / std[:, None]


def compute_training_channel_stats_from_arrays(recordings, training_patients):
    """Pure array reference for dataset-level training-patient channel stats.

    This pure helper is the reference policy for a split-aware normalizer. Each
    item is a mapping with dataset, patient_id, array, and independent quality
    permissions. Labels and recordings from validation/test patients are never
    consulted; production uses the same selection policy in SeizureDataset.
    """
    training_patients = {tuple(key) for key in training_patients}
    arrays_by_dataset = defaultdict(list)
    files_by_dataset = defaultdict(list)
    for recording in recordings:
        key = (recording["dataset"], recording["patient_id"])
        quality = recording.get("quality", {})
        if key not in training_patients:
            continue
        if not (quality.get("training_eligible", False)
                and quality.get("event_eval_eligible", False)):
            continue
        arrays_by_dataset[key[0]].append(np.asarray(recording["array"]))
        files_by_dataset[key[0]].append(recording.get("file", "<synthetic>"))
    if not arrays_by_dataset:
        raise ValueError("no quality-approved training recordings for normalization")
    stats = {name: compute_patient_stats_from_arrays(arrays)
             for name, arrays in arrays_by_dataset.items()}
    sources = {name: sorted(files) for name, files in files_by_dataset.items()}
    return stats, sources


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
                 exclude_patients: set = None,
                 normalization_mode: str = "per_patient"):
        if normalization_mode not in {"per_patient", "deferred", "none"}:
            raise ValueError(f"unknown normalization_mode: {normalization_mode}")
        self.normalization_mode = normalization_mode
        self.apply_augmentation = apply_augmentation
        self.augmentation_kwargs = augmentation_kwargs or {}
        self.rng = np.random.default_rng(seed)
        self.exclude_patients = set(EXCLUDE_PATIENTS if exclude_patients is None else exclude_patients)

        self.index = []                # dicts: dataset, patient_id, npy_path, start_sample, label, notch_freq
        self.patient_stats_cache = {}  # (dataset_name, patient_id) -> (mean, std)
        self._trusted_files_by_patient = {}
        self._patient_count = 0
        self.normalization_provenance = None

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

        within_file, quality_by_file = build_seizure_index(
            name, raw_dir, return_quality=True,
            confirmed_no_seizure_files=cfg.get("confirmed_no_seizure_files", ()))
        if name == "chbmit":
            file_start, patient_seizures = build_chbmit_timeline(raw_dir)
        else:
            file_start, patient_seizures = {}, {}

        candidates_by_patient = defaultdict(list)
        files_by_patient = defaultdict(list)
        excluded = defaultdict(int)
        review_required = defaultdict(int)
        for npy_file in sorted(processed_dir.glob("*.npy")):
            patient_id = get_patient_id(npy_file.stem, name)
            if (name, patient_id) in self.exclude_patients:
                excluded[patient_id] += 1
                continue
            candidates_by_patient[patient_id].append(npy_file)

        for patient_id, candidates in candidates_by_patient.items():
            trusted, withheld = filter_recordings_by_quality(
                candidates, quality_by_file, stem_getter=lambda path: path.stem)
            files_by_patient[patient_id].extend(trusted)
            review_required[patient_id] += len(withheld)

        counts = np.zeros(3, dtype=np.int64)
        n_timeline = n_within = 0

        for patient_id, files in files_by_patient.items():
            self._trusted_files_by_patient[(name, patient_id)] = list(files)
            if self.normalization_mode == "per_patient":
                mean, std = compute_patient_stats(files)
                self.patient_stats_cache[(name, patient_id)] = (mean, std)
            # time_sec: each patient gets its own 1e8-second block, so runs of windows
            # never connect across patients in event-level metrics
            patient_base = self._patient_count * 1e8
            self._patient_count += 1
            n_within_patient = 0

            for npy_file in files:
                n_samples = np.load(npy_file, mmap_mode="r").shape[1]
                stem = npy_file.stem

                if stem in file_start:
                    timeline_pid, offset = file_start[stem]
                    qualified_rows = quality_aware_window_entries(
                        stem, patient_seizures.get(timeline_pid, []), n_samples,
                        quality_by_file[stem], offset_sec=offset)
                    n_timeline += 1
                    file_time0 = patient_base + offset
                else:
                    qualified_rows = quality_aware_window_entries(
                        stem, within_file.get(stem, []), n_samples, quality_by_file[stem])
                    n_within += 1
                    n_within_patient += 1
                    file_time0 = patient_base + 5e7 + n_within_patient * 1e5

                for row in qualified_rows:
                    start_sample, label = row["start_sample"], row["label"]
                    counts[label] += 1
                    self.index.append({
                        "dataset": name,
                        "patient_id": patient_id,
                        "npy_path": npy_file,
                        "start_sample": start_sample,
                        "label": label,
                        "notch_freq": notch_freq,
                        "time_sec": file_time0 + start_sample / FS,
                        "data_quality": row["data_quality"],
                        "training_eligible": row["training_eligible"],
                        "event_eval_eligible": row["event_eval_eligible"],
                    })

        print(f"[{name}] windows: interictal={counts[0]:,} preictal={counts[1]:,} ictal={counts[2]:,} | "
              f"files labeled on patient timeline: {n_timeline}, within-file only: {n_within}")
        if excluded:
            print(f"[{name}] excluded patients: "
                  + ", ".join(f"{p} ({n} files)" for p, n in sorted(excluded.items())))
        if review_required:
            print(f"[{name}] recordings withheld for data-quality review: "
                  + ", ".join(f"{p} ({n} files)" for p, n in sorted(review_required.items())))

    def fit_training_normalization(self, train_indices):
        """Fit dataset-level channel stats from training patients and freeze them.

        Patient splitting must happen before this method. Only trusted files of
        patients represented by train_indices contribute; labels are not read.
        The resulting stats are applied to every indexed patient in that dataset.
        """
        if self.normalization_mode != "deferred":
            raise ValueError("training normalization requires normalization_mode='deferred'")
        train_indices = np.asarray(train_indices, dtype=np.int64)
        if train_indices.size == 0 or np.any(train_indices < 0) or np.any(train_indices >= len(self.index)):
            raise ValueError("training indices are empty or outside the dataset index")
        train_patients = {(self.index[int(i)]["dataset"], self.index[int(i)]["patient_id"])
                          for i in train_indices}
        files_by_dataset = defaultdict(list)
        fit_files = defaultdict(list)
        for key in sorted(train_patients):
            for path in self._trusted_files_by_patient.get(key, []):
                files_by_dataset[key[0]].append(path)
                fit_files[key[0]].append(Path(path).name)
        if not files_by_dataset:
            raise ValueError("training split has no quality-approved source recordings")
        stats_by_dataset = {
            name: compute_channel_stats_from_files(files)
            for name, files in files_by_dataset.items()
        }
        for entry in self.index:
            stats = stats_by_dataset.get(entry["dataset"])
            if stats is None:
                raise ValueError(f"training split has no normalization stats for {entry['dataset']}")
            self.patient_stats_cache[(entry["dataset"], entry["patient_id"])] = stats
        self.normalization_mode = "frozen_training"
        self.normalization_provenance = {
            "policy": "dataset_channel_zscore_from_training_patients_v1",
            "training_patients": sorted(f"{dataset}/{patient}"
                                         for dataset, patient in train_patients),
            "fit_files": {name: sorted(paths) for name, paths in fit_files.items()},
            "statistics": {name: {"mean": mean.tolist(), "std": std.tolist()}
                           for name, (mean, std) in stats_by_dataset.items()},
        }
        return self.normalization_provenance

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

        if self.normalization_mode != "none":
            key = (entry["dataset"], entry["patient_id"])
            if key not in self.patient_stats_cache:
                raise RuntimeError("normalization statistics have not been fit for this dataset")
            mean, std = self.patient_stats_cache[key]
            window = apply_channel_normalization(window, (mean, std))

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
