"""Cache and evaluation artifact provenance checks.

EDF identity uses header/metadata only. Processed NPY content digests are made
only by the explicit preparation helper and verified when provenance is built;
no array is deserialized here. Cache-array signatures remain metadata-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "preprocessing"))

PROVENANCE_SCHEMA_VERSION = 6
CACHE_SCHEMA_VERSION = 6
SHARD_SCHEMA_VERSION = 1
PROCESSED_NPY_MANIFEST_SCHEMA_VERSION = 2
ELIGIBILITY_POLICY_VERSION = "recording_quality_signal_and_processing_v2"
TRUSTED_QUALITY_STATES = {"VALIDATED", "NO_SEIZURE_CONFIRMED"}
RAW_FEATURE_NORMALIZATION_POLICY = "raw_signal_no_fitted_channel_stats_v1"
DEFAULT_AUGMENTATION_CONFIG = {
    "high_freq_noise_ratio": 0.15,
    "drift_amplitude_ratio": 0.6,
    "powerline_amplitude_ratio": 0.18,
    "apply_prob": 1.0,
}
SIGNAL_QUALITY_POLICY = {
    "version": "processed_signal_screen_v1",
    "expected_channels": 6,
    "expected_dtype": "float32",
    # 1 nV is far below ordinary EEG amplitudes but avoids accepting arrays
    # that vary only at numerical-noise scale; it is still only a screen.
    "absolute_std_floor_volts": 1e-9,
    # Screening only: flag channels whose variation is <0.01% of the median
    # usable channel variation. This is a conservative review trigger, not a
    # medical-quality threshold.
    "near_flat_relative_std": 1e-4,
    # Repeated extrema may indicate clipping, but filtered data cannot prove
    # ADC saturation. Flag for review only.
    "extreme_value_fraction_review": 0.01,
    "chunk_samples": 1_000_000,
}
ARTIFACT_PRODUCERS = {
    "mlp_split": "training/train_mlp.py",
    "mlp_branch": "training/train_mlp.py",
    "mlp_classifier": "training/train_mlp.py",
    "mlp_eval_data": "training/train_mlp.py",
    "tcn_split": "training/train_tcn.py",
    "tcn_branch": "training/train_tcn.py",
    "tcn_attention": "training/train_tcn.py",
    "tcn_classifier": "training/train_tcn.py",
    "tcn_resume": "training/train_tcn.py",
    "tcn_eval_probs": "training/train_tcn.py",
    "detection_oof": "evaluation/detection_loso.py",
    "loso_oof": "evaluation/loso_cache_analysis.py",
    "loso_oof_gap2": "evaluation/loso_cache_analysis_gap2.py",
}


def default_dataset_configs(root: Path = ROOT) -> list:
    root = Path(root)
    return [
        {"name": "chbmit", "processed_dir": root / "data/processed/chbmit",
         "raw_dir": root / "data/raw/chbmit", "notch_freq": 60.0},
        {"name": "siena", "processed_dir": root / "data/processed/siena",
         "raw_dir": root / "data/raw/siena", "notch_freq": 50.0},
    ]


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def fingerprint(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def processed_npy_manifest_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / "processed_npy_manifest.json"


def inventory_processed_files(processed_dir: Path) -> list:
    """List every NPY by metadata only. Inventory presence is not eligibility."""
    processed_dir = Path(processed_dir).resolve()
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed directory is missing: {processed_dir}")
    return [{"name": path.name, "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(processed_dir.glob("*.npy"))]


def _current_quality_policy(excluded_patients=()) -> dict:
    dataset_policy_path = Path(__file__).resolve().parents[1] / "training" / "dataset.py"
    preprocessing_policy_path = Path(__file__).resolve().parents[1] / "preprocessing" / "preprocess.py"
    normalized_exclusions = sorted({tuple(item) for item in excluded_patients})
    return {"version": ELIGIBILITY_POLICY_VERSION,
            "trusted_quality_states": sorted(TRUSTED_QUALITY_STATES),
            "requires_training_and_event_eval": True,
            "signal_quality_policy": SIGNAL_QUALITY_POLICY,
            "requires_verified_processing_evidence": True,
            "excluded_patients": [list(item) for item in normalized_exclusions],
            "classifier_code_sha256": {
                "cache_provenance.py": _file_sha256(Path(__file__)),
                "training/dataset.py": (_file_sha256(dataset_policy_path)
                                        if dataset_policy_path.is_file() else None),
                "preprocessing/preprocess.py": (_file_sha256(preprocessing_policy_path)
                                                if preprocessing_policy_path.is_file() else None),
            }}


def inspect_signal_array(array, *, policy=None) -> dict:
    """Read-only, chunked screening of a channels-by-samples processed signal.

    Reports numerical hazards and shape compatibility. A PASS is not evidence
    of clinical validity, correct annotations, or source provenance.
    """
    config = dict(SIGNAL_QUALITY_POLICY)
    if policy:
        config.update(policy)
    issues = []
    if getattr(array, "ndim", None) != 2:
        return {"status": "REVIEW_REQUIRED", "issues": ["array is not 2D"],
                "shape": list(getattr(array, "shape", ())),
                "policy_fingerprint": fingerprint(config), "channels": []}
    n_channels, n_samples = map(int, array.shape)
    if n_channels != int(config["expected_channels"]):
        issues.append(f"channel count {n_channels} != expected {config['expected_channels']}")
    dtype = str(getattr(array, "dtype", "unknown"))
    if dtype != config["expected_dtype"]:
        issues.append(f"dtype {dtype} != expected {config['expected_dtype']}")

    channels = []
    chunk_size = max(1, int(config["chunk_samples"]))
    for channel_index in range(n_channels):
        finite_count = 0
        nonfinite_count = 0
        zero_count = 0
        minimum = float("inf")
        maximum = float("-inf")
        total = 0.0
        total_sq = 0.0
        extreme_count = 0
        min_extreme_count = 0
        max_extreme_count = 0
        for begin in range(0, n_samples, chunk_size):
            values = np.asarray(array[channel_index, begin:min(begin + chunk_size, n_samples)],
                                dtype=np.float64)
            finite = np.isfinite(values)
            nonfinite_count += int(values.size - np.count_nonzero(finite))
            if not np.any(finite):
                continue
            valid = values[finite]
            finite_count += int(valid.size)
            zero_count += int(np.count_nonzero(valid == 0))
            block_min, block_max = float(np.min(valid)), float(np.max(valid))
            if block_min < minimum:
                minimum = block_min
                min_extreme_count = int(np.count_nonzero(valid == minimum))
            elif block_min == minimum:
                min_extreme_count += int(np.count_nonzero(valid == minimum))
            if block_max > maximum:
                maximum = block_max
                max_extreme_count = int(np.count_nonzero(valid == maximum))
            elif block_max == maximum:
                max_extreme_count += int(np.count_nonzero(valid == maximum))
            total += float(np.sum(valid, dtype=np.float64))
            total_sq += float(np.dot(valid, valid))
        mean = total / finite_count if finite_count else None
        variance = (max(0.0, total_sq / finite_count - mean * mean)
                    if finite_count else None)
        std = float(np.sqrt(variance)) if variance is not None else None
        if finite_count:
            extreme_count = min_extreme_count + max_extreme_count
        extreme_fraction = extreme_count / finite_count if finite_count else None
        clipping_suspected = (finite_count > 0 and maximum > minimum and
                              extreme_fraction is not None and
                              extreme_fraction >= float(config["extreme_value_fraction_review"]))
        row = {"index": channel_index, "finite_samples": finite_count,
               "nonfinite_samples": nonfinite_count, "zero_samples": zero_count,
               "minimum": minimum if finite_count else None,
               "maximum": maximum if finite_count else None,
               "range": (maximum - minimum) if finite_count else None,
               "mean": mean, "std": std,
               "extreme_value_fraction": extreme_fraction,
               "clipping_status": "suspected" if clipping_suspected
               else "not_flagged_not_proven_absent"}
        if nonfinite_count:
            row["issues"] = ["contains NaN or Inf"]
        else:
            row["issues"] = []
        if finite_count and (maximum == minimum or std <= float(config["absolute_std_floor_volts"])):
            row["issues"].append("constant or zero-variance channel")
        if row["clipping_status"] == "suspected":
            row["issues"].append("repeated exact extrema; clipping is only suspected")
        channels.append(row)

    positive_stds = [row["std"] for row in channels
                     if row["std"] is not None and row["std"] > float(config["absolute_std_floor_volts"])]
    reference_std = float(np.median(positive_stds)) if positive_stds else None
    for row in channels:
        if (reference_std is not None and row["std"] is not None and
                row["std"] > float(config["absolute_std_floor_volts"]) and
                row["std"] < reference_std * float(config["near_flat_relative_std"])):
            row["issues"].append("near-flat relative to other channels (screening threshold)")
    for row in channels:
        if row["issues"]:
            issues.extend(f"channel {row['index']}: {item}" for item in row["issues"])
    return {"status": "REVIEW_REQUIRED" if issues else "PASS",
            "issues": issues, "shape": [n_channels, n_samples], "dtype": dtype,
            "sample_count": n_samples, "channels": channels,
            "policy_fingerprint": fingerprint(config),
            "limitations": ["numerical screening does not establish clinical validity",
                            "cannot establish annotation correctness",
                            "clipping flag is heuristic; filtered data cannot prove ADC saturation",
                            "channel labels, units, sample rate, and source linkage require processing evidence"]}


def inspect_signal_file(path: Path, *, policy=None) -> dict:
    """Screen an NPY using a read-only memory map; no array is modified."""
    source = Path(path)
    try:
        array = np.load(source, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return {"status": "REVIEW_REQUIRED", "issues": [f"NPY unreadable: {type(exc).__name__}: {exc}"],
                "shape": None, "dtype": None,
                "policy_fingerprint": fingerprint(dict(SIGNAL_QUALITY_POLICY, **(policy or {})))}
    try:
        return inspect_signal_array(array, policy=policy)
    finally:
        del array


def _signal_gate_signature(report: dict) -> dict:
    """Fields that decide acceptance; descriptive measurements may vary a few ulps."""
    if not isinstance(report, dict):
        report = {}
    return {"status": report.get("status"), "issues": report.get("issues"),
            "shape": report.get("shape"), "dtype": report.get("dtype"),
            "policy_fingerprint": report.get("policy_fingerprint"),
            "channel_issues": [row.get("issues") for row in report.get("channels", [])]}


def preprocessing_config_fingerprint(dataset: str) -> str:
    """Fingerprint current deterministic preprocessing/channel configuration."""
    from preprocess import BANDPASS_HIGH, BANDPASS_LOW, DATASETS, TARGET_SFREQ
    import preprocess
    config = DATASETS.get(dataset, {})
    preprocess_path = Path(preprocess.__file__).resolve()
    return fingerprint({"dataset": dataset, "target_sfreq": TARGET_SFREQ,
                        "bandpass_low": BANDPASS_LOW, "bandpass_high": BANDPASS_HIGH,
                        "notch_freq": config.get("notch_freq"),
                        "channel_order": list(config.get("target_channels", [])),
                        "units": "V", "preprocess_code_sha256": _file_sha256(preprocess_path),
                        "mne_version": getattr(preprocess.mne, "__version__", None),
                        "numpy_version": np.__version__})


def _processing_evidence_status(evidence, *, stem, dataset, expected_channels,
                               source_registry, processed_path=None):
    if not isinstance(evidence, dict):
        return False, "verified processing evidence is missing"
    required = {"schema_version", "status", "source_file", "source_fingerprint",
                "preprocessing_config_fingerprint", "channel_order", "sample_rate_hz", "units",
                "dataset", "output_file", "output_size", "output_sha256"}
    if not required.issubset(evidence):
        return False, "processing evidence fields are incomplete"
    if evidence.get("schema_version") != 1 or evidence.get("status") != "VERIFIED":
        return False, evidence.get("reason") or "processing evidence schema/status is not verified"
    if evidence.get("dataset") != dataset:
        return False, "processing evidence dataset does not match"
    if processed_path is None or not Path(processed_path).is_file():
        return False, "processed NPY is missing"
    processed_path = Path(processed_path)
    if evidence.get("output_file") != processed_path.name:
        return False, "processing evidence output filename does not match"
    if evidence.get("output_size") != processed_path.stat().st_size:
        return False, "processed NPY size does not match its provenance sidecar"
    if evidence.get("output_sha256") != _file_sha256(processed_path):
        return False, "processed NPY content does not match its provenance sidecar"
    if not isinstance(evidence.get("source_size"), int) or evidence["source_size"] <= 0:
        return False, "processing evidence source size is missing or invalid"
    if str(evidence.get("source_file", "")).casefold() != f"{stem}.edf".casefold():
        return False, "processing source file does not match recording identity"
    if not evidence.get("source_fingerprint") or not evidence.get("preprocessing_config_fingerprint"):
        return False, "processing source or configuration fingerprint is missing"
    if evidence.get("preprocessing_config_fingerprint") != preprocessing_config_fingerprint(dataset):
        return False, "processing configuration fingerprint does not match current policy"
    source_matches = [source for source in source_registry.values()
                      if Path(str(source.get("path", ""))).name.casefold() ==
                      f"{stem}.edf".casefold()]
    if not any(source.get("fingerprint") == evidence.get("source_fingerprint")
               for source in source_matches):
        return False, "processing evidence is not bound to the inventoried source EDF"
    if list(evidence.get("channel_order", [])) != list(expected_channels):
        return False, "processed channel order does not match policy"
    if evidence.get("sample_rate_hz") != 250 or evidence.get("units") != "V":
        return False, "processed sample rate or units do not match policy"
    return True, "verified source and processing configuration evidence"


def _recording_decision(quality: dict, source_ids: list, excluded: bool,
                        signal_quality: dict = None, processing_evidence: dict = None,
                        stem: str = "", dataset: str = "", source_registry=None,
                        processed_path: Path = None) -> tuple:
    """Return a fail-closed shared-index decision from source-backed fields."""
    if excluded:
        return "REJECTED", "patient is excluded by dataset policy"
    if not isinstance(quality, dict):
        return "REVIEW_REQUIRED", "quality record is missing or malformed"
    status = quality.get("status")
    if status == "REVIEW_REQUIRED":
        return "REVIEW_REQUIRED", quality.get("reason") or "quality record requires review"
    if status not in TRUSTED_QUALITY_STATES:
        return "REVIEW_REQUIRED", f"quality status is missing or unknown: {status!r}"
    permission_keys = ("training_eligible", "event_eval_eligible")
    if any(key not in quality or type(quality[key]) is not bool for key in permission_keys):
        return "REVIEW_REQUIRED", "quality permissions are missing or malformed"
    if not source_ids:
        return "REVIEW_REQUIRED", "quality decision has no auditable source references"
    if not all(quality[key] for key in permission_keys):
        return "REJECTED", quality.get("reason") or "required quality permission is false"
    signal_status = (signal_quality or {}).get("status")
    if signal_status != "PASS":
        return "REVIEW_REQUIRED", "; ".join((signal_quality or {}).get("issues", [])) or "signal-quality check is missing or not passed"
    from preprocess import DATASETS
    expected_channels = DATASETS.get(dataset, {}).get("target_channels", [])
    processing_ok, processing_reason = _processing_evidence_status(
        processing_evidence, stem=stem, dataset=dataset,
        expected_channels=expected_channels,
        source_registry=source_registry or {}, processed_path=processed_path)
    if not processing_ok:
        return "REVIEW_REQUIRED", processing_reason
    return "ELIGIBLE", quality.get("reason") or "known trusted status and both permissions allow use"


def _record_source_evidence(dataset: str, stem: str, quality) -> dict:
    if not isinstance(quality, dict):
        return {"source": "missing quality record", "source_key": stem,
                "quality_status": "MISSING", "training_eligible": False,
                "event_eval_eligible": False, "reason": "quality record is missing"}
    return {
        "source": ("CHB-MIT summary/index inventory" if dataset == "chbmit"
                   else "SIENA seizure-list, RECORDS, and EDF metadata index"),
        "source_key": stem,
        "quality_status": quality.get("status", "MISSING"),
        "training_eligible": quality.get("training_eligible"),
        "event_eval_eligible": quality.get("event_eval_eligible"),
        "reason": quality.get("reason"),
    }


def classify_recording_inventory(dataset: str, processed_dir: Path, quality_by_file: dict,
                                 *, source_records_by_file: dict,
                                 signal_quality_by_file: dict = None,
                                 processing_evidence_by_file: dict = None,
                                 excluded_patients=None, patient_id_getter=None) -> dict:
    """Create a quality report over a complete processed-file inventory.

    A missing source decision is REVIEW_REQUIRED. Only explicit permissions for
    both shared-index uses, a passed signal screen, verified processing
    evidence, and no configured exclusion produce ELIGIBLE. Missing signal or
    processing evidence fails closed.
    """
    from dataset import EXCLUDE_PATIENTS, get_patient_id

    processed_dir = Path(processed_dir).resolve()
    getter = patient_id_getter or (lambda stem: get_patient_id(stem, dataset))
    excluded_patients = set(EXCLUDE_PATIENTS if excluded_patients is None else excluded_patients)
    inventory = inventory_processed_files(processed_dir)
    decisions, source_records = [], {}
    source_ids_by_file = {}
    for stem, evidence in (source_records_by_file or {}).items():
        ids = []
        for source in evidence if isinstance(evidence, (list, tuple)) else ():
            if not isinstance(source, dict):
                continue
            source_path = source.get("path")
            source_fingerprint = source.get("fingerprint")
            if not isinstance(source_path, str) or not source_path:
                continue
            if not isinstance(source_fingerprint, str) or not source_fingerprint:
                continue
            source_id = fingerprint({"path": source_path,
                                     "fingerprint": source_fingerprint})
            source_records[source_id] = {"path": source_path,
                                         "fingerprint": source_fingerprint}
            ids.append(source_id)
        source_ids_by_file[str(stem)] = sorted(set(ids))
    for item in inventory:
        stem = Path(item["name"]).stem
        patient_id = getter(stem)
        quality = quality_by_file.get(stem)
        excluded = (dataset, patient_id) in excluded_patients
        source_ids = source_ids_by_file.get(stem, [])
        supplied_signal = (signal_quality_by_file or {}).get(stem)
        processing = (processing_evidence_by_file or {}).get(stem)
        sidecar_path = processed_dir / (item["name"] + ".provenance.json")
        if processing is None and sidecar_path.is_file():
            try:
                processing = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                processing = {"status": "INVALID", "reason": "processing sidecar is unreadable"}
        signal = inspect_signal_file(processed_dir / item["name"])
        if supplied_signal is not None and supplied_signal != signal:
            signal = {**signal, "status": "REVIEW_REQUIRED",
                      "issues": list(signal.get("issues", [])) +
                      ["supplied signal-quality evidence conflicts with a fresh scan"]}
        status, reason = _recording_decision(
            quality, source_ids, excluded, signal, processing, stem=stem, dataset=dataset,
            source_registry=source_records, processed_path=processed_dir / item["name"])
        decisions.append({
            **item, "path": (processed_dir / item["name"]).as_posix(),
            "dataset": dataset, "stem": stem, "patient_id": patient_id,
            "decision": status, "reason": reason,
            "quality_record": quality,
            "source_ids": source_ids,
            "signal_quality": signal,
            "processing_evidence": processing,
            "source_evidence": _record_source_evidence(dataset, stem, quality),
        })
    policy = _current_quality_policy(excluded_patients)
    payload = {"dataset": dataset, "processed_dir": processed_dir.as_posix(),
               "inventory": inventory, "recordings": decisions,
               "sources": source_records,
               "quality_policy": policy, "quality_policy_fingerprint": fingerprint(policy)}
    return {**payload, "eligibility_fingerprint": fingerprint(payload)}


def validate_eligibility_report(report: dict) -> None:
    if not isinstance(report, dict) or not isinstance(report.get("recordings"), list):
        raise ValueError("eligibility report is missing or malformed")
    report_dir = report.get("processed_dir")
    if not isinstance(report_dir, str) or not Path(report_dir).is_dir():
        raise ValueError("eligibility report processed directory is missing")
    payload = {key: value for key, value in report.items()
               if key != "eligibility_fingerprint"}
    if fingerprint(payload) != report.get("eligibility_fingerprint"):
        raise ValueError("eligibility report fingerprint is invalid")
    from dataset import EXCLUDE_PATIENTS
    dataset = report.get("dataset")
    current_exclusions = (EXCLUDE_PATIENTS if dataset in {"chbmit", "siena"}
                          else report.get("quality_policy", {}).get("excluded_patients", ()))
    expected_policy = _current_quality_policy(current_exclusions)
    if report.get("quality_policy") != expected_policy:
        raise ValueError("eligibility report quality policy is stale or malformed")
    if report.get("quality_policy_fingerprint") != fingerprint(expected_policy):
        raise ValueError("eligibility report quality policy fingerprint is invalid")
    inventory_names = [item.get("name") for item in report.get("inventory", [])]
    record_names = [item.get("name") for item in report["recordings"]]
    if sorted(inventory_names) != sorted(record_names) or len(set(record_names)) != len(record_names):
        raise ValueError("eligibility report does not classify the complete inventory")
    if inventory_processed_files(Path(report_dir)) != report.get("inventory"):
        raise ValueError("eligibility report inventory is stale")
    inventory_by_name = {item["name"]: item for item in report["inventory"]}
    allowed = {"ELIGIBLE", "REJECTED", "REVIEW_REQUIRED"}
    sources = report.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("eligibility report source registry is missing")
    excluded = {tuple(item) for item in expected_policy["excluded_patients"]}
    for record in report["recordings"]:
        inventory_item = inventory_by_name.get(record.get("name"))
        expected_path = (Path(report["processed_dir"]) / str(record.get("name"))).resolve().as_posix()
        if (inventory_item is None
                or any(record.get(key) != inventory_item.get(key)
                       for key in ("name", "size", "mtime_ns"))
                or record.get("path") != expected_path
                or record.get("dataset") != report.get("dataset")
                or record.get("stem") != Path(str(record.get("name"))).stem):
            raise ValueError("eligibility report recording identity conflicts with inventory")
        if record.get("decision") not in allowed:
            raise ValueError("eligibility report contains an unknown decision")
        refs = record.get("source_ids")
        if not isinstance(refs, list) or any(source_id not in sources for source_id in refs):
            raise ValueError("eligibility report has missing or dangling source references")
        for source_id in refs:
            source = sources[source_id]
            if not isinstance(source, dict) or not source.get("path") or not source.get("fingerprint"):
                raise ValueError("eligibility report contains malformed source evidence")
            if fingerprint({"path": source["path"], "fingerprint": source["fingerprint"]}) != source_id:
                raise ValueError("eligibility report source identity is invalid")
        fresh_signal = inspect_signal_file(Path(report["processed_dir"]) / record["name"])
        if _signal_gate_signature(fresh_signal) != _signal_gate_signature(
                record.get("signal_quality", {})):
            raise ValueError("eligibility report signal-quality evidence is stale or inconsistent")
        expected_decision, expected_reason = _recording_decision(
            record.get("quality_record"), refs,
            (record.get("dataset"), record.get("patient_id")) in excluded,
            record.get("signal_quality"), record.get("processing_evidence"),
            stem=record.get("stem", ""), dataset=record.get("dataset", ""),
            source_registry=sources, processed_path=Path(report["processed_dir"]) / record["name"])
        if record.get("decision") != expected_decision or record.get("reason") != expected_reason:
            raise ValueError("eligibility report decision conflicts with its quality evidence")
        if record.get("source_evidence") != _record_source_evidence(
                record.get("dataset"), record.get("stem"), record.get("quality_record")):
            raise ValueError("eligibility report source evidence conflicts with its quality record")


def write_processed_npy_manifest(processed_dir: Path, *, eligibility_report: dict,
                                 approved_report_fingerprint: str,
                                 approved_eligible_files, overwrite=False) -> Path:
    """Hash only the caller-approved eligible list.

    The fingerprint/list arguments are a caller attestation, not authenticated
    human approval. This repository has no trusted approval identity service.
    """
    processed_dir = Path(processed_dir).resolve()
    path = processed_npy_manifest_path(processed_dir)
    if path.exists() and not overwrite:
        raise FileExistsError(f"processed NPY manifest already exists: {path}")
    validate_eligibility_report(eligibility_report)
    if Path(eligibility_report["processed_dir"]).resolve() != processed_dir:
        raise ValueError("eligibility report belongs to a different processed directory")
    if approved_report_fingerprint != eligibility_report["eligibility_fingerprint"]:
        raise ValueError("explicit eligibility approval does not match the report")
    eligible = sorted(record["name"] for record in eligibility_report["recordings"]
                      if record["decision"] == "ELIGIBLE")
    approved = sorted(map(str, approved_eligible_files))
    if approved != eligible:
        raise ValueError("approved file list must exactly match the report's eligible files")
    current_inventory = inventory_processed_files(processed_dir)
    if current_inventory != eligibility_report["inventory"]:
        raise ValueError("processed file inventory changed after eligibility review")
    files = []
    for name in approved:
        source = processed_dir / name
        files.append({"name": source.name, "size": source.stat().st_size,
                      "sha256": _file_sha256(source)})
    payload = {"schema_version": PROCESSED_NPY_MANIFEST_SCHEMA_VERSION,
               "processed_dir": str(processed_dir), "files": files,
               "inventory": eligibility_report["inventory"],
               "inventory_fingerprint": fingerprint(eligibility_report["inventory"]),
               "eligibility_report": eligibility_report,
               "eligibility_fingerprint": eligibility_report["eligibility_fingerprint"],
               "quality_policy_fingerprint": eligibility_report["quality_policy_fingerprint"]}
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode()
    with tempfile.NamedTemporaryFile(dir=processed_dir, prefix=".processed-manifest.",
                                     suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if overwrite:
            os.replace(temp_path, path)
        else:
            os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def validate_processed_npy_manifest(processed_dir: Path, *, eligibility_report: dict) -> list:
    """Verify inventory, current eligibility and hashes of eligible files only."""
    processed_dir = Path(processed_dir).resolve()
    path = processed_npy_manifest_path(processed_dir)
    if not path.is_file():
        raise ValueError(f"processed NPY content manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != PROCESSED_NPY_MANIFEST_SCHEMA_VERSION:
        raise ValueError("processed NPY content manifest schema is unsupported")
    if manifest.get("processed_dir") != str(processed_dir):
        raise ValueError("processed NPY manifest belongs to a different directory")
    validate_eligibility_report(eligibility_report)
    current_inventory = inventory_processed_files(processed_dir)
    if current_inventory != eligibility_report["inventory"]:
        raise ValueError("current processed inventory does not match eligibility report")
    if manifest.get("inventory") != current_inventory:
        raise ValueError("processed NPY manifest inventory does not match current files")
    if manifest.get("inventory_fingerprint") != fingerprint(current_inventory):
        raise ValueError("processed NPY inventory fingerprint is invalid")
    if manifest.get("eligibility_fingerprint") != eligibility_report.get("eligibility_fingerprint"):
        raise ValueError("processed NPY manifest eligibility does not match current report")
    if manifest.get("eligibility_report") != eligibility_report:
        raise ValueError("processed NPY manifest decisions or reasons do not match current report")
    if manifest.get("quality_policy_fingerprint") != eligibility_report.get("quality_policy_fingerprint"):
        raise ValueError("processed NPY manifest quality policy does not match")
    eligible = sorted(record["name"] for record in eligibility_report["recordings"]
                      if record["decision"] == "ELIGIBLE")
    recorded = manifest.get("files")
    if not isinstance(recorded, list):
        raise ValueError("processed NPY manifest file list is malformed")
    by_name = {entry.get("name"): entry for entry in recorded if isinstance(entry, dict)}
    if len(by_name) != len(recorded) or sorted(by_name) != eligible:
        raise ValueError("processed NPY manifest eligible file list does not match")
    verified = []
    for name in eligible:
        source = processed_dir / name
        entry = by_name[name]
        if entry.get("size") != source.stat().st_size or entry.get("sha256") != _file_sha256(source):
            raise ValueError(f"processed NPY content does not match manifest: {name}")
        verified.append({"name": name, "size": entry["size"], "sha256": entry["sha256"]})
    return verified


def shard_manifest_path(shard_path: Path) -> Path:
    path = Path(shard_path)
    return path.with_name(path.name + ".provenance.json")


def write_shard_manifest(shard_path: Path, *, shard_key: str,
                         build_fingerprint: str, expected_shape,
                         expected_dtype: str) -> Path:
    """Bind a newly written shard to its key, build policy, and content."""
    shard_path = Path(shard_path)
    if not shard_path.is_file():
        raise ValueError(f"cannot write provenance for missing shard: {shard_path}")
    path = shard_manifest_path(shard_path)
    stat = shard_path.stat()
    payload = {"schema_version": SHARD_SCHEMA_VERSION,
               "shard_key": shard_key,
               "build_fingerprint": build_fingerprint,
               "artifact_name": shard_path.name,
               "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
               "sha256": _file_sha256(shard_path),
               "shape": list(expected_shape), "dtype": expected_dtype}
    # Exclusive creation prevents silently replacing a pre-existing manifest.
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def validate_shard_manifest(shard_path: Path, *, shard_key: str,
                            build_fingerprint: str, expected_shape,
                            expected_dtype: str) -> dict:
    """Fail closed before a shard is reused or merged."""
    shard_path = Path(shard_path)
    path = shard_manifest_path(shard_path)
    if not path.is_file():
        raise ValueError(f"shard provenance manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    required = {"schema_version", "shard_key", "build_fingerprint", "artifact_name",
                "size", "mtime_ns", "sha256", "shape", "dtype"}
    if not required.issubset(manifest):
        raise ValueError("shard provenance manifest is incomplete")
    if manifest["schema_version"] != SHARD_SCHEMA_VERSION:
        raise ValueError("shard provenance schema is unsupported")
    expected = {"shard_key": shard_key, "build_fingerprint": build_fingerprint,
                "artifact_name": shard_path.name, "shape": list(expected_shape),
                "dtype": expected_dtype}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("shard provenance does not match current build settings")
    if not shard_path.is_file():
        raise ValueError(f"shard file is missing: {shard_path}")
    stat = shard_path.stat()
    if manifest["size"] != stat.st_size or manifest["mtime_ns"] != stat.st_mtime_ns:
        raise ValueError("shard file metadata does not match its manifest")
    if manifest["sha256"] != _file_sha256(shard_path):
        raise ValueError("shard content does not match its manifest")
    return manifest


def _producer_identity(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        rel = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    return {"path": rel, "sha256": _file_sha256(path)}


def shard_fingerprint(path: str, starts: bytes, mean: bytes, std: bytes,
                      feature_names, augmentation_config: dict,
                      augment: bool, seed: int, notch_freq: float,
                      cache_provenance_fingerprint: str) -> str:
    """Stable key covering every known input that can change shard contents."""
    h = hashlib.sha256()
    for part in (path.encode(), starts, mean, std, _canonical({
            "feature_names": list(feature_names), "augmentation": augmentation_config,
            "augment": bool(augment), "seed": int(seed),
            "notch_freq": float(notch_freq),
            "cache_provenance_fingerprint": cache_provenance_fingerprint,
            "cache_schema_version": CACHE_SCHEMA_VERSION})):
        h.update(part)
    return h.hexdigest()


def fold_normalization_manifest(training_patients, center, scale,
                                policy="robust_median_iqr_v1") -> dict:
    """Describe a fold-only feature scaler and the exact patients that fit it."""
    payload = {"policy": policy,
               "training_patients": sorted(map(str, training_patients)),
               "center": np.asarray(center, dtype=np.float32).tolist(),
               "scale": np.asarray(scale, dtype=np.float32).tolist()}
    return {**payload, "fingerprint": fingerprint(payload)}


def validate_fold_normalization_manifest(recorded: dict, expected: dict) -> None:
    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        raise ValueError("fold normalization provenance is missing or malformed")
    payload = {key: value for key, value in recorded.items() if key != "fingerprint"}
    if fingerprint(payload) != recorded.get("fingerprint"):
        raise ValueError("fold normalization provenance fingerprint is invalid")
    if recorded.get("fingerprint") != expected.get("fingerprint"):
        raise ValueError("fold normalization provenance does not match this training fold")


def validate_matching_normalization_manifests(*manifests: dict) -> str:
    """Ensure separately trained model branches use the same frozen signal stats."""
    fingerprints = [manifest.get("details", {}).get("normalization_fingerprint")
                    for manifest in manifests]
    if not fingerprints or any(not value for value in fingerprints):
        raise ValueError("model artifact is missing normalization provenance")
    if len(set(fingerprints)) != 1:
        raise ValueError("model artifacts use different normalization statistics")
    return fingerprints[0]


def _file_signature(path: Path, root: Path) -> dict:
    stat = path.stat()
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = str(path.resolve())
    return {"path": rel, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_metadata(raw_dir: Path, root: Path) -> list:
    if not raw_dir.exists():
        return []
    paths = [p for p in raw_dir.rglob("*") if p.is_file()
             and (p.name == "RECORDS" or p.name == "subject_info.csv"
                  or p.name.endswith("-summary.txt")
                  or p.name.startswith("Seizures-list-") and p.suffix.lower() == ".txt")]
    sources = []
    for path in sorted(paths):
        signature = _file_signature(path, root)
        signature["content_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(signature)
    # EDF size and mtime catch source replacement without reading signal data.
    for path in sorted(raw_dir.rglob("*.edf")):
        signature = _file_signature(path, root)
        try:
            with path.open("rb") as handle:
                signature["edf_header_sha256"] = hashlib.sha256(handle.read(256)).hexdigest()
        except OSError:
            signature["edf_header_sha256"] = None
        sources.append(signature)
    return sources


def _quality_source_records_by_file(dataset: str, processed_dir: Path,
                                   source_files: list) -> dict:
    """Bind recordings to auditable source metadata records by patient/file."""
    from dataset import get_patient_id

    result = {}
    for item in inventory_processed_files(processed_dir):
        stem = Path(item["name"]).stem
        patient = get_patient_id(stem, dataset)
        refs = []
        for source in source_files:
            source_path = source.get("path", "")
            base = Path(source_path).name
            low = base.lower()
            relevant = base.lower() == f"{stem}.edf".lower()
            if dataset == "siena":
                relevant |= low in {"records", "subject_info.csv"}
                relevant |= low == f"seizures-list-{patient}".lower() + ".txt"
            elif dataset == "chbmit":
                relevant |= low.startswith(patient.lower() + "-") and low.endswith("-summary.txt")
            digest = source.get("content_sha256") or source.get("edf_header_sha256")
            if relevant and digest:
                refs.append({"path": source_path, "fingerprint": digest})
        result[stem] = refs
    return result


def _git_identity(root: Path) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        commit = None
        status = None
    return {"commit": commit, "tracked_worktree_dirty": bool(status) if status is not None else None,
            "working_tree_assumed_clean": False}


def build_dataset_provenance(dataset_configs=None, root: Path = ROOT) -> dict:
    """Build current source, quality, label, preprocessing and code identity.

    Reads annotation text and EDF/NPY filesystem metadata only. EDF headers are
    parsed by the existing SIENA indexer; no signal samples are read here.
    """
    from dataset import (  # local import avoids imposing project paths on callers
        BUFFER_SEC, EXCLUDE_PATIENTS, FS, INTERICTAL, ICTAL, PREICTAL, POSTICTAL_SEC, PREICTAL_SEC, STEP_SEC,
        WINDOW_SEC, build_seizure_index,
    )
    import preprocess
    from handcrafted_features import FEATURE_NAMES

    root = Path(root)
    configs = dataset_configs or default_dataset_configs(root)
    source_files, eligible_recordings, quality_records = [], [], []
    eligibility_reports = []
    processed_npy_manifests = []
    preprocessing_config = {
        "target_sfreq": preprocess.TARGET_SFREQ,
        "bandpass_low": preprocess.BANDPASS_LOW,
        "bandpass_high": preprocess.BANDPASS_HIGH,
        "datasets": {},
    }

    for cfg in configs:
        name = cfg["name"]
        raw_dir, processed_dir = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
        dataset_source_files = _source_metadata(raw_dir, root)
        source_files.extend(dataset_source_files)
        preprocessing_config["datasets"][name] = {
            "notch_freq": float(cfg["notch_freq"]),
            "channels": preprocess.DATASETS.get(name, {}).get("target_channels", []),
        }
        _, quality_by_file = build_seizure_index(
            name, raw_dir, return_quality=True,
            confirmed_no_seizure_files=cfg.get("confirmed_no_seizure_files", ()))
        report = classify_recording_inventory(
            name, processed_dir, quality_by_file,
            source_records_by_file=_quality_source_records_by_file(
                name, processed_dir, dataset_source_files),
            excluded_patients=EXCLUDE_PATIENTS)
        eligibility_reports.append(report)
        verified_processed_files = validate_processed_npy_manifest(
            processed_dir, eligibility_report=report)
        processed_npy_manifests.append({
            "dataset": name,
            "processed_dir": str(processed_dir.resolve()),
            "inventory_fingerprint": fingerprint(report["inventory"]),
            "eligibility_fingerprint": report["eligibility_fingerprint"],
            "quality_policy_fingerprint": report["quality_policy_fingerprint"],
            "files": verified_processed_files,
        })
        content_by_name = {entry["name"]: entry["sha256"]
                           for entry in verified_processed_files}
        for decision in report["recordings"]:
            npy_path = processed_dir / decision["name"]
            record = {
                "dataset": name, "stem": decision["stem"],
                "patient_id": decision["patient_id"],
                "quality": decision["source_evidence"]["quality_status"],
                "eligibility": decision["decision"],
                "eligibility_reason": decision["reason"],
                "training_eligible": decision["source_evidence"]["training_eligible"],
                "event_eval_eligible": decision["source_evidence"]["event_eval_eligible"],
                "file": {**_file_signature(npy_path, root),
                         "inventory_size": decision["size"]},
            }
            if decision["decision"] == "ELIGIBLE":
                record["file"]["content_sha256"] = content_by_name[decision["name"]]
                eligible_recordings.append(record)
            quality_records.append(record)

    code_paths = [ROOT / rel for rel in (
        "training/dataset.py", "preprocessing/preprocess.py",
        "preprocessing/noise_injection.py", "features/handcrafted_features.py",
        "features/build_feature_cache.py", "features/cache_provenance.py",
        "training/train_mlp.py", "training/train_tcn.py",
        "evaluation/metrics.py", "models/mlp_branch.py", "models/tcn_branch.py",
        "models/channel_attention.py", "models/fusion_model.py")]
    code = [{"path": p.relative_to(ROOT).as_posix(),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in code_paths if p.is_file()]
    label_policy = {"fs": FS, "window_sec": WINDOW_SEC, "step_sec": STEP_SEC,
                    "preictal_sec": PREICTAL_SEC, "postictal_sec": POSTICTAL_SEC,
                    "buffer_sec": BUFFER_SEC,
                    "labels": {"interictal": INTERICTAL, "preictal": PREICTAL,
                               "ictal": ICTAL},
                    "exclude_patients": sorted(map(list, EXCLUDE_PATIENTS))}
    dataset_payload = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "code": {"identity": _git_identity(root), "files": code},
        "source_files": sorted(source_files, key=lambda item: item["path"]),
        "processed_npy_manifests": sorted(processed_npy_manifests,
                                           key=lambda item: item["dataset"]),
        "recordings": sorted(quality_records, key=lambda item: (item["dataset"], item["stem"])),
        "eligible_recordings": sorted(eligible_recordings,
                                      key=lambda item: (item["dataset"], item["stem"])),
        "eligibility_reports": sorted(eligibility_reports, key=lambda item: item["dataset"]),
        "quality_policy": {"version": ELIGIBILITY_POLICY_VERSION,
                           "states": ["VALIDATED", "REVIEW_REQUIRED", "NO_SEIZURE_CONFIRMED"],
                           "decisions": ["ELIGIBLE", "REJECTED", "REVIEW_REQUIRED"],
                           "shared_index_requires_training_and_event_eval": True},
        "label_policy": label_policy,
        "preprocessing": preprocessing_config,
        "feature_names": list(FEATURE_NAMES),
    }
    dataset_payload["dataset_fingerprint"] = fingerprint(dataset_payload)
    return dataset_payload


def build_cache_provenance(dataset_configs=None, *, augment=False, seed=42,
                           smoke=False, augmentation_config=None,
                           signal_normalization_policy=RAW_FEATURE_NORMALIZATION_POLICY,
                           fold_normalization=None, root: Path = ROOT) -> dict:
    dataset = build_dataset_provenance(dataset_configs, root)
    return compose_cache_provenance(
        dataset["dataset_fingerprint"], augment=augment, seed=seed, smoke=smoke,
        augmentation_config=augmentation_config, dataset_manifest=dataset,
        signal_normalization_policy=signal_normalization_policy,
        fold_normalization=fold_normalization)


def compose_cache_provenance(dataset_fingerprint: str, *, augment=False, seed=42,
                             smoke=False, augmentation_config=None,
                             dataset_manifest=None,
                             signal_normalization_policy=RAW_FEATURE_NORMALIZATION_POLICY,
                             fold_normalization=None) -> dict:
    if augmentation_config is None:
        augmentation_config = DEFAULT_AUGMENTATION_CONFIG if augment else {}
    if dataset_manifest is None:
        manifest_payload = {"source_identity": dataset_fingerprint,
                            "source_files": [], "recordings": [],
                            "eligible_recordings": [], "quality_policy": {},
                            "label_policy": {}, "preprocessing": {}, "feature_names": []}
        dataset_manifest = {**manifest_payload,
                            "dataset_fingerprint": fingerprint(manifest_payload)}
        dataset_fingerprint = dataset_manifest["dataset_fingerprint"]
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_manifest": dataset_manifest,
        "augmentation": {"enabled": bool(augment), "seed": int(seed),
                         "parameters": dict(augmentation_config or {})},
        "signal_normalization_policy": signal_normalization_policy,
        "fold_normalization": fold_normalization,
        "smoke": bool(smoke),
    }
    return {**payload, "fingerprint": fingerprint(payload)}


def validate_cache_info(info: dict, expected_provenance: dict, *,
                        augment=False, seed=42, smoke=False) -> None:
    """Fail closed before any cache array is opened."""
    required = {"cache_version", "cache_schema_version", "n_windows", "n_features",
                "feature_names", "augment", "seed", "smoke", "provenance"}
    missing = sorted(required - set(info or {}))
    if missing:
        raise ValueError(f"cache metadata is legacy or incomplete; missing {missing}")
    if info["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema version is unsupported")
    if info["cache_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError("cache implementation version is unsupported")
    if info["augment"] != bool(augment) or info["seed"] != int(seed) or info["smoke"] != bool(smoke):
        raise ValueError("cache build settings do not match the requested settings")
    provenance = info["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("cache provenance metadata is malformed")
    dataset_manifest = provenance.get("dataset_manifest")
    if not isinstance(dataset_manifest, dict):
        raise ValueError("cache provenance has no dataset manifest")
    manifest_payload = {key: value for key, value in dataset_manifest.items()
                        if key != "dataset_fingerprint"}
    if fingerprint(manifest_payload) != dataset_manifest.get("dataset_fingerprint"):
        raise ValueError("cache dataset manifest fingerprint is invalid")
    if dataset_manifest["dataset_fingerprint"] != provenance.get("dataset_fingerprint"):
        raise ValueError("cache dataset manifest and cache provenance disagree")
    provenance_payload = {key: value for key, value in provenance.items()
                          if key != "fingerprint"}
    if fingerprint(provenance_payload) != provenance.get("fingerprint"):
        raise ValueError("cache provenance fingerprint is invalid")
    if provenance.get("fingerprint") != expected_provenance.get("fingerprint"):
        raise ValueError("cache provenance does not match current code, sources, quality, or settings")


def validate_cache_directory(cache_dir: Path, expected_provenance=None, *,
                              dataset_configs=None, augment=False, seed=42, smoke=False,
                              root: Path = ROOT) -> dict:
    """Read and validate info.json only; callers may load arrays after success."""
    cache_dir = Path(cache_dir)
    info_path = cache_dir / "info.json"
    if not info_path.is_file():
        raise ValueError(f"cache metadata missing: {info_path}")
    info = json.loads(info_path.read_text())
    expected = expected_provenance or build_cache_provenance(
        dataset_configs, augment=augment, seed=seed, smoke=smoke, root=root)
    validate_cache_info(info, expected, augment=augment, seed=seed, smoke=smoke)
    validate_cache_file_signatures(info, cache_dir)
    from handcrafted_features import FEATURE_NAMES, N_FEATURES_PER_CHANNEL
    if info["feature_names"] != FEATURE_NAMES:
        raise ValueError("cache feature names do not match current extractor")
    if info["n_features"] != 6 * N_FEATURES_PER_CHANNEL:
        raise ValueError("cache feature count does not match current extractor")
    return info


def cache_file_signatures(cache_dir: Path) -> dict:
    """Return metadata-only signatures for the arrays bound to info.json."""
    cache_dir = Path(cache_dir)
    signatures = {}
    for name in ("features.npy", "meta.npz"):
        path = cache_dir / name
        if not path.is_file():
            raise ValueError(f"cache array is missing: {path}")
        stat = path.stat()
        signatures[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signatures


def validate_cache_file_signatures(info: dict, cache_dir: Path) -> None:
    """Reject missing/replaced cache arrays before any NumPy deserialization."""
    recorded = info.get("cache_files")
    expected_names = {"features.npy", "meta.npz"}
    if not isinstance(recorded, dict) or set(recorded) != expected_names:
        raise ValueError("cache metadata has no complete array file signatures")
    current = cache_file_signatures(cache_dir)
    if current != recorded:
        raise ValueError("cache array files do not match their provenance manifest")


def artifact_manifest_path(artifact_path: Path) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.name + ".provenance.json")


def write_artifact_manifest(artifact_path: Path, dataset_fingerprint: str,
                            artifact_kind: str, details=None) -> Path:
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise ValueError(f"cannot write provenance for missing artifact: {artifact_path}")
    producer_path = Path(sys._getframe(1).f_code.co_filename)
    path = artifact_manifest_path(artifact_path)
    payload = {"provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
               "dataset_fingerprint": dataset_fingerprint,
               "artifact_kind": artifact_kind,
               "artifact_name": artifact_path.name,
               "artifact_sha256": _file_sha256(artifact_path),
               "producer": _producer_identity(producer_path),
               "details": details or {}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def validate_artifact_manifest(artifact_path: Path, expected_dataset_fingerprint: str,
                               artifact_kind: str, expected_details=None,
                               expected_producer: str = None) -> dict:
    """Reject legacy or stale saved evaluation outputs before deserialization."""
    path = artifact_manifest_path(artifact_path)
    if not path.is_file():
        raise ValueError(f"saved evaluation artifact has no provenance manifest: {path}")
    manifest = json.loads(path.read_text())
    required = {"provenance_schema_version", "dataset_fingerprint", "artifact_kind",
                "artifact_name", "artifact_sha256", "producer"}
    if not required.issubset(manifest):
        raise ValueError("saved evaluation provenance is incomplete")
    if manifest["provenance_schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("saved evaluation provenance schema is unsupported")
    if manifest["artifact_kind"] != artifact_kind:
        raise ValueError("saved evaluation artifact kind does not match")
    artifact_path = Path(artifact_path)
    if manifest["artifact_name"] != artifact_path.name:
        raise ValueError("saved evaluation manifest belongs to a different artifact name")
    if not artifact_path.is_file():
        raise ValueError(f"saved evaluation artifact is missing: {artifact_path}")
    if _file_sha256(artifact_path) != manifest["artifact_sha256"]:
        raise ValueError("saved evaluation artifact content does not match its manifest")
    producer = manifest["producer"]
    if not isinstance(producer, dict) or not {"path", "sha256"}.issubset(producer):
        raise ValueError("saved evaluation producer identity is incomplete")
    recorded_producer = ROOT / producer["path"]
    if not recorded_producer.is_file() or _file_sha256(recorded_producer) != producer["sha256"]:
        raise ValueError("saved evaluation producer code is missing or has changed")
    expected_producer = expected_producer or ARTIFACT_PRODUCERS.get(artifact_kind)
    if expected_producer and producer["path"] != expected_producer:
        raise ValueError("saved evaluation manifest belongs to a different producer")
    if manifest["dataset_fingerprint"] != expected_dataset_fingerprint:
        raise ValueError("saved evaluation artifact does not match current data/code provenance")
    if expected_details:
        details = manifest.get("details", {})
        for key, value in expected_details.items():
            if details.get(key) != value:
                raise ValueError(f"saved evaluation artifact detail {key!r} does not match")
    return manifest


def current_dataset_fingerprint(dataset_configs=None, root: Path = ROOT) -> str:
    return build_dataset_provenance(dataset_configs, root)["dataset_fingerprint"]


def write_current_artifact_manifest(artifact_path: Path, artifact_kind: str,
                                    dataset_configs=None, details=None,
                                    root: Path = ROOT) -> Path:
    return write_artifact_manifest(
        artifact_path, current_dataset_fingerprint(dataset_configs, root),
        artifact_kind, details)


def validate_current_artifact(artifact_path: Path, artifact_kind: str,
                              dataset_configs=None, root: Path = ROOT,
                              expected_details=None, expected_producer=None) -> dict:
    return validate_artifact_manifest(
        artifact_path, current_dataset_fingerprint(dataset_configs, root), artifact_kind,
        expected_details=expected_details, expected_producer=expected_producer)
