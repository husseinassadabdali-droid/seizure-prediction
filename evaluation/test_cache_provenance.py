"""Small metadata-only tests for cache and saved-result provenance."""
import json
import os
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "features"))
from cache_provenance import (  # noqa: E402
    CACHE_SCHEMA_VERSION, DEFAULT_AUGMENTATION_CONFIG, SHARD_SCHEMA_VERSION,
    RAW_FEATURE_NORMALIZATION_POLICY, SIGNAL_QUALITY_POLICY,
    classify_recording_inventory, inspect_signal_array, inspect_signal_file,
    compose_cache_provenance, inventory_processed_files,
    artifact_manifest_path, cache_file_signatures,
    fold_normalization_manifest,
    preprocessing_config_fingerprint,
    processed_npy_manifest_path, shard_fingerprint, shard_manifest_path,
    validate_artifact_manifest, validate_cache_info, validate_processed_npy_manifest,
    validate_eligibility_report,
    validate_fold_normalization_manifest, validate_matching_normalization_manifests,
    validate_shard_manifest,
    write_artifact_manifest, write_processed_npy_manifest,
    write_shard_manifest,
)


def _save_synthetic_signal(path, array=None):
    if array is None:
        x = np.linspace(0, 40 * np.pi, 4096, dtype=np.float32)
        array = np.stack([np.sin(x + i * 0.2) * (i + 1) * 1e-5 for i in range(6)])
    np.save(path, np.asarray(array, dtype=np.float32))


def _classify(dataset, processed_dir, quality, *, source_records_by_file,
              excluded_patients=None, patient_id_getter=None):
    from preprocess import DATASETS
    processed_dir = Path(processed_dir)
    signals, processing = {}, {}
    target_channels = DATASETS.get(dataset, {}).get("target_channels", [])
    for item in inventory_processed_files(processed_dir):
        stem = Path(item["name"]).stem
        signals[stem] = inspect_signal_file(processed_dir / item["name"])
        source_records_by_file = dict(source_records_by_file or {})
        existing = list(source_records_by_file.get(stem, []))
        if existing:
            existing.append({"path": f"synthetic/{stem}.edf",
                             "fingerprint": f"synthetic-edf-{stem}"})
            source_records_by_file[stem] = existing
        npy_path = processed_dir / item["name"]
        processing[stem] = {
            "schema_version": 1, "status": "VERIFIED", "dataset": dataset,
            "source_size": 1024, "output_file": npy_path.name,
            "output_size": npy_path.stat().st_size,
            "output_sha256": hashlib.sha256(npy_path.read_bytes()).hexdigest(),
            "source_file": stem + ".edf",
            "source_fingerprint": f"synthetic-edf-{stem}",
            "preprocessing_config_fingerprint": preprocessing_config_fingerprint(dataset),
            "channel_order": target_channels, "sample_rate_hz": 250, "units": "V",
        }
    return classify_recording_inventory(
        dataset, processed_dir, quality, source_records_by_file=source_records_by_file,
        signal_quality_by_file=signals, processing_evidence_by_file=processing,
        excluded_patients=excluded_patients, patient_id_getter=patient_id_getter)


class CacheProvenanceTests(unittest.TestCase):
    def test_fusion_branch_validation_precedes_deserialization(self):
        sys.path.insert(0, str(ROOT / "models"))
        from fusion_model import _load_validated_branch

        with patch("cache_provenance.validate_artifact_manifest",
                   side_effect=ValueError("missing manifest")) as validate, \
                patch("torch.load") as load:
            with self.assertRaisesRegex(ValueError, "missing manifest"):
                _load_validated_branch("branch.pt", "tcn_branch", "fp", "cpu")
        validate.assert_called_once_with("branch.pt", "fp", "tcn_branch")
        load.assert_not_called()

    def test_fusion_branch_deserializes_only_after_validation(self):
        sys.path.insert(0, str(ROOT / "models"))
        from fusion_model import _load_validated_branch

        calls = []
        with patch("cache_provenance.validate_artifact_manifest",
                   side_effect=lambda *args: calls.append("validate")), \
                patch("torch.load", side_effect=lambda *args, **kwargs:
                      calls.append("load") or {"weights": "synthetic"}):
            result = _load_validated_branch("branch.pt", "tcn_branch", "fp", "cpu")
        self.assertEqual(result, {"weights": "synthetic"})
        self.assertEqual(calls, ["validate", "load"])

    def test_legacy_or_incomplete_cache_metadata_is_rejected(self):
        expected = compose_cache_provenance("dataset-a")
        with self.assertRaisesRegex(ValueError, "legacy or incomplete"):
            validate_cache_info({"cache_version": 1}, expected)

    def test_changed_quality_or_dataset_manifest_rejects_old_cache(self):
        old = compose_cache_provenance("dataset-with-validated-file")
        new = compose_cache_provenance("dataset-with-review-file")
        info = {"cache_version": CACHE_SCHEMA_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "n_windows": 1, "n_features": 1, "feature_names": ["x"],
                "augment": False, "seed": 42, "smoke": False,
                "provenance": old}
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_cache_info(info, new)

    def test_cache_input_normalization_policy_change_rejects_old_cache(self):
        old = compose_cache_provenance(
            "dataset-a", signal_normalization_policy=RAW_FEATURE_NORMALIZATION_POLICY)
        new = compose_cache_provenance(
            "dataset-a", signal_normalization_policy="per_patient_full_recording_legacy")
        info = {"cache_version": CACHE_SCHEMA_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "n_windows": 1, "n_features": 1, "feature_names": ["x"],
                "augment": False, "seed": 42, "smoke": False,
                "provenance": old}
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_cache_info(info, new)

    def test_augmentation_settings_change_shard_identity(self):
        common = ("record.npy", b"starts", b"mean", b"std", ["x"])
        original = shard_fingerprint(
            *common, DEFAULT_AUGMENTATION_CONFIG, True, 42, 50.0, "provenance")
        changed_config = dict(DEFAULT_AUGMENTATION_CONFIG, apply_prob=0.8)
        changed_probability = shard_fingerprint(
            *common, changed_config, True, 42, 50.0, "provenance")
        changed_policy = shard_fingerprint(
            *common, DEFAULT_AUGMENTATION_CONFIG, True, 42, 50.0, "new-provenance")
        self.assertNotEqual(original, changed_probability)
        self.assertNotEqual(original, changed_policy)

    def test_cache_settings_are_checked_before_array_loading(self):
        provenance = compose_cache_provenance("dataset-a", augment=True, seed=19)
        info = {"cache_version": CACHE_SCHEMA_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "n_windows": 1, "n_features": 1, "feature_names": ["x"],
                "augment": True, "seed": 19, "smoke": False,
                "provenance": provenance}
        validate_cache_info(info, provenance, augment=True, seed=19)
        with self.assertRaisesRegex(ValueError, "settings"):
            validate_cache_info(info, provenance, augment=False, seed=19)

    def test_saved_evaluation_artifact_requires_matching_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "results.bin"
            artifact.write_bytes(b"synthetic result A")
            write_artifact_manifest(artifact, "fingerprint-a", "synthetic-eval")
            accepted = validate_artifact_manifest(artifact, "fingerprint-a", "synthetic-eval")
            self.assertEqual(accepted["dataset_fingerprint"], "fingerprint-a")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_artifact_manifest(artifact, "fingerprint-b", "synthetic-eval")
            with self.assertRaisesRegex(ValueError, "kind"):
                validate_artifact_manifest(artifact, "fingerprint-a", "other-kind")
            with self.assertRaisesRegex(ValueError, "detail"):
                validate_artifact_manifest(artifact, "fingerprint-a", "synthetic-eval",
                                           expected_details={"seed": 7})
            with self.assertRaisesRegex(ValueError, "producer"):
                validate_artifact_manifest(artifact, "fingerprint-a", "synthetic-eval",
                                           expected_producer="features/cache_provenance.py")
            copied = Path(temp) / "different-result.bin"
            copied.write_bytes(artifact.read_bytes())
            artifact_manifest_path(copied).write_text(
                artifact_manifest_path(artifact).read_text())
            with self.assertRaisesRegex(ValueError, "different artifact name"):
                validate_artifact_manifest(copied, "fingerprint-a", "synthetic-eval")
            artifact.write_bytes(b"synthetic result B")
            with self.assertRaisesRegex(ValueError, "content does not match"):
                validate_artifact_manifest(artifact, "fingerprint-a", "synthetic-eval")

    def test_saved_result_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "legacy.bin"
            with self.assertRaisesRegex(ValueError, "no provenance manifest"):
                validate_artifact_manifest(artifact, "fingerprint", "synthetic-eval")

    def test_legacy_cache_is_rejected_before_numpy_load(self):
        sys.path.insert(0, str(ROOT / "training"))
        import build_feature_cache
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            (cache_dir / "info.json").write_text('{"cache_version": 1}')
            with patch("numpy.load") as load:
                with self.assertRaisesRegex(ValueError, "legacy or incomplete"):
                    build_feature_cache.load_cache(
                        cache_dir, expected_provenance=compose_cache_provenance("expected"))
                load.assert_not_called()

    def test_incompatible_cache_provenance_is_rejected_before_numpy_load(self):
        sys.path.insert(0, str(ROOT / "training"))
        import build_feature_cache
        expected = compose_cache_provenance("current-quality-policy")
        old = compose_cache_provenance("previous-quality-policy")
        info = {"cache_version": CACHE_SCHEMA_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "n_windows": 1, "n_features": 108, "feature_names": [],
                "augment": False, "seed": 42, "smoke": False,
                "provenance": old}
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            (cache_dir / "info.json").write_text(json.dumps(info))
            with patch("numpy.load") as load:
                with self.assertRaisesRegex(ValueError, "provenance"):
                    build_feature_cache.load_cache(
                        cache_dir, expected_provenance=expected)
                load.assert_not_called()

    def test_replaced_cache_array_is_rejected_before_numpy_load(self):
        sys.path.insert(0, str(ROOT / "training"))
        sys.path.insert(0, str(ROOT / "features"))
        import build_feature_cache
        from handcrafted_features import FEATURE_NAMES, N_FEATURES_PER_CHANNEL

        provenance = compose_cache_provenance("synthetic-dataset")
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp)
            (cache_dir / "features.npy").write_bytes(b"synthetic feature bytes")
            (cache_dir / "meta.npz").write_bytes(b"synthetic metadata bytes")
            info = {"cache_version": CACHE_SCHEMA_VERSION,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "n_windows": 1, "n_features": 6 * N_FEATURES_PER_CHANNEL,
                    "feature_names": FEATURE_NAMES, "augment": False,
                    "seed": 42, "smoke": False, "provenance": provenance,
                    "cache_files": cache_file_signatures(cache_dir)}
            (cache_dir / "info.json").write_text(json.dumps(info))
            # A copied/replaced array must be caught using metadata only.
            (cache_dir / "features.npy").write_bytes(b"different synthetic bytes")
            with patch("numpy.load") as load:
                with self.assertRaisesRegex(ValueError, "array files"):
                    build_feature_cache.load_cache(
                        cache_dir, expected_provenance=provenance)
                load.assert_not_called()

    def test_shard_manifest_missing_or_old_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp) / "shard.npy"
            shard.write_bytes(b"synthetic shard payload")
            kwargs = {"shard_key": "key-a", "build_fingerprint": "build-a",
                      "expected_shape": (1, 2), "expected_dtype": "float32"}
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_shard_manifest(shard, **kwargs)
            manifest_path = write_shard_manifest(shard, **kwargs)
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = SHARD_SCHEMA_VERSION - 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema"):
                validate_shard_manifest(shard, **kwargs)

    def test_shard_manifest_rejects_changed_build_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp) / "shard.npy"
            shard.write_bytes(b"synthetic shard payload")
            write_shard_manifest(shard, shard_key="key-a", build_fingerprint="build-a",
                                 expected_shape=(1, 2), expected_dtype="float32")
            with self.assertRaisesRegex(ValueError, "build settings"):
                validate_shard_manifest(shard, shard_key="key-a",
                                        build_fingerprint="build-b",
                                        expected_shape=(1, 2), expected_dtype="float32")

    def test_shard_content_change_with_same_size_and_mtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            shard = Path(temp) / "shard.npy"
            shard.write_bytes(b"synthetic shard A")
            kwargs = {"shard_key": "key-a", "build_fingerprint": "build-a",
                      "expected_shape": (1, 2), "expected_dtype": "float32"}
            write_shard_manifest(shard, **kwargs)
            old_stat = shard.stat()
            shard.write_bytes(b"synthetic shard B")
            os.utime(shard, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "content"):
                validate_shard_manifest(shard, **kwargs)

    def test_processed_npy_manifest_detects_same_size_content_change_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            source = processed_dir / "record.npy"
            _save_synthetic_signal(source)
            quality = {"record": {"status": "VALIDATED", "training_eligible": True,
                                  "event_eval_eligible": True, "reason": "synthetic source"}}
            report = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            manifest = write_processed_npy_manifest(
                processed_dir, eligibility_report=report,
                approved_report_fingerprint=report["eligibility_fingerprint"],
                approved_eligible_files=["record.npy"])
            self.assertEqual(manifest, processed_npy_manifest_path(processed_dir))
            old_stat = source.stat()
            changed = np.load(source)
            changed[0] = np.roll(changed[0], 1)
            _save_synthetic_signal(source, changed)
            os.utime(source, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            with patch("numpy.load", wraps=np.load) as load:
                with self.assertRaisesRegex(ValueError, "content"):
                    validate_processed_npy_manifest(processed_dir, eligibility_report=report)
                self.assertTrue(any(call.kwargs.get("mmap_mode") == "r"
                                    for call in load.call_args_list))

    def test_processed_npy_manifest_rejects_missing_and_changed_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            source = processed_dir / "record.npy"
            _save_synthetic_signal(source)
            quality = {"record": {"status": "VALIDATED", "training_eligible": True,
                                  "event_eval_eligible": True}}
            report = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_processed_npy_manifest(processed_dir, eligibility_report=report)
            write_processed_npy_manifest(
                processed_dir, eligibility_report=report,
                approved_report_fingerprint=report["eligibility_fingerprint"],
                approved_eligible_files=["record.npy"])
            _save_synthetic_signal(processed_dir / "second.npy")
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_processed_npy_manifest(processed_dir, eligibility_report=report)
            (processed_dir / "second.npy").unlink()
            source.unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                validate_processed_npy_manifest(processed_dir, eligibility_report=report)

    def test_inventory_presence_does_not_make_file_eligible_and_only_eligible_is_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            for name in ("good.npy", "bad.npy", "unknown.npy"):
                _save_synthetic_signal(processed_dir / name)
            quality = {
                "good": {"status": "VALIDATED", "training_eligible": True,
                         "event_eval_eligible": True, "reason": "explicitly trusted"},
                "bad": {"status": "NO_SEIZURE_CONFIRMED", "training_eligible": False,
                        "event_eval_eligible": False, "reason": "permission denied"},
                # unknown has no trusted source record and must remain review.
            }
            report = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={stem: [{"path": f"synthetic/{stem}.txt",
                                                "fingerprint": f"source-{stem}"}]
                                       for stem in ("good", "bad", "unknown")},
                patient_id_getter=lambda stem: "patient-a")
            self.assertEqual(inventory_processed_files(processed_dir), report["inventory"])
            self.assertTrue(all("sha256" not in item for item in report["inventory"]))
            self.assertEqual(len(report["inventory"]), 3)
            decisions = {row["name"]: row["decision"] for row in report["recordings"]}
            self.assertEqual(decisions, {"good.npy": "ELIGIBLE", "bad.npy": "REJECTED",
                                         "unknown.npy": "REVIEW_REQUIRED"})
            with patch("cache_provenance._file_sha256", wraps=__import__(
                    "cache_provenance")._file_sha256) as digest:
                write_processed_npy_manifest(
                    processed_dir, eligibility_report=report,
                    approved_report_fingerprint=report["eligibility_fingerprint"],
                    approved_eligible_files=["good.npy"])
                data_hashes = [Path(call.args[0]).name for call in digest.call_args_list
                               if Path(call.args[0]).parent == processed_dir]
                self.assertEqual(data_hashes, ["good.npy"])
            manifest = json.loads(processed_npy_manifest_path(processed_dir).read_text())
            self.assertEqual([row["name"] for row in manifest["files"]], ["good.npy"])
            self.assertEqual(sorted(row["name"] for row in manifest["inventory"]),
                             ["bad.npy", "good.npy", "unknown.npy"])
            report_rows = {row["name"]: row for row in manifest["eligibility_report"]["recordings"]}
            self.assertEqual(report_rows["bad.npy"]["decision"], "REJECTED")
            self.assertEqual(report_rows["bad.npy"]["reason"], "permission denied")
            self.assertEqual(report_rows["unknown.npy"]["decision"], "REVIEW_REQUIRED")
            self.assertIn("quality record is missing",
                          report_rows["unknown.npy"]["reason"])

    def test_quality_classification_is_fail_closed_for_unknown_or_incomplete_records(self):
        cases = {
            "missing-status": {"training_eligible": True, "event_eval_eligible": True},
            "unknown-status": {"status": "TRUSTED_BY_GUESS", "training_eligible": True,
                               "event_eval_eligible": True},
            "missing-permission": {"status": "VALIDATED", "training_eligible": True},
            "review-with-true-permissions": {"status": "REVIEW_REQUIRED",
                                              "training_eligible": True,
                                              "event_eval_eligible": True},
        }
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            source_map = {}
            for stem in cases:
                _save_synthetic_signal(processed_dir / f"{stem}.npy")
                source_map[stem] = [{"path": f"source/{stem}.txt",
                                     "fingerprint": f"source-{stem}"}]
            report = _classify(
                "synthetic", processed_dir, cases, source_records_by_file=source_map,
                patient_id_getter=lambda stem: "patient-a")
            self.assertTrue(all(row["decision"] == "REVIEW_REQUIRED"
                                for row in report["recordings"]))
            no_source = _classify(
                "synthetic", processed_dir,
                {stem: {"status": "VALIDATED", "training_eligible": True,
                        "event_eval_eligible": True} for stem in cases},
                source_records_by_file={}, patient_id_getter=lambda stem: "patient-a")
            self.assertTrue(all(row["decision"] == "REVIEW_REQUIRED"
                                for row in no_source["recordings"]))

            good_dir = Path(temp) / "good"
            good_dir.mkdir()
            _save_synthetic_signal(good_dir / "record.npy")
            good = _classify(
                "synthetic", good_dir,
                {"record": {"status": "VALIDATED", "training_eligible": True,
                            "event_eval_eligible": True}},
                source_records_by_file={"record": [{"path": "source/record.txt",
                                                     "fingerprint": "source-record"}]},
                patient_id_getter=lambda stem: "patient-a")
            self.assertEqual(good["recordings"][0]["decision"], "ELIGIBLE")
            validate_eligibility_report(good)
            forged = json.loads(json.dumps(good))
            forged["recordings"][0]["decision"] = "REVIEW_REQUIRED"
            payload = {key: value for key, value in forged.items()
                       if key != "eligibility_fingerprint"}
            forged["eligibility_fingerprint"] = __import__(
                "cache_provenance").fingerprint(payload)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                validate_eligibility_report(forged)

    def test_eligibility_change_or_policy_change_rejects_old_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            _save_synthetic_signal(processed_dir / "record.npy")
            allowed = {"record": {"status": "VALIDATED", "training_eligible": True,
                                  "event_eval_eligible": True}}
            original = _classify(
                "synthetic", processed_dir, allowed,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            write_processed_npy_manifest(
                processed_dir, eligibility_report=original,
                approved_report_fingerprint=original["eligibility_fingerprint"],
                approved_eligible_files=["record.npy"])
            denied = {"record": {"status": "REVIEW_REQUIRED", "training_eligible": False,
                                 "event_eval_eligible": False, "reason": "source conflict"}}
            changed = _classify(
                "synthetic", processed_dir, denied,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            with self.assertRaisesRegex(ValueError, "eligibility"):
                validate_processed_npy_manifest(processed_dir, eligibility_report=changed)
            with patch("cache_provenance.ELIGIBILITY_POLICY_VERSION", "synthetic-policy-v2"):
                with self.assertRaisesRegex(ValueError, "policy"):
                    validate_processed_npy_manifest(processed_dir,
                                                    eligibility_report=original)
                changed_policy = _classify(
                    "synthetic", processed_dir, allowed,
                    source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                         "fingerprint": "source-fp"}]},
                    patient_id_getter=lambda stem: "patient-a")
                with self.assertRaisesRegex(ValueError, "eligibility"):
                    validate_processed_npy_manifest(processed_dir,
                                                    eligibility_report=changed_policy)

    def test_eligibility_requires_explicit_complete_approval_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            _save_synthetic_signal(processed_dir / "record.npy")
            quality = {"record": {"status": "VALIDATED", "training_eligible": True,
                                  "event_eval_eligible": True}}
            report = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            changed_quality = {"record": {"status": "REVIEW_REQUIRED",
                                          "training_eligible": False,
                                          "event_eval_eligible": False,
                                          "reason": "changed source decision"}}
            changed_report = _classify(
                "synthetic", processed_dir, changed_quality,
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            with self.assertRaisesRegex(ValueError, "approval"):
                write_processed_npy_manifest(
                    processed_dir, eligibility_report=changed_report,
                    approved_report_fingerprint=report["eligibility_fingerprint"],
                    approved_eligible_files=[])
            with self.assertRaisesRegex(ValueError, "explicit eligibility approval"):
                write_processed_npy_manifest(
                    processed_dir, eligibility_report=report,
                    approved_report_fingerprint="", approved_eligible_files=["record.npy"])
            with self.assertRaisesRegex(ValueError, "exactly match"):
                write_processed_npy_manifest(
                    processed_dir, eligibility_report=report,
                    approved_report_fingerprint=report["eligibility_fingerprint"],
                    approved_eligible_files=[])
            path = write_processed_npy_manifest(
                processed_dir, eligibility_report=report,
                approved_report_fingerprint=report["eligibility_fingerprint"],
                approved_eligible_files=["record.npy"])
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_processed_npy_manifest(
                    processed_dir, eligibility_report=report,
                    approved_report_fingerprint=report["eligibility_fingerprint"],
                    approved_eligible_files=["record.npy"])
            self.assertEqual(path.read_bytes(), before)
            replacement_report = _classify(
                "synthetic", processed_dir,
                {"record": {"status": "VALIDATED", "training_eligible": True,
                            "event_eval_eligible": True, "reason": "reviewed synthetic source"}},
                source_records_by_file={"record": [{"path": "synthetic/source.txt",
                                                     "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            write_processed_npy_manifest(
                processed_dir, eligibility_report=replacement_report,
                approved_report_fingerprint=replacement_report["eligibility_fingerprint"],
                approved_eligible_files=["record.npy"], overwrite=True)
            self.assertNotEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(path.read_text())["eligibility_report"]["recordings"][0]["reason"],
                             "reviewed synthetic source")

    def test_inventory_addition_or_removal_changes_eligibility_report(self):
        with tempfile.TemporaryDirectory() as temp:
            processed_dir = Path(temp)
            first = processed_dir / "one.npy"
            _save_synthetic_signal(first)
            quality = {"one": {"status": "VALIDATED", "training_eligible": True,
                               "event_eval_eligible": True}}
            report_a = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"one": [{"path": "synthetic/source.txt",
                                                  "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            second = processed_dir / "two.npy"
            _save_synthetic_signal(second)
            report_b = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"one": [{"path": "synthetic/source.txt",
                                                  "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            self.assertNotEqual(report_a["eligibility_fingerprint"],
                                report_b["eligibility_fingerprint"])
            second.unlink()
            report_c = _classify(
                "synthetic", processed_dir, quality,
                source_records_by_file={"one": [{"path": "synthetic/source.txt",
                                                  "fingerprint": "source-fp"}]},
                patient_id_getter=lambda stem: "patient-a")
            self.assertEqual(report_a["eligibility_fingerprint"],
                             report_c["eligibility_fingerprint"])

    def test_loso_scaler_ignores_heldout_features_and_binds_fit_patients(self):
        sys.path.insert(0, str(ROOT / "evaluation"))
        from loso_cache_analysis import fit_fold_feature_scaler

        train_x = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        X_a = np.concatenate([train_x, [[1000.0, -1000.0]]], axis=0)
        X_b = np.concatenate([train_x, [[-1e6, 1e6]]], axis=0)
        center_a, scale_a, manifest_a = fit_fold_feature_scaler(
            X_a, [0, 1], ["train-patient"])
        center_b, scale_b, manifest_b = fit_fold_feature_scaler(
            X_b, [0, 1], ["train-patient"])
        np.testing.assert_array_equal(center_a, center_b)
        np.testing.assert_array_equal(scale_a, scale_b)
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(center_a.tolist(), [1.0, 3.0])

    def test_fold_training_list_or_stats_change_rejects_fold_cache_context(self):
        recorded = fold_normalization_manifest(["patient-a"], [1.0, 2.0], [3.0, 4.0])
        changed_patients = fold_normalization_manifest(
            ["patient-a", "patient-b"], [1.0, 2.0], [3.0, 4.0])
        changed_stats = fold_normalization_manifest(
            ["patient-a"], [1.0, 2.5], [3.0, 4.0])
        for expected in (changed_patients, changed_stats):
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_fold_normalization_manifest(recorded, expected)

        old_cache = compose_cache_provenance(
            "synthetic-dataset", fold_normalization=recorded)
        new_cache = compose_cache_provenance(
            "synthetic-dataset", fold_normalization=changed_stats)
        info = {"cache_version": CACHE_SCHEMA_VERSION,
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "n_windows": 1, "n_features": 1, "feature_names": ["x"],
                "augment": False, "seed": 42, "smoke": False,
                "provenance": old_cache}
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_cache_info(info, new_cache)

    def test_fusion_branches_must_share_frozen_normalization(self):
        mlp = {"details": {"normalization_fingerprint": "norm-a"}}
        tcn = {"details": {"normalization_fingerprint": "norm-a"}}
        self.assertEqual(validate_matching_normalization_manifests(mlp, tcn), "norm-a")
        tcn_other = {"details": {"normalization_fingerprint": "norm-b"}}
        with self.assertRaisesRegex(ValueError, "different normalization"):
            validate_matching_normalization_manifests(mlp, tcn_other)

    def test_signal_screen_detects_all_zero_and_zero_channel_without_mutation(self):
        all_zero = np.zeros((6, 64), dtype=np.float32)
        all_zero_before = all_zero.copy()
        all_zero_result = inspect_signal_array(all_zero)
        self.assertEqual(all_zero_result["status"], "REVIEW_REQUIRED")
        self.assertTrue(all(channel["clipping_status"] == "not_flagged_not_proven_absent"
                            for channel in all_zero_result["channels"]))
        np.testing.assert_array_equal(all_zero, all_zero_before)

        x = np.linspace(0, 8 * np.pi, 512, dtype=np.float32)
        signal = np.stack([np.sin(x + i * 0.1) for i in range(6)]).astype(np.float32)
        signal[2] = 0
        signal.setflags(write=False)
        result = inspect_signal_array(signal)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(any("channel 2" in issue and "zero-variance" in issue
                            for issue in result["issues"]))
        self.assertEqual(float(signal[2].sum()), 0.0)

    def test_signal_screen_detects_near_flat_nan_and_inf(self):
        x = np.linspace(0, 12 * np.pi, 2048, dtype=np.float32)
        signal = np.stack([np.sin(x + i * 0.2) for i in range(6)]).astype(np.float32)
        signal[4] *= 1e-6
        result = inspect_signal_array(signal)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(any("near-flat" in issue for issue in result["issues"]))

        signal[0, 0] = np.nan
        signal[1, 1] = np.inf
        result = inspect_signal_array(signal)
        self.assertTrue(any("NaN or Inf" in issue for issue in result["issues"]))
        self.assertEqual(result["status"], "REVIEW_REQUIRED")

    def test_signal_screen_checks_channel_count_and_dtype(self):
        wrong_channels = np.ones((5, 64), dtype=np.float32)
        result = inspect_signal_array(wrong_channels)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(any("channel count" in issue for issue in result["issues"]))
        wrong_dtype = np.stack([np.sin(np.arange(64) + i) for i in range(6)])
        result = inspect_signal_array(wrong_dtype)
        self.assertTrue(any("dtype" in issue for issue in result["issues"]))

    def test_channel_match_requires_exact_unique_labels_without_substitution(self):
        sys.path.insert(0, str(ROOT / "preprocessing"))
        from preprocess import match_target_channels

        indices, missing, ambiguous = match_target_channels(
            ["FP1-F7", "FP1-F7", "F7-T7"], ["FP1-F7", "FP2-F8", "F7-T7"])
        self.assertEqual(indices, [None, None, 2])
        self.assertEqual(missing, ["FP2-F8"])
        self.assertEqual(ambiguous, ["FP1-F7"])

    def test_unresolved_or_unverified_recording_cannot_become_eligible(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _save_synthetic_signal(directory / "record.npy")
            quality = {"record": {"status": "VALIDATED", "training_eligible": True,
                                  "event_eval_eligible": True}}
            source = {"record": [{"path": "synthetic/record.edf",
                                  "fingerprint": "edf-fp"}]}
            report = classify_recording_inventory(
                "synthetic", directory, quality, source_records_by_file=source,
                patient_id_getter=lambda _stem: "patient-a")
            self.assertEqual(report["recordings"][0]["decision"], "REVIEW_REQUIRED")
            self.assertIn("processing evidence", report["recordings"][0]["reason"])

            review_quality = {"record": {"status": "REVIEW_REQUIRED",
                                          "training_eligible": False,
                                          "event_eval_eligible": False}}
            review = _classify("synthetic", directory, review_quality,
                               source_records_by_file=source,
                               patient_id_getter=lambda _stem: "patient-a")
            self.assertEqual(review["recordings"][0]["decision"], "REVIEW_REQUIRED")

    def test_processing_channel_order_mismatch_and_siena_reviews_stay_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _save_synthetic_signal(directory / "record.npy")
            signal = inspect_signal_file(directory / "record.npy")
            source = {"record": [{"path": "synthetic/record.edf",
                                  "fingerprint": "edf-fp"}]}
            npy_path = directory / "record.npy"
            evidence = {"record": {
                "schema_version": 1, "status": "VERIFIED", "dataset": "chbmit",
                "source_size": 1024, "output_file": npy_path.name,
                "output_size": npy_path.stat().st_size,
                "output_sha256": hashlib.sha256(npy_path.read_bytes()).hexdigest(),
                "source_file": "record.edf", "source_fingerprint": "edf-fp",
                "preprocessing_config_fingerprint": preprocessing_config_fingerprint("chbmit"),
                "channel_order": ["wrong"] * 6, "sample_rate_hz": 250, "units": "V"}}
            report = classify_recording_inventory(
                "chbmit", directory,
                {"record": {"status": "VALIDATED", "training_eligible": True,
                            "event_eval_eligible": True}},
                source_records_by_file=source, signal_quality_by_file={"record": signal},
                processing_evidence_by_file=evidence,
                patient_id_getter=lambda _stem: "patient-a")
            self.assertEqual(report["recordings"][0]["decision"], "REVIEW_REQUIRED")
            self.assertIn("channel order", report["recordings"][0]["reason"])

            for stem in ("PN00-3", "PN05-3", "PN10-2", "PN10-3", "PN14-3"):
                _save_synthetic_signal(directory / f"{stem}.npy")
            siena_quality = {
                stem: {"status": "REVIEW_REQUIRED", "training_eligible": False,
                       "event_eval_eligible": False, "reason": "annotation review pending"}
                for stem in ("PN00-3", "PN05-3", "PN10-2", "PN10-3", "PN14-3")}
            siena_sources = {
                stem: [{"path": f"synthetic/{stem}.edf", "fingerprint": f"edf-{stem}"}]
                for stem in siena_quality}
            siena = _classify("siena", directory, siena_quality,
                              source_records_by_file=siena_sources,
                              patient_id_getter=lambda _stem: "patient-a")
            rows = {row["stem"]: row for row in siena["recordings"]}
            self.assertTrue(all(rows[stem]["decision"] == "REVIEW_REQUIRED"
                                for stem in siena_quality))


if __name__ == "__main__":
    unittest.main()
