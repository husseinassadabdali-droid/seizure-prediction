"""Focused, synthetic checks for SIENA annotation parsing and EDF validation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from dataset import (  # noqa: E402
    INTERICTAL, PREICTAL, ICTAL, VALIDATED, REVIEW_REQUIRED, NO_SEIZURE_CONFIRMED,
    _chbmit_quality_from_summary, _quality_for_recording,
    compare_siena_event_counts,
    SeizureDataset, apply_channel_normalization,
    build_seizure_index, compute_patient_stats_from_arrays,
    compute_training_channel_stats_from_arrays, filter_recordings_by_quality,
    label_windows_for_file, parse_siena_seizure_list, quality_aware_window_entries,
    _siena_time_candidates,
    windows_for_quality,
)


class SienaParserTests(unittest.TestCase):
    def parse(self, text, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "Seizures-list-test.txt"
        path.write_text(text)
        diagnostics = []
        result = parse_siena_seizure_list(path, diagnostics=diagnostics, **kwargs)
        return result, diagnostics

    def parse_times(self, start, end, duration=5000):
        text = ("Seizure n 1\nFile name: sample.edf\nRegistration start time: 10.00.00\n"
                f"Seizure start time: {start}\nSeizure end time: {end}\n")
        return self.parse(text, available_files={"sample": Path("sample.edf")},
                          edf_durations={"sample": duration})

    def test_single_valid_time_is_linked(self):
        result, diagnostics = self.parse_times("10.03.20", "10.03.59")
        self.assertEqual(result["sample"], [(200, 239)])
        self.assertEqual(diagnostics[0]["time_resolution"], "resolved")

    def test_compact_and_spaced_event_headings_are_both_parsed(self):
        text = """PN03
Seizure n1
File name: first.edf
Registration start time: 10.00.00
Seizure start time: 10.01.00
Seizure end time: 10.01.10
Seizure n 2
File name: second.edf
Registration start time: 11.00.00
Seizure start time: 11.01.00
Seizure end time: 11.01.10
"""
        result, diagnostics = self.parse(
            text, available_files={"first": Path("first"), "second": Path("second")},
            edf_durations={"first": 100, "second": 100})
        self.assertEqual(result, {"first": [(60, 70)], "second": [(60, 70)]})
        self.assertEqual([d["status"] for d in diagnostics], ["linked", "linked"])

    def test_event_count_audit_reports_match_and_keeps_mismatch_unresolved(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "subject_info.csv").write_text(
            "patient_id,number_seizures\nPN01,2\nPN02,2\n")
        (root / "Seizures-list-PN01.txt").write_text(
            "Seizure n1\nSeizure n 2\n")
        (root / "Seizures-list-PN02.txt").write_text("Seizure n 1\n")
        audit = compare_siena_event_counts(root)
        self.assertEqual(audit["PN01"]["status"], "match")
        self.assertEqual(audit["PN02"]["status"], "review")
        self.assertEqual(audit["PN02"]["subject_info_count"], 2)
        self.assertEqual(audit["PN02"]["annotation_heading_count"], 1)
        self.assertIn("neither source was selected", audit["PN02"]["reason"])

    def test_source_count_mismatch_quarantines_patient_without_choosing_a_source(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "RECORDS").write_text("PN02-1.edf\n")
        (root / "subject_info.csv").write_text(
            "patient_id,number_seizures\nPN02,2\n")
        (root / "Seizures-list-PN02.txt").write_text(
            "Seizure n1\nFile name: PN02-1.edf\n"
            "Registration start time: 10.00.00\n"
            "Seizure start time: 10.01.00\nSeizure end time: 10.01.10\n")
        _, qualities = build_seizure_index("siena", root, return_quality=True)
        self.assertEqual(qualities["PN02-1"]["status"], REVIEW_REQUIRED)
        self.assertIn("event count unresolved", qualities["PN02-1"]["reason"])
        self.assertNotEqual(qualities["PN02-1"]["status"], NO_SEIZURE_CONFIRMED)

    def test_equivalent_time_formats_are_retained_and_can_be_collapsed(self):
        result, diagnostics = self.parse_times("10.03.20; 10:03:20", "10.03.59")
        self.assertEqual(result["sample"], [(200, 239)])
        item = diagnostics[0]
        self.assertEqual(item["time_resolution"], "equivalent")
        self.assertEqual(item["raw_time_fields"]["seizure_start"], "10.03.20; 10:03:20")
        self.assertEqual(len(item["time_candidates"]["seizure_start"]), 2)
        self.assertIn("same event", item["reason"])

    def test_distinct_times_without_label_change_still_require_review(self):
        result, diagnostics = self.parse_times("10.03.20; 10.03.21", "10.03.59")
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["status"], "review")
        self.assertFalse(diagnostics[0]["window_labels_differ"])
        self.assertEqual(len(diagnostics[0]["candidate_intervals"]), 2)

    def test_distinct_end_times_that_change_window_labels_require_review(self):
        result, diagnostics = self.parse_times("10.03.20", "10.03.59; 10.04.02")
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["status"], "review")
        self.assertTrue(diagnostics[0]["window_labels_differ"])

    def test_clinical_and_electrical_onsets_remain_distinct_semantics(self):
        result, diagnostics = self.parse_times(
            "10.03.20 (CLINICAL ONSET); 10.03.25 (ELECTRICAL ONSET)", "10.03.59")
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["time_resolution"], "review")
        candidates = diagnostics[0]["time_candidates"]["seizure_start"]
        self.assertEqual([c["semantic"] for c in candidates],
                         ["clinical_onset", "electrical_onset"])

    def test_candidate_outside_edf_is_rejected_with_diagnostic(self):
        result, diagnostics = self.parse_times("10.03.20", "10.20.00", duration=500)
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["status"], "review")
        self.assertIn("outside EDF", diagnostics[0]["reason"])
        self.assertEqual(diagnostics[0]["candidate_intervals"], [(200, 1200)])

    def test_unparseable_time_keeps_raw_field_and_review_reason(self):
        result, diagnostics = self.parse_times("onset not recorded", "10.03.59")
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["status"], "review")
        self.assertEqual(diagnostics[0]["raw_time_fields"]["seizure_start"], "onset not recorded")
        self.assertEqual(diagnostics[0]["time_candidates"]["seizure_start"], [])
        self.assertIn("unparseable", diagnostics[0]["reason"])

    def test_candidate_retains_context_span_and_interpretive_correction(self):
        candidates = _siena_time_candidates("1 6.49.25", "registration_start")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["raw"], "1 6.49.25")
        self.assertEqual(candidates[0]["normalized"], "16:49:25")
        self.assertEqual(candidates[0]["correction"], "16.49.25")
        self.assertIn("1 6.49.25", candidates[0]["context"])
        self.assertEqual(candidates[0]["span"], (0, len("1 6.49.25")))

    def test_valid_and_out_of_range_candidates_are_both_preserved_and_rejected(self):
        result, diagnostics = self.parse_times("10.03.20; 99.99.99", "10.03.59")
        self.assertEqual(result, {})
        item = diagnostics[0]
        candidates = item["time_candidates"]["seizure_start"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[0]["valid"])
        self.assertFalse(candidates[1]["valid"])
        self.assertEqual(candidates[1]["raw"], "99.99.99")
        self.assertEqual(candidates[1]["seconds_since_midnight"], None)
        self.assertIn("hour outside", candidates[1]["invalid_reason"])
        self.assertEqual(len(item["invalid_time_candidates"]), 1)
        self.assertFalse(item["combination_coverage_complete"])
        self.assertEqual(item["status"], "review")

    def test_valid_candidate_with_unparseable_clock_candidate_is_reviewed(self):
        result, diagnostics = self.parse_times("10.03.20; 10.xx.30", "10.03.59")
        self.assertEqual(result, {})
        item = diagnostics[0]
        candidates = item["time_candidates"]["seizure_start"]
        self.assertEqual([candidate["raw"] for candidate in candidates], ["10.03.20", "10.xx.30"])
        self.assertFalse(candidates[1]["valid"])
        self.assertEqual(candidates[1]["invalid_reason"], "time components are not numeric")
        self.assertEqual(item["status"], "review")

    def test_long_clock_component_is_preserved_as_one_invalid_candidate(self):
        result, diagnostics = self.parse_times("10.03.20; 10.1000.30", "10.03.59")
        self.assertEqual(result, {})
        candidates = diagnostics[0]["time_candidates"]["seizure_start"]
        self.assertEqual([c["raw"] for c in candidates], ["10.03.20", "10.1000.30"])
        self.assertFalse(candidates[1]["valid"])
        self.assertEqual(candidates[1]["invalid_reason"],
                         "minute and second must contain two digits")
        self.assertEqual(diagnostics[0]["status"], "review")

    def test_same_numeric_clinical_and_electrical_onsets_are_reviewed(self):
        result, diagnostics = self.parse_times(
            "10.03.20 (clinical onset); 10.03.20 (electrical onset)", "10.03.59")
        self.assertEqual(result, {})
        item = diagnostics[0]
        self.assertFalse(item["window_labels_differ"])
        self.assertEqual([c["semantic"] for c in item["time_candidates"]["seizure_start"]],
                         ["clinical_onset", "electrical_onset"])
        self.assertEqual(item["status"], "review")

    def test_slash_or_connectors_do_not_merge_onset_semantics(self):
        for connector in ("/", "or"):
            with self.subTest(connector=connector):
                result, diagnostics = self.parse_times(
                    f"10.03.20 clinical onset {connector} 10.03.20 electrical onset", "10.03.59")
                self.assertEqual(result, {})
                item = diagnostics[0]
                self.assertEqual([c["semantic"] for c in item["time_candidates"]["seizure_start"]],
                                 ["clinical_onset", "electrical_onset"])
                self.assertEqual(item["status"], "review")

    def test_mixed_onset_descriptions_without_delimiter_are_not_smeared(self):
        result, diagnostics = self.parse_times(
            "10.03.20 clinical onset 10.03.20 electrical onset", "10.03.59")
        self.assertEqual(result, {})
        self.assertTrue(all(c["semantic_conflict"]
                            for c in diagnostics[0]["time_candidates"]["seizure_start"]))
        self.assertEqual(diagnostics[0]["status"], "review")

    def test_all_start_end_combinations_are_retained_when_one_exceeds_edf(self):
        result, diagnostics = self.parse_times(
            "10.03.20; 10.04.00", "10.04.10; 10.05.01", duration=300)
        self.assertEqual(result, {})
        item = diagnostics[0]
        self.assertEqual(len(item["candidate_combinations"]), 4)
        self.assertEqual(len(item["candidate_intervals"]), 4)
        self.assertTrue(any(not combo["within_edf"] for combo in item["candidate_combinations"]))
        self.assertTrue(any(not combo["valid_interval"] for combo in item["candidate_combinations"]))
        self.assertEqual(item["expected_candidate_combinations"], 4)
        self.assertEqual(item["recorded_candidate_combinations"], 4)
        self.assertTrue(item["combination_coverage_complete"])
        self.assertEqual(item["status"], "review")

    def test_zero_length_combination_is_recorded_but_never_accepted(self):
        result, diagnostics = self.parse_times("10.03.20", "10.03.20")
        self.assertEqual(result, {})
        item = diagnostics[0]
        self.assertEqual(item["expected_candidate_combinations"], 1)
        self.assertEqual(item["recorded_candidate_combinations"], 1)
        self.assertEqual(len(item["candidate_combinations"]), 1)
        self.assertFalse(item["candidate_combinations"][0]["valid_interval"])
        self.assertTrue(item["combination_coverage_complete"])
        self.assertEqual(item["status"], "review")

    def test_missing_edf_duration_is_unknown_and_prevents_event_acceptance(self):
        text = ("Seizure n 1\nFile name: sample.edf\nRegistration start time: 10.00.00\n"
                "Seizure start time: 10.03.20\nSeizure end time: 10.03.59\n")
        result, diagnostics = self.parse(
            text, available_files={"sample": Path("sample.edf")})
        self.assertEqual(result, {})
        item = diagnostics[0]
        self.assertEqual(item["status"], "review")
        self.assertIn("EDF duration unavailable", item["reason"])
        combo = item["candidate_combinations"][0]
        self.assertIsNone(combo["within_edf"])
        self.assertEqual(combo["boundary_status"], "unknown")
        self.assertIsNone(combo["valid_interval"])

    def test_rejected_event_does_not_become_confirmed_negative_or_trusted_window(self):
        result, diagnostics = self.parse_times("10.03.20; 10.03.21", "10.03.59")
        item = diagnostics[0]
        quality = _quality_for_recording(item["matched_file"], result,
                                         {item["matched_file"]: item["reason"]})
        self.assertEqual(quality["status"], REVIEW_REQUIRED)
        self.assertNotEqual(quality["status"], NO_SEIZURE_CONFIRMED)
        self.assertEqual(quality_aware_window_entries(
            item["matched_file"], result.get(item["matched_file"], []), 5000 * 250, quality), [])

    def test_review_event_cannot_enter_trusted_window_index(self):
        result, diagnostics = self.parse_times("10.03.20; 10.03.21", "10.03.59")
        self.assertEqual(result, {})
        item = diagnostics[0]
        quality = _quality_for_recording(item["matched_file"], result,
                                         {item["matched_file"]: item["reason"]})
        self.assertEqual(quality["status"], REVIEW_REQUIRED)
        self.assertEqual(quality_aware_window_entries(
            item["matched_file"], result.get(item["matched_file"], []), 5000 * 250, quality), [])

    def test_pn01_alternate_fields_and_explicit_record_alias(self):
        text = """File name: PN01.edf
Registration start time: 19.00.44
Seizure n 1
Start time: 21:51:02
End time: 21.51.56
Seizure n 2
Start time: 07.53.17
End time: 07.54.31
"""
        result, diagnostics = self.parse(
            text, available_files={"PN01-1": Path("PN01-1.edf")},
            edf_durations={"PN01-1": 48557})
        self.assertEqual(len(result["PN01-1"]), 2)
        self.assertEqual([x["status"] for x in diagnostics], ["linked", "linked"])

    def test_pn10_malformed_registration_is_corrected_and_preserved(self):
        text = """Seizure n 7
File name: PN10-7.8.9.edf
Registration start time:1 6.49.25
Seizure start time: 17.35.13
Seizure end time: 17.36.01
"""
        result, diagnostics = self.parse(
            text, available_files={"PN10-7.8.9": Path("record.edf")},
            edf_durations={"PN10-7.8.9": 14955},
            edf_starts={"PN10-7.8.9": ("01.01.17", "16.49.25")})
        self.assertEqual(result["PN10-7.8.9"], [(2748, 2796)])
        self.assertEqual(diagnostics[0]["raw_registration_start"], "1 6.49.25")
        self.assertEqual(diagnostics[0]["registration_correction"], "16.49.25")

    def test_same_file_inherits_registration_for_second_seizure(self):
        text = """Seizure n 1
File name: PN12-1.2.edf
Registration start time: 15.51.31
Seizure start time: 16.13.23
Seizure end time: 16.14.26
Seizure n 2
File name: PN12-1.2.edf
Seizure start time: 18.31.01
Seizure end time: 18.32.09
"""
        result, diagnostics = self.parse(text, available_files={"PN12-1.2": Path("x")},
                                         edf_durations={"PN12-1.2": 9774})
        self.assertEqual(result["PN12-1.2"], [(1312, 1375), (9570, 9638)])
        self.assertEqual(len(diagnostics), 2)

    def test_new_file_does_not_inherit_registration(self):
        text = """Seizure n 1
File name: first.edf
Registration start time: 10.00.00
Seizure start time: 10.01.00
Seizure end time: 10.01.10
Seizure n 2
File name: second.edf
Seizure start time: 10.02.00
Seizure end time: 10.02.10
"""
        result, diagnostics = self.parse(
            text, available_files={"first": Path("1"), "second": Path("2")},
            edf_durations={"first": 100, "second": 100})
        self.assertEqual(result, {"first": [(60, 70)]})
        self.assertEqual(diagnostics[1]["reason"], "missing file or registration time")

    def test_missing_registration_and_unmatched_filename_are_reported(self):
        missing, missing_diag = self.parse(
            "Seizure n 1\nFile name: a.edf\nSeizure start time: 10.01.00\nSeizure end time: 10.01.10\n",
            available_files={"a": Path("a")})
        unmatched, unmatched_diag = self.parse(
            "Seizure n 1\nFile name: absent.edf\nRegistration start time: 10.00.00\n"
            "Seizure start time: 10.01.00\nSeizure end time: 10.01.10\n",
            available_files={"present": Path("p")})
        self.assertEqual(missing, {})
        self.assertEqual(missing_diag[0]["status"], "review")
        self.assertEqual(unmatched, {})
        self.assertEqual(unmatched_diag[0]["status"], "unlinked")

    def test_header_registration_mismatch_and_end_beyond_edf_are_reviewed(self):
        text = "Seizure n 1\nFile name: a.edf\nRegistration start time: 10.00.05\n"
        text += "Seizure start time: 10.00.10\nSeizure end time: 10.01.00\n"
        result, diagnostics = self.parse(
            text, available_files={"a": Path("a")}, edf_durations={"a": 50},
            edf_starts={"a": ("01.01.17", "10.00.00")})
        self.assertEqual(result, {})
        self.assertEqual(diagnostics[0]["status"], "review")
        self.assertIn("differs from EDF header", diagnostics[0]["reason"])
        self.assertIn("outside EDF duration", diagnostics[0]["reason"])

    def test_unresolved_event_never_enters_trusted_window_index(self):
        quality = _quality_for_recording(
            "PN00-3", {}, {"PN00-3": "event outside EDF duration"})
        self.assertEqual(quality["status"], REVIEW_REQUIRED)
        self.assertEqual(
            quality_aware_window_entries("PN00-3", [(765, 4425)], 2509 * 250, quality), [])

    def test_untrusted_region_is_not_filled_as_interictal(self):
        review = _quality_for_recording("PN05-3", {}, {"PN05-3": "registration mismatch"})
        windows = quality_aware_window_entries("PN05-3", [], 600 * 250, review)
        self.assertEqual(windows, [])
        self.assertNotIn(INTERICTAL, [label for _, label in windows])

    def test_missing_event_is_not_a_confirmed_negative(self):
        unknown = _quality_for_recording("PN00-2", {})
        self.assertEqual(unknown["status"], REVIEW_REQUIRED)
        confirmed = _quality_for_recording("PN00-2", {}, confirmed_no_seizure_files={"PN00-2"})
        self.assertEqual(confirmed["status"], NO_SEIZURE_CONFIRMED)
        self.assertTrue(confirmed["training_eligible"])
        self.assertTrue(confirmed["event_eval_eligible"])

    def test_training_and_event_evaluation_permissions_are_independent(self):
        quality = {"training_eligible": True, "event_eval_eligible": False}
        seizures = [(4000, 4060)]
        train_windows = windows_for_quality(seizures, 25000 * 250, quality, purpose="training")
        event_windows = windows_for_quality(seizures, 25000 * 250, quality, purpose="event_eval")
        self.assertEqual(train_windows, label_windows_for_file(seizures, 25000 * 250))
        self.assertEqual(event_windows, [])

    def test_quality_filter_excludes_review_recording_from_normalization_arrays(self):
        memory_files = [
            ("validated", np.array([[1.0, 3.0]], dtype=np.float32)),
            ("review", np.array([[1000.0, 3000.0]], dtype=np.float32)),
        ]
        qualities = {
            "validated": {"training_eligible": True, "event_eval_eligible": True},
            "review": {"training_eligible": False, "event_eval_eligible": False},
        }
        selected, withheld = filter_recordings_by_quality(
            memory_files, qualities, stem_getter=lambda item: item[0])
        mean, std = compute_patient_stats_from_arrays([array for _, array in selected])
        self.assertEqual([name for name, _ in selected], ["validated"])
        self.assertEqual([name for name, _ in withheld], ["review"])
        np.testing.assert_allclose(mean, [2.0])
        np.testing.assert_allclose(std, [1.0])

    def test_training_normalizer_is_invariant_to_heldout_signals(self):
        quality = {"training_eligible": True, "event_eval_eligible": True}
        train = {"dataset": "chbmit", "patient_id": "train-patient",
                 "file": "train.edf", "array": np.array([[1.0, 3.0]]),
                 "quality": quality, "label": 0}
        test_a = {"dataset": "chbmit", "patient_id": "test-patient",
                  "file": "test.edf", "array": np.array([[10.0, 20.0]]),
                  "quality": quality, "label": 1}
        test_b = {**test_a, "array": np.array([[1e6, -1e6]])}
        fit_patients = {("chbmit", "train-patient")}
        stats_a, sources_a = compute_training_channel_stats_from_arrays(
            [train, test_a], fit_patients)
        stats_b, sources_b = compute_training_channel_stats_from_arrays(
            [train, test_b], fit_patients)
        np.testing.assert_array_equal(stats_a["chbmit"][0], stats_b["chbmit"][0])
        np.testing.assert_array_equal(stats_a["chbmit"][1], stats_b["chbmit"][1])
        self.assertEqual(sources_a, {"chbmit": ["train.edf"]})
        self.assertEqual(sources_b, sources_a)

    def test_training_normalizer_is_invariant_to_heldout_labels(self):
        quality = {"training_eligible": True, "event_eval_eligible": True}
        train = {"dataset": "siena", "patient_id": "train-patient",
                 "array": np.array([[2.0, 4.0]]), "quality": quality,
                 "label": PREICTAL}
        test = {"dataset": "siena", "patient_id": "test-patient",
                "array": np.array([[50.0, 90.0]]), "quality": quality,
                "label": ICTAL}
        changed_test_label = {**test, "label": INTERICTAL}
        fit_patients = {("siena", "train-patient")}
        first, _ = compute_training_channel_stats_from_arrays(
            [train, test], fit_patients)
        second, _ = compute_training_channel_stats_from_arrays(
            [train, changed_test_label], fit_patients)
        np.testing.assert_array_equal(first["siena"][0], second["siena"][0])
        np.testing.assert_array_equal(first["siena"][1], second["siena"][1])

    def test_review_and_future_patient_calibration_do_not_fit_training_stats(self):
        approved = {"training_eligible": True, "event_eval_eligible": True}
        review = {"training_eligible": False, "event_eval_eligible": False}
        recordings = [
            {"dataset": "siena", "patient_id": "train-patient", "file": "train.edf",
             "array": np.array([[1.0, 3.0]]), "quality": approved},
            {"dataset": "siena", "patient_id": "train-patient", "file": "review.edf",
             "array": np.array([[1000.0, 3000.0]]), "quality": review},
            {"dataset": "siena", "patient_id": "test-patient", "file": "calibration.edf",
             "array": np.array([[1e6, 2e6]]), "quality": approved,
             "purpose": "patient_calibration"},
        ]
        stats, sources = compute_training_channel_stats_from_arrays(
            recordings, {("siena", "train-patient")})
        np.testing.assert_allclose(stats["siena"][0], [2.0])
        self.assertEqual(sources, {"siena": ["train.edf"]})
        self.assertEqual(quality_aware_window_entries(
            "review", [], 1000, _quality_for_recording(
                "review", {}, {"review": "manual review"})), [])

    def test_deferred_dataset_fits_then_applies_only_frozen_training_stats(self):
        arrays = {
            "train-file.npy": np.array([[1.0, 3.0]], dtype=np.float32),
            "heldout-file.npy": np.array([[100.0, 200.0]], dtype=np.float32),
            "review-file.npy": np.array([[1e6, 2e6]], dtype=np.float32),
        }

        def fit_files(paths):
            return compute_patient_stats_from_arrays(
                [arrays[Path(path).name] for path in paths])

        def make_deferred_dataset(heldout_label):
            ds = SeizureDataset.__new__(SeizureDataset)
            ds.normalization_mode = "deferred"
            ds.index = [
                {"dataset": "siena", "patient_id": "train", "label": PREICTAL},
                {"dataset": "siena", "patient_id": "heldout", "label": heldout_label},
            ]
            # This map is populated after the shared quality gate; review-file is absent.
            ds._trusted_files_by_patient = {
                ("siena", "train"): [Path("train-file.npy")],
                ("siena", "heldout"): [Path("heldout-file.npy")],
            }
            ds.patient_stats_cache = {}
            with patch("dataset.compute_channel_stats_from_files", side_effect=fit_files) as fit:
                manifest = ds.fit_training_normalization(np.array([0], dtype=np.int64))
            fit.assert_called_once_with([Path("train-file.npy")])
            return ds, manifest

        ds_a, manifest_a = make_deferred_dataset(ICTAL)
        arrays["heldout-file.npy"] = np.array([[-1e6, 1e6]], dtype=np.float32)
        ds_b, manifest_b = make_deferred_dataset(INTERICTAL)
        self.assertEqual(manifest_a, manifest_b)
        np.testing.assert_array_equal(ds_a.patient_stats_cache[("siena", "train")][0], [2.0])
        np.testing.assert_array_equal(ds_a.patient_stats_cache[("siena", "heldout")][0], [2.0])
        np.testing.assert_array_equal(ds_b.patient_stats_cache[("siena", "heldout")][0], [2.0])

        frozen = ds_a.patient_stats_cache[("siena", "heldout")]
        np.testing.assert_allclose(apply_channel_normalization(
            np.array([[2.0, 4.0]], dtype=np.float32), frozen), [[0.0, 2.0]])
        # New patients use the persisted training stats; their future recordings are not fit data.
        np.testing.assert_allclose(apply_channel_normalization(
            np.array([[5.0, 7.0]], dtype=np.float32), frozen), [[3.0, 5.0]])

    def test_chbmit_no_seizure_requires_an_explicit_zero_count(self):
        confirmed = _chbmit_quality_from_summary([], 0, 0, 0)
        malformed = _chbmit_quality_from_summary([], None, 0, 0)
        inconsistent = _chbmit_quality_from_summary([], 1, 0, 0)
        contradictory = _chbmit_quality_from_summary([], 0, 1, 0)
        malformed_zero = _chbmit_quality_from_summary([], 0, 0, 0, True)
        self.assertEqual(confirmed["status"], NO_SEIZURE_CONFIRMED)
        self.assertEqual(malformed["status"], REVIEW_REQUIRED)
        self.assertEqual(inconsistent["status"], REVIEW_REQUIRED)
        self.assertEqual(contradictory["status"], REVIEW_REQUIRED)
        self.assertEqual(malformed_zero["status"], REVIEW_REQUIRED)

    def test_quality_is_resolved_per_file(self):
        review = _quality_for_recording("PN14-3", {}, {"PN14-3": "clock mismatch"})
        valid = _quality_for_recording("PN14-4", {"PN14-4": [(100, 120)]})
        self.assertEqual(review["status"], REVIEW_REQUIRED)
        self.assertEqual(valid["status"], VALIDATED)
        self.assertTrue(valid["training_eligible"])
        self.assertTrue(valid["event_eval_eligible"])

    def test_validated_labels_match_existing_labeler(self):
        seizures = [(4000, 4060)]
        quality = _quality_for_recording("valid", {"valid": seizures})
        actual = windows_for_quality(seizures, 25000 * 250, quality)
        expected = label_windows_for_file(seizures, 25000 * 250)
        self.assertEqual(actual, expected)
        labels = {label for _, label in actual}
        self.assertTrue({INTERICTAL, PREICTAL, ICTAL}.issubset(labels))


if __name__ == "__main__":
    unittest.main()
