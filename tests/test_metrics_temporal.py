import unittest

import numpy as np

from evaluation.metrics import event_metrics, make_alarm


class TemporalAlarmTests(unittest.TestCase):
    def test_consecutive_alarm_resets_at_time_gap(self):
        probs = np.array([[0.1, 0.8, 0.1]] * 4)
        times = np.array([0.0, 2.0, 120.0, 122.0])
        alarm = make_alarm(probs, min_consecutive=3, times=times, max_gap_sec=60)
        np.testing.assert_array_equal(alarm, [False, False, False, False])

    def test_consecutive_alarm_resets_at_group_boundary(self):
        probs = np.array([[0.1, 0.8, 0.1]] * 4)
        times = np.array([0.0, 2.0, 0.0, 2.0])
        groups = np.array([0, 0, 1, 1])
        alarm = make_alarm(probs, min_consecutive=3, times=times,
                           group_ids=groups, max_gap_sec=60)
        np.testing.assert_array_equal(alarm, [False, False, False, False])

    def test_alarm_fires_after_required_contiguous_windows(self):
        probs = np.array([[0.1, 0.8, 0.1]] * 4)
        times = np.arange(4) * 2.0
        alarm = make_alarm(probs, min_consecutive=3, times=times, max_gap_sec=60)
        np.testing.assert_array_equal(alarm, [False, False, True, True])

    def test_event_metrics_isolates_patients_with_overlapping_clocks(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        alarm = np.array([False, True, False, False, False, False])
        times = np.array([0, 2, 4, 0, 2, 4], dtype=float)
        groups = np.array([0, 0, 0, 1, 1, 1])
        result = event_metrics(labels, alarm, times, group_ids=groups)
        self.assertEqual(result["n_seizures"], 2)
        self.assertEqual(result["predicted_in_advance"], 1)
        self.assertEqual(result["missed"], 1)

    def test_mismatched_group_length_is_rejected(self):
        with self.assertRaises(ValueError):
            event_metrics(np.array([0, 1]), np.array([False, True]),
                          np.array([0.0, 2.0]), group_ids=np.array([0]))


if __name__ == "__main__":
    unittest.main()
