"""Synthetic tests for fail-closed preprocessing channel handling."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "preprocessing"))
import preprocess


class FakeRaw:
    def __init__(self, names, data):
        self.info = {"ch_names": list(names)}
        self.n_times = data.shape[1]
        self._data = np.asarray(data, dtype=np.float64)

    def get_data(self, picks):
        return self._data[picks, :]


class PreprocessChannelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.targets = ["A", "B", "C", "D", "E", "F"]
        self.data = np.vstack([np.full(16, i + 1.0) for i in range(6)])

    def test_missing_target_raises_and_does_not_save(self):
        raw = FakeRaw(["A", "B", "C", "D", "E", "OTHER"], self.data)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(preprocess, "load_and_filter", return_value=raw):
                with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
                    preprocess.process_file(Path("record.edf"), out, self.targets, 50.0)
            self.assertEqual(list(out.glob("*.npy")), [])

    def test_ambiguous_duplicate_target_raises_and_does_not_save(self):
        names = ["A", "B", "C", "D", "E", "F", "F"]
        data = np.vstack([np.full(16, i + 1.0) for i in range(7)])
        raw = FakeRaw(names, data)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(preprocess, "load_and_filter", return_value=raw):
                with self.assertRaisesRegex(ValueError, "ambiguous duplicate label"):
                    preprocess.process_file(Path("record.edf"), out, self.targets, 50.0)
            self.assertEqual(list(out.glob("*.npy")), [])

    def test_complete_recording_saves_target_channels_in_requested_order(self):
        # EDF channels are deliberately shuffled; output must follow target order.
        names = ["D", "A", "F", "C", "B", "E"]
        row_values = {"A": 10, "B": 20, "C": 30, "D": 40, "E": 50, "F": 60}
        data = np.vstack([np.full(16, row_values[name]) for name in names])
        raw = FakeRaw(names, data)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(preprocess, "load_and_filter", return_value=raw):
                result = preprocess.process_file(Path("record.edf"), out, self.targets, 50.0)
            self.assertEqual(result, [])
            saved = np.load(out / "record.npy", allow_pickle=False)
            self.assertEqual(saved.shape, (6, 16))
            self.assertEqual(saved.dtype, np.float32)
            np.testing.assert_array_equal(
                saved[:, 0], np.array([10, 20, 30, 40, 50, 60], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
