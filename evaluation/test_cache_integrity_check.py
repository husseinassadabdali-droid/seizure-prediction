import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.cache_integrity_check import load_cache, compare_caches


def write_cache(root, *, augment=False, labels=None, index=None, features=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    labels = np.array([0, 1, 2], dtype=np.int8) if labels is None else np.asarray(labels)
    index = np.arange(len(labels), dtype=np.int64) if index is None else np.asarray(index)
    features = np.ones((len(labels), 4), dtype=np.float32) if features is None else np.asarray(features)
    np.save(root / "features.npy", features)
    np.savez(
        root / "meta.npz",
        y=labels,
        time_sec=np.arange(len(labels), dtype=np.float64),
        start_sample=np.arange(len(labels), dtype=np.int64),
        index_idx=index,
        file_id=np.zeros(len(labels), dtype=np.int32),
        patient_code=np.zeros(len(labels), dtype=np.int32),
        patient_names=np.array(["chbmit/chb01"]),
        dataset_code=np.zeros(len(labels), dtype=np.int8),
        dataset_names=np.array(["chbmit"]),
    )
    info = {
        "n_windows": int(len(labels)),
        "n_features": int(features.shape[1]),
        "feature_names": [f"f{i}" for i in range(features.shape[1])],
        "augment": augment,
        "label_counts": {
            "interictal": int(np.count_nonzero(labels == 0)),
            "preictal": int(np.count_nonzero(labels == 1)),
            "ictal": int(np.count_nonzero(labels == 2)),
        },
    }
    (root / "info.json").write_text(json.dumps(info), encoding="utf-8")


class CacheIntegrityCheckTests(unittest.TestCase):
    def test_valid_cache_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clean"
            write_cache(path)
            X, meta, info, counts = load_cache(path, expected_augment=False)
            self.assertEqual(X.shape, (3, 4))
            self.assertEqual(meta["y"].tolist(), [0, 1, 2])
            self.assertEqual(info["n_windows"], 3)
            self.assertEqual(counts, {"0": 1, "1": 1, "2": 1})

    def test_nonfinite_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clean"
            values = np.ones((3, 4), dtype=np.float32)
            values[1, 2] = np.nan
            write_cache(path, features=values)
            with self.assertRaisesRegex(ValueError, "non-finite feature"):
                load_cache(path)

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clean"
            write_cache(path, labels=[0, 1, 7])
            with self.assertRaisesRegex(ValueError, "labels contain"):
                load_cache(path)

    def test_duplicate_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clean"
            write_cache(path, index=[0, 0, 2])
            with self.assertRaisesRegex(ValueError, "duplicate rows"):
                load_cache(path)

    def test_clean_and_noisy_metadata_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            clean = Path(temp) / "clean"
            noisy = Path(temp) / "noisy"
            write_cache(clean, augment=False)
            write_cache(noisy, augment=True, labels=[0, 2, 1])
            with self.assertRaisesRegex(ValueError, "metadata differs in y"):
                compare_caches(clean, noisy)


if __name__ == "__main__":
    unittest.main()
