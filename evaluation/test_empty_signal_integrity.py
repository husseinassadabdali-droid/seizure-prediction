import numpy as np
import pytest

from features.cache_provenance import inspect_signal_array
from preprocessing import preprocess


def test_empty_sample_axis_is_review_required():
    result = inspect_signal_array(np.empty((6, 0), dtype=np.float32))
    assert result["status"] == "REVIEW_REQUIRED"
    assert "array has zero samples" in result["issues"]


def test_zero_channel_axis_is_review_required():
    result = inspect_signal_array(np.empty((0, 100), dtype=np.float32))
    assert result["status"] == "REVIEW_REQUIRED"
    assert "array has zero channels" in result["issues"]


def test_invalid_dataset_is_rejected_before_loading(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("EDF loading must not occur for an invalid dataset")

    monkeypatch.setattr(preprocess, "load_and_filter", fail_if_called)
    with pytest.raises(ValueError, match="unknown dataset"):
        preprocess.process_file(
            tmp_path / "record.edf", tmp_path / "processed",
            ["A"], 50.0, dataset_name="unknown"
        )


def test_failed_provenance_does_not_leave_stale_sidecar(monkeypatch, tmp_path):
    class FakeRaw:
        info = {"ch_names": list(preprocess.DATASETS["chbmit"]["target_channels"])}
        n_times = 4

        def get_data(self, picks):
            return np.array([[1e-6, 2e-6, 3e-6, 4e-6]], dtype=np.float64)

    edf_path = tmp_path / "record.edf"
    edf_path.write_bytes(b"edf-source")
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    output_path = output_dir / "record.npy"
    sidecar = output_dir / "record.npy.provenance.json"
    np.save(output_path, np.ones((6, 4), dtype=np.float32))
    sidecar.write_text('{"status":"VERIFIED","old":true}', encoding="utf-8")

    monkeypatch.setattr(preprocess, "load_and_filter", lambda *args, **kwargs: FakeRaw())
    monkeypatch.setattr(
        preprocess, "_processing_config_fingerprint",
        lambda dataset: (_ for _ in ()).throw(RuntimeError("provenance failure"))
    )

    with pytest.raises(RuntimeError, match="provenance failure"):
        preprocess.process_file(
            edf_path, output_dir,
            preprocess.DATASETS["chbmit"]["target_channels"],
            60.0, dataset_name="chbmit"
        )

    assert output_path.is_file()
    assert not sidecar.exists()
