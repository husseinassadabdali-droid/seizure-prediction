"""
Hand-crafted feature extraction for the MLP branch of the hybrid
TCN+MLP seizure prediction model.

Per channel: band powers (delta/theta/alpha/beta), relative band
powers, theta/alpha and theta/beta ratios, spectral entropy, Hjorth
parameters, and statistical moments.

NaN/Inf SAFETY: near-flat signals (zero-padded missing channels, or
flat recording segments) make skewness/kurtosis/ratios numerically
unstable. Flat channels skip those computations, and every feature is
clamped to a finite value as a last line of defense.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

FS = 250  # must match training/dataset.py FS
FLAT_SIGNAL_STD_THRESHOLD = 1e-6


def _safe(value: float) -> float:
    """Replace NaN/Inf with 0.0."""
    if not np.isfinite(value):
        return 0.0
    return float(value)


def compute_band_powers(signal_1d: np.ndarray, fs: int = FS) -> dict:
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    powers = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = _safe(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def compute_relative_band_powers(band_powers: dict) -> dict:
    total = sum(band_powers.values())
    if total <= 1e-10:
        return {f"rel_{k}": 0.0 for k in band_powers}
    return {f"rel_{k}": _safe(v / total) for k, v in band_powers.items()}


def compute_band_ratios(band_powers: dict) -> dict:
    eps = 1e-6
    return {
        "theta_alpha_ratio": _safe(band_powers["theta"] / (band_powers["alpha"] + eps)),
        "theta_beta_ratio": _safe(band_powers["theta"] / (band_powers["beta"] + eps)),
    }


def compute_spectral_entropy(signal_1d: np.ndarray, fs: int = FS) -> float:
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    total = np.sum(psd)
    if total <= 1e-10:
        return 0.0
    psd_norm = psd / total
    psd_norm = psd_norm[psd_norm > 1e-12]
    if len(psd_norm) == 0:
        return 0.0
    return _safe(-np.sum(psd_norm * np.log2(psd_norm)))


def compute_hjorth_params(signal_1d: np.ndarray) -> dict:
    first_deriv = np.diff(signal_1d)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal_1d)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 1e-10 else 0.0
    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 1e-10 else 0.0
    complexity = mobility_d1 / mobility if mobility > 1e-10 else 0.0

    return {"hjorth_activity": _safe(activity), "hjorth_mobility": _safe(mobility),
            "hjorth_complexity": _safe(complexity)}


def compute_statistical_features(signal_1d: np.ndarray) -> dict:
    std = np.std(signal_1d)
    if std < FLAT_SIGNAL_STD_THRESHOLD:
        return {
            "mean": _safe(float(np.mean(signal_1d))),
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }
    return {
        "mean": _safe(float(np.mean(signal_1d))),
        "std": _safe(float(std)),
        "skewness": _safe(float(skew(signal_1d))),
        "kurtosis": _safe(float(kurtosis(signal_1d))),
    }


def extract_channel_features(signal_1d: np.ndarray, fs: int = FS) -> dict:
    band_powers = compute_band_powers(signal_1d, fs)
    features = {}
    features.update({f"bp_{k}": v for k, v in band_powers.items()})
    features.update(compute_relative_band_powers(band_powers))
    features.update(compute_band_ratios(band_powers))
    features["spectral_entropy"] = compute_spectral_entropy(signal_1d, fs)
    features.update(compute_hjorth_params(signal_1d))
    features.update(compute_statistical_features(signal_1d))
    return features


# Fixed feature order -- must stay consistent across training and inference.
FEATURE_NAMES = [
    "bp_delta", "bp_theta", "bp_alpha", "bp_beta",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta",
    "theta_alpha_ratio", "theta_beta_ratio",
    "spectral_entropy",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "mean", "std", "skewness", "kurtosis",
]
N_FEATURES_PER_CHANNEL = len(FEATURE_NAMES)


def extract_features_from_window(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    window: (n_channels, n_samples)
    Returns a flat float32 vector of length n_channels * N_FEATURES_PER_CHANNEL,
    channel-major order, guaranteed finite.
    """
    n_channels = window.shape[0]
    all_features = np.zeros(n_channels * N_FEATURES_PER_CHANNEL, dtype=np.float32)

    for ch in range(n_channels):
        ch_features = extract_channel_features(window[ch], fs)
        for i, name in enumerate(FEATURE_NAMES):
            all_features[ch * N_FEATURES_PER_CHANNEL + i] = ch_features[name]

    return np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    from dataset import SeizureDataset  # noqa: E402

    project_root = Path(__file__).resolve().parent.parent
    configs = [
        {
            "name": "chbmit",
            "processed_dir": project_root / "data/processed/chbmit",
            "raw_dir": project_root / "data/raw/chbmit",
            "notch_freq": 60.0,
        },
    ]

    ds = SeizureDataset(configs, apply_augmentation=False)
    window_tensor, label = ds[0]
    features = extract_features_from_window(window_tensor.numpy())

    print(f"Feature vector length: {len(features)}")
    print(f"All features finite: {np.all(np.isfinite(features))}")

    # Stress test: flat and zero-padded channels must NOT produce NaN/Inf
    flat_window = np.zeros((6, 1000), dtype=np.float32)
    flat_window[1] = 5.0  # constant non-zero channel
    flat_features = extract_features_from_window(flat_window)
    print(f"Flat/zero window finite: {np.all(np.isfinite(flat_features))}")
