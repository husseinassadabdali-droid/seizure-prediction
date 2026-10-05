"""
Hand-crafted feature extraction for the MLP branch of the hybrid
TCN+MLP seizure prediction model.

For each (channels, samples) window, extracts per-channel:
  1. Band powers (Delta, Theta, Alpha, Beta) via Welch PSD
  2. Relative band powers (domain-invariant: robust to amplitude/age
     differences across patients and recording devices)
  3. Band power ratios (Theta/Alpha, Theta/Beta -- classic seizure
     biomarkers, also domain-invariant)
  4. Spectral entropy (domain-invariant measure of signal regularity)
  5. Hjorth parameters: Activity, Mobility, Complexity
  6. Statistical moments: mean, std, skewness, kurtosis

All features are computed per-channel then concatenated into one flat
vector per window. Designed to be called with the SAME (possibly
noise-augmented) window array that the TCN branch receives, so both
branches see identical signal content during phased training.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# ---- Band definitions (Hz) ----
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

FS = 250  # must match training/dataset.py FS


def compute_band_powers(signal_1d: np.ndarray, fs: int = FS) -> dict:
    """
    Welch PSD-based absolute band power for one channel's signal.
    Returns {"delta": p, "theta": p, "alpha": p, "beta": p}.
    """
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))

    powers = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        powers[band_name] = np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
    return powers


def compute_relative_band_powers(band_powers: dict) -> dict:
    """
    Normalize each band's power by total power across all defined bands.
    Domain-invariant: removes absolute-amplitude differences caused by
    skull thickness, electrode impedance, or recording device gain.
    """
    total = sum(band_powers.values())
    if total <= 0:
        return {f"rel_{k}": 0.0 for k in band_powers}
    return {f"rel_{k}": v / total for k, v in band_powers.items()}


def compute_band_ratios(band_powers: dict) -> dict:
    """
    Classic seizure-related spectral ratios. Domain-invariant for the
    same reason as relative band powers (ratio of two powers cancels
    out absolute-amplitude differences).
    """
    eps = 1e-10
    return {
        "theta_alpha_ratio": band_powers["theta"] / (band_powers["alpha"] + eps),
        "theta_beta_ratio": band_powers["theta"] / (band_powers["beta"] + eps),
    }


def compute_spectral_entropy(signal_1d: np.ndarray, fs: int = FS) -> float:
    """
    Shannon entropy of the normalized power spectrum. Low entropy =
    more regular/rhythmic signal (often seen during seizures); high
    entropy = more complex/irregular (typical interictal activity).
    Domain-invariant: depends on spectral SHAPE, not absolute amplitude.
    """
    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    return float(-np.sum(psd_norm * np.log2(psd_norm)))


def compute_hjorth_params(signal_1d: np.ndarray) -> dict:
    """
    Hjorth Activity, Mobility, Complexity -- measure signal power,
    mean frequency, and change in frequency, respectively.
    """
    first_deriv = np.diff(signal_1d)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal_1d)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    mobility = np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0.0
    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0.0
    complexity = mobility_d1 / mobility if mobility > 0 else 0.0

    return {"hjorth_activity": activity, "hjorth_mobility": mobility,
            "hjorth_complexity": complexity}


def compute_statistical_features(signal_1d: np.ndarray) -> dict:
    """Basic statistical moments."""
    return {
        "mean": float(np.mean(signal_1d)),
        "std": float(np.std(signal_1d)),
        "skewness": float(skew(signal_1d)),
        "kurtosis": float(kurtosis(signal_1d)),
    }


def extract_channel_features(signal_1d: np.ndarray, fs: int = FS) -> dict:
    """All features for ONE channel's 1D signal, as a flat dict."""
    band_powers = compute_band_powers(signal_1d, fs)
    features = {}
    features.update({f"bp_{k}": v for k, v in band_powers.items()})
    features.update(compute_relative_band_powers(band_powers))
    features.update(compute_band_ratios(band_powers))
    features["spectral_entropy"] = compute_spectral_entropy(signal_1d, fs)
    features.update(compute_hjorth_params(signal_1d))
    features.update(compute_statistical_features(signal_1d))
    return features


# Fixed feature name order -- MUST stay consistent across every call,
# since mlp_branch.py will consume these as a plain ordered vector.
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
    window: (n_channels, n_samples) array -- same window passed to the
            TCN branch (post-normalization, optionally noise-augmented).
    Returns: flat 1D feature vector, length = n_channels * N_FEATURES_PER_CHANNEL,
             channel-major order (all of channel 0's features, then channel 1's, ...).
    """
    n_channels = window.shape[0]
    all_features = np.zeros(n_channels * N_FEATURES_PER_CHANNEL, dtype=np.float32)

    for ch in range(n_channels):
        ch_features = extract_channel_features(window[ch], fs)
        for i, name in enumerate(FEATURE_NAMES):
            all_features[ch * N_FEATURES_PER_CHANNEL + i] = ch_features[name]

    return all_features


if __name__ == "__main__":
    # Sanity check against a real processed window via SeizureDataset
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
    window = window_tensor.numpy()

    features = extract_features_from_window(window)

    print(f"Window shape: {window.shape}")
    print(f"Feature vector length: {len(features)} "
          f"(expected {window.shape[0]} channels x {N_FEATURES_PER_CHANNEL} features)")
    print(f"Label: {label.item()}")
    print()
    print("First channel's features:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<20} {features[i]:.6f}")
