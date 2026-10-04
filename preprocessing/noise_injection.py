"""
Synthetic dry-electrode noise injection for EEG seizure prediction.

Since no clinical seizure dataset recorded with dry electrodes exists,
this module simulates dry-electrode signal characteristics on top of
clean wet-electrode recordings (CHB-MIT, Siena) during training. This
exposes the model to realistic noise conditions it will face with the
actual dry-electrode hardware.

Three noise components, matching real dry-electrode artifacts:
  1. High-frequency Gaussian noise -- higher electrode-skin impedance
     picks up more ambient electrical noise.
  2. Low-frequency baseline drift -- unstable skin contact causes slow
     amplitude wandering.
  3. Power-line interference -- weaker electrode contact is more
     susceptible to 50/60Hz mains coupling.

Default ratios were increased from an initial conservative pass (which
produced only ~2.5% std increase) to better approximate real dry-electrode
SNR degradation (~8-10% std increase), based on the project's dry-electrode
hardware design.

IMPORTANT: Call apply_dry_electrode_augmentation() ONCE per training
sample, and feed the SAME augmented array to both the TCN branch (raw
signal) and the MLP branch (handcrafted features computed from this
same noisy array). This mirrors what will happen on the real device --
both branches see the same noisy signal, not independently noised
copies -- and keeps branch comparisons during phased training fair.
"""

import numpy as np


def add_high_freq_noise(signal: np.ndarray, noise_std_ratio: float = 0.15,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    Add high-frequency Gaussian noise, scaled per-channel relative to
    that channel's own signal amplitude (so a flat/near-zero channel
    doesn't get disproportionate noise).

    noise_std_ratio: noise std as a fraction of each channel's signal std.
                      0.05 = light noise, 0.15 = moderate-heavy noise
                      (current default, approximating dry-electrode SNR).
    """
    rng = rng or np.random.default_rng()
    noisy = signal.copy()
    for ch in range(signal.shape[0]):
        ch_std = np.std(signal[ch])
        if ch_std > 0:
            noise = rng.normal(0, ch_std * noise_std_ratio, size=signal.shape[1])
            noisy[ch] += noise
    return noisy


def add_baseline_drift(signal: np.ndarray, fs: float,
                        drift_amplitude_ratio: float = 0.6,
                        drift_freq_range: tuple = (0.05, 0.3),
                        rng: np.random.Generator = None) -> np.ndarray:
    """
    Add slow-moving baseline drift per channel, simulating unstable
    skin-electrode contact over time.

    drift_amplitude_ratio: drift peak amplitude as a fraction of each
                            channel's signal std.
    drift_freq_range: (min_hz, max_hz) -- a random frequency in this
                       range is picked per channel per call.
    """
    rng = rng or np.random.default_rng()
    n_samples = signal.shape[1]
    t = np.arange(n_samples) / fs

    drifted = signal.copy()
    for ch in range(signal.shape[0]):
        ch_std = np.std(signal[ch])
        if ch_std > 0:
            drift_freq = rng.uniform(*drift_freq_range)
            phase = rng.uniform(0, 2 * np.pi)
            amplitude = ch_std * drift_amplitude_ratio
            drift = amplitude * np.sin(2 * np.pi * drift_freq * t + phase)
            drifted[ch] += drift
    return drifted


def add_powerline_noise(signal: np.ndarray, fs: float, notch_freq: float = 50.0,
                         amplitude_ratio: float = 0.18,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    Add residual power-line interference at the mains frequency (and its
    2nd harmonic), simulating imperfect notch filtering under weaker
    electrode contact.

    notch_freq: 50.0 or 60.0, matching the dataset's recording region.
    amplitude_ratio: interference amplitude as a fraction of each
                      channel's signal std.
    """
    rng = rng or np.random.default_rng()
    n_samples = signal.shape[1]
    t = np.arange(n_samples) / fs

    noisy = signal.copy()
    for ch in range(signal.shape[0]):
        ch_std = np.std(signal[ch])
        if ch_std > 0:
            phase1 = rng.uniform(0, 2 * np.pi)
            phase2 = rng.uniform(0, 2 * np.pi)
            amplitude = ch_std * amplitude_ratio
            interference = (
                amplitude * np.sin(2 * np.pi * notch_freq * t + phase1)
                + 0.3 * amplitude * np.sin(2 * np.pi * 2 * notch_freq * t + phase2)
            )
            noisy[ch] += interference
    return noisy


def apply_dry_electrode_augmentation(
    signal: np.ndarray,
    fs: float,
    notch_freq: float = 50.0,
    high_freq_noise_ratio: float = 0.15,
    drift_amplitude_ratio: float = 0.6,
    powerline_amplitude_ratio: float = 0.18,
    apply_prob: float = 0.8,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply all three dry-electrode noise components to a (channels, samples)
    signal array. Call this once per training sample; feed the result to
    BOTH the TCN branch (raw) and the MLP branch (feature extraction) so
    they see identical noise -- not independently noised copies.

    apply_prob: probability of applying augmentation at all for this call
                (set < 1.0 so the model also sees some clean samples during
                training, not just noisy ones).

    Returns a NEW array; the input `signal` is not modified in place.
    """
    rng = rng or np.random.default_rng()

    if rng.random() > apply_prob:
        return signal.copy()

    augmented = signal.copy()
    augmented = add_high_freq_noise(augmented, high_freq_noise_ratio, rng)
    augmented = add_baseline_drift(augmented, fs, drift_amplitude_ratio, rng=rng)
    augmented = add_powerline_noise(augmented, fs, notch_freq,
                                     powerline_amplitude_ratio, rng)
    return augmented


if __name__ == "__main__":
    # Quick sanity check on one processed file: compare signal stats
    # before and after augmentation.
    from pathlib import Path

    sample_file = next(Path("../data/processed/chbmit").glob("*.npy"), None)
    if sample_file is None:
        print("No processed CHB-MIT files found -- run preprocess.py first.")
    else:
        clean = np.load(sample_file)
        rng = np.random.default_rng(seed=42)
        noisy = apply_dry_electrode_augmentation(
            clean, fs=250.0, notch_freq=60.0, apply_prob=1.0, rng=rng
        )

        print(f"Sample file: {sample_file.name}")
        print(f"Shape: {clean.shape}")
        print()
        print(f"{'Channel':<10}{'Clean std':<15}{'Noisy std':<15}{'Increase':<10}")
        for ch in range(clean.shape[0]):
            clean_std = np.std(clean[ch])
            noisy_std = np.std(noisy[ch])
            increase = (noisy_std / clean_std - 1) * 100 if clean_std > 0 else 0
            print(f"{ch:<10}{clean_std:<15.4f}{noisy_std:<15.4f}{increase:<10.1f}%")
