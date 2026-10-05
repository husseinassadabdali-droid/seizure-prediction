"""
MLP branch for the hybrid TCN+MLP seizure prediction model.

Consumes the flat hand-crafted feature vector (108 values: 6 channels
x 18 features, from features/handcrafted_features.py) and projects it
through a small dense network into a fixed-size embedding, matching
the TCN branch's role in the fusion design.

Architecture: three Linear layers with BatchNorm + ReLU + Dropout,
progressively narrowing: 108 -> 128 -> 64 -> 64 (output embedding).

Input:  (batch, n_features=108) -- raw hand-crafted feature vector,
        NOT pre-normalized by this module. Feature-level normalization
        (e.g. per-patient Z-score) should happen upstream if needed;
        BatchNorm here handles batch-level scale differences during
        training.
Output: (batch, 64) -- fixed-size embedding, fed into the fusion
        layer alongside the TCN branch's 128-dim output.
"""

import torch
import torch.nn as nn


class MLPBranch(nn.Module):
    def __init__(self, n_input_features: int = 108, hidden_dim: int = 128,
                 output_dim: int = 64, dropout: float = 0.3):
        """
        n_input_features: length of the hand-crafted feature vector
                           (6 channels x 18 features = 108, must match
                           features/handcrafted_features.py exactly).
        hidden_dim: width of the first hidden layer.
        output_dim: final embedding size (64, per the agreed fusion
                    design: TCN=128 + MLP=64 -> concat=192).
        dropout: higher than the TCN branch's (0.2) since the MLP
                 branch has proportionally more capacity relative to
                 its small input size, and is more prone to
                 overfitting on 108 hand-crafted numbers.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_input_features)
        returns: (batch, output_dim)
        """
        return self.net(x)


if __name__ == "__main__":
    batch_size, n_features = 8, 108

    model = MLPBranch(n_input_features=n_features, hidden_dim=128,
                       output_dim=64, dropout=0.3)

    dummy_input = torch.randn(batch_size, n_features)
    model.eval()  # BatchNorm needs eval mode for a clean single-batch test
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 64), "Output shape mismatch!"
    print("Shape check: OK")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # Sanity check: with a real feature vector (via handcrafted_features.py
    # + dataset.py), confirm the full pipeline connects end-to-end.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
    from dataset import SeizureDataset  # noqa: E402
    from handcrafted_features import extract_features_from_window  # noqa: E402

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
    features_tensor = torch.from_numpy(features).unsqueeze(0)  # add batch dim -> (1, 108)

    with torch.no_grad():
        real_output = model(features_tensor)

    print(f"Real feature vector shape: {features_tensor.shape}")
    print(f"Real MLP branch output shape: {real_output.shape}")
    print(f"Label for this sample: {label.item()}")
