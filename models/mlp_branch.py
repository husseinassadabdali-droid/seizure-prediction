"""
MLP branch for the hybrid TCN+MLP seizure prediction model.

Consumes the flat hand-crafted feature vector (6 channels x 18 features
= 108, from features/handcrafted_features.py) and projects it into a
fixed-size embedding (64) for the fusion layer.

Input scaling lives INSIDE this module (robust center/scale buffers +
clamp to +-FEATURE_CLIP). The buffers are part of state_dict, so the
saved branch weights carry their own normalization statistics and the
fusion model can load them without any extra preprocessing. Defaults
(center=0, scale=1) make it an identity until set_feature_stats() is
called by train_mlp.py.
"""

import torch
import torch.nn as nn

FEATURE_CLIP = 10.0


class MLPBranch(nn.Module):
    def __init__(self, n_input_features: int = 108, hidden_dim: int = 128,
                 output_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("feature_center", torch.zeros(n_input_features))
        self.register_buffer("feature_scale", torch.ones(n_input_features))

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

    @torch.no_grad()
    def set_feature_stats(self, center, scale):
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32).clamp_min(1e-6)
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x - self.feature_center) / self.feature_scale).clamp(-FEATURE_CLIP, FEATURE_CLIP)
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MLPBranch(n_input_features=108)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(8, 108))
    assert out.shape == (8, 64)
    print(f"Shape check OK: {tuple(out.shape)}, params={sum(p.numel() for p in model.parameters()):,}")

    # normalization buffers must travel with state_dict
    model.set_feature_stats(torch.full((108,), 3.0), torch.full((108,), 2.0))
    state = model.state_dict()
    assert "feature_center" in state and "feature_scale" in state
    fresh = MLPBranch(n_input_features=108)
    fresh.load_state_dict(state)
    assert torch.allclose(fresh.feature_center, torch.full((108,), 3.0))
    print("State-dict carries normalization stats: OK")

    # extreme inputs must stay finite thanks to the clamp
    fresh.eval()
    with torch.no_grad():
        out_huge = fresh(torch.full((2, 108), 1e9))
    assert torch.isfinite(out_huge).all()
    print("Extreme-input clamp: OK")
