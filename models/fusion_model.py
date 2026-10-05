"""
Fusion model for hybrid TCN+MLP seizure prediction (3-class:
interictal/preictal/ictal).

Flow (as agreed in design discussion):
  Raw window (B, 6, 1000)
    -> ChannelAttention (applied ONCE, shared by both branches)
    -> attended (B, 6, 1000)
       |
       +--> TCN branch (fully differentiable path) -> (B, 128)
       |
       +--> detach() -> numpy -> handcrafted feature extraction
            (NOT differentiable -- a bridge, not a trainable layer)
            -> (B, 108) -> MLP branch -> (B, 64)
       |
    concat(128 + 64) -> (B, 192)
    -> Dropout(0.4-0.5)
    -> Linear(192 -> 3)   [interictal / preictal / ictal logits]

Note on the MLP path: handcrafted_features.py runs on NumPy/SciPy and
is not differentiable, so the attended signal is detached from the
autograd graph before feature extraction. Gradients still flow fully
through ChannelAttention via the TCN path, so the attention weights
are trained normally -- see the design discussion for why a fully
differentiable feature path was considered and not adopted here.

Supports phased training: train_tcn.py and train_mlp.py train each
branch independently first; load_pretrained_branches() then loads
those weights here as initialization before joint fusion training
(train_fusion.py), rather than starting the branches from random
weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))

from channel_attention import ChannelAttention  # noqa: E402
from tcn_branch import TCNBranch  # noqa: E402
from mlp_branch import MLPBranch  # noqa: E402
from handcrafted_features import extract_features_from_window, N_FEATURES_PER_CHANNEL  # noqa: E402

N_CLASSES = 3  # interictal=0, preictal=1, ictal=2
FS = 250  # must match training/dataset.py and features/handcrafted_features.py


class FusionModel(nn.Module):
    def __init__(self, n_channels: int = 6, tcn_output_dim: int = 128,
                 mlp_output_dim: int = 64, fusion_dropout: float = 0.45,
                 n_classes: int = N_CLASSES, fs: int = FS):
        super().__init__()
        self.n_channels = n_channels
        self.fs = fs

        self.attention = ChannelAttention(n_channels=n_channels)
        self.tcn_branch = TCNBranch(n_input_channels=n_channels,
                                     output_dim=tcn_output_dim)
        self.mlp_branch = MLPBranch(n_input_features=n_channels * N_FEATURES_PER_CHANNEL,
                                     output_dim=mlp_output_dim)

        fusion_input_dim = tcn_output_dim + mlp_output_dim  # 128 + 64 = 192
        self.classifier = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_input_dim, n_classes),
        )

    def _extract_handcrafted_batch(self, attended: torch.Tensor) -> torch.Tensor:
        """
        Bridge from the (detached, differentiable-path-ended) attended
        signal to a batch of hand-crafted feature vectors. Runs per-
        sample in a Python loop since handcrafted_features.py operates
        on one (channels, samples) array at a time -- acceptable for
        training-time use, not intended for the low-latency inference
        path (see server/inference.py for the real-time feature path).
        """
        attended_np = attended.detach().cpu().numpy()  # (B, C, L)
        batch_size = attended_np.shape[0]

        feature_vectors = np.zeros(
            (batch_size, self.n_channels * N_FEATURES_PER_CHANNEL),
            dtype=np.float32,
        )
        for i in range(batch_size):
            feature_vectors[i] = extract_features_from_window(attended_np[i], fs=self.fs)

        return torch.from_numpy(feature_vectors).to(attended.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, length) -- raw (normalized, possibly
           noise-augmented) window, same input dataset.py produces.
        returns: (batch, n_classes) raw logits (apply softmax/CE loss
                 externally, not inside this module).
        """
        attended = self.attention(x)  # (B, C, L) -- differentiable

        tcn_embedding = self.tcn_branch(attended)  # (B, 128) -- differentiable path

        handcrafted = self._extract_handcrafted_batch(attended)  # (B, 108) -- detached
        mlp_embedding = self.mlp_branch(handcrafted)  # (B, 64)

        fused = torch.cat([tcn_embedding, mlp_embedding], dim=1)  # (B, 192)
        logits = self.classifier(fused)  # (B, 3)
        return logits

    def load_pretrained_branches(self, tcn_weights_path: str = None,
                                  mlp_weights_path: str = None,
                                  map_location: str = "cpu"):
        """
        Load independently pre-trained branch weights (from
        train_tcn.py / train_mlp.py checkpoints) as initialization
        before joint fusion training -- implements the agreed phased
        training strategy (Phase 1: MLP alone, Phase 2: TCN alone,
        Phase 3: load both here and train the fusion layer jointly).
        """
        if tcn_weights_path is not None:
            state = torch.load(tcn_weights_path, map_location=map_location)
            self.tcn_branch.load_state_dict(state)
            print(f"Loaded pretrained TCN branch weights from {tcn_weights_path}")

        if mlp_weights_path is not None:
            state = torch.load(mlp_weights_path, map_location=map_location)
            self.mlp_branch.load_state_dict(state)
            print(f"Loaded pretrained MLP branch weights from {mlp_weights_path}")


if __name__ == "__main__":
    batch_size, n_channels, length = 8, 6, 1000

    model = FusionModel(n_channels=n_channels)
    dummy_input = torch.randn(batch_size, n_channels, length)

    # --- Shape check ---
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, N_CLASSES), "Output shape mismatch!"
    print("Shape check: OK")
    print()

    # --- Gradient flow check: TCN path + attention must receive
    #     gradients; MLP path must also receive gradients (its own
    #     weights), but the attention module's gradient should come
    #     ONLY via the TCN path (the MLP path is detached upstream) ---
    model.train()
    dummy_input2 = torch.randn(batch_size, n_channels, length, requires_grad=False)
    logits2 = model(dummy_input2)
    dummy_target = torch.randint(0, N_CLASSES, (batch_size,))
    loss = nn.functional.cross_entropy(logits2, dummy_target)
    loss.backward()

    attn_grad = model.attention.squeeze_excite[0].weight.grad
    tcn_grad = model.tcn_branch.output_proj.weight.grad
    mlp_grad = model.mlp_branch.net[0].weight.grad

    print(f"Loss value: {loss.item():.4f}")
    print(f"Attention layer received gradient: {attn_grad is not None and attn_grad.abs().sum().item() > 0}")
    print(f"TCN branch received gradient:       {tcn_grad is not None and tcn_grad.abs().sum().item() > 0}")
    print(f"MLP branch received gradient:       {mlp_grad is not None and mlp_grad.abs().sum().item() > 0}")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # --- Real data sanity check ---
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
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

    real_batch = torch.stack([ds[i][0] for i in range(4)])  # (4, 6, 1000)
    real_labels = torch.stack([ds[i][1] for i in range(4)])

    model.eval()
    with torch.no_grad():
        real_logits = model(real_batch)
    print(f"Real batch input shape:  {real_batch.shape}")
    print(f"Real batch output shape: {real_logits.shape}")
    print(f"Real labels: {real_labels.tolist()}")
