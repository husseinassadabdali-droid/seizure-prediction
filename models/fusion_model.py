"""Frozen pretrained EEG branches used by Fusion experiments.
TCN receives sanitized/clipped EEG and pretrained channel attention. MLP receives
handcrafted features from the original normalized window, without attention.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "features", ROOT / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from channel_attention import ChannelAttention
from tcn_branch import TCNBranch
from mlp_branch import MLPBranch
from handcrafted_features import extract_features_from_window, N_FEATURES_PER_CHANNEL

N_CLASSES, FS, INPUT_CLIP = 3, 250, 20.0

class FusionModel(nn.Module):
    """Embedding backbone. F1/F2 start with frozen pretrained branches."""
    def __init__(self, n_channels=6, tcn_output_dim=128, mlp_output_dim=64,
                 fs=FS, freeze_branches=True):
        super().__init__()
        self.n_channels, self.fs = n_channels, fs
        self.tcn_output_dim, self.mlp_output_dim = tcn_output_dim, mlp_output_dim
        self.attention = ChannelAttention(n_channels=n_channels)
        self.tcn_branch = TCNBranch(n_input_channels=n_channels, output_dim=tcn_output_dim)
        self.mlp_branch = MLPBranch(n_input_features=n_channels*N_FEATURES_PER_CHANNEL,
                                   output_dim=mlp_output_dim)
        self.freeze_branches = bool(freeze_branches)
        if self.freeze_branches:
            self._freeze_branch_modules()
        self.train(False)

    def _freeze_branch_modules(self):
        for m in (self.attention, self.tcn_branch, self.mlp_branch):
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_branches:
            self.attention.eval()
            self.tcn_branch.eval()
            self.mlp_branch.eval()
        return self

    @staticmethod
    def sanitize_tcn_input(x):
        return torch.nan_to_num(x, nan=0.0, posinf=INPUT_CLIP, neginf=-INPUT_CLIP).clamp(-INPUT_CLIP, INPUT_CLIP)

    def _extract_features(self, x):
        arr = x.detach().cpu().numpy()
        out = np.empty((arr.shape[0], self.n_channels*N_FEATURES_PER_CHANNEL), np.float32)
        for i, sample in enumerate(arr):
            out[i] = np.nan_to_num(extract_features_from_window(sample, fs=self.fs),
                                   nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)

    def extract_tcn_embedding(self, x):
        return self.tcn_branch(self.attention(self.sanitize_tcn_input(x)))

    def extract_mlp_embedding(self, x):
        return self.mlp_branch(self._extract_features(x))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected EEG [B,{self.n_channels},T], got {tuple(x.shape)}")
        return {"tcn_embedding": self.extract_tcn_embedding(x),
                "mlp_embedding": self.extract_mlp_embedding(x)}

    def load_pretrained_branches(self, tcn_weights_path, mlp_weights_path,
                                attention_weights_path, dataset_fingerprint,
                                map_location="cpu"):
        from cache_provenance import (validate_artifact_manifest,
                                      validate_matching_normalization_manifests)
        paths_kinds = ((tcn_weights_path, "tcn_branch"),
                       (mlp_weights_path, "mlp_branch"),
                       (attention_weights_path, "tcn_attention"))
        manifests = [validate_artifact_manifest(p, dataset_fingerprint, k)
                    for p, k in paths_kinds]
        validate_matching_normalization_manifests(manifests[0], manifests[1])
        states = [torch.load(p, map_location=map_location, weights_only=True)
                  for p, _ in paths_kinds]
        self.tcn_branch.load_state_dict(states[0], strict=True)
        self.mlp_branch.load_state_dict(states[1], strict=True)
        self.attention.load_state_dict(states[2], strict=True)
        self._freeze_branch_modules()
        self.train(False)
        return {"tcn": str(tcn_weights_path), "mlp": str(mlp_weights_path),
                "attention": str(attention_weights_path)}
