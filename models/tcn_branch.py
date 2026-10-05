"""
TCN (Temporal Convolutional Network) branch for the hybrid TCN+MLP
seizure prediction model.

Architecture: 8 stacked causal residual blocks with exponentially
increasing dilation (1, 2, 4, 8, 16, 32, 64, 128), following Bai et
al. (2018). Each block:
  Conv1d (causal, dilated) -> WeightNorm -> ReLU -> Dropout
  Conv1d (causal, dilated) -> WeightNorm -> ReLU -> Dropout
  + residual connection (1x1 conv if channel count changes)
  -> ReLU

Causality is enforced via left-only padding + "chomp" (trimming the
extra right-side output that a standard same-padding conv would
otherwise produce) -- the output at time t depends ONLY on inputs at
times <= t, never on future samples. This matches the no-future-
leakage requirement for real-time, streaming inference on the server.

Input:  (batch, n_channels=6, length=1000)  -- already Channel-
        Attention-weighted upstream (see channel_attention.py),
        NOT applied again here.
Output: (batch, 128) -- fixed-size embedding, fed into the fusion
        layer alongside the MLP branch's output.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Removes the extra right-side padding a same-padding dilated conv
    produces, so the layer's output length matches its input length
    while remaining strictly causal (no peeking at future samples)."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One causal residual block: two dilated conv layers + skip connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # left-pad amount for causality

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )

        # 1x1 conv to match channel counts for the residual add, only
        # when in_channels != out_channels
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


class TCNBranch(nn.Module):
    def __init__(self, n_input_channels: int = 6, n_layers: int = 8,
                 hidden_channels: int = 32, kernel_size: int = 7,
                 dropout: float = 0.2, output_dim: int = 128):
        """
        n_input_channels: EEG channels (6).
        n_layers: number of TemporalBlocks; dilation doubles each layer
                  (1, 2, 4, ..., 2^(n_layers-1)). 8 layers -> dilation
                  up to 128, giving a receptive field well beyond the
                  1000-sample (4s) window.
        hidden_channels: channel width inside each TemporalBlock.
        output_dim: final embedding size (128, per the agreed fusion
                    design: TCN=128 + MLP=64 -> concat=192).
        """
        super().__init__()

        layers = []
        in_ch = n_input_channels
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(TemporalBlock(
                in_ch, hidden_channels, kernel_size,
                dilation=dilation, dropout=dropout))
            in_ch = hidden_channels

        self.tcn_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_input_channels, length)
        returns: (batch, output_dim)
        """
        features = self.tcn_layers(x)          # (B, hidden_channels, length)
        pooled = self.global_pool(features)      # (B, hidden_channels, 1)
        pooled = pooled.squeeze(-1)              # (B, hidden_channels)
        return self.output_proj(pooled)          # (B, output_dim)

    def receptive_field(self, kernel_size: int = 7, n_layers: int = 8) -> int:
        """Informational: total receptive field in samples."""
        total = 1
        for i in range(n_layers):
            dilation = 2 ** i
            total += 2 * (kernel_size - 1) * dilation
        return total


if __name__ == "__main__":
    batch_size, n_channels, length = 8, 6, 1000

    model = TCNBranch(n_input_channels=n_channels, n_layers=8,
                       hidden_channels=32, kernel_size=7,
                       dropout=0.2, output_dim=128)

    dummy_input = torch.randn(batch_size, n_channels, length)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 128), "Output shape mismatch!"
    print("Shape check: OK")
    print()

    rf = model.receptive_field(kernel_size=7, n_layers=8)
    print(f"Receptive field: {rf} samples ({rf / 250:.2f} seconds at 250Hz)")
    print(f"Window length:   {length} samples (4.00 seconds)")
    print(f"Receptive field {'covers' if rf >= length else 'does NOT cover'} the full window")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Causality check: a change in a LATER time step must not affect
    # an EARLIER output position's features (verified at the feature-map
    # level, before global pooling, where per-timestep causality is
    # directly observable)
    model.eval()
    with torch.no_grad():
        x1 = torch.randn(1, n_channels, length)
        x2 = x1.clone()
        x2[:, :, 500:] += 10.0  # perturb only the SECOND half

        feat1 = model.tcn_layers(x1)
        feat2 = model.tcn_layers(x2)

        early_diff = (feat1[:, :, :500] - feat2[:, :, :500]).abs().max().item()
        print()
        print(f"Causality check -- max difference in early timesteps "
              f"after perturbing only later timesteps: {early_diff:.8f}")
        print("(Should be ~0.0 if causal; any large value indicates future leakage)")
