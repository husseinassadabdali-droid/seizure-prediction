"""
Channel Attention Block for the hybrid TCN+MLP seizure prediction model.

Dry electrodes mean individual channels (e.g. F7/F8, closer to jaw
muscles) are more prone to motion/EMG artifacts than others. This
block learns a per-channel weight at inference time, down-weighting
noisy channels and up-weighting clean ones BEFORE the signal enters
the TCN branch -- rather than treating all 6 channels equally.

Architecture (as agreed in design discussion):
  Input (B, C, L)
    -> Global Average Pooling over time -> (B, C)
    -> Dense C -> C//2 -> ReLU
    -> Dense C//2 -> C -> Sigmoid         (per-channel weights, 0-1)
    -> Element-wise multiply with input   (B, C, L) * (B, C, 1)
  Output (B, C, L), same shape as input.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, n_channels: int = 6, reduction: int = 2):
        """
        n_channels: number of EEG channels (6 for this project).
        reduction: bottleneck factor for the squeeze layer.
                   n_channels=6, reduction=2 -> hidden size 3,
                   matching the agreed 6 -> 3 -> 6 design.
        """
        super().__init__()
        hidden = max(1, n_channels // reduction)

        self.squeeze_excite = nn.Sequential(
            nn.Linear(n_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, length)
        returns: (batch, n_channels, length), channel-reweighted
        """
        # Global average pooling over the time dimension -> (B, C)
        channel_summary = x.mean(dim=2)

        # Per-channel weights, 0-1 -> (B, C)
        channel_weights = self.squeeze_excite(channel_summary)

        # Reshape for broadcasting: (B, C) -> (B, C, 1)
        channel_weights = channel_weights.unsqueeze(-1)

        # Scale each channel by its learned weight
        return x * channel_weights


if __name__ == "__main__":
    # Sanity check 1: synthetic batch, verify shapes and weight range
    batch_size, n_channels, length = 8, 6, 1000
    dummy_input = torch.randn(batch_size, n_channels, length)

    attn = ChannelAttention(n_channels=n_channels)
    output = attn(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Shape mismatch!"
    print("Shape check: OK (input/output match)")
    print()

    # Inspect the actual learned weights for this dummy batch
    with torch.no_grad():
        summary = dummy_input.mean(dim=2)
        weights = attn.squeeze_excite(summary)
    print(f"Channel weights for sample 0: {weights[0].numpy()}")
    print(f"All weights in [0, 1]: {bool((weights >= 0).all() and (weights <= 1).all())}")
    print()

    # Sanity check 2: a near-silent (noisy/dead) channel should tend to
    # get a different weight than an active one, on average over many
    # random batches with a fixed, trained-like contrast -- here we just
    # confirm the module responds differently to different inputs.
    quiet_input = dummy_input.clone()
    quiet_input[:, 2, :] *= 0.01  # simulate a near-flat/dead channel 2
    with torch.no_grad():
        summary_q = quiet_input.mean(dim=2)
        weights_q = attn.squeeze_excite(summary_q)
    print(f"Channel weights with channel 2 silenced: {weights_q[0].numpy()}")
    print("(Note: weights are meaningful only after training -- this is an"
          " untrained, randomly-initialized module, shown here only to"
          " confirm the forward pass responds to input changes.)")
