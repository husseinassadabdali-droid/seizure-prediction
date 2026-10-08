"""
Phase 1 of phased training: train the MLP branch alone on hand-crafted
features, with a classification head, before it's used as a
pre-trained initializer in fusion_model.py (Phase 3).

Class imbalance strategy:
  - Per-epoch undersampling of the majority (interictal) class to
    2x the preictal count (all ictal + all preictal windows are kept
    every epoch; interictal is randomly resampled each epoch).
  - Weighted CrossEntropyLoss on top, to handle the remaining
    imbalance between preictal and ictal counts.

Split strategy: patient-level (grouped) train/val split -- entire
patients go to one side only, never split across train and val, to
avoid leaking a patient's signal characteristics across the split
(same principle LOSO-CV will apply more rigorously in evaluation/).

Stability safeguards (added after a NaN-loss training collapse was
traced to near-flat signals in handcrafted_features.py):
  - Gradient norm clipping (max_norm=5.0) after every backward pass --
    a second line of defense against exploding gradients from any
    future NaN/Inf source, not just the one already fixed upstream.
  - A batch-level NaN/Inf guard on the loss itself: if a batch somehow
    still produces a non-finite loss, that batch's update is skipped
    (not applied) rather than corrupting the model's weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))

from dataset import SeizureDataset, INTERICTAL, PREICTAL, ICTAL  # noqa: E402
from mlp_branch import MLPBranch  # noqa: E402
from handcrafted_features import extract_features_from_window, N_FEATURES_PER_CHANNEL  # noqa: E402

N_CLASSES = 3
SEED = 42
GRAD_CLIP_MAX_NORM = 5.0


class MLPClassifier(nn.Module):
    """MLP branch + a temporary classification head for Phase 1 training.
    Only mlp_branch's weights are saved/reused later in fusion_model.py --
    this head is discarded after this phase."""

    def __init__(self, n_input_features: int, output_dim: int = 64):
        super().__init__()
        self.mlp_branch = MLPBranch(n_input_features=n_input_features, output_dim=output_dim)
        self.head = nn.Linear(output_dim, N_CLASSES)

    def forward(self, x):
        return self.head(self.mlp_branch(x))


def patient_level_split(ds: SeizureDataset, val_fraction: float = 0.2, seed: int = SEED):
    """Split by (dataset_name, patient_id) so no patient appears in both splits."""
    patient_keys = sorted({(e["dataset"], e["patient_id"]) for e in ds.index})
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_keys)

    n_val = max(1, int(len(patient_keys) * val_fraction))
    val_patients = set(patient_keys[:n_val])
    train_patients = set(patient_keys[n_val:])

    train_idx, val_idx = [], []
    for i, e in enumerate(ds.index):
        key = (e["dataset"], e["patient_id"])
        (val_idx if key in val_patients else train_idx).append(i)

    print(f"Patients -- train: {len(train_patients)}, val: {len(val_patients)}")
    return train_idx, val_idx


def balanced_epoch_indices(ds: SeizureDataset, all_indices: list,
                            interictal_multiplier: int = 2, rng: np.random.Generator = None):
    """Build one epoch's worth of indices: all preictal + all ictal
    windows from `all_indices`, plus a random sample of interictal
    windows sized interictal_multiplier * n_preictal."""
    rng = rng or np.random.default_rng()

    by_label = {INTERICTAL: [], PREICTAL: [], ICTAL: []}
    for i in all_indices:
        by_label[ds.index[i]["label"]].append(i)

    n_preictal = len(by_label[PREICTAL])
    n_interictal_target = min(len(by_label[INTERICTAL]), interictal_multiplier * n_preictal)
    sampled_interictal = rng.choice(by_label[INTERICTAL], size=n_interictal_target, replace=False).tolist()

    epoch_indices = by_label[PREICTAL] + by_label[ICTAL] + sampled_interictal
    rng.shuffle(epoch_indices)
    return epoch_indices


def compute_class_weights(ds: SeizureDataset, indices: list) -> torch.Tensor:
    """Inverse-frequency class weights based on the GIVEN indices (e.g.
    one balanced epoch's composition), normalized to mean 1.0."""
    counts = np.zeros(N_CLASSES)
    for i in indices:
        counts[ds.index[i]["label"]] += 1
    counts[counts == 0] = 1  # avoid div-by-zero
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FeatureDataset(torch.utils.data.Dataset):
    """Wraps SeizureDataset windows, extracting hand-crafted features
    on the fly (so train_mlp.py never needs raw TCN-style windows)."""

    def __init__(self, base_ds: SeizureDataset, indices: list, fs: int = 250):
        self.base_ds = base_ds
        self.indices = indices
        self.fs = fs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        window_tensor, label = self.base_ds[self.indices[i]]
        features = extract_features_from_window(window_tensor.numpy(), fs=self.fs)
        # Belt-and-suspenders: guarantee a finite tensor even if an
        # unexpected edge case slips past handcrafted_features.py's
        # own guards.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(features), label


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = np.zeros(N_CLASSES)
    per_class_total = np.zeros(N_CLASSES)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            for c in range(N_CLASSES):
                mask = y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()

    acc = correct / max(total, 1)
    per_class_recall = per_class_correct / np.maximum(per_class_total, 1)
    macro_recall = per_class_recall.mean()
    return acc, macro_recall, per_class_recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    configs = [
        {"name": "chbmit", "processed_dir": project_root / "data/processed/chbmit",
         "raw_dir": project_root / "data/raw/chbmit", "notch_freq": 60.0},
        {"name": "siena", "processed_dir": project_root / "data/processed/siena",
         "raw_dir": project_root / "data/raw/siena", "notch_freq": 50.0},
    ]

    base_ds = SeizureDataset(configs, apply_augmentation=True)
    train_idx, val_idx = patient_level_split(base_ds, val_fraction=0.2, seed=SEED)

    val_ds = FeatureDataset(base_ds, val_idx)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = MLPClassifier(n_input_features=6 * N_FEATURES_PER_CHANNEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    rng = np.random.default_rng(SEED)
    n_epochs = 15
    best_macro_recall = -1.0
    checkpoint_path = project_root / "checkpoints" / "mlp_branch.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        epoch_train_idx = balanced_epoch_indices(base_ds, train_idx, interictal_multiplier=2, rng=rng)
        class_weights = compute_class_weights(base_ds, epoch_train_idx).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_ds = FeatureDataset(base_ds, epoch_train_idx)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)

        model.train()
        running_loss = 0.0
        n_samples_seen = 0
        n_batches_skipped = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if not torch.isfinite(loss):
                # Skip this batch entirely rather than let a non-finite
                # loss corrupt the model's weights via backward().
                n_batches_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_samples_seen += x.size(0)

        train_loss = running_loss / max(n_samples_seen, 1)
        if n_batches_skipped > 0:
            print(f"  [!] Skipped {n_batches_skipped} batch(es) with non-finite loss this epoch")

        val_acc, val_macro_recall, val_per_class_recall = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:2d}/{n_epochs} | train_loss={train_loss:.4f} | "
              f"val_acc={val_acc:.4f} | val_macro_recall={val_macro_recall:.4f} | "
              f"per_class_recall=[inter={val_per_class_recall[0]:.3f}, "
              f"pre={val_per_class_recall[1]:.3f}, ictal={val_per_class_recall[2]:.3f}]")

        if val_macro_recall > best_macro_recall:
            best_macro_recall = val_macro_recall
            torch.save(model.mlp_branch.state_dict(), checkpoint_path)
            print(f"  -> New best (macro_recall={best_macro_recall:.4f}), saved to {checkpoint_path}")

    print(f"\nTraining complete. Best val macro_recall: {best_macro_recall:.4f}")
    print(f"Best MLP branch weights saved at: {checkpoint_path}")


if __name__ == "__main__":
    main()
