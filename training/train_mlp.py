"""
Phase 1 of phased training: train the MLP branch alone on hand-crafted
features (with a temporary classification head), 3 classes:
0=interictal, 1=preictal, 2=ictal.

Design decisions
  * Patient-level 3-way split (train/val/test): whole patients go to one
    side only. chb21 is the same subject as chb01 (recorded 1.5 years
    later, per the CHB-MIT documentation), so they are grouped together.
    Val is used for model selection / early stopping; test is touched
    only once, at the end, for the final numbers.
  * Training windows get dry-electrode noise augmentation; validation
    and test windows are CLEAN (deterministic, reproducible metrics).
  * Class imbalance: each epoch keeps all preictal + ictal windows and
    a random 2x-preictal sample of interictal windows, plus inverse-
    frequency class weights.
  * Robust feature scaling (median/IQR from a training sample) is stored
    inside MLPBranch so the saved weights are self-contained.
  * Stability: gradient clipping, non-finite-loss batch skipping,
    drop_last (BatchNorm needs >1 sample), fixed seeds, per-worker RNG
    reseeding for augmentation noise.

Outputs (checkpoints/):
  mlp_branch.pt        branch weights + normalization stats  -> fusion
  mlp_classifier.pt    branch + head                         -> evaluation
  mlp_split.json       which patients are in train/val/test
  mlp_eval_data.pt     cached val/test features, labels, times, groups
  mlp_history.json     per-epoch metrics
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("training", "models", "features", "evaluation"):
    sys.path.insert(0, str(ROOT / _sub))

from dataset import SeizureDataset, INTERICTAL, PREICTAL, ICTAL, FS, STEP_SEC  # noqa: E402
from mlp_branch import MLPBranch  # noqa: E402
from handcrafted_features import extract_features_from_window, N_FEATURES_PER_CHANNEL  # noqa: E402
from metrics import (confusion_matrix, per_class_metrics, window_level_report,  # noqa: E402
                     make_alarm, event_metrics, CLASS_NAMES)

# ---- Config ----
N_CLASSES = 3
N_FEATURES = 6 * N_FEATURES_PER_CHANNEL
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
N_EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 256
NUM_WORKERS = 4            # lower to 2 if the server runs out of RAM
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 5.0
INTERICTAL_MULTIPLIER = 2
STATS_SAMPLE_SIZE = 20000
ALARM_SETTINGS = [(0.5, 1), (0.8, 3), (0.9, 5)]   # (threshold, min consecutive windows)

# Same physical subject recorded twice -> must stay on the same side of the split
GROUP_ALIASES = {("chbmit", "chb21"): ("chbmit", "chb01")}


# ---------------- model ----------------

class MLPClassifier(nn.Module):
    """MLPBranch + a temporary head. Only the branch is reused later."""

    def __init__(self, n_input_features: int, output_dim: int = 64):
        super().__init__()
        self.mlp_branch = MLPBranch(n_input_features=n_input_features, output_dim=output_dim)
        self.head = nn.Linear(output_dim, N_CLASSES)

    def forward(self, x):
        return self.head(self.mlp_branch(x))


# ---------------- data ----------------

class FeatureDataset(Dataset):
    """Windows from a SeizureDataset -> hand-crafted feature vectors."""

    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        window, label = self.base_ds[int(self.indices[i])]
        feats = extract_features_from_window(window.numpy(), fs=FS)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(feats), label


def _seed_worker(_worker_id):
    """Give every worker (and every epoch) its own augmentation RNG."""
    info = get_worker_info()
    info.dataset.base_ds.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def collect(dataset, label: str = ""):
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=NUM_WORKERS, worker_init_fn=_seed_worker)
    xs, ys = [], []
    n_batches = len(loader)
    for b, (x, y) in enumerate(loader):
        xs.append(x)
        ys.append(y)
        if (b + 1) % 100 == 0 or b + 1 == n_batches:
            print(f"    {label}: {b + 1}/{n_batches} batches", flush=True)
    return torch.cat(xs), torch.cat(ys)


def group_key(entry):
    key = (entry["dataset"], entry["patient_id"])
    return GROUP_ALIASES.get(key, key)


def patient_level_split(ds):
    groups = sorted({group_key(e) for e in ds.index})
    rng = np.random.default_rng(SEED)
    groups = [groups[i] for i in rng.permutation(len(groups))]

    n_val = max(1, round(len(groups) * VAL_FRACTION))
    n_test = max(1, round(len(groups) * TEST_FRACTION))
    val_g = set(groups[:n_val])
    test_g = set(groups[n_val:n_val + n_test])
    train_g = set(groups[n_val + n_test:])

    split = {"train": [], "val": [], "test": []}
    for i, e in enumerate(ds.index):
        g = group_key(e)
        name = "val" if g in val_g else "test" if g in test_g else "train"
        split[name].append(i)

    meta = {name: sorted(f"{d}/{p}" for d, p in g)
            for name, g in (("train", train_g), ("val", val_g), ("test", test_g))}
    return {k: np.asarray(v, dtype=np.int64) for k, v in split.items()}, meta


def indices_by_label(ds, indices):
    by = {INTERICTAL: [], PREICTAL: [], ICTAL: []}
    for i in indices:
        by[ds.index[int(i)]["label"]].append(int(i))
    return {k: np.asarray(v, dtype=np.int64) for k, v in by.items()}


def sample_balanced_epoch(by_label, rng):
    n_pre = len(by_label[PREICTAL])
    n_inter = min(len(by_label[INTERICTAL]), INTERICTAL_MULTIPLIER * n_pre)
    inter = rng.choice(by_label[INTERICTAL], size=n_inter, replace=False)
    epoch = np.concatenate([by_label[PREICTAL], by_label[ICTAL], inter])
    rng.shuffle(epoch)
    counts = np.array([n_inter, n_pre, len(by_label[ICTAL])], dtype=np.float64)  # label order 0,1,2
    return epoch, counts


def class_weights_from_counts(counts):
    w = 1.0 / np.maximum(counts, 1.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def file_order(ds):
    """File ids in order of first appearance, so times stay monotonic along the index."""
    ids = {}
    for e in ds.index:
        ids.setdefault(e["npy_path"], len(ids))
    return ids


def build_eval_data(clean_ds, indices, name, file_ids):
    assert np.all(np.diff(indices) > 0), "indices must be strictly increasing (temporal order)"
    print(f"Extracting {name} features ({len(indices):,} windows, clean)...", flush=True)
    X, y = collect(FeatureDataset(clean_ds, indices), label=name)

    entries = [clean_ds.index[int(i)] for i in indices]
    names = sorted({f"{e['dataset']}/{e['patient_id']}" for e in entries})
    code = {n: k for k, n in enumerate(names)}
    times = np.array([e["time_sec"] for e in entries], dtype=np.float64)
    groups = np.array([code[f"{e['dataset']}/{e['patient_id']}"] for e in entries], dtype=np.int64)
    same_patient = groups[1:] == groups[:-1]
    assert np.all(np.diff(times)[same_patient] >= 0), "window times must not decrease within a patient"
    return {"X": X, "y": y, "times": torch.from_numpy(times),
            "groups": torch.from_numpy(groups), "group_names": names}


def estimate_feature_stats(base_ds, indices, rng):
    """Robust center/scale (median, IQR/1.349) from an augmented TRAIN sample."""
    n = min(STATS_SAMPLE_SIZE, len(indices))
    sample = rng.choice(indices, size=n, replace=False)
    print(f"Estimating feature statistics from {n:,} training windows...", flush=True)
    X, _ = collect(FeatureDataset(base_ds, sample), label="stats")
    X = X.numpy()
    q25, med, q75 = np.percentile(X, [25, 50, 75], axis=0)
    robust = (q75 - q25) / 1.349
    std = X.std(axis=0)
    scale = np.where(robust > 1e-8, robust, np.where(std > 1e-8, std, 1.0))
    return med.astype(np.float32), scale.astype(np.float32)


# ---------------- evaluation ----------------

@torch.no_grad()
def predict_proba(model, X, device, batch_size=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.softmax(model(X[i:i + batch_size].to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def macro_recall(y, probs):
    cm = confusion_matrix(y, probs.argmax(axis=1))
    recall, _, _, support = per_class_metrics(cm)
    return float(recall[support > 0].mean()), recall


def print_report(name, data, probs):
    y = data["y"].numpy()
    rep = window_level_report(y, probs)
    print(f"\n===== {name}: window-level =====")
    print(f"balanced_accuracy={rep['balanced_accuracy']:.4f}  macro_f1={rep['macro_f1']:.4f}  "
          f"alarm_auc={rep['alarm_auc']:.4f}")
    for c in CLASS_NAMES:
        print(f"  {c:<11} recall={rep['recall'][c]:.3f}  precision={rep['precision'][c]:.3f}  "
              f"support={rep['support'][c]}")
    print("  confusion matrix (rows=true, cols=predicted):")
    print(rep["confusion_matrix"])

    print(f"===== {name}: event-level (alarm score = 1 - P(interictal)) =====")
    times = data["times"].numpy()
    for thr, k in ALARM_SETTINGS:
        alarm = make_alarm(probs, threshold=thr, min_consecutive=k)
        ev = event_metrics(y, alarm, times, step_sec=STEP_SEC, max_gap_sec=60.0)
        print(f"  thr={thr:.2f} consec={k}: seizures={ev['n_seizures']} "
              f"early={ev['predicted_in_advance']} during_only={ev['detected_only_during_seizure']} "
              f"missed={ev['missed']} mean_lead={ev['mean_lead_time_sec']:.0f}s "
              f"false_alarms/h={ev['false_alarms_per_hour']:.2f}")


# ---------------- main ----------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    configs = [
        {"name": "chbmit", "processed_dir": ROOT / "data/processed/chbmit",
         "raw_dir": ROOT / "data/raw/chbmit", "notch_freq": 60.0},
        {"name": "siena", "processed_dir": ROOT / "data/processed/siena",
         "raw_dir": ROOT / "data/raw/siena", "notch_freq": 50.0},
    ]

    base_ds = SeizureDataset(configs, apply_augmentation=True)     # training view (noisy)
    clean_ds = copy.copy(base_ds)                                   # shares the index, no noise
    clean_ds.apply_augmentation = False
    file_ids = file_order(base_ds)

    split, split_meta = patient_level_split(base_ds)
    for name in ("train", "val", "test"):
        print(f"{name:<5}: {len(split_meta[name]):2d} patient groups, {len(split[name]):>9,} windows")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    branch_path = ckpt_dir / "mlp_branch.pt"
    classifier_path = ckpt_dir / "mlp_classifier.pt"
    (ckpt_dir / "mlp_split.json").write_text(json.dumps(
        {"patients": split_meta, "seed": SEED, "val_fraction": VAL_FRACTION,
         "test_fraction": TEST_FRACTION}, indent=2))

    rng = np.random.default_rng(SEED)
    by_label_train = indices_by_label(base_ds, split["train"])

    val_data = build_eval_data(clean_ds, split["val"], "validation", file_ids)
    y_val = val_data["y"].numpy()

    first_epoch, _ = sample_balanced_epoch(by_label_train, rng)
    center, scale = estimate_feature_stats(base_ds, first_epoch, rng)

    model = MLPClassifier(N_FEATURES).to(device)
    model.mlp_branch.set_feature_stats(center, scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        epoch_idx, counts = sample_balanced_epoch(by_label_train, rng)
        criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(counts).to(device))
        loader = DataLoader(FeatureDataset(base_ds, epoch_idx), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS,
                            worker_init_fn=_seed_worker, drop_last=True)

        model.train()
        running, seen, skipped = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(seen, 1)
        if skipped:
            print(f"  [!] skipped {skipped} batch(es) with non-finite loss")

        probs = predict_proba(model, val_data["X"], device)
        val_bal, val_recall = macro_recall(y_val, probs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_balanced_acc": val_bal,
                        "val_recall": [float(r) for r in val_recall]})

        print(f"Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | val_bal_acc={val_bal:.4f} | "
              f"recall[inter={val_recall[0]:.3f} pre={val_recall[1]:.3f} ictal={val_recall[2]:.3f}] | "
              f"{time.time() - t0:.0f}s", flush=True)

        if val_bal > best + 1e-4:
            best, bad_epochs = val_bal, 0
            torch.save(model.mlp_branch.state_dict(), branch_path)
            torch.save(model.state_dict(), classifier_path)
            print(f"  -> new best ({best:.4f}); saved branch + classifier", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"  Early stopping: no improvement for {PATIENCE} epochs")
                break

    (ckpt_dir / "mlp_history.json").write_text(json.dumps(history, indent=2))

    # ---- final evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(classifier_path, map_location=device))
    print(f"\nBest validation balanced accuracy: {best:.4f}")
    print_report("VALIDATION", val_data, predict_proba(model, val_data["X"], device))

    test_data = build_eval_data(clean_ds, split["test"], "test", file_ids)
    print_report("TEST (final numbers)", test_data, predict_proba(model, test_data["X"], device))
    torch.save({"val": val_data, "test": test_data}, ckpt_dir / "mlp_eval_data.pt")
    print(f"\nSaved: {branch_path.name}, {classifier_path.name}, mlp_split.json, "
          f"mlp_eval_data.pt, mlp_history.json  (in {ckpt_dir})")


if __name__ == "__main__":
    main()
