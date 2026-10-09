"""
Phase 2 of phased training: train the TCN branch (with Channel Attention
in front of it) on raw windows, 3 classes: 0=interictal, 1=preictal, 2=ictal.

Mirrors train_mlp.py on purpose so the two branches are comparable and can
be fused later without leakage:
  * SAME patient-level train/val/test split (function imported from
    train_mlp.py, then checked against checkpoints/mlp_split.json if present).
  * Training windows get dry-electrode noise augmentation; validation and
    test windows are clean.
  * Same balanced-epoch sampling + inverse-frequency class weights.

TCN-specific decisions
  * Per-epoch validation uses a FIXED SUBSAMPLE of the validation windows
    (all preictal + ictal, plus a random interictal sample), because a full
    pass is expensive. Full validation and the test set are evaluated once,
    at the end, with window-level and event-level reports.
  * Model input is sanitized and clamped to +-INPUT_CLIP (z-scored units)
    inside TCNClassifier. The TCN has no BatchNorm, so one extreme sample
    (e.g. a disconnected electrode) could otherwise blow up training.
    fusion_model.py and the server must apply the same clamp.
  * Mixed precision on CUDA, gradient clipping, non-finite batch skipping,
    LR reduction on plateau, early stopping.

Modes
  python3 train_tcn.py --benchmark    speed test on synthetic data, then exit
  python3 train_tcn.py                train, then evaluate val + test
  python3 train_tcn.py --resume       continue an interrupted run
  python3 train_tcn.py --eval-only    evaluate checkpoints/tcn_classifier.pt

Outputs (checkpoints/):
  tcn_branch.pt       TCN branch weights                     -> fusion
  tcn_attention.pt    Channel Attention weights trained here -> fusion
  tcn_classifier.pt   attention + branch + head              -> evaluation
  tcn_resume.pt       full training state                    -> --resume
  tcn_history.json    per-epoch metrics
  tcn_split.json      patient split used
  tcn_eval_probs.npz  probabilities for full val/test (later threshold work)
"""

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("training", "models", "features", "evaluation"):
    sys.path.insert(0, str(ROOT / _sub))

from dataset import SeizureDataset, INTERICTAL, PREICTAL, ICTAL, FS, WINDOW_SEC  # noqa: E402
from channel_attention import ChannelAttention  # noqa: E402
from tcn_branch import TCNBranch  # noqa: E402
from train_mlp import (patient_level_split, indices_by_label, sample_balanced_epoch,  # noqa: E402
                       class_weights_from_counts, file_order, macro_recall, print_report)

# ---- Config ----
N_CLASSES = 3
SEED = 42
WINDOW_SAMPLES = int(WINDOW_SEC * FS)
INPUT_CLIP = 20.0
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM = 5.0
PATIENCE = 5


# ---------------- model ----------------

class TCNClassifier(nn.Module):
    """ChannelAttention -> TCNBranch (128-d) -> temporary head.
    The head is discarded after this phase; attention and branch are reused."""

    def __init__(self, n_channels: int = 6, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.attention = ChannelAttention(n_channels=n_channels)
        self.tcn_branch = TCNBranch(n_input_channels=n_channels, output_dim=embed_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, N_CLASSES))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=INPUT_CLIP, neginf=-INPUT_CLIP)
        x = x.clamp(-INPUT_CLIP, INPUT_CLIP)
        return self.head(self.tcn_branch(self.attention(x)))


# ---------------- data ----------------

class RawWindowDataset(Dataset):
    """Raw (channels, samples) windows from a SeizureDataset, by index."""

    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[int(self.indices[i])]


def _seed_worker(_worker_id):
    """Each worker (and each epoch) gets its own augmentation RNG."""
    info = get_worker_info()
    info.dataset.base_ds.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def make_loader(ds, indices, batch_size, shuffle, num_workers, device, drop_last=False):
    return DataLoader(RawWindowDataset(ds, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=_seed_worker,
                      pin_memory=(device.type == "cuda"), drop_last=drop_last)


def sample_epoch(by_label, labels_all, rng, cap):
    """Balanced epoch (as in train_mlp.py), optionally capped to `cap` windows."""
    epoch, counts = sample_balanced_epoch(by_label, rng)
    if cap and len(epoch) > cap:
        epoch = rng.choice(epoch, size=cap, replace=False)
        counts = np.bincount(labels_all[epoch], minlength=N_CLASSES).astype(np.float64)
    return epoch, counts


def make_val_subsample(ds, val_idx, n_interictal, seed=SEED):
    by = indices_by_label(ds, val_idx)
    rng = np.random.default_rng(seed)
    inter = by[INTERICTAL]
    if len(inter) > n_interictal:
        inter = rng.choice(inter, size=n_interictal, replace=False)
    return np.sort(np.concatenate([by[PREICTAL], by[ICTAL], inter]))


def check_split_consistency(split_meta, ckpt_dir):
    path = ckpt_dir / "mlp_split.json"
    if not path.exists():
        print("Note: mlp_split.json not found yet; the MLP script computes the same split "
              "deterministically (same seed/fractions).")
        return
    saved = json.loads(path.read_text())["patients"]
    if saved != split_meta:
        raise SystemExit("ERROR: this patient split differs from checkpoints/mlp_split.json. "
                         "Fusion would leak test patients. Make sure both scripts use the "
                         "same data and the same split settings.")
    print("Patient split matches the MLP split (mlp_split.json): OK")


# ---------------- train / evaluate ----------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running, seen, skipped = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        if not torch.isfinite(grad_norm):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        seen += x.size(0)
    return running / max(seen, 1), skipped


@torch.no_grad()
def predict(model, ds, indices, batch_size, num_workers, device, use_amp, desc=""):
    model.eval()
    loader = make_loader(ds, indices, batch_size, False, num_workers, device)
    probs, ys = [], []
    n_batches = len(loader)
    for b, (x, y) in enumerate(loader):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        ys.append(y)
        if desc and ((b + 1) % 200 == 0 or b + 1 == n_batches):
            print(f"    {desc}: {b + 1}/{n_batches} batches", flush=True)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def evaluate_full(model, ds, indices, file_ids, batch_size, num_workers, device, use_amp, name):
    assert np.all(np.diff(indices) > 0), "indices must be strictly increasing (temporal order)"
    print(f"Evaluating {name} ({len(indices):,} windows, clean)...", flush=True)
    probs, y = predict(model, ds, indices, batch_size, num_workers, device, use_amp, desc=name)

    entries = [ds.index[int(i)] for i in indices]
    names = sorted({f"{e['dataset']}/{e['patient_id']}" for e in entries})
    code = {n: k for k, n in enumerate(names)}
    times = np.array([file_ids[e["npy_path"]] * 1e7 + e["start_sample"] / FS for e in entries])
    groups = np.array([code[f"{e['dataset']}/{e['patient_id']}"] for e in entries], dtype=np.int64)
    return {"probs": probs, "y": y, "times": times, "groups": groups, "group_names": names}


def run_training(args, model, device, base_ds, clean_ds, split, ckpt_dir):
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    branch_path = ckpt_dir / "tcn_branch.pt"
    attention_path = ckpt_dir / "tcn_attention.pt"
    classifier_path = ckpt_dir / "tcn_classifier.pt"
    resume_path = ckpt_dir / "tcn_resume.pt"
    history_path = ckpt_dir / "tcn_history.json"

    labels_all = np.fromiter((e["label"] for e in base_ds.index), dtype=np.int64,
                             count=len(base_ds.index))
    by_label_train = indices_by_label(base_ds, split["train"])
    val_sub = make_val_subsample(base_ds, split["val"], args.val_interictal_windows)
    sub_counts = np.bincount(labels_all[val_sub], minlength=N_CLASSES)
    print(f"Per-epoch validation subsample: {len(val_sub):,} windows "
          f"(interictal={sub_counts[0]:,}, preictal={sub_counts[1]:,}, ictal={sub_counts[2]:,})")
    print(f"AMP: {use_amp} | batch size: {args.batch_size} | workers: {args.num_workers} | "
          f"max epochs: {args.epochs} | epoch cap: {args.max_epoch_windows or 'none'}")

    start_epoch, best, bad_epochs, history = 1, -1.0, 0, []
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        bad_epochs, history = ck["bad_epochs"], ck["history"]
        print(f"Resumed from epoch {ck['epoch']} (best so far: {best:.4f})")
    elif args.resume:
        print("--resume given but no tcn_resume.pt found; starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        rng = np.random.default_rng([SEED, epoch])
        epoch_idx, counts = sample_epoch(by_label_train, labels_all, rng, args.max_epoch_windows)
        criterion = nn.CrossEntropyLoss(weight=class_weights_from_counts(counts).to(device))
        loader = make_loader(base_ds, epoch_idx, args.batch_size, True, args.num_workers,
                             device, drop_last=True)

        train_loss, skipped = train_one_epoch(model, loader, criterion, optimizer,
                                              scaler, device, use_amp)
        if skipped:
            print(f"  [!] skipped {skipped} batch(es) (non-finite loss/gradient; "
                  f"a handful is normal with mixed precision)")

        probs, y_sub = predict(model, clean_ds, val_sub, 256, args.num_workers, device, use_amp)
        val_bal, val_recall = macro_recall(y_sub, probs)
        scheduler.step(val_bal)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "train_loss": train_loss, "val_balanced_acc": val_bal,
                        "val_recall": [float(r) for r in val_recall], "lr": lr_now,
                        "epoch_windows": int(len(epoch_idx)), "seconds": time.time() - t0})
        history_path.write_text(json.dumps(history, indent=2))

        print(f"Epoch {epoch:2d}/{args.epochs} | loss={train_loss:.4f} | val_bal_acc={val_bal:.4f} | "
              f"recall[inter={val_recall[0]:.3f} pre={val_recall[1]:.3f} ictal={val_recall[2]:.3f}] | "
              f"lr={lr_now:.1e} | {time.time() - t0:.0f}s", flush=True)

        if val_bal > best + 1e-4:
            best, bad_epochs = val_bal, 0
            torch.save(model.tcn_branch.state_dict(), branch_path)
            torch.save(model.attention.state_dict(), attention_path)
            torch.save(model.state_dict(), classifier_path)
            print(f"  -> new best ({best:.4f}); saved branch + attention + classifier", flush=True)
        else:
            bad_epochs += 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "bad_epochs": bad_epochs,
                    "history": history}, resume_path)

        if bad_epochs >= PATIENCE:
            print(f"  Early stopping: no improvement for {PATIENCE} epochs")
            break

    print(f"\nBest per-epoch validation balanced accuracy (subsample): {best:.4f}")


def final_evaluation(args, model, device, clean_ds, split, file_ids, ckpt_dir):
    use_amp = device.type == "cuda"
    classifier_path = ckpt_dir / "tcn_classifier.pt"
    if not classifier_path.exists():
        raise SystemExit(f"No checkpoint at {classifier_path}. Train first.")
    model.load_state_dict(torch.load(classifier_path, map_location=device))

    saved = {}
    for name, key in (("VALIDATION (full)", "val"), ("TEST (final numbers)", "test")):
        ev = evaluate_full(model, clean_ds, split[key], file_ids, 256,
                           args.num_workers, device, use_amp, name)
        print_report(name, {"y": torch.from_numpy(ev["y"]),
                            "times": torch.from_numpy(ev["times"])}, ev["probs"])
        for field in ("probs", "y", "times", "groups"):
            saved[f"{key}_{field}"] = ev[field]
        saved[f"{key}_group_names"] = np.array(ev["group_names"])

    np.savez_compressed(ckpt_dir / "tcn_eval_probs.npz", **saved)
    print(f"\nSaved evaluation probabilities to {ckpt_dir / 'tcn_eval_probs.npz'}")


# ---------------- benchmark ----------------

def fmt_time(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_benchmark(args, device):
    use_amp = device.type == "cuda"
    model = TCNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device} | torch threads: {torch.get_num_threads()} | AMP: {use_amp}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("(synthetic data; compute only -- real data loading adds on top of this)\n")
    print(f"{'batch':>6} {'train ms/batch':>16} {'train win/s':>13} {'eval ms/batch':>15} {'eval win/s':>12}")

    best_train, best_eval = 0.0, 0.0
    warmup, iters = 2, 5
    for bs in (32, 64, 128):
        x = torch.randn(bs, 6, WINDOW_SAMPLES, device=device)
        y = torch.randint(0, N_CLASSES, (bs,), device=device)

        model.train()
        for i in range(warmup + iters):
            if i == warmup:
                _sync(device)
                t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        _sync(device)
        train_s = (time.perf_counter() - t0) / iters

        model.eval()
        with torch.no_grad():
            for i in range(warmup + iters):
                if i == warmup:
                    _sync(device)
                    t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    model(x)
        _sync(device)
        eval_s = (time.perf_counter() - t0) / iters

        best_train = max(best_train, bs / train_s)
        best_eval = max(best_eval, bs / eval_s)
        print(f"{bs:>6} {train_s * 1000:>16.0f} {bs / train_s:>13.1f} "
              f"{eval_s * 1000:>15.0f} {bs / eval_s:>12.1f}")

    print("\nRough compute-only estimates (best batch size, data loading excluded):")
    print(f"  one training epoch of {args.est_epoch_windows:,} windows : "
          f"{fmt_time(args.est_epoch_windows / best_train)}")
    print(f"  full val+test evaluation of {args.est_eval_windows:,} windows : "
          f"{fmt_time(args.est_eval_windows / best_eval)}")
    print("Adjust the window counts with --est-epoch-windows / --est-eval-windows "
          "(the real counts are printed when training starts).")


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the TCN branch (phase 2).")
    p.add_argument("--benchmark", action="store_true",
                   help="measure speed on synthetic data and exit")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; evaluate checkpoints/tcn_classifier.pt")
    p.add_argument("--resume", action="store_true",
                   help="continue from checkpoints/tcn_resume.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epoch-windows", type=int, default=0,
                   help="cap windows per epoch (0 = no cap)")
    p.add_argument("--val-interictal-windows", type=int, default=30000,
                   help="interictal windows in the per-epoch validation subsample")
    p.add_argument("--est-epoch-windows", type=int, default=250000)
    p.add_argument("--est-eval-windows", type=int, default=500000)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.benchmark:
        run_benchmark(args, device)
        return

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
    check_split_consistency(split_meta, ckpt_dir)
    (ckpt_dir / "tcn_split.json").write_text(json.dumps({"patients": split_meta, "seed": SEED}, indent=2))

    model = TCNClassifier().to(device)
    if not args.eval_only:
        run_training(args, model, device, base_ds, clean_ds, split, ckpt_dir)
    final_evaluation(args, model, device, clean_ds, split, file_ids, ckpt_dir)


if __name__ == "__main__":
    main()
