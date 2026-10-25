"""Which label types does each recording file contain? Uses cached val/test data.
File id is recovered from times = file_id * 1e7 + start_seconds."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "features"))
from cache_provenance import validate_current_artifact  # noqa: E402

validate_current_artifact(ROOT / "checkpoints" / "mlp_eval_data.pt", "mlp_eval_data", root=ROOT,
                          expected_details={"seed": 42})
data = torch.load(ROOT / "checkpoints" / "mlp_eval_data.pt")

print(f"{'patient':<14}{'files':>6}{'w/ inter':>10}{'w/ pre':>8}{'w/ both':>9}{'w/ ictal':>10}")
for split in ("val", "test"):
    d = data[split]
    y = d["y"].numpy()
    g = d["groups"].numpy()
    fid = np.floor_divide(d["times"].numpy(), 1e7).astype(np.int64)
    print(f"--- {split} ---")
    for k, name in enumerate(d["group_names"]):
        m = g == k
        files = np.unique(fid[m])
        n_i = n_p = n_b = n_c = 0
        for f in files:
            labels = set(y[m & (fid == f)].tolist())
            has_i, has_p = 0 in labels, 1 in labels
            n_i += has_i
            n_p += has_p
            n_b += has_i and has_p
            n_c += 2 in labels
        print(f"{name:<14}{len(files):>6}{n_i:>10}{n_p:>8}{n_b:>9}{n_c:>10}")
