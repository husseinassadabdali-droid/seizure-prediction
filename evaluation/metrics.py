"""
Evaluation metrics for 3-class seizure prediction/detection
(0=interictal, 1=preictal, 2=ictal).

Two levels:
  1. Window-level: confusion matrix, per-class recall/precision/F1,
     balanced accuracy, one-vs-rest ROC-AUC, and an "alarm AUC"
     (interictal vs. preictal+ictal).
  2. Event-level (what actually matters for the device): how many
     seizures were warned in advance (and by how many seconds), how
     many were only caught during the seizure, how many were missed,
     and false alarms per hour of interictal time.

Event-level metrics need windows in temporal order with their start
times in seconds. When concatenating several recordings, add a large
offset per recording (e.g. file_index * 1e7) to `times` so runs never
bridge two different files.

Raw accuracy is deliberately not the headline number: with ~90%
interictal windows it is misleading.
"""

import numpy as np
from scipy.stats import rankdata

INTERICTAL, PREICTAL, ICTAL = 0, 1, 2
CLASS_NAMES = ["interictal", "preictal", "ictal"]


# ---------------- window-level ----------------

def confusion_matrix(y_true, y_pred, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_metrics(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return recall, precision, f1, support


def roc_auc_binary(scores, positives) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC; NaN if a class is absent."""
    positives = np.asarray(positives).astype(bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def window_level_report(y_true, probs) -> dict:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    recall, precision, f1, support = per_class_metrics(cm)
    present = support > 0

    return {
        "confusion_matrix": cm,
        "recall": dict(zip(CLASS_NAMES, recall)),
        "precision": dict(zip(CLASS_NAMES, precision)),
        "f1": dict(zip(CLASS_NAMES, f1)),
        "support": dict(zip(CLASS_NAMES, support.astype(int))),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "auc_ovr": {CLASS_NAMES[c]: roc_auc_binary(probs[:, c], y_true == c) for c in range(3)},
        "alarm_auc": roc_auc_binary(1.0 - probs[:, 0], y_true != INTERICTAL),
    }


# ---------------- event-level ----------------

def make_alarm(probs, threshold: float = 0.5, min_consecutive: int = 1) -> np.ndarray:
    """
    Alarm score = P(preictal) + P(ictal) = 1 - P(interictal).
    min_consecutive > 1 requires that many consecutive windows above the
    threshold before the alarm fires (suppresses isolated false alarms).
    Note: the consecutive check ignores time gaps between windows.
    """
    raw = (1.0 - np.asarray(probs)[:, 0]) >= threshold
    if min_consecutive <= 1:
        return raw
    counts = np.convolve(raw.astype(int), np.ones(min_consecutive, dtype=int), mode="full")[: len(raw)]
    return counts >= min_consecutive


def _runs(mask, times, max_gap):
    """(start, end) index pairs of runs where mask is True and
    consecutive windows are no more than max_gap seconds apart."""
    runs, start = [], None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
            elif times[i] - times[i - 1] > max_gap:
                runs.append((start, i - 1))
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def event_metrics(y_true, alarm, times, step_sec: float = 2.0) -> dict:
    y_true = np.asarray(y_true)
    alarm = np.asarray(alarm, dtype=bool)
    times = np.asarray(times, dtype=float)
    max_gap = 1.5 * step_sec

    n_early = n_ictal_only = n_missed = 0
    lead_times = []

    seizures = _runs(y_true == ICTAL, times, max_gap)
    for (s, e) in seizures:
        # contiguous preictal block immediately before this seizure
        j = s - 1
        while j >= 0 and y_true[j] == PREICTAL and times[j + 1] - times[j] <= max_gap:
            j -= 1
        pre_start = j + 1

        early = np.flatnonzero(alarm[pre_start:s])
        if early.size > 0:
            n_early += 1
            lead_times.append(times[s] - times[pre_start + early[0]])
        elif alarm[s:e + 1].any():
            n_ictal_only += 1
        else:
            n_missed += 1

    false_alarms = sum(
        1 for (s, e) in _runs(alarm, times, max_gap)
        if np.all(y_true[s:e + 1] == INTERICTAL)
    )
    interictal_hours = (y_true == INTERICTAL).sum() * step_sec / 3600.0

    n_seiz = len(seizures)
    return {
        "n_seizures": n_seiz,
        "predicted_in_advance": n_early,
        "detected_only_during_seizure": n_ictal_only,
        "missed": n_missed,
        "event_sensitivity_prediction": n_early / n_seiz if n_seiz else float("nan"),
        "event_sensitivity_any": (n_early + n_ictal_only) / n_seiz if n_seiz else float("nan"),
        "mean_lead_time_sec": float(np.mean(lead_times)) if lead_times else float("nan"),
        "false_alarms": false_alarms,
        "interictal_hours": float(interictal_hours),
        "false_alarms_per_hour": false_alarms / interictal_hours if interictal_hours > 0 else float("nan"),
    }


if __name__ == "__main__":
    # Synthetic self-test with a known answer (lightweight).
    step = 2.0
    y = np.array([INTERICTAL] * 200 + [PREICTAL] * 30 + [ICTAL] * 10 + [INTERICTAL] * 100)
    times = np.arange(len(y)) * step

    alarm = np.zeros(len(y), dtype=bool)
    alarm[50:53] = True      # one false alarm during interictal
    alarm[215:240] = True    # fires 15 windows into the preictal block

    ev = event_metrics(y, alarm, times, step_sec=step)
    for k, v in ev.items():
        print(f"{k:<32} {v}")

    assert ev["n_seizures"] == 1 and ev["predicted_in_advance"] == 1
    assert abs(ev["mean_lead_time_sec"] - 30.0) < 1e-9
    assert ev["false_alarms"] == 1
    assert abs(ev["false_alarms_per_hour"] - 6.0) < 1e-9
    print("\nEvent-level self-test: OK")

    # Window-level smoke test with random probabilities
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(y), 3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    rep = window_level_report(y, probs)
    print(f"Random-probs balanced accuracy (expect ~0.33): {rep['balanced_accuracy']:.3f}")

    # A perfect scorer must get alarm AUC = 1.0
    perfect = np.zeros((len(y), 3)); perfect[np.arange(len(y)), y] = 1.0
    print(f"Perfect-probs alarm AUC (expect 1.0): {window_level_report(y, perfect)['alarm_auc']:.3f}")
