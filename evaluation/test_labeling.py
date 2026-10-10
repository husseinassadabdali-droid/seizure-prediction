"""Fast checks (seconds, no data) for the labeling logic in training/dataset.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from dataset import (label_windows_for_file, place_on_timeline, get_patient_id,  # noqa: E402
                     FS, WINDOW_SEC, STEP_SEC, PREICTAL_SEC, POSTICTAL_SEC, BUFFER_SEC,
                     INTERICTAL, PREICTAL, ICTAL)

WIN, STEP = int(WINDOW_SEC * FS), int(STEP_SEC * FS)


def reference_label(ws, we, seizures):
    """Plain scalar implementation of the documented priority rules."""
    for ss, se in seizures:
        if we > ss and ws < se:
            return ICTAL
    for ss, se in seizures:
        if se < we <= se + POSTICTAL_SEC:
            return -1
    for ss, se in seizures:
        if ss - PREICTAL_SEC <= ws < ss:
            return PREICTAL
    for ss, se in seizures:
        if (ss - BUFFER_SEC <= ws < ss - PREICTAL_SEC) or (se + POSTICTAL_SEC < we <= se + BUFFER_SEC):
            return -1
    return INTERICTAL


def label_dict(seizures, n_samples, offset=0.0):
    return dict(label_windows_for_file(seizures, n_samples, offset_sec=offset))


# A. vectorized == scalar reference on random configurations
rng = np.random.default_rng(1)
n_samples = int(2.5 * 3600 * FS)
for trial in range(40):
    offset = float(rng.choice([0.0, 5000.0, 123456.5]))
    seizures = []
    for _ in range(int(rng.integers(0, 5))):
        s = offset + float(rng.uniform(-20000, 30000))
        seizures.append((s, s + float(rng.uniform(10, 200))))
    got = label_dict(seizures, n_samples, offset)
    expected = {}
    for start in range(0, n_samples - WIN + 1, STEP):
        ws = offset + start / FS
        lab = reference_label(ws, ws + WINDOW_SEC, seizures)
        if lab != -1:
            expected[start] = lab
    assert got == expected, f"mismatch on trial {trial}"
print("A. vectorized labels == scalar reference on 40 random cases: OK")

# B. two seizures 16 min apart (the v1 bug): the second one's windows must stay ictal
two = [(1000.0, 1060.0), (2000.0, 2060.0)]
got = label_dict(two, int(4000 * FS))
assert got[int(2000 * FS)] == ICTAL and got[int(2030 * FS)] == ICTAL
assert int(1500 * FS) not in got  # postictal of #1 AND preictal of #2 -> excluded
print("B. second seizure keeps its ictal windows; ambiguous window excluded: OK")

# C. stitching: seizure at t=3700 on the patient timeline, file 1 spans 0..3600
across = label_dict([(3700.0, 3760.0)], int(3600 * FS), offset=0.0)
within = label_dict([], int(3600 * FS))
assert across[int(2000 * FS)] == PREICTAL      # 1700 s before onset, in the previous file
assert within[int(2000 * FS)] == INTERICTAL    # what within-file labeling would say
assert int(1000 * FS) not in across            # buffer zone
print("C. preictal found in the PREVIOUS file when files are stitched: OK")

# D. patient ids
assert get_patient_id("chb17a_03", "chbmit") == "chb17"
assert get_patient_id("chb17c_20", "chbmit") == "chb17"
assert get_patient_id("chb01_03", "chbmit") == "chb01"
assert get_patient_id("chb24_01", "chbmit") == "chb24"
assert get_patient_id("PN10-7.8.9", "siena") == "PN10"
print("D. patient ids (chb17a/b/c merge): OK")

# E. timeline placement
wrap = [("f1", 22 * 3600), ("f2", 23 * 3600 + 1800), ("f3", 1800), ("f4", 5400)]
assert list(place_on_timeline(wrap).values()) == [79200, 84600, 88200, 91800]
mono = [("f1", 22 * 3600), ("f2", 25 * 3600), ("f3", 26 * 3600)]
assert list(place_on_timeline(mono).values()) == [79200, 90000, 93600]
assert place_on_timeline([("x", None), ("y", 100)]) == {"y": 100}
print("E. timeline placement (midnight wrap, hours >= 24, missing times): OK")

print("\nALL LABELING TESTS PASSED")
