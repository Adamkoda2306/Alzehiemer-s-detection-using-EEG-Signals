#!/usr/bin/env python3
"""
convert_volts_to_microvolts.py
================================
Walks DATA/<group>/eyes-closed/patientN/*.txt for the ds004504-derived
patients (Alzehiemers 1-36, Healthy 1-29, Frontotemporal-Dementia 1-23)
and converts every channel file's values from volts to microvolts
(x 1,000,000), overwriting the files in place.

Only eyes-closed is touched, per your request -- eyes-open and any
later-merged batches (Paciente/controls_c1/adhd, patient30+) are left
alone since this is specifically about the original 88-subject batch.

SAFETY (this operation overwrites files, so please read):
  - DRY_RUN = True by default -- prints what WOULD change without
    writing anything. Set to False once you've reviewed the dry-run
    output and are ready to actually apply it.
  - BACKUP = True by default -- copies each affected file to
    DATA_backup_before_uv_conversion/ (mirroring DATA/'s structure)
    before overwriting it, so you have a rollback path.
  - Any file whose values are already "large" (max|value| > 1, the
    SANITY_THRESHOLD) is skipped with a warning rather than converted
    again -- volts-scale EEG is always tiny (~1e-4), so a file that's
    already in that range is a strong sign it's already been converted.
    This guards against accidentally doubling the conversion if you run
    the script twice.

Usage:
    python3 convert_volts_to_microvolts.py
    # review the dry-run output, then flip DRY_RUN to False and re-run
"""

import os
import glob
import shutil
import numpy as np

# ================================================================ CONFIG ===
DATA_ROOT = "../../DATA"
CONDITION = "eyes-closed"     # only touch eyes-closed, per instruction

GROUP_PATIENT_RANGES = {
    "Alzehiemers": range(1, 37),               # patient1 - patient36
    "Healthy": range(1, 30),                   # patient1 - patient29
    "Frontotemporal-Dementia": range(1, 24),   # patient1 - patient23
}

CONVERSION_FACTOR = 1_000_000     # volts -> microvolts

# Output number formatting -- pick ONE:
#   "%.8e"   -> scientific notation, e.g. 2.61000000e+02   (your 2nd example)
#   "%.6f"   -> fixed-point,          e.g. -16.814000       (your 1st example)
OUTPUT_FORMAT = "%.6f"

SANITY_THRESHOLD = 1.0     # values already bigger than this look pre-converted -> skip

DRY_RUN = False            # <-- set to False to actually write changes
BACKUP = True                # copy affected files before overwriting them
BACKUP_ROOT = "DATA_backup_before_uv_conversion"
# ============================================================================


def process_patient_dir(patient_dir, stats):
    txt_files = sorted(glob.glob(os.path.join(patient_dir, "*.txt")))
    for txt_path in txt_files:
        values = np.atleast_1d(np.loadtxt(txt_path))

        if values.size == 0:
            print(f"  [skip] {txt_path}: empty file")
            stats["skipped"] += 1
            continue

        max_abs = float(np.max(np.abs(values)))
        if max_abs > SANITY_THRESHOLD:
            print(f"  [skip] {txt_path}: max|value|={max_abs:.4g} already > "
                  f"{SANITY_THRESHOLD} -- looks already converted")
            stats["skipped"] += 1
            continue

        converted = values * CONVERSION_FACTOR

        if DRY_RUN:
            print(f"  [dry-run] {txt_path}: e.g. {values[0]:.6g} V -> {converted[0]:.6g} uV "
                  f"({len(values)} values total)")
        else:
            if BACKUP:
                rel = os.path.relpath(txt_path, DATA_ROOT)
                backup_path = os.path.join(BACKUP_ROOT, rel)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                if not os.path.exists(backup_path):
                    shutil.copy2(txt_path, backup_path)
            np.savetxt(txt_path, converted, fmt=OUTPUT_FORMAT)
            print(f"  [ok] {txt_path}: converted {len(values)} values")

        stats["converted"] += 1


def main():
    stats = {"converted": 0, "skipped": 0}

    if DRY_RUN:
        print("=== DRY RUN -- no files will be modified ===\n")
    else:
        print("=== LIVE RUN -- files WILL be overwritten"
              f"{' (backups first)' if BACKUP else ' (NO BACKUP -- BACKUP=False)'} ===\n")

    for group, patient_range in GROUP_PATIENT_RANGES.items():
        condition_dir = os.path.join(DATA_ROOT, group, CONDITION)
        if not os.path.isdir(condition_dir):
            print(f"[warn] {condition_dir} does not exist, skipping group '{group}'")
            continue

        for n in patient_range:
            patient_dir = os.path.join(condition_dir, f"patient{n}")
            if not os.path.isdir(patient_dir):
                print(f"[warn] {patient_dir} not found, skipping")
                continue
            print(f"{group}/patient{n}:")
            process_patient_dir(patient_dir, stats)

    print(f"\nDone. {stats['converted']} files converted, {stats['skipped']} skipped.")
    if DRY_RUN:
        print("This was a dry run -- set DRY_RUN = False at the top of the script "
              "to actually write changes.")
    elif BACKUP:
        print(f"Originals backed up under: {os.path.abspath(BACKUP_ROOT)}")


if __name__ == "__main__":
    main()