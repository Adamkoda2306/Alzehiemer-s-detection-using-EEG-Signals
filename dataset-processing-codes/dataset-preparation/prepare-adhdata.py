#!/usr/bin/env python3
"""
extract_adhd_controls.py
==========================
Extracts the "Control" (healthy) subjects from adhdata.csv and writes
them out in the same patientN/channel.txt format used elsewhere.

No downsampling is needed -- this data is already sampled at 128 Hz.
All of these recordings are eyes-closed (per your confirmation).

Channel names are remapped from the modern 10-20 nomenclature used in
this CSV (T7, T8, P7, P8) to the older nomenclature used everywhere
else in this project (T3, T4, T5, T6), so channel filenames stay
consistent across every patient folder in the combined dataset:
    T7 -> T3   T8 -> T4   P7 -> T5   P8 -> T6
(these are the same physical electrode positions under the two
different 10-20 naming eras -- not a guess, this is a standard,
well-documented equivalence.)

Output goes to a NEW, separate folder (DST_ROOT below) -- not merged
into DATA/ or DATA_new_controls/. Patient numbering starts at
START_PATIENT (42) independently within this new folder, since it's a
parallel staging area, same as the controls_c1_new.mat batch was.

Usage:
    pip install pandas numpy --break-system-packages
    python3 extract_adhd_controls.py
"""

import os
import csv
import numpy as np
import pandas as pd

# ================================================================ CONFIG ===
CSV_PATH = "adhdata.csv"
DST_ROOT = "DATA_new_adhd_controls"    # separate output folder

GROUP_FOLDER = "Healthy"
CONDITION_FOLDER = "eyes-closed"       # confirmed: all Control subjects are eyes-closed

START_PATIENT = 42                     # local numbering, scoped to this new folder

# Rename modern 10-20 names to the older nomenclature used elsewhere in
# this project, so every patient folder across every batch has matching
# channel filenames.
CHANNEL_RENAME = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}

# Source column names to extract, in the order they appear in the CSV
SOURCE_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

TXT_FMT = "%.8e"
# ============================================================================


def main():
    out_condition_dir = os.path.join(DST_ROOT, GROUP_FOLDER, CONDITION_FOLDER)
    os.makedirs(out_condition_dir, exist_ok=True)

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    missing_cols = [c for c in SOURCE_CHANNELS + ["Class", "ID"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}")

    controls = df[df["Class"] == "Control"]
    print(f"Total rows: {len(df)}  |  Control rows: {len(controls)}")

    # Deterministic, reproducible patient ordering -- alphabetical by ID.
    # If you'd rather number them in first-appearance order instead, use:
    #   unique_ids = list(dict.fromkeys(controls["ID"]))
    unique_ids = sorted(controls["ID"].unique())
    print(f"Unique Control patient IDs: {len(unique_ids)}\n")

    manifest_rows = []

    for rank, subject_id in enumerate(unique_ids):
        patient_num = START_PATIENT + rank
        patient_id = f"patient{patient_num}"

        subject_df = controls[controls["ID"] == subject_id]   # preserves original row order
        n_samples = len(subject_df)

        out_dir = os.path.join(out_condition_dir, patient_id)
        os.makedirs(out_dir, exist_ok=True)

        for src_ch in SOURCE_CHANNELS:
            dst_ch = CHANNEL_RENAME.get(src_ch, src_ch)
            values = subject_df[src_ch].to_numpy(dtype=np.float64)
            out_path = os.path.join(out_dir, f"{dst_ch}.txt")
            np.savetxt(out_path, values, fmt=TXT_FMT)

        print(f"[ok] ID={subject_id!r:<8} -> {patient_id}  ({n_samples} samples)")

        manifest_rows.append({
            "orig_id": subject_id, "patient_id": patient_id, "group": GROUP_FOLDER,
            "condition": CONDITION_FOLDER, "n_samples": n_samples,
        })

    manifest_path = os.path.join(DST_ROOT, "manifest_adhd_controls.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\nManifest written to: {manifest_path}")
    print(f"Output written to:   {os.path.abspath(DST_ROOT)}")
    if manifest_rows:
        print(f"Patient range: patient{START_PATIENT} .. patient{START_PATIENT + len(unique_ids) - 1}")


if __name__ == "__main__":
    main()