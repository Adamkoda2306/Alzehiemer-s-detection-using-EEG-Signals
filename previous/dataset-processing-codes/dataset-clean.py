#!/usr/bin/env python3
"""
dataset-clean.py
=================

Walks the `dataset-processed/` export folder (88 subjects, sub-001 .. sub-088),
for each subject:

  1. Reads the `*_metadata.csv` to find the diagnostic `group_code` (A/F/C).
  2. Reads the matching `*_channels_timeseries.csv` (19 channels @ 500 Hz).
  3. Downsamples every channel 500 Hz -> 128 Hz using polyphase resampling
     (anti-aliased, good for EEG — not a naive "take every Nth sample").
  4. Works out eyes-closed / eyes-open from the filename (`task-eyesclosed`
     / `task-eyesopen`).
  5. Writes each channel to its own .txt file (one sample per line) under:

        DATA/
        ├── Alzehiemers/
        │   ├── eyes-closed/patient1/Fp1.txt, Fp2.txt, ...
        │   └── eyes-open/...
        ├── Healthy/
        │   ├── eyes-closed/...
        │   └── eyes-open/...
        └── Frontotemporal-Dementia/      <- see GROUP_FOLDER note below
            ├── eyes-closed/...
            └── eyes-open/...

  6. Writes DATA/manifest.csv logging original subject id -> patient id ->
     group -> task -> sample counts, so you can always trace a "patientN"
     folder back to the real subject.

Requirements:
    pip install pandas numpy scipy
"""

import os
import re
import glob
import csv
from math import gcd

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

# ============================================================= CONFIG =====

SRC_ROOT = "../../dataset-processed"   # input root: contains sub-XXX_eeg/exports/...
DST_ROOT = "DATA"                # output root

ORIG_FS = 500                    # original sampling rate (Hz)
TARGET_FS = 128                  # target sampling rate (Hz)

# group_code -> output folder name.
# NOTE: verified against the source dataset (ds004504, Miltiadous et al. 2023):
#   A = Alzheimer's Disease (36 subjects)
#   F = Frontotemporal Dementia (23 subjects)   <-- NOT healthy
#   C = Healthy / Control (29 subjects)         <-- NOT mild-Alzheimer's
# If you deliberately want your original A/C/F -> Alzheimer's/Mild/Healthy
# scheme instead, just edit the dict below.
GROUP_FOLDER = {
    "A": "Alzehiemers",
    "F": "Frontotemporal-Dementia",
    "C": "Healthy",
}

TXT_FMT = "%.8e"   # numeric format for the per-channel .txt files

# ============================================================================


def get_task_label(filename: str) -> str:
    """Return 'eyes-closed' / 'eyes-open' based on the filename's task tag."""
    fname = filename.lower()
    if "eyesclosed" in fname:
        return "eyes-closed"
    if "eyesopen" in fname:
        return "eyes-open"
    raise ValueError(f"Could not determine eyes-open/closed task from filename: {filename}")


def downsample(data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Polyphase downsample data (n_samples, n_channels) orig_fs -> target_fs."""
    g = gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g   # 128,500 -> up=32, down=125
    return resample_poly(data, up, down, axis=0)


def find_subject_exports(subject_dir: str):
    """Return list of (timeseries_csv, metadata_csv) pairs found in a subject folder."""
    exports_dir = os.path.join(subject_dir, "exports")
    ts_files = sorted(glob.glob(os.path.join(exports_dir, "*_channels_timeseries.csv")))
    pairs = []
    for ts_path in ts_files:
        meta_path = ts_path.replace("_channels_timeseries.csv", "_metadata.csv")
        if os.path.exists(meta_path):
            pairs.append((ts_path, meta_path))
        else:
            print(f"  [skip] no metadata found for {os.path.basename(ts_path)}")
    return pairs


def process_subject(subject_dir, group_counters, subject_patient_id, manifest_rows):
    subject_name = os.path.basename(subject_dir)
    pairs = find_subject_exports(subject_dir)
    if not pairs:
        print(f"[skip] {subject_name}: no export CSVs found")
        return

    for ts_path, meta_path in pairs:
        base = os.path.basename(ts_path)

        # ---- metadata -------------------------------------------------
        meta = pd.read_csv(meta_path)
        group_code = str(meta.loc[0, "group_code"]).strip().upper()
        if group_code not in GROUP_FOLDER:
            print(f"  [skip] {base}: unknown group_code '{group_code}'")
            continue
        group_folder = GROUP_FOLDER[group_code]
        orig_subject_id = str(meta.loc[0, "subject"]).strip() if "subject" in meta.columns else subject_name

        # ---- task (eyes-closed / eyes-open) ----------------------------
        task_label = get_task_label(base)

        # ---- consistent patient id per subject (shared across tasks) --
        if subject_dir not in subject_patient_id:
            group_counters[group_folder] = group_counters.get(group_folder, 0) + 1
            subject_patient_id[subject_dir] = f"patient{group_counters[group_folder]}"
        patient_id = subject_patient_id[subject_dir]

        # ---- read channel timeseries ------------------------------------
        df = pd.read_csv(ts_path)
        time_col = "time_s" if "time_s" in df.columns else df.columns[0]
        channel_names = [c for c in df.columns if c != time_col]
        data = df[channel_names].to_numpy(dtype=np.float64)
        n_samples_orig = data.shape[0]
        del df

        # ---- downsample 500 Hz -> 128 Hz --------------------------------
        data_ds = downsample(data, ORIG_FS, TARGET_FS)
        n_samples_ds = data_ds.shape[0]
        del data

        # ---- write one .txt file per channel ----------------------------
        out_dir = os.path.join(DST_ROOT, group_folder, task_label, patient_id)
        os.makedirs(out_dir, exist_ok=True)
        for ch_idx, ch_name in enumerate(channel_names):
            out_path = os.path.join(out_dir, f"{ch_name}.txt")
            np.savetxt(out_path, data_ds[:, ch_idx], fmt=TXT_FMT)

        print(f"[ok] {subject_name} (group={group_code} -> {group_folder}) "
              f"{task_label} -> {patient_id}  "
              f"({n_samples_orig} @ {ORIG_FS}Hz -> {n_samples_ds} @ {TARGET_FS}Hz, "
              f"{len(channel_names)} channels)")

        manifest_rows.append({
            "original_subject_folder": subject_name,
            "original_subject_id": orig_subject_id,
            "group_code": group_code,
            "group_folder": group_folder,
            "task": task_label,
            "patient_id": patient_id,
            "n_channels": len(channel_names),
            "n_samples_orig_500hz": n_samples_orig,
            "n_samples_downsampled_128hz": n_samples_ds,
            "channel_names": ";".join(channel_names),
        })

        del data_ds


def main():
    subject_dirs = sorted(
        d for d in glob.glob(os.path.join(SRC_ROOT, "sub-*_eeg")) if os.path.isdir(d)
    )
    print(f"Found {len(subject_dirs)} subject folders under '{SRC_ROOT}'\n")

    # pre-create the full requested directory tree
    for folder in GROUP_FOLDER.values():
        for task in ("eyes-closed", "eyes-open"):
            os.makedirs(os.path.join(DST_ROOT, folder, task), exist_ok=True)

    group_counters = {}       # group_folder -> running patient count
    subject_patient_id = {}   # subject_dir -> "patientN" (fixed once per subject)
    manifest_rows = []

    for subject_dir in subject_dirs:
        process_subject(subject_dir, group_counters, subject_patient_id, manifest_rows)

    # ---- manifest -------------------------------------------------------
    manifest_path = os.path.join(DST_ROOT, "manifest.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("\n===== Summary =====")
    for (group_folder), count in sorted(group_counters.items()):
        print(f"  {group_folder}: {count} subjects")
    print(f"\nManifest written to: {manifest_path}")
    print(f"Output written to:   {os.path.abspath(DST_ROOT)}")


if __name__ == "__main__":
    main()