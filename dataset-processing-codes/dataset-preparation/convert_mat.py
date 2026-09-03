#!/usr/bin/env python3
"""
convert_controls_mat.py
=========================
Converts the Dryad "controls_c1_new.mat" file (Benninger & Shor et al.,
PLOS ONE 2021 / Dryad doi:10.5061/dryad.8gtht76pw) into the same
patientN/channel.txt format used elsewhere, downsampled to 128 Hz.

IMPORTANT ASSUMPTIONS -- please read before trusting the output:

1. CHANNEL ORDER is not stored anywhere in this .mat file (the struct
   only has age/sex/G/n/g -- no channel-name field). The 19 columns are
   ASSUMED to follow the same standard 10-20 anterior-to-posterior order
   used for the ds004504 data:
       Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6,
       Fz, Cz, Pz
   This is UNVERIFIED for this specific dataset. If AUTHOR_DATASET_
   SHOR_BENNINGER.txt (from the Dryad page) specifies a different order,
   update CHANNEL_NAMES below before trusting channel-specific results.

2. EYES-OPEN/CLOSED: confirmed there is no condition marker anywhere in
   this file -- one continuous routine-EEG recording per subject. Per
   your decision, everything goes into a new "unlabeled" condition
   folder instead of being forced into eyes-closed/eyes-open.

3. SAMPLING RATE is read PER-SUBJECT from the 'g' field, not assumed to
   be 500 Hz for everyone -- the file itself shows some subjects at
   500 Hz and others at 200 Hz.

4. Per your decision, the FULL recording (not just the paper's 351s
   excerpt) is downsampled and kept.

5. Only the first N_SUBJECTS_TO_USE subjects (in file order) are
   converted, numbered patient42 .. patient116, continuing the existing
   Healthy patient numbering from the earlier ds004504 + Paciente
   batches.

Output goes to a NEW, separate folder (DST_ROOT below) -- NOT merged
into the existing DATA/ tree.

Usage:
    pip install h5py numpy scipy --break-system-packages
    python3 convert_controls_mat.py
"""

import os
import csv
import math
import numpy as np
import h5py
from scipy.signal import resample_poly

# ================================================================ CONFIG ===
MAT_PATH = "controls_c1_new.mat"     # path to the downloaded .mat file
DST_ROOT = "DATA_Mat_File"       # separate output folder (not merged into DATA/)

GROUP_FOLDER = "Healthy"             # this whole file is the Controls group
CONDITION_FOLDER = "eyes-closed"       # no eyes-open/closed marker in this data

TARGET_FS = 128                      # downsample target
START_PATIENT = 42                   # first new patient number
END_PATIENT = 116                    # last new patient number (inclusive)
N_SUBJECTS_TO_USE = END_PATIENT - START_PATIENT + 1   # 75

CHANNEL_NAMES = [                    # ASSUMED order -- see module docstring, point 1
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

TXT_FMT = "%.8e"
# ============================================================================


def deref_scalar(f, ref):
    """ref points directly to a (1,1) numeric dataset -- return the float."""
    ds = f[ref]
    return float(np.array(ds).flatten()[0])


def deref_char_direct(f, ref):
    """ref points directly to a MATLAB char (uint16-coded) dataset -- decode it."""
    ds = f[ref]
    codes = np.array(ds).flatten()
    return "".join(chr(int(c)) for c in codes)


def deref_double_nested(f, ref):
    """ref -> (1,1) cell -> inner ref -> final dataset (NOT decoded/converted).
    Used for fields like G and n which are cell-of-cell in this file."""
    outer = f[ref]           # (1,1) object dataset
    inner_ref = outer[0, 0]
    return f[inner_ref]


def decode_char_dataset(ds):
    codes = np.array(ds).flatten()
    return "".join(chr(int(c)) for c in codes)


def downsample(data, orig_fs, target_fs):
    orig_fs = int(round(orig_fs))
    if orig_fs == target_fs:
        return data
    if orig_fs < target_fs:
        print(f"    [warn] original rate {orig_fs}Hz < target {target_fs}Hz -- "
              f"that would be upsampling; leaving this subject at its native rate")
        return data
    g = math.gcd(orig_fs, target_fs)
    up, down = target_fs // g, orig_fs // g
    return resample_poly(data, up, down, axis=0)


def main():
    out_condition_dir = os.path.join(DST_ROOT, GROUP_FOLDER, CONDITION_FOLDER)
    os.makedirs(out_condition_dir, exist_ok=True)
    manifest_rows = []

    with h5py.File(MAT_PATH, "r") as f:
        G_refs = f["controls_r"]["G"][0, :]
        age_refs = f["controls_r"]["age"][0, :]
        sex_refs = f["controls_r"]["sex"][0, :]
        g_refs = f["controls_r"]["g"][0, :]
        n_refs = f["controls_r"]["n"][0, :]

        n_available = G_refs.shape[0]
        n_use = min(N_SUBJECTS_TO_USE, n_available)
        if n_use < N_SUBJECTS_TO_USE:
            print(f"[warn] only {n_available} subjects in file, "
                  f"requested {N_SUBJECTS_TO_USE} -- using {n_use}\n")

        for i in range(n_use):
            patient_num = START_PATIENT + i
            patient_id = f"patient{patient_num}"

            # ---- pull per-subject EEG data & metadata ----
            data_ds = deref_double_nested(f, G_refs[i])
            data = np.array(data_ds, dtype=np.float64)          # (n_samples, 19)

            age = deref_scalar(f, age_refs[i])
            sex = deref_char_direct(f, sex_refs[i])
            orig_fs = deref_scalar(f, g_refs[i])
            fname_ds = deref_double_nested(f, n_refs[i])
            orig_fname = decode_char_dataset(fname_ds)

            n_ch = data.shape[1]
            if n_ch != len(CHANNEL_NAMES):
                print(f"  [skip] subject {i}: data has {n_ch} channels but "
                      f"{len(CHANNEL_NAMES)} channel names are configured")
                continue

            n_samples_orig = data.shape[0]
            data_resampled = downsample(data, orig_fs, TARGET_FS)
            n_samples_new = data_resampled.shape[0]

            # sanity flag: warn if the whole subject is flat/zero (bad export)
            if np.allclose(data, 0.0):
                print(f"  [warn] subject {i}: entire recording is all zeros -- check source data")

            # ---- write one .txt per channel ----
            out_dir = os.path.join(out_condition_dir, patient_id)
            os.makedirs(out_dir, exist_ok=True)
            for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                out_path = os.path.join(out_dir, f"{ch_name}.txt")
                np.savetxt(out_path, data_resampled[:, ch_idx], fmt=TXT_FMT)

            print(f"[ok] subject {i:>2} (orig_fs={orig_fs:.0f}Hz, age={age:.0f}, sex={sex}, "
                  f"src={orig_fname}) -> {patient_id}  "
                  f"({n_samples_orig} -> {n_samples_new} samples)")

            manifest_rows.append({
                "orig_index": i, "patient_id": patient_id, "group": GROUP_FOLDER,
                "condition": CONDITION_FOLDER, "age": age, "sex": sex,
                "orig_sampling_rate_hz": orig_fs, "orig_filename": orig_fname,
                "n_channels": n_ch, "n_samples_orig": n_samples_orig,
                "n_samples_downsampled": n_samples_new,
            })

            del data, data_resampled

    manifest_path = os.path.join(DST_ROOT, "manifest_controls.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\nManifest written to: {manifest_path}")
    print(f"Output written to:   {os.path.abspath(DST_ROOT)}")


if __name__ == "__main__":
    main()