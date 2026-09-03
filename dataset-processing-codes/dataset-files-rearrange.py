#!/usr/bin/env python3
"""
dataset-files-rearrange.py
===========================
Cleans and merges a second batch of "Paciente"-named EEG data
(EEG_data/AD/..., EEG_data/Healthy/...) into the existing DATA/ output
tree produced earlier by dataset-clean.py.

For every source Paciente folder under:
    EEG_data/AD/Eyes_closed/PacienteN
    EEG_data/AD/Eyes_open/PacienteN
    EEG_data/Healthy/Eyes_closed/PacienteN
    EEG_data/Healthy/Eyes_open/PacienteN

it:
  1. Copies every channel .txt file EXCEPT F1.txt and F2.txt (dropped so
     the channel set stays the standard 19-channel 10-20 montage used
     elsewhere in DATA/).
  2. Renumbers each Paciente, continuing on from whatever patient
     numbers already exist in DATA/<group>/eyes-closed/ (auto-detected
     — for your current data that comes out to patient37 for AD and
     patient30 for Healthy).
  3. Uses the SAME new patient number for a given Paciente in both
     Eyes_closed and Eyes_open (mapping is built from Eyes_closed
     first), so the same person keeps the same id across both
     conditions.
  4. Writes into DATA/<group>/<eyes-closed|eyes-open>/patientM/*.txt
     without touching the existing DATA/ patient1..N folders.

By default this COPIES files — EEG_data/ is left completely untouched.
Set MOVE_FILES = True below if you'd rather move (delete from source).

NOTE: the "healthy_output" (sub-037 .. sub-065) folder under
EEG_data/Healthy/ is intentionally NOT touched by this script since it
wasn't part of the request — worth double-checking separately, since
in the original ds004504 dataset subject numbers in that range
(sub-037 onward) are mostly Frontotemporal Dementia, not Healthy
controls.
"""

import os
import re
import shutil
import csv

# ================================================================ CONFIG ===
SRC_ROOT = "../../EEG_data"     # contains AD/ and Healthy/
DST_ROOT = "DATA"         # existing output root from dataset-clean.py

GROUP_DIRS = {
    "AD": "Alzehiemers",       # EEG_data/AD      -> DATA/Alzehiemers
    "Healthy": "Healthy",      # EEG_data/Healthy -> DATA/Healthy
}

CHANNELS_TO_DROP = {"F1.txt", "F2.txt"}

MOVE_FILES = False   # False = copy (safe, keeps EEG_data/ untouched)
                      # True  = move (deletes source files after copying)

# Purely a sanity-check print — the script does NOT rely on these values,
# it auto-detects the real starting number from DATA/. If the printed
# "auto-detected" number doesn't match these, something about DATA/'s
# current state is different than expected, so double check before trusting
# the run.
EXPECTED_START = {"AD": 37, "Healthy": 30}
# ============================================================================


def natural_patient_number(folder_name: str):
    """Extract the integer N from a 'PacienteN' folder name, else None."""
    m = re.match(r'^Paciente(\d+)$', folder_name)
    return int(m.group(1)) if m else None


def get_next_patient_number(dst_root, group_folder, task_folder="eyes-closed"):
    """Look at DATA/<group>/<task> and return (max existing patientN) + 1."""
    existing_dir = os.path.join(dst_root, group_folder, task_folder)
    max_n = 0
    if os.path.isdir(existing_dir):
        for name in os.listdir(existing_dir):
            m = re.match(r'^patient(\d+)$', name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def list_paciente_folders(base_dir):
    """Return {paciente_number: folder_path} for an Eyes_closed/Eyes_open dir."""
    result = {}
    if not os.path.isdir(base_dir):
        return result
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        num = natural_patient_number(name)
        if num is not None:
            result[num] = full
    return result


def transfer_patient(src_dir, dst_dir):
    """Copy (or move) every channel file from src_dir to dst_dir, dropping
    F1.txt/F2.txt. Returns the number of files transferred."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    if not os.path.isdir(src_dir):
        return n
    for fname in os.listdir(src_dir):
        if fname in CHANNELS_TO_DROP:
            continue
        if not fname.lower().endswith(".txt"):
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if MOVE_FILES:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        n += 1
    return n


def process_group(group_key, manifest_rows):
    group_folder = GROUP_DIRS[group_key]
    closed_src = os.path.join(SRC_ROOT, group_key, "Eyes_closed")
    open_src = os.path.join(SRC_ROOT, group_key, "Eyes_open")

    closed_pacientes = list_paciente_folders(closed_src)
    open_pacientes = list_paciente_folders(open_src)

    if not closed_pacientes and not open_pacientes:
        print(f"[skip] no Paciente folders found for {group_key}")
        return

    start_num = get_next_patient_number(DST_ROOT, group_folder, "eyes-closed")
    expected = EXPECTED_START.get(group_key)
    flag = "" if expected is None or expected == start_num else \
        f"  <-- WARNING: expected patient{expected}, check DATA/{group_folder}/eyes-closed/ before trusting this run"
    print(f"\n=== {group_key} -> {group_folder}: auto-detected start = patient{start_num}{flag} ===")
    print(f"    ({len(closed_pacientes)} Eyes_closed Pacientes, {len(open_pacientes)} Eyes_open Pacientes found)")

    # Build paciente_number -> new_patient_number from Eyes_closed first,
    # so the same person keeps the same id in both conditions.
    mapping = {}
    for rank, paciente_num in enumerate(sorted(closed_pacientes.keys())):
        mapping[paciente_num] = start_num + rank

    # Any Paciente only present in Eyes_open gets appended after those.
    next_free = start_num + len(mapping)
    for paciente_num in sorted(open_pacientes.keys()):
        if paciente_num not in mapping:
            mapping[paciente_num] = next_free
            print(f"  [note] Paciente{paciente_num} only found in Eyes_open -> patient{next_free}")
            next_free += 1

    # ---- transfer Eyes_closed ----
    for paciente_num in sorted(closed_pacientes.keys()):
        src_dir = closed_pacientes[paciente_num]
        new_id = mapping[paciente_num]
        dst_dir = os.path.join(DST_ROOT, group_folder, "eyes-closed", f"patient{new_id}")
        n = transfer_patient(src_dir, dst_dir)
        print(f"  [ok] Paciente{paciente_num:<3} (Eyes_closed) -> patient{new_id}  ({n} channel files)")
        manifest_rows.append({"group": group_key, "task": "eyes-closed",
                               "original_folder": os.path.basename(src_dir),
                               "new_patient_id": f"patient{new_id}", "n_files": n})

    # ---- transfer Eyes_open (same mapping) ----
    for paciente_num in sorted(open_pacientes.keys()):
        src_dir = open_pacientes[paciente_num]
        new_id = mapping[paciente_num]
        dst_dir = os.path.join(DST_ROOT, group_folder, "eyes-open", f"patient{new_id}")
        n = transfer_patient(src_dir, dst_dir)
        print(f"  [ok] Paciente{paciente_num:<3} (Eyes_open)   -> patient{new_id}  ({n} channel files)")
        manifest_rows.append({"group": group_key, "task": "eyes-open",
                               "original_folder": os.path.basename(src_dir),
                               "new_patient_id": f"patient{new_id}", "n_files": n})


def main():
    manifest_rows = []
    for group_key in GROUP_DIRS:
        process_group(group_key, manifest_rows)

    manifest_path = os.path.join(DST_ROOT, "manifest_rearrange.csv")
    if manifest_rows:
        os.makedirs(DST_ROOT, exist_ok=True)
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\nMapping log written to: {manifest_path}")

    print(f"Mode: {'MOVE (source files deleted from EEG_data/)' if MOVE_FILES else 'COPY (EEG_data/ left untouched)'}")


if __name__ == "__main__":
    main()