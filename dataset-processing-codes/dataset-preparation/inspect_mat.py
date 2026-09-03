#!/usr/bin/env python3
"""
inspect_mat_file.py
====================
Safely inspects a .mat file's structure WITHOUT loading its full
contents into memory — safe to run on multi-GB files.

Handles both:
  - MATLAB v7.3 files (HDF5-based; what a 7 GB .mat almost certainly is)
    -> uses h5py, walks groups/datasets, dereferences MATLAB cell/struct
       object references, and only reads small preview slices.
  - Legacy MATLAB v5/v7 files (rare at this size)
    -> uses scipy.io.loadmat.

Usage:
    pip install h5py numpy scipy --break-system-packages
    python3 inspect_mat_file.py /path/to/controls_c1_new.mat

Writes a full report to mat_inspection_report.txt (next to the script)
AND prints it to the console, so you can just copy/paste that file's
contents back.
"""

import sys
import os
import io

SAMPLE_N = 5          # how many values to preview per dataset
MAX_ITEMS = 300        # safety cap on how many HDF5 items we'll describe
MAX_REF_PREVIEW = 3     # how many elements of a cell/struct array to expand


class Tee:
    """Writes to both the console and a report file at the same time."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")

    def write(self, text):
        sys.__stdout__.write(text)
        self.file.write(text)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def is_hdf5(path):
    """MATLAB v7.3 files prefix the real HDF5 data with a 512-byte text
    'userblock' (the classic 'MATLAB 7.3 MAT-file...' header), so the
    HDF5 signature isn't at byte 0. h5py.is_hdf5() checks the standard
    offsets (0, 512, 1024, 2048, ...) correctly, unlike a raw byte check."""
    import h5py
    return h5py.is_hdf5(path)


# --------------------------------------------------------------- HDF5 / v7.3 ---

def inspect_hdf5(path):
    import h5py
    import numpy as np

    f = h5py.File(path, "r")

    def is_ref_dtype(dtype):
        try:
            return h5py.check_dtype(ref=dtype) is not None
        except Exception:
            return False

    def preview_dataset(obj, pad):
        """Print shape/dtype/attrs and a small safe sample of a Dataset."""
        shape, dtype = obj.shape, obj.dtype
        print(f"{pad}[Dataset] shape={shape}  dtype={dtype}")

        for k, v in obj.attrs.items():
            print(f"{pad}    attr: {k} = {v}")

        # MATLAB stores char arrays as uint16 codes -- try to decode strings
        if dtype.kind in ("u", "i") and obj.attrs.get("MATLAB_class") == b"char":
            try:
                raw = obj[()]
                text = "".join(chr(c) for c in np.array(raw).flatten()[:200])
                print(f"{pad}    string preview: {text!r}")
                return
            except Exception:
                pass

        if is_ref_dtype(dtype):
            # cell array / struct array of references to other datasets
            flat_refs = obj[()].flatten()
            print(f"{pad}    (reference array, {flat_refs.size} elements) "
                  f"-- expanding first {min(MAX_REF_PREVIEW, flat_refs.size)}:")
            for i, ref in enumerate(flat_refs[:MAX_REF_PREVIEW]):
                if not ref:
                    print(f"{pad}      [{i}] <null ref>")
                    continue
                target = f[ref]
                print(f"{pad}      [{i}] -> {target.name}")
                if isinstance(target, h5py.Dataset):
                    preview_dataset(target, pad + "        ")
                elif isinstance(target, h5py.Group):
                    print(f"{pad}          [Group] keys={list(target.keys())}")
            return

        # plain numeric dataset -- sample a small corner of it, never the whole thing
        try:
            if obj.size == 0:
                print(f"{pad}    (empty)")
                return
            if obj.ndim == 0:
                print(f"{pad}    value: {obj[()]}")
                return
            slicer = tuple(slice(0, min(SAMPLE_N, s)) for s in shape)
            sample = np.array(obj[slicer])
            print(f"{pad}    sample corner {sample.shape}: {sample.flatten()[:SAMPLE_N]}")
        except Exception as e:
            print(f"{pad}    [could not sample: {e}]")

    print(f"=== MATLAB v7.3 / HDF5 file: {path} ===")
    print(f"Top-level keys: {list(f.keys())}\n")

    count = [0]

    def visitor(name, obj):
        if count[0] >= MAX_ITEMS:
            return
        # skip MATLAB's internal reference-storage group in the walk (we
        # dereference into it explicitly when we hit a reference dataset)
        if name.startswith("#refs#"):
            return
        depth = name.count("/")
        pad = "  " * depth
        kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
        print(f"{pad}{name}  [{kind}]")
        if isinstance(obj, h5py.Dataset):
            preview_dataset(obj, pad + "  ")
        elif isinstance(obj, h5py.Group) and obj.attrs:
            for k, v in obj.attrs.items():
                print(f"{pad}  attr: {k} = {v}")
        print()
        count[0] += 1

    f.visititems(visitor)
    if count[0] >= MAX_ITEMS:
        print(f"... truncated after {MAX_ITEMS} items (file has more) ...")

    f.close()


# --------------------------------------------------------- legacy v5 / v7 mat ---

def inspect_legacy_mat(path):
    from scipy.io import loadmat
    import numpy as np

    print(f"=== Legacy MATLAB (v5/v7) file: {path} ===")
    print("(This is unusual at this file size -- loading may use a lot of RAM)\n")

    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Top-level variables: {keys}\n")

    def describe(val, name, indent=0, depth=0):
        pad = "  " * indent
        if depth > 4:
            print(f"{pad}{name}: ... (max depth reached)")
            return
        if isinstance(val, np.ndarray):
            print(f"{pad}{name}: ndarray shape={val.shape} dtype={val.dtype}")
            if val.dtype == object:
                flat = val.flatten()
                print(f"{pad}  object array, {flat.size} elements -- first {min(MAX_REF_PREVIEW, flat.size)}:")
                for i, item in enumerate(flat[:MAX_REF_PREVIEW]):
                    describe(item, f"{name}[{i}]", indent + 2, depth + 1)
            else:
                try:
                    print(f"{pad}  sample: {val.flatten()[:SAMPLE_N]}")
                except Exception:
                    pass
        elif hasattr(val, "_fieldnames"):
            print(f"{pad}{name}: struct, fields={val._fieldnames}")
            for fn in val._fieldnames:
                describe(getattr(val, fn), f"{name}.{fn}", indent + 1, depth + 1)
        else:
            s = repr(val)
            print(f"{pad}{name}: {type(val).__name__} = {s[:200]}")

    for k in keys:
        describe(mat[k], k)
        print()


# ------------------------------------------------------------------- main ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inspect_mat_file.py /path/to/controls_c1_new.mat")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "mat_inspection_report.txt")
    tee = Tee(report_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"File: {path}  ({size_gb:.2f} GB)\n")

        if is_hdf5(path):
            inspect_hdf5(path)
        else:
            inspect_legacy_mat(path)
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"\nFull report also written to: {report_path}")
    print("Share that file's contents back and I'll write the conversion script from it.")


if __name__ == "__main__":
    main()