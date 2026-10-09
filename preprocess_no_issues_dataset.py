#!/usr/bin/env python3
"""Prepare EEG windows without fitting preprocessing on held-out subjects.

1. python preprocess_no_issues_dataset.py --make-manifest eeg_manifest.csv
2. Fill source, sampling_rate_hz, unit and verified person_id in that CSV.
3. python preprocess_no_issues_dataset.py --manifest eeg_manifest.csv --output prepared_eeg

Requires numpy, scipy. No source files are changed. Output must be a new directory.
Units accepted: uV, mV, V. sampling_rate_hz MUST describe the exported text files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

CHANNELS = 'Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2'.split()
LABELS = {'Alzehiemers': 1, 'Healthy': 0}
UNITS = {'uV': 1., 'mV': 1000., 'V': 1e6}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else ['status']
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_signal(root, relative):
    folder = (root / relative).resolve()
    if not folder.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside dataset: {relative}')
    actual = {p.stem for p in folder.glob('*.txt')}
    if actual != set(CHANNELS):
        raise ValueError(f'{relative}: expected exactly the documented 19 channels')
    xs = [np.loadtxt(folder / (ch + '.txt'), dtype=np.float64, ndmin=1) for ch in CHANNELS]
    if any(x.ndim != 1 or not len(x) for x in xs) or len({len(x) for x in xs}) != 1:
        raise ValueError(f'{relative}: empty, multidimensional, or unequal channel lengths')
    return np.stack(xs)


def make_manifest(root, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Refusing to overwrite {destination}')
    rows = []
    for label, number in LABELS.items():
        for p in sorted((root / label / 'eyes-closed').glob('patient*')):
            if not p.is_dir():
                continue
            rows.append(dict(subject_id=f'{label}_{p.name}', relative_path=str(p.relative_to(root)),
                             label=number, person_id='', source='', sampling_rate_hz='', unit=''))
    if not rows:
        raise ValueError('No subject directories found')
    write_csv(destination, rows)
    print(f'Wrote {len(rows)} rows to {destination}. Fill every blank from verified provenance.')


def read_manifest(root, path):
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError('Empty manifest')
    seen = set()
    paths = set()
    for row in rows:
        for key in ('subject_id', 'relative_path', 'label', 'person_id', 'source', 'sampling_rate_hz', 'unit'):
            if not row.get(key, '').strip():
                raise ValueError(f'Missing {key} for {row.get("relative_path", "row")}; do not infer rates from length')
            row[key] = row[key].strip()
        sid = row['subject_id']
        if sid in seen or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in sid):
            raise ValueError('subject_id must be unique and contain only letters, digits, _ or -')
        seen.add(sid)
        row['label'] = int(row['label'])
        row['sampling_rate_hz'] = float(row['sampling_rate_hz'])
        if row['unit'] not in UNITS or not np.isfinite(row['sampling_rate_hz']) or row['sampling_rate_hz'] <= 0:
            raise ValueError(f'{sid}: invalid rate or unit (use uV, mV, V)')
        relative = Path(row['relative_path'])
        folder = (root / relative).resolve()
        if folder in paths:
            raise ValueError('Same directory appears twice in manifest')
        paths.add(folder)
        if not folder.is_relative_to(root.resolve()) or len(relative.parts) != 3:
            raise ValueError(f'Invalid relative dataset path: {relative}')
        if row['label'] != LABELS.get(relative.parts[0]):
            raise ValueError(f'{sid}: label disagrees with class directory')
    return rows


def assign_groups(rows):
    """Union person identities AND exact numeric recording hashes, retaining all rows."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    person, recording = {}, {}
    for i, r in enumerate(rows):
        # person_id must be globally unique across sources, or identical for the same person.
        for table, key in ((person, r['person_id']), (recording, r['signal_sha256'])):
            if key in table:
                union(i, table[key])
            table[key] = i
    members = defaultdict(list)
    for i in range(len(rows)):
        members[find(i)].append(i)
    groups = []
    for indices in members.values():
        group_rows = [rows[i] for i in indices]
        if len({r['label'] for r in group_rows}) != 1:
            raise ValueError('Same person or exact recording has conflicting labels; resolve before training')
        gid = min(r['subject_id'] for r in group_rows)
        sources = '|'.join(sorted({r['source'] for r in group_rows}))
        for r in group_rows:
            r['group_id'] = gid
        groups.append(dict(group_id=gid, label=group_rows[0]['label'], source_stratum=sources))
    return groups


def assign_splits(groups, seed, val_fraction, test_fraction):
    """Stratify by source and label at independent group level."""
    strata = defaultdict(list)
    for g in groups:
        strata[(g['source_stratum'], g['label'])].append(g['group_id'])
    rng = np.random.default_rng(seed)
    assigned = {}
    for key, ids in sorted(strata.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            raise ValueError(f'Stratum {key} has only {n} independent groups; cannot split into train/val/test')
        nv = max(1, int(round(n * val_fraction)))
        nt = max(1, int(round(n * test_fraction)))
        while nv + nt >= n:
            if nv >= nt and nv > 1:
                nv -= 1
            elif nt > 1:
                nt -= 1
            else:
                raise ValueError(f'Cannot split {key}')
        for i, gid in enumerate(ids):
            assigned[gid] = 'val' if i < nv else 'test' if i < nv + nt else 'train'
    return assigned


def continuous_intervals(valid):
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def clean_windows(x, fs, args):
    from scipy.signal import butter, sosfiltfilt, resample_poly
    if args.high_hz >= min(fs, args.target_hz) / 2:
        raise ValueError('High cutoff must be below both native and target Nyquist frequencies')
    if fs < args.target_hz:
        raise ValueError('Refusing upsampling: select a target rate no higher than any input rate')
    # Invalid intervals break continuity. Do not filter across missing data or join gaps.
    valid = np.isfinite(x).all(axis=0) & ~np.all(x == 0, axis=0)
    sos = butter(4, [args.low_hz, args.high_hz], btype='bandpass', fs=fs, output='sos')
    ratio = Fraction(args.target_hz / fs).limit_denominator(100000)
    if abs(float(ratio) * fs - args.target_hz) > 1e-6:
        raise ValueError('Sampling-rate ratio is not represented accurately')
    size = int(round(args.window_seconds * args.target_hz))
    edge = int(np.ceil(args.edge_seconds * args.target_hz))
    kept, positions = [], []
    quality = [dict(start_sample=int(a), end_sample=int(b), accepted=False, reason='nonfinite_or_all_channel_zero_gap')
               for a, b in continuous_intervals(~valid)]
    for start, end in continuous_intervals(valid):
        if end - start < np.ceil((args.window_seconds + 2 * args.edge_seconds) * fs):
            quality.append(dict(start_sample=int(start), end_sample=int(end), accepted=False, reason='short_continuous_segment'))
            continue
        segment = x[:, start:end]
        filtered = sosfiltfilt(sos, segment, axis=1)
        filtered = resample_poly(filtered, ratio.numerator, ratio.denominator, axis=1)
        for offset in range(edge, filtered.shape[1] - edge - size + 1, size):
            a = start + int(np.floor(offset * fs / args.target_hz))
            b = min(end, start + int(np.ceil((offset + size) * fs / args.target_hz)))
            raw = x[:, a:b]
            w = filtered[:, offset:offset + size].copy()
            reasons = []
            # Gate artifacts before average referencing, which could hide common artifacts.
            if np.ptp(raw, axis=1).max() > args.max_ptp_uv:
                reasons.append('raw_peak_to_peak')
            if raw.shape[1] > 1 and np.abs(np.diff(raw, axis=1)).max() > args.max_step_uv:
                reasons.append('raw_step')
            if np.ptp(w, axis=1).min() < args.min_ptp_uv:
                reasons.append('flat_channel')
            flat_size = max(2, int(round(args.flat_seconds * fs)))
            for channel in raw:
                edges = np.flatnonzero(np.r_[True, channel[1:] != channel[:-1], True])
                if np.diff(edges).max() >= flat_size:
                    reasons.append('constant_run')
                    break
            quality.append(dict(start_sample=int(a), end_sample=int(b), accepted=not reasons, reason=';'.join(reasons) or 'accepted'))
            if reasons:
                continue
            if args.reference == 'average':
                w -= w.mean(axis=0, keepdims=True)
            kept.append(w.astype(np.float32))
            positions.append((a, b))
    shape = (0, len(CHANNELS), size)
    windows = np.stack(kept) if kept else np.empty(shape, dtype=np.float32)
    return windows, np.asarray(positions, dtype=np.int64).reshape(-1, 2), quality, int((~valid).sum())


def compute_scaler(rows, output):
    """Equal weight per leakage group, then per recording within each group."""
    groups = defaultdict(list)
    for r in rows:
        if r['split'] == 'train' and r['n_windows']:
            with np.load(output / r['window_file']) as d:
                x = d['x'].astype(np.float64)
            groups[r['group_id']].append((x.mean(axis=(0, 2)), (x*x).mean(axis=(0, 2))))
    if not groups:
        raise ValueError('No training windows survived cleaning')
    means, seconds = [], []
    for members in groups.values():
        means.append(np.mean([m[0] for m in members], axis=0))
        seconds.append(np.mean([m[1] for m in members], axis=0))
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds, axis=0) - mean**2, 1e-12))
    return mean, std


def run(args):
    root = Path(args.dataset).resolve()
    if args.make_manifest:
        make_manifest(root, args.make_manifest)
        return
    if not args.manifest:
        raise ValueError('Supply --manifest, or first use --make-manifest eeg_manifest.csv')
    if not (0 < args.val_fraction < .5 and 0 < args.test_fraction < .5 and args.val_fraction + args.test_fraction < 1):
        raise ValueError('Invalid validation/test fractions')
    if not (0 < args.low_hz < args.high_hz and args.target_hz > 0 and args.window_seconds > 0 and args.edge_seconds >= 0):
        raise ValueError('Invalid filter, window or sampling-rate settings')
    if min(args.max_ptp_uv, args.max_step_uv, args.min_ptp_uv, args.flat_seconds) <= 0:
        raise ValueError('Quality thresholds must be positive')
    samples = args.window_seconds * args.target_hz
    if not np.isclose(samples, round(samples)) or samples < 16:
        raise ValueError('Window duration must give an integer count of at least 16 target samples')
    rows = read_manifest(root, args.manifest)
    from scipy.signal import butter, sosfiltfilt, resample_poly  # Validate dependency before reading gigabytes.
    output = Path(args.output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Output must be outside the original dataset')
    if output.exists():
        raise ValueError(f'Output exists: {output}. Use a new directory to preserve prior results.')
    output.mkdir(parents=True)
    (output / 'windows').mkdir()
    save_json(output / 'config.json', dict(vars(args), channels=CHANNELS, label_mapping=LABELS))
    print('Hashing all numeric recordings to keep matching arrays in one split...', flush=True)
    for i, r in enumerate(rows):
        x = load_signal(root, r['relative_path'])
        # Hash the numeric exported representation before any unit conversion.
        canonical = x.copy()
        canonical[canonical == 0] = 0.0  # Canonicalize negative zero.
        r['signal_sha256'] = hashlib.sha256(canonical.astype('<f8').tobytes()).hexdigest()
        r['n_samples'] = x.shape[1]
        if (i+1) % 20 == 0:
            print(f'  hashed {i+1}/{len(rows)}', flush=True)
    groups = assign_groups(rows)
    assignments = assign_splits(groups, args.seed, args.val_fraction, args.test_fraction)
    duplicate_rows = defaultdict(list)
    for r in rows:
        duplicate_rows[r['signal_sha256']].append(r['subject_id'])
    save_json(output / 'identical_recording_groups.json', [v for v in duplicate_rows.values() if len(v) > 1])
    qc, rejects = [], []
    for i, r in enumerate(rows):
        r['split'] = assignments[r['group_id']]
        x = load_signal(root, r['relative_path']) * UNITS[r['unit']]
        w, positions, details, invalid = clean_windows(x, r['sampling_rate_hz'], args)
        r['n_windows'] = len(w)
        r['invalid_timepoints'] = invalid
        r['retained_seconds'] = len(w) * args.window_seconds
        r['window_file'] = f'windows/{r["subject_id"]}.npz'
        np.savez_compressed(output / r['window_file'], x=w, original_sample_intervals=positions)
        for detail in details:
            qc.append(dict(subject_id=r['subject_id'], split=r['split'], **detail))
        if not len(w):
            rejects.append(r['subject_id'])
        print(f'{i+1}/{len(rows)} {r["subject_id"]}: {len(w)} windows ({r["split"]})', flush=True)
    write_csv(output / 'subjects.csv', rows)
    write_csv(output / 'window_quality.csv', qc)
    save_json(output / 'excluded_no_clean_windows.json', rejects)
    summary = {}
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split and r['n_windows']]
        summary[split] = dict(recordings=len(subset), groups=len({r['group_id'] for r in subset}),
                              windows=sum(r['n_windows'] for r in subset),
                              groups_by_class={str(y): len({r['group_id'] for r in subset if r['label'] == y}) for y in (0, 1)},
                              recordings_by_source={src: sum(r['source'] == src for r in subset) for src in sorted({r['source'] for r in rows})})
    save_json(output / 'split_summary.json', summary)
    if any(min(v['groups_by_class'].values()) == 0 for v in summary.values()):
        raise ValueError('Cleaning left a partition without both classes. Review saved QC; do not tune cleaning on test accuracy.')
    mean, std = compute_scaler(rows, output)
    np.savez(output / 'training_scaler.npz', mean=mean.astype(np.float32), std=std.astype(np.float32))
    save_json(output / 'READY.json', dict(status='complete', normalization='training_only_group_weighted_channel_zscore',
              recordings=len(rows), independent_groups=len(groups), excluded_no_clean_windows=rejects))
    print(f'Prepared data saved to {output}. Review window_quality.csv and split_summary.json before training.')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='FINAL_DATASET')
    p.add_argument('--make-manifest')
    p.add_argument('--manifest')
    p.add_argument('--output', default='prepared_eeg')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=.15)
    p.add_argument('--test-fraction', type=float, default=.15)
    p.add_argument('--target-hz', type=float, default=128.)
    p.add_argument('--window-seconds', type=float, default=4.)
    p.add_argument('--edge-seconds', type=float, default=.5, help='Discard at both ends of each continuous filtered segment')
    p.add_argument('--low-hz', type=float, default=.5)
    p.add_argument('--high-hz', type=float, default=40.)
    p.add_argument('--reference', choices=['average', 'keep'], default='keep', help='Use average only after checking original reference compatibility')
    p.add_argument('--max-ptp-uv', type=float, default=500.)
    p.add_argument('--max-step-uv', type=float, default=150.)
    p.add_argument('--min-ptp-uv', type=float, default=.1)
    p.add_argument('--flat-seconds', type=float, default=.5)
    return p


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
