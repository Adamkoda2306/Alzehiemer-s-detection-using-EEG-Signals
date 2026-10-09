#!/usr/bin/env python3
"""Train a compact EEG CNN; no accuracy target is promised.

python cnn.py --data prepared_eeg --output cnn_results --epochs 150
Requires numpy, torch, scikit-learn, matplotlib.
Selection/early stopping uses validation groups only. Test is evaluated once at the end.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def save_json(path, x):
    Path(path).write_text(json.dumps(jsonable(x), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


class EEGDataset(Dataset):
    def __init__(self, root, rows, mean, std):
        self.rows = rows
        self.arrays = []
        self.index = []
        self.mean = mean[None, :, None]
        self.std = std[None, :, None]
        for i, row in enumerate(rows):
            with np.load(root / row['window_file']) as data:
                x = data['x'].astype(np.float32)
            if len(x) != row['n_windows'] or x.ndim != 3 or not np.isfinite(x).all():
                raise ValueError('Invalid prepared window file')
            x = (x - self.mean) / self.std
            self.arrays.append(x)
            self.index.extend((i, j) for j in range(len(x)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, j = self.index[index]
        return torch.from_numpy(self.arrays[i][j]).unsqueeze(0), torch.tensor(self.rows[i]['label'], dtype=torch.float32), index

    def balanced_weights(self):
        # Equal classes -> equal independent groups -> equal recordings -> equal windows.
        by_class = defaultdict(set)
        members = defaultdict(int)
        for r in self.rows:
            by_class[r['label']].add(r['group_id'])
            members[r['group_id']] += 1
        return [1. / (len(by_class[self.rows[i]['label']]) * members[self.rows[i]['group_id']] * self.rows[i]['n_windows']) for i, j in self.index]


class EEGCNN(nn.Module):
    """Temporal filters -> depthwise electrode mixing -> separable temporal blocks.

    Input [batch, 1, electrodes, time]. Adaptive pooling supports the configured
    fixed window length. Model is small because there are few independent people.
    """
    def __init__(self, channels=19, temporal_filters=16, depth_multiplier=2, dropout=.5):
        super().__init__()
        f = temporal_filters
        d = f * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, f, (1, 65), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f),
            nn.Conv2d(f, d, (channels, 1), groups=f, bias=False),
            nn.BatchNorm2d(d), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(d, d, (1, 17), padding=(0, 8), groups=d, bias=False),
            nn.Conv2d(d, d * 2, 1, bias=False), nn.BatchNorm2d(d * 2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(d * 2, d * 2, (1, 9), padding=(0, 4), groups=d * 2, bias=False),
            nn.Conv2d(d * 2, d * 2, 1, bias=False), nn.BatchNorm2d(d * 2), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 4)), nn.Flatten(), nn.Dropout(dropout))
        self.classifier = nn.Linear(d * 2 * 4, 1)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    result = np.empty(len(loader.dataset), dtype=np.float64)
    for x, _, indices in loader:
        result[indices.numpy()] = torch.sigmoid(model(x.to(device))).cpu().numpy()
    return result


def aggregate(dataset, probabilities):
    by_recording = defaultdict(list)
    windows = []
    for index, (i, j) in enumerate(dataset.index):
        r = dataset.rows[i]
        p = float(probabilities[index])
        by_recording[i].append(p)
        windows.append(dict(subject_id=r['subject_id'], group_id=r['group_id'], source=r['source'],
                            window_index=j, label=r['label'], probability_ad=p))
    recordings = []
    for i, values in by_recording.items():
        r = dataset.rows[i]
        recordings.append(dict(subject_id=r['subject_id'], group_id=r['group_id'], source=r['source'],
                               label=r['label'], n_windows=len(values), probability_ad=float(np.mean(values))))
    grouped = defaultdict(list)
    for r in recordings:
        grouped[r['group_id']].append(r)
    groups = []
    for gid, members in grouped.items():
        groups.append(dict(group_id=gid, source='|'.join(sorted({r['source'] for r in members})),
                           label=members[0]['label'], n_recordings=len(members),
                           n_windows=sum(r['n_windows'] for r in members),
                           probability_ad=float(np.mean([r['probability_ad'] for r in members]))))
    return windows, recordings, groups


def metrics(rows, threshold):
    from sklearn.metrics import (confusion_matrix, accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
        matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss, classification_report)
    y = np.array([r['label'] for r in rows], dtype=int)
    p = np.array([r['probability_ad'] for r in rows])
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n=len(y), threshold=threshold, class_counts={'Healthy': int((y == 0).sum()), 'AD': int((y == 1).sum())},
                classification_report=classification_report(y, pred, labels=[0, 1], target_names=['Healthy', 'AD'], output_dict=True, zero_division=0),
                accuracy=accuracy_score(y, pred),
                balanced_accuracy=balanced_accuracy_score(y, pred) if both else None,
                sensitivity=recall_score(y, pred, zero_division=0) if (y == 1).any() else None,
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                precision=precision_score(y, pred, zero_division=0),
                npv=float(tn / (tn + fn)) if tn + fn else None,
                f1=f1_score(y, pred, zero_division=0),
                mcc=matthews_corrcoef(y, pred) if both else None,
                cohen_kappa=cohen_kappa_score(y, pred) if both else None,
                roc_auc=roc_auc_score(y, p) if both else None,
                average_precision=average_precision_score(y, p) if both else None,
                log_loss=log_loss(y, np.clip(p, 1e-7, 1-1e-7), labels=[0, 1]),
                brier_score=brier_score_loss(y, p), confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def choose_threshold(rows):
    # Prefer 0.5 on ties to reduce arbitrary movement on a small validation set.
    best = (.5, -1.)
    candidates = sorted(np.linspace(.05, .95, 91), key=lambda t: abs(t-.5))
    y = np.array([r['label'] for r in rows])
    p = np.array([r['probability_ad'] for r in rows])
    for t in candidates:
        pred = p >= t
        score = .5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0]))
        if score > best[1] + 1e-12:
            best = (float(t), float(score))
    return best[0]


def bootstrap_ci(rows, threshold, repetitions, seed):
    # Bootstrap independent leakage groups, never individual correlated windows.
    rng = np.random.default_rng(seed)
    strata = [[r for r in rows if r['label'] == y] for y in (0, 1)]
    if not all(strata) or repetitions <= 0:
        return {}
    result = defaultdict(list)
    for _ in range(repetitions):
        sample = [g[i] for g in strata for i in rng.integers(0, len(g), size=len(g))]
        m = metrics(sample, threshold)
        for name in ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc', 'f1'):
            result[name].append(m[name])
    return {key: dict(lower_95=float(np.percentile(v, 2.5)), upper_95=float(np.percentile(v, 97.5))) for key, v in result.items()}


def plot_results(output, history, rows, threshold):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [h['epoch'] for h in history]
    ax[0].plot(epochs, [h['train_loss'] for h in history], label='Train window loss')
    ax[0].plot(epochs, [h['val_group_log_loss'] for h in history], label='Validation group loss')
    ax[0].set(xlabel='Epoch', ylabel='BCE / log loss'); ax[0].legend()
    ax[1].plot(epochs, [h['val_group_balanced_accuracy_05'] for h in history], label='Validation balanced accuracy at 0.5')
    ax[1].plot(epochs, [h['val_group_auc'] for h in history], label='Validation AUROC')
    ax[1].set(xlabel='Epoch', ylabel='Score', ylim=(0, 1)); ax[1].legend()
    fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=160); plt.close(fig)
    y = np.array([r['label'] for r in rows]); p = np.array([r['probability_ad'] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cm = confusion_matrix(y, p >= threshold, labels=[0, 1])
    axes[0, 0].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2): axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
    axes[0, 0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Healthy', 'AD'], yticklabels=['Healthy', 'AD'], xlabel='Predicted', ylabel='True', title='Test group confusion matrix')
    fpr, tpr, _ = roc_curve(y, p); axes[0, 1].plot(fpr, tpr); axes[0, 1].plot([0, 1], [0, 1], '--')
    axes[0, 1].set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC')
    precision, recall, _ = precision_recall_curve(y, p); axes[0, 2].plot(recall, precision)
    axes[0, 2].axhline(y.mean(), ls='--', color='grey'); axes[0, 2].set(xlabel='Recall', ylabel='Precision', title='Test precision–recall')
    observed, predicted = calibration_curve(y, p, n_bins=5, strategy='quantile')
    axes[1, 0].plot(predicted, observed, 'o-'); axes[1, 0].plot([0, 1], [0, 1], '--')
    axes[1, 0].set(xlabel='Mean predicted probability', ylabel='Observed AD fraction', title='Test calibration (small sample)')
    for label, name in [(0, 'Healthy'), (1, 'AD')]:
        axes[1, 1].hist(p[y == label], bins=np.linspace(0, 1, 11), alpha=.5, label=name)
    axes[1, 1].axvline(threshold, color='black', ls='--'); axes[1, 1].legend()
    axes[1, 1].set(xlabel='Predicted AD probability', ylabel='Groups', title='Test probabilities')
    sources = sorted({r['source'] for r in rows})
    scores = [metrics([r for r in rows if r['source'] == s], threshold)['accuracy'] for s in sources]
    axes[1, 2].bar(sources, scores); axes[1, 2].tick_params(axis='x', labelrotation=20)
    axes[1, 2].set(ylim=(0, 1), ylabel='Accuracy', title='Test accuracy by source; check class counts')
    fig.tight_layout(); fig.savefig(output / 'test_diagnostics.png', dpi=160); plt.close(fig)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.windows_per_group < 1:
        raise ValueError('Epochs, batch size, patience and windows per group must be positive')
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid optimizer or dropout settings')
    from sklearn.metrics import confusion_matrix  # Validate dependency before training.
    root = Path(args.data).resolve()
    if not (root / 'READY.json').exists():
        raise ValueError('Preprocessing has not completed successfully (missing READY.json)')
    config = json.loads((root / 'config.json').read_text())
    rows = list(csv.DictReader((root / 'subjects.csv').open()))
    for r in rows:
        r['label'] = int(r['label']); r['n_windows'] = int(r['n_windows'])
    rows = [r for r in rows if r['n_windows'] > 0]
    memberships = defaultdict(set)
    for r in rows:
        memberships[r['group_id']].add(r['split'])
    if any(len(v) != 1 for v in memberships.values()):
        raise ValueError('Group leakage detected in supplied prepared data')
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != {0, 1}:
            raise ValueError(f'{split} must contain both classes')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing experiment results will not be overwritten')
    output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    with np.load(root / 'training_scaler.npz') as scale:
        mean, std = scale['mean'], scale['std']
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid training scaler')
    sets = {split: EEGDataset(root, [r for r in rows if r['split'] == split], mean, std) for split in ('train', 'val')}
    ngroups = len({r['group_id'] for r in sets['train'].rows})
    sampler = WeightedRandomSampler(sets['train'].balanced_weights(), num_samples=ngroups * args.windows_per_group,
                                   replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(sets['train'], batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(sets['val'], batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_args = dict(channels=len(config['channels']), temporal_filters=16, depth_multiplier=2, dropout=args.dropout)
    model = EEGCNN(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=6)
    criterion = nn.BCEWithLogitsLoss()
    metadata = dict(training=vars(args), preprocessing=config, architecture=model_args,
                    parameter_count=sum(p.numel() for p in model.parameters()), device=str(device),
                    versions=dict(torch=torch.__version__, numpy=np.__version__),
                    selection='minimum validation independent-group log loss',
                    notes='Exact duplicate recordings share a group and one evaluation vote. No promised accuracy.')
    save_json(output / 'experiment.json', metadata)
    write_csv(output / 'split_manifest.csv', rows)
    (output / 'architecture.txt').write_text(str(model) + '\n')
    best_loss = float('inf'); best_epoch = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.; count = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; inspect inputs and settings')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += loss.item() * len(y); count += len(y)
        _, _, val_groups = aggregate(sets['val'], predict(model, val_loader, device))
        m = metrics(val_groups, .5)
        scheduler.step(m['log_loss'])
        record = dict(epoch=epoch, train_loss=total/count, val_group_log_loss=m['log_loss'],
                      val_group_balanced_accuracy_05=m['balanced_accuracy'], val_group_auc=m['roc_auc'],
                      learning_rate=optimizer.param_groups[0]['lr'])
        history.append(record); write_csv(output / 'history.csv', history)
        checkpoint = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                          scheduler_state=scheduler.state_dict(), epoch=epoch, architecture=model_args,
                          channel_order=config['channels'], target_hz=config['target_hz'], window_seconds=config['window_seconds'],
                          normalization_mean=torch.tensor(mean), normalization_std=torch.tensor(std),
                          preprocessing=config, validation_log_loss=m['log_loss'])
        torch.save(checkpoint, output / 'last_model.pt')
        if m['log_loss'] < best_loss - 1e-5:
            best_loss, best_epoch = m['log_loss'], epoch
            torch.save(checkpoint, output / 'best_model.pt')
        print(f'Epoch {epoch:03d}: train loss={total/count:.4f}, validation group loss={m["log_loss"]:.4f}, BA={m["balanced_accuracy"]:.3f}, AUC={m["roc_auc"]:.3f}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    # Load only this run's checkpoint. Validation selects the threshold; test does not.
    checkpoint = torch.load(output / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    val_probs = predict(model, val_loader, device)
    _, _, val_groups = aggregate(sets['val'], val_probs)
    threshold = choose_threshold(val_groups)
    save_json(output / 'decision_threshold.json', dict(threshold=threshold, chosen_on='validation groups only', best_epoch=best_epoch))
    checkpoint['decision_threshold'] = threshold
    torch.save(checkpoint, output / 'best_model.pt')
    # Test arrays are first loaded after architecture/training/threshold selection is complete.
    sets['test'] = EEGDataset(root, [r for r in rows if r['split'] == 'test'], mean, std)
    results = {}
    test_groups = None
    for split in ('train', 'val', 'test'):
        ds = sets[split]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        probs = val_probs if split == 'val' else predict(model, loader, device)
        window_rows, recording_rows, group_rows = aggregate(ds, probs)
        for level, level_rows in [('window', window_rows), ('recording', recording_rows), ('group', group_rows)]:
            for r in level_rows: r['prediction'] = int(r['probability_ad'] >= threshold)
            write_csv(output / f'{split}_{level}_predictions.csv', level_rows)
        per_source = {src: metrics([r for r in group_rows if r['source'] == src], threshold)
                      for src in sorted({r['source'] for r in group_rows})}
        results[split] = dict(group=metrics(group_rows, threshold), group_threshold_05=metrics(group_rows, .5),
                              recording=metrics(recording_rows, threshold), window=metrics(window_rows, threshold),
                              per_source_group=per_source)
        if split == 'test':
            write_csv(output / 'test_misclassified_groups.csv', [r for r in group_rows if r['prediction'] != r['label']])
            test_groups = group_rows
            results[split]['group_bootstrap_95ci'] = bootstrap_ci(group_rows, threshold, args.bootstrap, args.seed)
            # Equal window count sensitivity analysis: earliest K valid windows of every recording.
            k = min(r['n_windows'] for r in ds.rows)
            equal = [r for r in window_rows if r['window_index'] < k]
            by_subject = defaultdict(list)
            for r in equal: by_subject[r['subject_id']].append(r)
            by_group = defaultdict(list)
            for subject, rr in by_subject.items():
                by_group[rr[0]['group_id']].append(dict(label=rr[0]['label'], probability_ad=float(np.mean([v['probability_ad'] for v in rr]))))
            fixed = [dict(group_id=g, label=rr[0]['label'], probability_ad=float(np.mean([v['probability_ad'] for v in rr]))) for g, rr in by_group.items()]
            write_csv(output / 'test_equal_window_group_predictions.csv', fixed)
            results[split]['equal_window_group'] = dict(windows_per_recording=k, metrics=metrics(fixed, threshold))
    save_json(output / 'metrics.json', results)
    plot_results(output, history, test_groups, threshold)
    (output / 'RESULTS.md').write_text(
        f'Best epoch: {best_epoch}. Validation-selected threshold: {threshold:.2f}.\n\n'
        f'Test independent-group accuracy: {results["test"]["group"]["accuracy"]:.3%}. '
        f'Balanced accuracy: {results["test"]["group"]["balanced_accuracy"]:.3%}.\n\n'
        'See metrics.json for sensitivity, specificity, AUROC, calibration loss, confidence intervals and source-level results. '
        'Window metrics are secondary: windows are correlated. Duplicate arrays and repeated people share one group vote. '
        'Bootstrap intervals describe this fixed test split, not all training/split uncertainty. '
        'Repeated model selection using these test results invalidates their held-out interpretation. '
        'Random source-stratified splitting alone cannot establish generalization to a new acquisition source.\n')
    print(f'Saved model and results to {output}', flush=True)
    print(json.dumps(jsonable(results['test']['group']), indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='prepared_eeg')
    p.add_argument('--output', default='cnn_results')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--windows-per-group', type=int, default=16, help='Expected sampled windows per independent training group per epoch')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    return p


if __name__ == '__main__':
    try:
        train(parser().parse_args())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
