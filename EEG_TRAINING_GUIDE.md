The two requested scripts are preprocess_no_issues_dataset.py and cnn.py. They implement a compact temporal/spatial EEG CNN, not a guaranteed 96–97% classifier. Honest accuracy depends on independent subjects, label quality, acquisition differences and usable data.

The pipeline was smoke-tested with Python 3.10, NumPy 2.2.6, SciPy 1.14.1 and PyTorch 2.11.0. The existing system SciPy 1.15.3 binary failed to load on this machine; the requirements pin the tested SciPy version. Use a fresh virtual environment:

```bash
python3 -m venv .venv-eeg
source .venv-eeg/bin/activate
python -m pip install -r eeg_training_requirements.txt
```

Fill eeg_manifest_template.csv before preprocessing. Each row already identifies a subject folder and its assigned class. Supply:

- person_id: verified, globally unique biological identity. All sessions from the same person must share this ID, including across sources. Different verified people should have different IDs.
- source: the originating dataset, consistently named.
- sampling_rate_hz: the actual rate of the exported arrays, not automatically the original dataset's native rate.
- unit: uV, mV or V, as verified from the export. Every signal is converted to microvolts before amplitude screening.

The scripts deliberately do not guess these fields from patient numbers, signal scale or recording length. Using the supplied original source rates without confirming whether exports were already resampled could distort the data. Review reference/montage compatibility too; default --reference keep preserves the existing reference. Use --reference average only when appropriate for the original recordings.

Run:

```bash
python preprocess_no_issues_dataset.py --manifest eeg_manifest_template.csv --output prepared_eeg
python cnn.py --data prepared_eeg --output cnn_results --epochs 150
```

Inspect preprocessing outputs before starting the second command. Defaults are a 0.5–40 Hz fourth-order Butterworth bandpass applied forward/backward, anti-aliased resampling to 128 Hz, four-second nonoverlapping windows and 0.5-second discarded boundaries at both ends of every valid continuous stretch. Forward/backward filtering is offline, not a causal streaming implementation. These settings are starting choices, not an assurance that every artifact is removed. Eight seconds at 128 Hz usually yields ONE accepted four-second window under these edge margins, not two. Extra margins may be needed for low-frequency filter transients. Filtering never bridges invalid/all-channel-zero intervals.

Default review thresholds reject any window with raw peak-to-peak above 500 microvolts on any channel, an adjacent raw step above 150 microvolts, a filtered channel peak-to-peak below 0.1 microvolts, or a constant run of at least 0.5 seconds. Raw step thresholds have rate-dependent behavior: inspect retention per source. These conservative gates may reject many windows in this dataset. They are configurable with --max-ptp-uv, --max-step-uv, --min-ptp-uv and --flat-seconds. Do not tune them using test accuracy. The code rejects suspect windows rather than automatically reconstructing bad electrodes or removing components with ICA. No mains-notch is imposed without evidence. A bandpass does not guarantee removal of mains contamination.

The script groups repeated person IDs and exact matching numeric recordings into the same partition; it does not delete the distinct-person records you identified. Because identical arrays provide no independent signal evidence, these groups share training weight and one main evaluation vote. It does not claim to detect re-filtered copies or all overlapping excerpts. For a group linking inconsistent labels, preprocessing stops. Splits are roughly 70/15/15, stratified by source and class; small strata use rounded group counts. At least three independent groups per source/class stratum are required. Cleaning exclusions are logged without silently changing partitions. If exclusions empty a class in a partition, preparation stops for review.

Normalization is per channel, fitted using training groups only, with equal weight per group and equal weight per recording inside each group. The .npz window files remain in microvolts; cnn.py applies the saved training mean/std in memory. No test or validation values are used to estimate them. Training samples classes equally, then independent groups equally in expectation, then recordings and windows equally within groups. This avoids long records dominating training, but repeated short windows remain correlated. Windows are loaded into RAM; plan for several GB for a full retained dataset.

The CNN contains temporal convolutions, depthwise mixing across electrodes, separable temporal convolutions, batch normalization, ELU, pooling and dropout. It uses AdamW, gradient clipping, learning-rate reduction and early stopping. Minimum validation group log loss selects the epoch; validation balanced accuracy selects a threshold on a fixed grid. It also reports the untuned 0.5 threshold. Test data are evaluated only after selection. The default 16 sampled windows per group per epoch is an expected contribution, not 16 newly independent examples. A GPU or Apple MPS is selected automatically when available; use --device cpu to force CPU. Deterministic settings are requested, but exact results can differ across devices and library versions.

Preprocessing outputs:

- config.json, subjects.csv, split_summary.json: all settings, splits and retained counts.
- windows/*.npz: float32 [windows, 19 channels, time] arrays and original source-sample intervals.
- window_quality.csv: accepted/rejected intervals and reasons, including invalid gaps.
- identical_recording_groups.json and excluded_no_clean_windows.json.
- training_scaler.npz and READY.json. Training refuses an incomplete preparation.

Training outputs:

- best_model.pt: selected network weights, architecture, threshold, channel order, target rate, window length, normalization values, preprocessing settings and optimizer/scheduler state.
- last_model.pt: final training-epoch checkpoint (not necessarily the selected model).
- experiment.json, architecture.txt, split_manifest.csv, history.csv, decision_threshold.json and RESULTS.md.
- Train/validation/test prediction CSVs at window, recording and independent-group levels.
- metrics.json: accuracy, balanced accuracy, sensitivity, specificity, precision, NPV, F1, MCC, Cohen's kappa, AUROC, average precision, log loss, Brier score, confusion matrices, class reports and per-source group results.
- Independent-group stratified bootstrap 95% intervals for test accuracy, balanced accuracy, sensitivity, specificity, AUROC and F1. Small test sets still give uncertain estimates; these intervals do not include training/split uncertainty.
- test_equal_window_group_predictions.csv and associated metrics: every recording contributes its earliest K accepted windows, with K equal to the smallest retained test-window count. This is an additional duration-controlled analysis, not a replacement for cross-source testing.
- learning_curves.png and test_diagnostics.png: loss, validation balanced accuracy/AUROC, test confusion matrix, ROC, precision–recall, calibration, probability distributions and source accuracy.
- test_misclassified_groups.csv when there are errors.

Reloading the best model for already cleaned windows:

```python
import torch
from cnn import EEGCNN
checkpoint = torch.load('cnn_results/best_model.pt', map_location='cpu', weights_only=True)
model = EEGCNN(**checkpoint['architecture'])
model.load_state_dict(checkpoint['model_state'])
model.eval()
# windows_uv must already follow the saved channel order, filtering, reference,
# rate and duration; shape is [batch, channels, samples].
x = torch.as_tensor(windows_uv, dtype=torch.float32)
x = (x - checkpoint['normalization_mean'][None, :, None]) / checkpoint['normalization_std'][None, :, None]
with torch.no_grad():
    probability_ad = torch.sigmoid(model(x[:, None]))
prediction_ad = probability_ad >= checkpoint['decision_threshold']
```

No actual dataset training has been performed: verified metadata are still required. Do not repeatedly optimize the model on the reported test scores. The source-stratified split includes the same sources in train and test and cannot establish new-source generalization. In particular, a Healthy-only source remains a confound even after normalization; interpret its accuracy as specificity, not full Alzheimer’s classification performance. A primary experiment restricted to sources containing both classes, plus separate source-held-out evaluation, is advisable before making generalization claims. Different subsets can be specified in separately reviewed manifests.
