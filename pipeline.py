import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import butter, filtfilt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten,
    Input
)
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
FS = 128
EPOCH_SEC = 1
BAND = (0.5, 45)

ROOT = "FINAL_DATASET"
HEALTHY_FOLDERS = ["eyes_closed"]
AD_FOLDER = "eyes_closed"

N_FOLDS = 5

SAVE_DIR = "saved_models"
METRIC_DIR = "metrics"
CM_DIR = "confusion_matrices"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

CHANNELS = [
    "Fp1","Fp2","F3","F4","F7","F8","Fz",
    "C3","C4","Cz",
    "P3","P4","Pz",
    "T3","T4","T5","T6",
    "O1","O2"
]

# ============ SIGNAL PROCESSING ============
def butter_bandpass(low, high, fs, order=6):
    nyq = 0.5 * fs
    return butter(order, [low/nyq, high/nyq], btype="band")

def bandpass_filter(eeg):
    b, a = butter_bandpass(*BAND, FS)
    eeg = filtfilt(b, a, eeg, axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-8)
    return eeg.astype(np.float32)

def epoch_signal(eeg):
    samples = FS * EPOCH_SEC
    epochs = []
    for i in range(0, eeg.shape[0], samples):
        ep = eeg[i:i+samples]
        if ep.shape[0] == samples:
            epochs.append(ep.astype(np.float32))
    return np.asarray(epochs, dtype=np.float32)

# ============ DATA LOADING ============
def load_patient(folder):
    signals = []
    for ch in CHANNELS:
        path = os.path.join(folder, f"{ch}.txt")
        if not os.path.exists(path):
            return None
        signals.append(np.loadtxt(path, dtype=np.float32))
    return np.stack(signals, axis=1).astype(np.float32)

def load_all():
    X, y = [], []

    healthy_base = os.path.join(ROOT, "Healthy")
    healthy_count = 0

    for folder in HEALTHY_FOLDERS:
        base = os.path.join(healthy_base, folder)
        if not os.path.exists(base):
            continue

        for patient in os.listdir(base):
            pdir = os.path.join(base, patient)
            if not os.path.isdir(pdir):
                continue

            eeg = load_patient(pdir)
            if eeg is None:
                continue

            eeg = bandpass_filter(eeg)
            epochs = epoch_signal(eeg)

            if len(epochs) == 0:
                continue

            X.append(epochs)
            y.append(0)
            healthy_count += 1

    ad_base = os.path.join(ROOT, "Alzehiemers", AD_FOLDER)
    ad_count = 0

    for patient in os.listdir(ad_base):
        pdir = os.path.join(ad_base, patient)
        if not os.path.isdir(pdir):
            continue

        eeg = load_patient(pdir)
        if eeg is None:
            continue

        eeg = bandpass_filter(eeg)
        epochs = epoch_signal(eeg)

        if len(epochs) == 0:
            continue

        X.append(epochs)
        y.append(1)
        ad_count += 1

    print(f"\nDataset → Healthy: {healthy_count} | AD: {ad_count}")
    return X, np.array(y, dtype=np.int32)

# ============ CNN MODEL ============
def build_cnn(filters, dense_units, lr):
    model = Sequential([Input(shape=(128,19,1))])

    for f in filters:
        model.add(Conv2D(f, 3, padding="same", activation="relu"))
        model.add(MaxPooling2D())

    model.add(Flatten())

    for d in dense_units:
        model.add(Dense(d, activation="relu"))

    model.add(Dense(2, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============ BALANCE ============
def balance_epochs(X, y):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    min_count = min(len(idx0), len(idx1))

    sel0 = np.random.choice(idx0, min_count, replace=False)
    sel1 = np.random.choice(idx1, min_count, replace=False)

    sel = np.concatenate([sel0, sel1])
    np.random.shuffle(sel)

    return X[sel], y[sel]

# ============ MODELS ============
PAPER_CNNS = [
    {"name": "CNN1", "filters": [32,64,128,256], "dense":[512,256,128,64,32,16], "lr":1e-4, "epochs":12},
    {"name": "CNN2", "filters": [32,64,128,256], "dense":[512,256,128,64], "lr":1e-3, "epochs":10},
    {"name": "CNN3", "filters": [32,64,128], "dense":[512,256,128,64,32,16], "lr":1e-3, "epochs":8},
    {"name": "CNN4", "filters": [128,128,128], "dense":[512,256,128,64,32,16], "lr":1e-3, "epochs":10},
    {"name": "CNN5", "filters": [128,64,32], "dense":[512,256,128,64,32,16], "lr":1e-2, "epochs":10}
]

# ============ MAIN ============
def main():
    Xp, yp = load_all()

    for cfg in PAPER_CNNS:
        print(f"\n======= {cfg['name']} =======")

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        for fold, (tr, te) in enumerate(skf.split(Xp, yp), 1):

            Xtr = np.vstack([Xp[i] for i in tr])[..., None]
            ytr = np.hstack([[yp[i]] * len(Xp[i]) for i in tr])

            Xtr, ytr = balance_epochs(Xtr, ytr)

            classes = np.unique(ytr)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=ytr
            )
            class_weights = dict(zip(classes, weights))

            model = build_cnn(cfg["filters"], cfg["dense"], cfg["lr"])

            model.fit(
                Xtr, ytr,
                epochs=cfg["epochs"],
                batch_size=32,
                class_weight=class_weights,
                verbose=0
            )

            # Patient-level evaluation
            y_true, y_pred = [], []

            for i in te:
                patient_epochs = Xp[i][..., None]
                true_label = yp[i]

                probs = model.predict(patient_epochs, verbose=0)
                avg_probs = np.mean(probs, axis=0)
                final_pred = np.argmax(avg_probs)

                y_true.append(true_label)
                y_pred.append(final_pred)

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=1)
            rec  = recall_score(y_true, y_pred, zero_division=1)
            f1   = f1_score(y_true, y_pred, zero_division=1)

            print(f"{cfg['name']} – Fold {fold}")
            print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

            # -------- CONFUSION MATRIX --------
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(4,4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "AD"],
                yticklabels=["Healthy", "AD"]
            )
            plt.title(f"{cfg['name']} – Fold {fold}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(f"{CM_DIR}/{cfg['name']}_fold{fold}.png")
            plt.close()

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
