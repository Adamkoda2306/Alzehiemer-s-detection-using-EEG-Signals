# FINAL IMPROVEDDNN LOSO PIPELINE
# Alzheimer\'s vs Healthy EEG classification
# Dataset:
# FINAL_DATASET/Alzehiemers/eyes-closed/patientXX/<19 channels>.txt
# FINAL_DATASET/Healthy/eyes-closed/patientXX/<19 channels>.txt

import os, gc, re, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, detrend
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss, roc_curve,
    precision_recall_curve)

warnings.filterwarnings("ignore")

# ================= CONFIG =================
SEED=42
FS=128
EPOCH_SEC=8
EPOCH_SAMPLES=FS*EPOCH_SEC
BAND=(0.5,45.0)
ROOT=Path("FINAL_DATASET")
ALZ_FOLDER=ROOT/"Alzehiemers"/"eyes-closed"
HEALTHY_FOLDER=ROOT/"Healthy"/"eyes-closed"
CHANNELS=["Fp1","Fp2","F3","F4","F7","F8","Fz","C3","C4","Cz","P3","P4","Pz","T3","T4","T5","T6","O1","O2"]

BATCH_SIZE=32
MAX_EPOCHS=100
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-5
L2_REG=1e-4
DROPOUT_RATE=0.35
VALIDATION_SIZE=0.15
EARLY_STOPPING_PATIENCE=15
REDUCE_LR_PATIENCE=6

# Set None for all epochs. Use e.g. 30 only for debugging.
LOSO_MAX_EPOCHS_PER_SUBJECT=None
MAX_TRAIN_EPOCHS_PER_SUBJECT=None
LOSO_START_INDEX=0
LOSO_END_INDEX=None

RESULTS_DIR=Path("results_improved_dnn_loso")
PLOTS_DIR=RESULTS_DIR/"plots"
CACHE_DIR=RESULTS_DIR/"preprocessed_cache"
MODELS_DIR=RESULTS_DIR/"models"
for d in [RESULTS_DIR,PLOTS_DIR,CACHE_DIR,MODELS_DIR]: d.mkdir(parents=True,exist_ok=True)
CHECKPOINT_PATH=RESULTS_DIR/"improved_dnn_loso_subject_results.csv"
PROB_COL="Subject Probability Alzheimer's"
USE_DISK_CACHE=True
FORCE_REBUILD_CACHE=False
SAVE_MODELS=False

# ================= REPRODUCIBILITY =================
def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
seed_everything()

# ================= DATA LOADING =================
def load_single_channel_txt(path):
    text=Path(path).read_text(encoding="utf-8",errors="ignore")
    text=text.replace(","," ").replace(";"," ").replace("\t"," ")
    vals=re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",text)
    if not vals: raise ValueError(f"No numeric samples: {path}")
    x=np.asarray(vals,dtype=np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0: raise ValueError(f"No finite samples: {path}")
    return x

def load_subject_raw(folder):
    data={}; missing=[]
    for ch in CHANNELS:
        p=Path(folder)/f"{ch}.txt"
        if not p.exists(): missing.append(ch)
        else: data[ch]=load_single_channel_txt(p)
    if missing: raise FileNotFoundError(f"{Path(folder).name}: missing {missing}")
    n=min(len(data[ch]) for ch in CHANNELS)
    if n<EPOCH_SAMPLES: raise ValueError(f"Recording too short: {n}")
    return np.column_stack([data[ch][:n] for ch in CHANNELS])

def preprocess_subject(folder):
    x=load_subject_raw(folder)
    x=detrend(x,axis=0,type="linear")
    sos=butter(4,BAND,btype="bandpass",fs=FS,output="sos")
    x=sosfiltfilt(sos,x,axis=0)
    n_epochs=len(x)//EPOCH_SAMPLES
    x=x[:n_epochs*EPOCH_SAMPLES].reshape(n_epochs,EPOCH_SAMPLES,len(CHANNELS))
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    return x.astype(np.float32)

# ================= SUBJECT DISCOVERY =================
def discover_subjects():
    if not ALZ_FOLDER.exists(): raise FileNotFoundError(f"Not found: {ALZ_FOLDER.resolve()}")
    if not HEALTHY_FOLDER.exists(): raise FileNotFoundError(f"Not found: {HEALTHY_FOLDER.resolve()}")
    rows=[]
    for f in sorted(ALZ_FOLDER.iterdir()):
        if f.is_dir(): rows.append({"subject_id":f"AD__{f.name}","original_subject_id":f.name,"path":str(f.resolve()),"label":1,"class_name":"Alzheimer's"})
    for f in sorted(HEALTHY_FOLDER.iterdir()):
        if f.is_dir(): rows.append({"subject_id":f"HC__{f.name}","original_subject_id":f.name,"path":str(f.resolve()),"label":0,"class_name":"Healthy"})
    return pd.DataFrame(rows)

def build_cache(subjects):
    cache={}; failures=[]
    for i,row in subjects.iterrows():
        sid=row.subject_id
        print(f"[{i+1}/{len(subjects)}] Loading {sid}")
        try:
            cp=CACHE_DIR/(re.sub(r"[^A-Za-z0-9_.-]","_",sid)+".npy")
            if USE_DISK_CACHE and cp.exists() and not FORCE_REBUILD_CACHE:
                X=np.load(cp)
            else:
                X=preprocess_subject(row.path)
                if USE_DISK_CACHE: np.save(cp,X)
            if X.ndim!=3 or X.shape[1:]!=(EPOCH_SAMPLES,len(CHANNELS)):
                raise ValueError(f"Invalid shape {X.shape}")
            if LOSO_MAX_EPOCHS_PER_SUBJECT is not None: X=X[:LOSO_MAX_EPOCHS_PER_SUBJECT]
            if len(X)==0: raise ValueError("Zero epochs")
            cache[sid]={"X":X.astype(np.float32),"label":int(row.label),"class_name":row.class_name,"original_subject_id":row.original_subject_id}
            print("  OK",X.shape)
        except Exception as e:
            print("  FAILED:",e); failures.append({"subject_id":sid,"path":row.path,"error":str(e)})
    pd.DataFrame(failures).to_csv(RESULTS_DIR/"failed_subjects.csv",index=False)
    return cache

# ================= IMPROVED DNN =================
def se_block(x,reduction=8,name="se"):
    c=int(x.shape[-1])
    s=layers.GlobalAveragePooling1D(name=name+"_gap")(x)
    s=layers.Dense(max(c//reduction,4),activation="relu",name=name+"_fc1")(s)
    s=layers.Dense(c,activation="sigmoid",name=name+"_fc2")(s)
    s=layers.Reshape((1,c),name=name+"_reshape")(s)
    return layers.Multiply(name=name+"_scale")([x,s])

def residual_block(x,filters,kernel,name):
    shortcut=x
    y=layers.Conv1D(filters,kernel,padding="same",kernel_regularizer=regularizers.l2(L2_REG),name=name+"_conv")(x)
    y=layers.BatchNormalization(name=name+"_bn1")(y)
    y=layers.Activation("relu",name=name+"_relu1")(y)
    y=layers.SeparableConv1D(filters,kernel,padding="same",depthwise_regularizer=regularizers.l2(L2_REG),pointwise_regularizer=regularizers.l2(L2_REG),name=name+"_sep")(y)
    y=layers.BatchNormalization(name=name+"_bn2")(y)
    if int(shortcut.shape[-1])!=filters:
        shortcut=layers.Conv1D(filters,1,padding="same",kernel_regularizer=regularizers.l2(L2_REG),name=name+"_proj")(shortcut)
        shortcut=layers.BatchNormalization(name=name+"_projbn")(shortcut)
    y=layers.Add(name=name+"_add")([shortcut,y])
    y=layers.Activation("relu",name=name+"_relu2")(y)
    return layers.Dropout(DROPOUT_RATE,name=name+"_drop")(y)

def build_improved_dnn():
    inp=layers.Input(shape=(EPOCH_SAMPLES,len(CHANNELS)),name="eeg_input")
    x=layers.Conv1D(32,9,strides=2,padding="same",kernel_regularizer=regularizers.l2(L2_REG),name="stem_conv")(inp)
    x=layers.BatchNormalization(name="stem_bn")(x)
    x=layers.Activation("relu",name="stem_relu")(x)
    x=layers.MaxPooling1D(2,name="stem_pool")(x)
    x=residual_block(x,32,7,"res1"); x=layers.MaxPooling1D(2,name="pool1")(x)
    x=residual_block(x,64,5,"res2"); x=layers.MaxPooling1D(2,name="pool2")(x)
    x=residual_block(x,96,3,"res3")
    x=se_block(x,8,"se_attention")
    x=layers.GlobalAveragePooling1D(name="gap")(x)
    x=layers.Dense(64,activation="relu",kernel_regularizer=regularizers.l2(L2_REG),name="dense1")(x)
    x=layers.Dropout(DROPOUT_RATE,name="classifier_dropout")(x)
    out=layers.Dense(1,activation="sigmoid",name="output")(x)
    return models.Model(inp,out,name="ImprovedDNN")

# ================= HELPERS =================
def build_epochs(ids,cache,max_per_subject=None):
    xs=[]; ys=[]; sids=[]
    for sid in ids:
        X=cache[sid]["X"]
        if max_per_subject is not None and len(X)>max_per_subject:
            idx=np.linspace(0,len(X)-1,max_per_subject,dtype=int); X=X[idx]
        xs.append(X); ys.append(np.full(len(X),cache[sid]["label"],dtype=np.int32)); sids.extend([sid]*len(X))
    return np.concatenate(xs),np.concatenate(ys),np.asarray(sids)

def normalize_fit(X):
    mean=X.mean(axis=(0,1),keepdims=True); std=np.maximum(X.std(axis=(0,1),keepdims=True),1e-8)
    return mean.astype(np.float32),std.astype(np.float32)

def normalize(X,mean,std): return ((X-mean)/std).astype(np.float32)

def class_weights(y):
    cls=np.unique(y); w=compute_class_weight("balanced",classes=cls,y=y)
    return {int(c):float(v) for c,v in zip(cls,w)}

def epoch_probs(model,X):
    return model.predict(X,batch_size=BATCH_SIZE,verbose=0).reshape(-1)

def aggregate(p,method):
    return float(np.mean(p)) if method=="mean" else float(np.median(p))

def subject_predictions(model,X,sids,cache,method):
    p=epoch_probs(model,X); rows=[]
    for sid in np.unique(sids):
        q=p[sids==sid]
        rows.append({"subject_id":sid,"true_label":cache[sid]["label"],"probability":aggregate(q,method)})
    return pd.DataFrame(rows)

def select_threshold(model,X_val,val_sids,cache):
    rows=[]
    for method in ["mean","median"]:
        df=subject_predictions(model,X_val,val_sids,cache,method)
        yt=df.true_label.to_numpy(int); prob=df.probability.to_numpy(float)
        for t in np.arange(.05,.951,.01):
            yp=(prob>=t).astype(int)
            rows.append({"method":method,"threshold":float(t),
                         "balanced_accuracy":balanced_accuracy_score(yt,yp),
                         "f1":f1_score(yt,yp,zero_division=0),
                         "distance":abs(float(t)-.5)})
    out=pd.DataFrame(rows).sort_values(["balanced_accuracy","f1","distance"],ascending=[False,False,True]).reset_index(drop=True)
    return out.iloc[0].to_dict(),out

# ================= LOSO =================
def run_loso(cache):
    all_ids=sorted(cache)
    existing=pd.read_csv(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else pd.DataFrame()
    done=set(existing["Subject ID"].astype(str)) if len(existing) and "Subject ID" in existing else set()
    end=len(all_ids) if LOSO_END_INDEX is None else min(LOSO_END_INDEX,len(all_ids))
    results=existing.copy()

    for fold,test_id in enumerate(all_ids[LOSO_START_INDEX:end],start=LOSO_START_INDEX+1):
        if test_id in done:
            print(f"Skipping completed {test_id}"); continue
        print("\n"+"#"*80); print(f"LOSO {fold}/{len(all_ids)} TEST={test_id}")

        tf.keras.backend.clear_session(); gc.collect(); seed_everything(SEED+fold)
        remaining=[s for s in all_ids if s!=test_id]
        labels=np.array([cache[s]["label"] for s in remaining])
        splitter=StratifiedShuffleSplit(n_splits=1,test_size=VALIDATION_SIZE,random_state=SEED+fold)
        tr_idx,val_idx=next(splitter.split(np.zeros(len(remaining)),labels))
        train_ids=[remaining[i] for i in tr_idx]; val_ids=[remaining[i] for i in val_idx]

        Xtr,ytr,_=build_epochs(train_ids,cache,MAX_TRAIN_EPOCHS_PER_SUBJECT)
        Xv,yv,val_sids=build_epochs(val_ids,cache)
        Xtest=cache[test_id]["X"]

        mean,std=normalize_fit(Xtr)
        Xtr=normalize(Xtr,mean,std); Xv=normalize(Xv,mean,std); Xtest=normalize(Xtest,mean,std)

        model=build_improved_dnn()
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE,weight_decay=WEIGHT_DECAY),
                      loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(name="auc"),"accuracy",
                               tf.keras.metrics.Precision(name="precision"),
                               tf.keras.metrics.Recall(name="recall")])

        hist=model.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,
                       class_weight=class_weights(ytr),
                       callbacks=[
                           callbacks.EarlyStopping(monitor="val_auc",mode="max",patience=EARLY_STOPPING_PATIENCE,restore_best_weights=True,verbose=0),
                           callbacks.ReduceLROnPlateau(monitor="val_auc",mode="max",factor=.5,patience=REDUCE_LR_PATIENCE,min_lr=1e-6,verbose=0)
                       ],verbose=0)

        best,search=select_threshold(model,Xv,val_sids,cache)
        search.to_csv(RESULTS_DIR/f"threshold_search_{test_id}.csv",index=False)

        p_epochs=epoch_probs(model,Xtest)
        prob=aggregate(p_epochs,best["method"]); threshold=float(best["threshold"])
        pred=int(prob>=threshold); true=cache[test_id]["label"]; correct=pred==true
        margin=abs(prob-threshold)

        if correct: error="Correct - Low Confidence" if margin<.10 else "Correct - High Confidence"
        elif true==0: error="Strong False Positive" if prob>=.70 else "False Positive"
        else: error="Strong False Negative" if prob<=.30 else "False Negative"

        result={"LOSO Fold":fold,"Subject ID":test_id,"Original Subject ID":cache[test_id]["original_subject_id"],
                "Class":cache[test_id]["class_name"],"True Label":true,"Predicted Label":pred,
                "Predicted Class":"Alzheimer's" if pred else "Healthy",PROB_COL:prob,"Threshold":threshold,
                "Aggregation Method":best["method"],"Validation Balanced Accuracy":float(best["balanced_accuracy"]),
                "Epoch Count":len(p_epochs),"Correct":bool(correct),"Confidence Margin":margin,
                "Mean Epoch Probability":float(np.mean(p_epochs)),"Median Epoch Probability":float(np.median(p_epochs)),
                "Std Epoch Probability":float(np.std(p_epochs)),"Min Epoch Probability":float(np.min(p_epochs)),
                "Max Epoch Probability":float(np.max(p_epochs)),"Error Type":error,
                "Train Subjects":len(train_ids),"Validation Subjects":len(val_ids),
                "Epochs Trained":len(hist.history.get("loss",[])),
                "Best Validation AUC":float(max(hist.history.get("val_auc",[np.nan])))}

        results=pd.concat([results,pd.DataFrame([result])],ignore_index=True)
        results=results.drop_duplicates("Subject ID",keep="last").reset_index(drop=True)
        results.to_csv(CHECKPOINT_PATH,index=False)
        print(f"Result: true={true}, pred={pred}, prob={prob:.4f}, threshold={threshold:.2f}, {error}")

        del model,hist,Xtr,ytr,Xv,yv,val_sids,Xtest
        tf.keras.backend.clear_session(); gc.collect()

    return pd.read_csv(CHECKPOINT_PATH)

# ================= FINAL ANALYSIS =================
def metrics(y_true,y_pred,y_prob):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return {"Accuracy":accuracy_score(y_true,y_pred),"Balanced Accuracy":balanced_accuracy_score(y_true,y_pred),
            "Sensitivity / Recall":tp/(tp+fn) if tp+fn else np.nan,
            "Specificity":tn/(tn+fp) if tn+fp else np.nan,
            "Precision":precision_score(y_true,y_pred,zero_division=0),"F1 Score":f1_score(y_true,y_pred,zero_division=0),
            "ROC-AUC":roc_auc_score(y_true,y_prob),"PR-AUC":average_precision_score(y_true,y_prob),
            "MCC":matthews_corrcoef(y_true,y_pred),"Cohen Kappa":cohen_kappa_score(y_true,y_pred),
            "Brier Score":brier_score_loss(y_true,y_prob),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def final_analysis(df):
    df=df.drop_duplicates("Subject ID",keep="last").sort_values(["Class","Original Subject ID"]).reset_index(drop=True)
    df.to_csv(RESULTS_DIR/"improved_dnn_loso_all_subjects.csv",index=False)
    y=df["True Label"].astype(int).to_numpy(); yp=df["Predicted Label"].astype(int).to_numpy(); p=df[PROB_COL].astype(float).to_numpy()
    cm=confusion_matrix(y,yp,labels=[0,1])
    pd.DataFrame(cm,index=["True Healthy","True Alzheimer's"],columns=["Pred Healthy","Pred Alzheimer's"]).to_csv(RESULTS_DIR/"confusion_matrix.csv")

    # Raw and normalized confusion matrices
    for normalized,name in [(False,"confusion_matrix.png"),(True,"normalized_confusion_matrix.png")]:
        v=cm.astype(float)/np.maximum(cm.sum(axis=1,keepdims=True),1) if normalized else cm
        plt.figure(figsize=(7,6)); plt.imshow(v); plt.colorbar()
        plt.xticks([0,1],["Healthy","Alzheimer's"]); plt.yticks([0,1],["Healthy","Alzheimer's"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Normalized Confusion Matrix" if normalized else "Final LOSO Confusion Matrix")
        for i in range(2):
            for j in range(2): plt.text(j,i,f"{v[i,j]*100:.2f}%" if normalized else str(int(v[i,j])),ha="center",va="center",fontsize=15)
        plt.tight_layout(); plt.savefig(PLOTS_DIR/name,dpi=300,bbox_inches="tight"); plt.close()

    m=metrics(y,yp,p)
    mdf=pd.DataFrame(list(m.items()),columns=["Metric","Value"]); mdf.to_csv(RESULTS_DIR/"final_metrics.csv",index=False)

    fpr,tpr,_=roc_curve(y,p); plt.figure(figsize=(7,6)); plt.plot(fpr,tpr,label=f"ROC-AUC={m['ROC-AUC']:.4f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Final LOSO ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"roc_curve.png",dpi=300,bbox_inches="tight"); plt.close()

    precision,recall,_=precision_recall_curve(y,p); plt.figure(figsize=(7,6)); plt.plot(recall,precision,label=f"PR-AUC={m['PR-AUC']:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Final LOSO Precision-Recall Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"pr_curve.png",dpi=300,bbox_inches="tight"); plt.close()

    healthy=df[df["True Label"]==0].copy(); alz=df[df["True Label"]==1].copy()
    healthy.to_csv(RESULTS_DIR/"healthy_subjects.csv",index=False); alz.to_csv(RESULTS_DIR/"alzheimers_subjects.csv",index=False)

    severity={"Strong False Negative":1,"Strong False Positive":1,"False Negative":2,"False Positive":2,"Correct - Low Confidence":3,"Correct - High Confidence":4}
    difficult=df.copy(); difficult["Severity Rank"]=difficult["Error Type"].map(severity).fillna(5)
    difficult=difficult.sort_values(["Severity Rank","Confidence Margin","Std Epoch Probability"],ascending=[True,True,False])
    difficult.to_csv(RESULTS_DIR/"difficult_subjects.csv",index=False)
    difficult.head(20).to_csv(RESULTS_DIR/"top_20_difficult_subjects.csv",index=False)

    wrong=df[df["Correct"]==False].copy(); wrong.to_csv(RESULTS_DIR/"misclassified_subjects.csv",index=False)

    summary={"Total Subjects":len(df),"Healthy Subjects":int((y==0).sum()),"Alzheimer's Subjects":int((y==1).sum()),
             "Correct Predictions":int((y==yp).sum()),"Incorrect Predictions":int((y!=yp).sum()),**m}
    sdf=pd.DataFrame(list(summary.items()),columns=["Metric","Value"]); sdf.to_csv(RESULTS_DIR/"complete_summary.csv",index=False)

    print("\nFINAL CONFUSION MATRIX\n",cm)
    print("\nFINAL METRICS")
    for k,v in m.items(): print(f"{k:25}: {v:.4f}" if isinstance(v,float) else f"{k:25}: {v}")
    print("\nMISCLASSIFIED SUBJECTS")
    print(wrong[["Subject ID","Class","Predicted Class",PROB_COL,"Threshold","Error Type"]].to_string(index=False) if len(wrong) else "None")
    return df,m

# ================= MAIN =================
if __name__=="__main__":
    print("="*80); print("FINAL IMPROVEDDNN SUBJECT-LEVEL LOSO PIPELINE"); print("="*80)
    subjects=discover_subjects()
    subjects.to_csv(RESULTS_DIR/"discovered_subjects.csv",index=False)
    print(subjects["class_name"].value_counts())

    cache=build_cache(subjects)
    print(f"\nValid subjects loaded: {len(cache)}")
    if len(cache)<3: raise RuntimeError("Too few valid subjects")

    results=run_loso(cache)
    final_analysis(results)

    print("\nDONE")
    print("Results:",RESULTS_DIR.resolve())
