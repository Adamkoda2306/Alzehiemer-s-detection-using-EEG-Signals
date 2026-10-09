from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
def welch(x,fs=1,nperseg=2048):
    if len(x)<2:return np.array([0.]),np.array([0.])
    size=nperseg; hop=max(1,size//2)
    blocks=np.lib.stride_tricks.sliding_window_view(x,size)[::hop]
    win=np.hanning(size)
    power=np.zeros(size//2+1)
    for start in range(0,len(blocks),256):
        b=blocks[start:start+256];b=(b-b.mean(axis=1,keepdims=True))*win
        power+=(abs(np.fft.rfft(b,axis=1))**2).sum(axis=0)
    power/=len(blocks);power[1:-1]*=2
    return np.fft.rfftfreq(size,1/fs),power
import hashlib,json,csv,time
ROOT=Path('/Users/adamk/Work/BTP/FINAL_DATASET')
OUT=Path('/Users/adamk/Work/BTP/final_dataset_value_audit_20261009');OUT.mkdir(exist_ok=True)
channels=[];subjects=[];pairs=[];windows=[];errors=[];fingerprints=[];hashes=defaultdict(list)
expected=set('Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2'.split())
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def maxrun(x):
 idx=np.flatnonzero(np.r_[True,x[1:]!=x[:-1],True]);return int(np.diff(idx).max()) if x.size else 0
paths=sorted([p for p in ROOT.glob('*/eyes-closed/patient*') if p.is_dir()],key=lambda p:(p.parts[-3],int(p.name[7:])))
start=time.time()
for si,p in enumerate(paths):
 sid=str(p.relative_to(ROOT)); arr=[];names=[];sr=[]
 files=sorted(p.glob('*'));actual={f.stem for f in files if f.suffix=='.txt'}
 if actual!=expected:errors.append({'path':sid,'error':'channel set mismatch','missing':sorted(expected-actual),'extra':sorted(actual-expected)})
 for f in files:
  if f.suffix!='.txt':continue
  rel=str(f.relative_to(ROOT)); raw=f.read_bytes();sha=hashlib.sha256(raw).hexdigest();hashes[sha].append(rel)
  try:
   x=np.loadtxt(raw.splitlines(),dtype=np.float64,ndmin=1)
   if x.ndim!=1:raise ValueError(f'not single column: {x.shape}')
  except Exception as e:errors.append({'path':rel,'error':str(e)});continue
  n=x.size;finite=np.isfinite(x);good=x[finite]
  if not n or not finite.all():errors.append({'path':rel,'error':f'empty/nonfinite {n-int(finite.sum())}'})
  if not good.size:continue
  d=np.diff(good);med=np.median(good);mad=np.median(abs(good-med));std=float(good.std());qs=np.quantile(good,[.001,.01,.5,.99,.999]);absq=np.quantile(abs(good),[.99,.999]);
  freq,power=welch(good,fs=1,nperseg=min(2048,n));total=power[1:].sum();low=power[(freq>0)&(freq<.002)].sum()/total if total else 0;high=power[freq>.2].sum()/total if total else 0
  row=dict(path=rel,subject=sid,label=p.parts[-3],channel=f.stem,n=n,bytes=len(raw),nonfinite=int(n-finite.sum()),minimum=float(good.min()),maximum=float(good.max()),mean=float(good.mean()),std=std,median=float(med),mad=float(mad),p001=float(qs[0]),p01=float(qs[1]),p99=float(qs[3]),p999=float(qs[4]),abs_p99=float(absq[0]),abs_p999=float(absq[1]),zeros=int((good==0).sum()),equal_adjacent_fraction=float((d==0).mean()) if d.size else 0,longest_equal_run_samples=maxrun(good),min_count=int((good==good.min()).sum()),max_count=int((good==good.max()).sum()),max_abs_step=float(abs(d).max()) if d.size else 0,robust_10sigma_count=int((abs(good-med)>10*1.4826*mad).sum()) if mad else -1,low_power_fraction_under_002_cycles_per_sample=float(low),high_power_fraction_over_02_cycles_per_sample=float(high),sha256=sha)
  channels.append(row);sr.append(row);arr.append(x);names.append(f.stem)
  idx=np.unique(np.r_[np.arange(min(n,512)),np.linspace(0,n-1,min(n,512),dtype=int)])
  v=x[idx];v=v-v.mean();v=v/np.linalg.norm(v) if np.linalg.norm(v) else v
  fingerprints.append((rel,n,idx,v))
  block=1024;count=n//block
  if count:
   b=x[:count*block].reshape(count,block);bs=b.std(axis=1);ptp=np.ptp(b,axis=1);bm=b.mean(axis=1)
   flags=(bs==0)|(ptp>max(20*1.4826*mad,1e-12))|(abs(bm-med)>max(10*1.4826*mad,1e-12))
   for j in np.flatnonzero(flags):windows.append(dict(path=rel,start_sample=int(j*block),end_sample_exclusive=int((j+1)*block),std=float(bs[j]),peak_to_peak=float(ptp[j]),mean=float(bm[j]),flat=bool(bs[j]==0)))
 if not sr:continue
 lens=[len(x) for x in arr]
 if len(set(lens))==1 and len(arr)>1:
  X=np.array(arr);sd=X.std(axis=1);valid=sd>0
  if valid.all():
   corr=np.corrcoef(X)
   for a,b in zip(*np.where(np.triu(abs(corr)>.98,1))):pairs.append(dict(subject=sid,channel_a=names[a],channel_b=names[b],correlation=float(corr[a,b]),exact_equal=bool(np.array_equal(X[a],X[b]))))
  avg=X.mean(axis=0);avg_ratio=float(avg.std()/np.median(sd)) if np.median(sd)>0 else 0
 else:avg_ratio=None
 subjects.append(dict(subject=sid,label=p.parts[-3],patient=int(p.name[7:]),channels=len(sr),min_n=min(lens),max_n=max(lens),lengths_match=len(set(lens))==1,total_values=sum(lens),median_channel_std=float(np.median([r['std'] for r in sr])),min_channel_std=min(r['std'] for r in sr),max_channel_std=max(r['std'] for r in sr),maximum_absolute_value=max(max(abs(r['minimum']),abs(r['maximum'])) for r in sr),median_equal_adjacent_fraction=float(np.median([r['equal_adjacent_fraction'] for r in sr])),mean_signal_std_ratio=avg_ratio))
 if (si+1)%10==0:print(f'{si+1}/{len(paths)} subjects; {len(channels)} channels; {time.time()-start:.1f}s',flush=True)
write('channel_statistics.csv',channels);write('subject_statistics.csv',subjects);write('high_channel_correlations.csv',pairs);write('flagged_1024_sample_windows.csv',windows)
duplicates=[v for v in hashes.values() if len(v)>1]
# Screen same named channels for scaled/offset copies, then verify every sample of candidates.
candidates=[]
for ch in sorted(expected):
 groups=defaultdict(list)
 for item in fingerprints:
  if Path(item[0]).stem==ch:groups[item[1]].append(item)
 for n,g in groups.items():
  if len(g)<2:continue
  V=np.array([t[3] for t in g]);C=V@V.T
  for a,b in zip(*np.where(np.triu(abs(C)>.99999,1))):
   pa,pb=g[a][0],g[b][0];x=np.loadtxt(ROOT/pa);y=np.loadtxt(ROOT/pb);xc=x-x.mean();yc=y-y.mean();slope=float(np.dot(xc,yc)/np.dot(xc,xc));intercept=float(y.mean()-slope*x.mean());res=y-(slope*x+intercept);ratio=float(np.std(res)/np.std(y)) if y.std() else 0
   candidates.append(dict(path_a=pa,path_b=pb,n=n,correlation=float(np.corrcoef(x,y)[0,1]),slope_b_from_a=slope,intercept=intercept,residual_std_ratio=ratio,max_abs_residual=float(abs(res).max()),exact_numeric_equal=bool(np.array_equal(x,y))))
write('scaled_duplicate_channels.csv',candidates)
summary=dict(subjects=len(subjects),channel_files=len(channels),total_values=sum(r['n'] for r in channels),total_bytes=sum(r['bytes'] for r in channels),class_counts=dict(Counter(r['label'] for r in subjects)),channel_names=sorted(expected),errors=errors,nonfinite_values=sum(r['nonfinite'] for r in channels),length_mismatch_subjects=[r['subject'] for r in subjects if not r['lengths_match']],constant_channels=[r['path'] for r in channels if r['std']==0],exact_byte_duplicate_groups=duplicates,scaled_duplicate_channel_pairs=len(candidates),high_correlation_channel_pairs=len(pairs),flagged_1024_sample_windows=len(windows),elapsed_seconds=time.time()-start)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
