from pathlib import Path
import csv,json,hashlib,numpy as np
from collections import defaultdict,Counter
R=Path('/Users/adamk/Work/BTP/FINAL_DATASET');O=Path('/Users/adamk/Work/BTP/final_dataset_value_audit_20261009')
s=list(csv.DictReader((O/'subject_statistics.csv').open()));c=list(csv.DictReader((O/'channel_statistics.csv').open()))
hashes=defaultdict(list);runs=[];allzero=[];prefixes=defaultdict(list);qcs=[];blockhashes=defaultdict(list)
def write(name,rows):
 if rows:
  with (O/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for i,row in enumerate(s):
 p=R/row['subject'];xs=[]
 for f in sorted(p.glob('*.txt')):
  x=np.loadtxt(f);xs.append(x);rel=str(f.relative_to(R));hashes[hashlib.sha256(x.tobytes()).hexdigest()].append(rel)
  inds=np.flatnonzero(np.r_[True,x[1:]!=x[:-1],True])
  for j in np.flatnonzero(np.diff(inds)>=16):runs.append(dict(path=rel,start_sample=int(inds[j]),end_sample_exclusive=int(inds[j+1]),length=int(inds[j+1]-inds[j]),value=float(x[inds[j]])))
  v=x[:512];v=v-v.mean();norm=np.linalg.norm(v);v=v/norm if norm else v;prefixes[f.stem].append((rel,v))
  for start in range(0,len(x)-1023,1024):
   b=x[start:start+1024]
   if np.std(b)>0:blockhashes[hashlib.sha256(b.tobytes()).hexdigest()].append((rel,start))
 X=np.array(xs);z=np.all(X==0,axis=0);edges=np.diff(np.r_[False,z,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
 for a,b in zip(starts,ends):allzero.append(dict(subject=row['subject'],start_sample=int(a),end_sample_exclusive=int(b),length=int(b-a)))
 if (i+1)%50==0:print(i+1,flush=True)
write('constant_runs_at_least_16_samples.csv',runs);write('simultaneous_zero_intervals.csv',allzero)
numdups=[g for g in hashes.values() if len(g)>1]
(O/'numeric_duplicate_groups.json').write_text(json.dumps(numdups,indent=2))
blocks=[g for g in blockhashes.values() if len(g)>1]
(O/'duplicate_aligned_1024_sample_blocks.json').write_text(json.dumps(blocks,indent=2))
pairs=[]
for ch,items in prefixes.items():
 V=np.array([v for _,v in items]);C=np.einsum('ik,jk->ij',V,V,optimize=False)
 for a,b in zip(*np.where(np.triu(abs(C)>.99999,1))):
  pa,pb=items[a][0],items[b][0];x=np.loadtxt(R/pa);y=np.loadtxt(R/pb);n=min(len(x),len(y));x=x[:n];y=y[:n];xc=x-x.mean();yc=y-y.mean();sl=float(np.sum(xc*yc)/np.sum(xc*xc));it=float(y.mean()-sl*x.mean());res=y-sl*x-it
  pairs.append(dict(path_a=pa,path_b=pb,overlap_n=n,full_overlap_correlation=float(np.sum(xc*yc)/np.sqrt(np.sum(xc*xc)*np.sum(yc*yc))),slope=sl,residual_std_ratio=float(res.std()/y.std())))
write('prefix_duplicate_candidates_verified.csv',pairs)
summary=dict(numeric_duplicate_groups=len(numdups),constant_runs_16_samples=len(runs),subjects_with_constant_runs=len(set(str(Path(r['path']).parent) for r in runs)),simultaneous_zero_intervals=len(allzero),subjects_with_simultaneous_zeros=len(set(r['subject'] for r in allzero)),simultaneous_zero_timepoints=sum(r['length'] for r in allzero),duplicate_aligned_block_groups=len(blocks),prefix_duplicate_pairs=len(pairs),prefix_duplicate_pairs_other_than_known_group=[r for r in pairs if r['residual_std_ratio']>1e-6])
(O/'followup_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
