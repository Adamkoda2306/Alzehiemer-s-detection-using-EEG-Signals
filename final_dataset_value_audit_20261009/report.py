from pathlib import Path
import csv,json,numpy as np
from collections import Counter
R=Path('/Users/adamk/Work/BTP/FINAL_DATASET');O=Path('/Users/adamk/Work/BTP/final_dataset_value_audit_20261009')
s=list(csv.DictReader((O/'subject_statistics.csv').open()));c=list(csv.DictReader((O/'channel_statistics.csv').open()));z=list(csv.DictReader((O/'simultaneous_zero_intervals.csv').open()));corr=list(csv.DictReader((O/'high_channel_correlations.csv').open()))
issues=[]
for r in c:
 raw=(R/r['path']).read_bytes();nlines=raw.count(b'\n')+int(bool(raw) and not raw.endswith(b'\n'))
 if nlines!=int(r['n']) or b'#' in raw:issues.append(dict(path=r['path'],physical_lines=nlines,numeric_values=int(r['n']),comments=b'#' in raw))
(O/'physical_line_validation.json').write_text(json.dumps(dict(files_checked=len(c),issues=issues),indent=2))
counts=Counter(r['subject'] for r in corr);zs=Counter(r['subject'] for r in z);cr={}
for r in c:cr.setdefault(r['subject'],[]).append(r)
with (O/'subject_review_queue.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['subject','duplicate_group','samples_per_channel','simultaneous_zero_intervals','high_correlation_pairs_abs_gt_098','robust_10sigma_values','max_absolute_value','max_absolute_step','action'])
 for r in s:
  sid=r['subject'];cc=cr[sid];dup=r['label']=='Alzehiemers' and 76<=int(r['patient'])<=80
  actions=['verify sampling rate, units, source and subject identity']
  if dup:actions.append('resolve identical patient76-80 recordings before splitting')
  if int(r['min_n'])==1024:actions.append('confirm 1024-sample export is complete')
  if zs[sid]:actions.append('review synchronous zero intervals')
  if counts[sid]:actions.append('review common signal and reference')
  w.writerow([sid,'AD76-80' if dup else '',r['min_n'],zs[sid],counts[sid],sum(int(t['robust_10sigma_count']) for t in cc),r['maximum_absolute_value'],max(float(t['max_abs_step']) for t in cc),'; '.join(actions)])
print('zero subjects',dict(zs));print('zero longest',sorted(z,key=lambda r:int(r['length']),reverse=True)[:5]);print('physical line issues',issues)
# Standalone figure; all amplitude axes are in unknown stored units.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for label,col in [('Alzehiemers','#b54343'),('Healthy','#24778d')]:
 g=[r for r in s if r['label']==label]
 axes[0,0].scatter([int(r['min_n']) for r in g],[float(r['median_channel_std']) for r in g],c=col,label=label,alpha=.65,s=25)
axes[0,0].set(xscale='log',yscale='log',xlabel='Samples per channel (log scale)',ylabel='Median channel SD, stored units (log scale)',title='Length and amplitude differ sharply');axes[0,0].legend()
axes[0,1].bar(['AD subjects','Healthy subjects','AD samples','Healthy samples'],[50,50,100*3845555/20106182,100*16260627/20106182],color=['#b54343','#24778d']*2);axes[0,1].set(ylabel='Percent within subjects or samples',title='Balanced folders do not mean balanced samples',ylim=(0,100));axes[0,1].tick_params(axis='x',labelrotation=12)
x=np.loadtxt(R/'Alzehiemers/eyes-closed/patient22/Fp1.txt');axes[1,0].plot(np.arange(400),x[:400],lw=1);axes[1,0].axvspan(0,129,color='red',alpha=.15);axes[1,0].set(xlabel='Sample index (zero-based)',ylabel='Stored units',title='AD patient22 / Fp1: first 130 samples are zero')
x=np.loadtxt(R/'Alzehiemers/eyes-closed/patient27/T4.txt');j=int(np.argmax(abs(np.diff(x))));a=max(0,j-100);b=min(len(x),j+101);axes[1,1].plot(np.arange(a,b),x[a:b],lw=1);axes[1,1].set(xlabel='Sample index (zero-based)',ylabel='Stored units',title='AD patient27 / T4: largest one-sample jump')
fig.suptitle('FINAL_DATASET: full-value audit, 9 October 2026',fontsize=15);fig.savefig(O/'audit_overview.png',dpi=160);plt.close(fig)
