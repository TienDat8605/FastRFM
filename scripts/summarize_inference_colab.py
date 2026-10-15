"""Make per-mask tables and scientific plots from a completed Colab benchmark."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('results', type=Path)
args = p.parse_args()
root = args.results
records = json.loads((root/'results.json').read_text())
state = json.loads((root/'state.json').read_text())
if state['status'] != 'complete_exploratory' or len(records) != 32:
    raise ValueError('Require all 32 completed benchmark cells')
order = ['euler100', 'euler50', 'adjoint100', 'adjoint50', 'midpoint2', 'midpoint4', 'student2', 'student4']
summary = []
for mask in ['legs', 'torso']:
    reference = [r for r in records if r['mask']==mask and r['method']=='euler100']
    reference_time = np.mean([t['seconds'] for r in reference for t in r['timings']])
    reference_error = np.mean([r['metrics']['mpvpe_mm'] for r in reference])
    for method in order:
        cells = [r for r in records if r['mask']==mask and r['method']==method]
        if len(cells) != 2: raise ValueError('Missing paired inference seed')
        seconds = np.mean([t['seconds'] for r in cells for t in r['timings']])
        error = np.mean([r['metrics']['mpvpe_mm'] for r in cells])
        summary.append({'mask':mask, 'method':method, 'solver_seconds':float(seconds),
            'speedup':float(reference_time/seconds), 'mpvpe_mm':float(error),
            'mpvpe_change_mm':float(error-reference_error),
            'visible_rad':float(np.mean([r['metrics']['visible_rad'] for r in cells])),
            'apd_cm':float(np.mean([r['metrics']['apd_cm'] for r in cells])),
            'peak_allocated_mib':max(t['peak_allocated_mib'] for r in cells for t in r['timings']),
            'incremental_peak_mib':max(t['incremental_peak_mib'] for r in cells for t in r['timings'])})
with (root/'summary_by_mask.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)
lines = ['# Results by completion mask', '',
         'Eight validation poses, two hypotheses, two inference seeds; two timing repeats. '
         'Changes are paired against the 100-step full-graph teacher within each mask.', '',
         '| Mask | Method | Solver s/batch | Speedup | MPVPE mm | Change mm | Visible rad | Incremental peak MiB |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['mask']} | {r['method']} | {r['solver_seconds']:.3f} | {r['speedup']:.2f}x | "
                 f"{r['mpvpe_mm']:.2f} | {r['mpvpe_change_mm']:+.2f} | {r['visible_rad']:.3f} | {r['incremental_peak_mib']:.2f} |")
training = json.loads((root/'training.json').read_text())
if training['student4']['selected_update'] == 0:
    lines += ['', 'The four-step student selected update zero; it is identical to the '
             'undistilled four-step midpoint control. Distillation did not improve its selection loss.']
audit_path = root/'adjoint_source_audit.json'
if audit_path.exists():
    audit = json.loads(audit_path.read_text())
    lines += ['', f"Actual Haar-source task-gradient audit: {sum(r['passed'] for r in audit)}/{len(audit)} checks passed; "
              f"maximum relative gradient error {max(r['gradient_relative_error'] for r in audit):.8f}. "
              'These checks use the first two validation poses, two hypotheses each, both masks/seeds, and 50/100/200 steps.']
lines += ['', 'Adjoint gradient/memory probes use two validation poses as initial states. '
          'The first 25-step incremental memory readings include one-time allocations; use warmed '
          '50/100/200-step measurements to assess solver-depth scaling. Restoring the source anchor '
          'makes its final residual zero; max_anchor_drift records pre-reset drift. '
          'Timing and quality are exploratory, not population confidence intervals.']
(root/'report_by_mask.md').write_text('\n'.join(lines)+'\n')
colors = plt.get_cmap('tab10')
labels = {'euler100':'Euler 100', 'euler50':'Euler 50', 'adjoint100':'Adjoint 100',
          'adjoint50':'Adjoint 50', 'midpoint2':'Midpoint 2', 'midpoint4':'Midpoint 4',
          'student2':'Distilled midpoint 2', 'student4':'Distilled midpoint 4'}
fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
plot_methods = order[:-1] if training['student4']['selected_update']==0 else order
for ax, mask in zip(axes, ['legs', 'torso']):
    for i, method in enumerate(plot_methods):
        r = next(x for x in summary if x['mask']==mask and x['method']==method)
        ax.scatter(r['solver_seconds'], r['mpvpe_mm'], color=colors(i), s=65,
                   marker='s' if method.startswith('adjoint') else 'o', label=labels[method])
    ax.set_xscale('log')
    ax.set_xlabel('Solver time per 16-hypothesis batch (s, log scale)')
    ax.set_ylabel('Hidden-region MPVPE (mm)')
    ax.set_title(mask.capitalize()+' completion')
    ax.grid(alpha=.2)
axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02,1))
for extension in ['png','pdf']:
    fig.savefig(root/f'speed_quality.{extension}', dpi=180, bbox_inches='tight')
plt.close(fig)
memory = json.loads((root/'memory_gradient.json').read_text())
fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
for method, label in [('full_graph','Full backpropagation'),('adjoint','Fixed-anchor discrete adjoint')]:
    values = [r for r in memory if r['method']==method and r['steps']>=50]
    ax.plot([r['steps'] for r in values], [r['incremental_peak_mib'] for r in values], 'o-', label=label)
ax.set_xlabel('ODE steps')
ax.set_ylabel('Additional peak allocated GPU memory (MiB)')
ax.set_title('Two-pose backward probe on T4; warmed measurements')
ax.grid(alpha=.2); ax.legend()
for extension in ['png','pdf']:
    fig.savefig(root/f'memory_scaling.{extension}', dpi=180)
plt.close(fig)
print(json.dumps(summary, indent=2))
