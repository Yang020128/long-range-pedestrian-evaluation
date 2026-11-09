#!/usr/bin/env python3
"""P2/P3 supplementary figures from saved results. No re-scoring / bootstrap.
Run: cd figures/new && python3 plot_figs.py  (data files are in this directory)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 8})

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = ['<20', '20-32', '>=32']
LBL = {'<20': '<20 px', '20-32': '20\u201332 px', '>=32': '\u226532 px'}

# ---- P2 ----
p2 = json.load(open(os.path.join(HERE, 'p2_fig_data.json')))
tgt = p2['target_precision']
arms = [('T', 'T (baseline)'), ('H', 'H (cross-frame)'), ('F', 'F (same-frame)'), ('SO', 'SO (score-only)')]
x = np.arange(len(LAYERS)); w = 0.19
fig, ax = plt.subplots(figsize=(7.0, 3.2))
colors = ['#666666', '#d62728', '#2ca02c', '#1f77b4']
for i, (k, lab) in enumerate(arms):
    a = p2['arms'][k]; tau = a['tau']; prec = a['precision']
    vals = [a['strata'][L]['recall'] for L in LAYERS]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab + ' (\u03c4 = ' + str(tau) + ', p = ' + f'{prec:.4f}' + ')', color=colors[i])
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1.5) * w, v + 0.010, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([LBL[L] for L in LAYERS])
ax.set_ylim(0, 0.82); ax.set_ylabel('Recall (absolute)')
ax.legend(ncol=2, fontsize=7.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_p2_strata.pdf'))
fig.savefig(os.path.join(HERE, 'fig_p2_strata.png'), dpi=300)
plt.close(fig)

# ---- P3 ----
p3 = json.load(open(os.path.join(HERE, 'p3_fig_data.json')))
p3arms = [('full', 'full'), ('BTm', 'BT-m (full\u222atile)'), ('SAHI', 'SAHI (slice-only)'), ('fullSAHI', 'fullSAHI (full\u222aSAHI)')]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
colors2 = ['#666666', '#1f77b4', '#ff7f0e', '#2ca02c']
for i, (k, lab) in enumerate(p3arms):
    a = p3['arms'][k]; prec = a['precision']
    vals = [a['recall'][L] for L in LAYERS]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab + ' (p = ' + f'{prec:.4f}' + ')', color=colors2[i])
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1.5) * w, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([LBL[L] for L in LAYERS])
ax.set_ylim(0, 0.75); ax.set_ylabel('Recall (absolute)')
ax.legend(ncol=2, fontsize=7.5, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_p3_slicing.pdf'))
fig.savefig(os.path.join(HERE, 'fig_p3_slicing.png'), dpi=300)
plt.close(fig)
print('generated: fig_p2_strata.pdf/png, fig_p3_slicing.pdf/png')
