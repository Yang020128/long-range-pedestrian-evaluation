#!/usr/bin/env python3
"""Three standalone figures with unified styling (Arial, colorblind-friendly, >=8pt).
(a) fig_p2_strata: exploratory ablation (absolute recall at matched precision).
(b) fig_p3_slicing: external slicing baseline (absolute recall at fixed tau=0.25).
(c) fig_r1r3: R1 no-candidate share.
Run: cd figures/new && python3 unified_figs.py"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# unified style: Arial, TrueType, >=8pt
matplotlib.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7})
LAYERS = ['<20', '20-32', '>=32']
LBL = {'<20': '<20 px', '20-32': '20\u201332 px', '>=32': '\u226532 px'}
# colorblind-friendly palette
C = {'T': '#999999', 'H': '#E69F00', 'F': '#0072B2', 'SO': '#009E73',
     'full': '#999999', 'BTm': '#0072B2', 'SAHI': '#E69F00', 'fullSAHI': '#009E73'}

def style_ax(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)

# ============ (a) exploratory ablation ============
p2 = json.load(open(os.path.join(HERE, 'p2_fig_data.json')))
fig, ax = plt.subplots(figsize=(6.6, 2.9))
arms = [('T', 'T (baseline)'), ('H', 'H (cross-frame)'), ('F', 'F (same-frame)'), ('SO', 'SO (score-only)')]
x = np.arange(len(LAYERS)); w = 0.19
for i, (k, lab) in enumerate(arms):
    a = p2['arms'][k]
    vals = [a['strata'][L]['recall'] for L in LAYERS]
    ax.bar(x + (i - 1.5) * w, vals, w,
           label=lab + ' (\u03c4=' + str(a['tau']) + ', p=' + f"{a['precision']:.4f}" + ')',
           color=C[k], edgecolor='none')
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1.5) * w, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([LBL[L] for L in LAYERS])
ax.set_ylim(0, 0.85); ax.set_ylabel('Recall (absolute)')
ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=7.5)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_p2_strata.pdf'))
fig.savefig(os.path.join(HERE, 'fig_p2_strata.png'), dpi=300)
plt.close(fig)

# ============ (b) external slicing baseline ============
p3 = json.load(open(os.path.join(HERE, 'p3_fig_data.json')))
fig, ax = plt.subplots(figsize=(6.0, 2.9))
p3arms = [('full', 'full'), ('BTm', 'BT-m (full+tile)'), ('SAHI', 'SAHI (slice-only)'), ('fullSAHI', 'fullSAHI (full+SAHI)')]
for i, (k, lab) in enumerate(p3arms):
    a = p3['arms'][k]
    vals = [a['recall'][L] for L in LAYERS]
    ax.bar(x + (i - 1.5) * w, vals, w, label=lab + ' (p=' + f"{a['precision']:.3f}" + ')', color=C[k], edgecolor='none')
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1.5) * w, v + 0.012, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([LBL[L] for L in LAYERS])
ax.set_ylim(0, 0.78); ax.set_ylabel('Recall (absolute)')
ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=7.5)
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_p3_slicing.pdf'))
fig.savefig(os.path.join(HERE, 'fig_p3_slicing.png'), dpi=300)
plt.close(fig)

# ============ (c) R1 no-candidate share ============
Ccell = {'lt20': '#1F3A5F', '20to32': '#7FA6D3'}
cells = [('VIRAT\n\u00d7 YOLO11m', None),
         ('VIRAT\n\u00d7 RT-DETR-L', {'lt20': (0.9734, 0.9481, 0.9933), '20to32': (0.9884, 0.9780, 0.9964)}),
         ('VisDrone-VID\n\u00d7 YOLO11m', {'lt20': (0.7723, 0.4694, 0.9604), '20to32': (0.2827, 0.1892, 0.4011)}),
         ('VisDrone-VID\n\u00d7 RT-DETR-L', {'lt20': (0.8449, 0.5849, 1.0000), '20to32': (0.7602, 0.6701, 0.8607)})]
fig, ax = plt.subplots(figsize=(5.4, 2.7))
off = {'lt20': -0.12, '20to32': 0.12}; mk = {'lt20': 'o', '20to32': 's'}
for i, (lab, d) in enumerate(cells):
    if d is None:
        ax.bar(i, 0.529, 0.42, facecolor='white', edgecolor='#777777', hatch='////', lw=0.8, zorder=2)
        ax.text(i, 0.529 + 0.04, '<32 px merged\n0.529, no CI', ha='center', va='bottom', fontsize=6.5, color='#555555')
        continue
    for k, (e, lo, hi) in d.items():
        x = i + off[k]
        ax.plot([x, x], [lo, hi], color=Ccell[k], lw=1.0, zorder=3)
        ax.plot(x, e, mk[k], color=Ccell[k], ms=4.5, mec='white', mew=0.6, zorder=4)
ax.axhline(0.5, color='#888888', lw=0.8, ls='--', zorder=1)
ax.text(3.62, 0.5, 'decision\nlevel 0.5', va='center', ha='left', fontsize=7, color='#555555')
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cells], fontsize=7.5)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 1.08); ax.set_ylabel('No-candidate share')
style_ax(ax)
h = [Line2D([0], [0], color=Ccell['lt20'], marker='o', ms=4.5, mec='white', lw=1.0, label='<20 px'),
     Line2D([0], [0], color=Ccell['20to32'], marker='s', ms=4.5, mec='white', lw=1.0, label='20\u201332 px')]
ax.legend(handles=h, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_r1r3.pdf'))
fig.savefig(os.path.join(HERE, 'fig_r1r3.png'), dpi=300)
plt.close(fig)
print('generated: fig_p2_strata, fig_p3_slicing, fig_r1r3 (unified Arial >=8pt)')
