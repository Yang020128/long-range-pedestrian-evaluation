"""Fig. 6 (R1 no-candidate share) redrawn from Table 3 (tables/tab_r123.tex) values.
Run: cd figures && python3 fig_r1r3_v2.py"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
matplotlib.rcParams.update({'pdf.fonttype': 42, 'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 7})
C = {'lt20': '#1F3A5F', '20to32': '#7FA6D3'}
# cell -> {stratum: (estimate, lo, hi)}; values from Table 3
cells = [('VIRAT\n× YOLO11m', None),
         ('VIRAT\n× RT-DETR-L', {'lt20': (0.9734, 0.9481, 0.9933), '20to32': (0.9884, 0.9780, 0.9964)}),
         ('VisDrone-VID\n× YOLO11m', {'lt20': (0.7723, 0.4694, 0.9604), '20to32': (0.2827, 0.1892, 0.4011)}),
         ('VisDrone-VID\n× RT-DETR-L', {'lt20': (0.8449, 0.5849, 1.0000), '20to32': (0.7602, 0.6701, 0.8607)})]
fig, ax = plt.subplots(figsize=(5.2, 2.4))
off = {'lt20': -0.12, '20to32': 0.12}; mk = {'lt20': 'o', '20to32': 's'}
for i, (lab, d) in enumerate(cells):
    if d is None:
        ax.bar(i, 0.529, 0.42, facecolor='white', edgecolor='#777777', hatch='////', lw=0.8, zorder=2)
        ax.text(i, 0.529 + 0.03, '<32 px merged\n0.529, no CI', ha='center', va='bottom', fontsize=6.5, color='#555555')
        continue
    for k, (e, lo, hi) in d.items():
        x = i + off[k]
        ax.plot([x, x], [lo, hi], color=C[k], lw=1.0, zorder=3)
        ax.plot(x, e, mk[k], color=C[k], ms=4, mec='white', mew=0.5, zorder=4)
ax.axhline(0.5, color='#888888', lw=0.8, ls='--', zorder=1)
ax.text(3.62, 0.5, 'decision\nlevel 0.5', va='center', ha='left', fontsize=6.5, color='#555555')
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cells])
ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 1.05); ax.set_ylabel('No-candidate share')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.tick_params(length=2, width=0.6)
h = [Line2D([0], [0], color=C['lt20'], marker='o', ms=4, mec='white', lw=1.0, label='<20 px'),
     Line2D([0], [0], color=C['20to32'], marker='s', ms=4, mec='white', lw=1.0, label='20\u201332 px')]
ax.legend(handles=h, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig('fig_r1r3_v2.pdf'); fig.savefig('fig_r1r3_v2.png', dpi=300)
