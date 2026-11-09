# Research materials — local review draft, not published

This is a proposed subset of the material for the manuscript. All files require author and institutional rights review; third-party terms and final licenses remain unconfirmed. Do not publish this directory until approved. No DOI has been assigned.

## Contents

- `code/`: six historical evaluation, matching and merging scripts. They are archival material, not a validated turnkey reproduction workflow.
- `results/`: four stored summary or pooled result files; `r4_v12.csv` includes stored R4 rows for u1, rt and u2, but not every row of manuscript Table 2.
- `figures/`: three plotting scripts and two aggregated plotting inputs. Some data are hard-coded in the plotting scripts.

There is no separate configuration file in this candidate. Some historical settings appear within source code; they are not a complete, independently verified run configuration. The package does not include per-frame predictions, raw videos, official annotations, derived VIRAT reference annotations, model weights, per-frame/per-block bootstrap counts, or the P3 per-sequence curve file `p3_reconcile.json`. Aggregated P3 plotting values remain in `figures/p3_fig_data.json`. Other files, including `results/r4_v12.csv`, may include per-sequence or per-clip summaries. Rights to each asset and any redistribution restrictions must be established separately. Check current terms at https://viratdata.org/ and https://github.com/VisDrone/VisDrone-Dataset.

## Running and limitations

Python, NumPy, SciPy and Matplotlib may be required. Several scripts embed the placeholder `/PATH/TO/PROJECT` and expect specific historical project paths and external inputs; `scommon.py` reads data on import, and evaluation/merging scripts can write to paths under the configured project root. Inspect the source and use isolated copies and output locations before running. Adding the included modules alone does not establish that the analysis can run or reproduce historical results. No execution of evaluation, matching or bootstrap was performed in this review. A drawing script may generate plots from bundled inputs but this does not verify the underlying experiments or the published figures byte for byte.

See the accompanying review files for the original source-change manifest, selected table-source evidence and unresolved rights decisions.

## Licenses and attribution (local review draft)

**Code:** PolyForm Noncommercial License 1.0.0.  
**Summary and plotting data:** CC BY-NC 4.0.  
**Attribution:** Please cite the accompanying paper.  
**Permissions beyond these licenses:** yangzb2557230@buaa.edu.cn

The code license applies to `code/*.py` and `figures/*.py`; the data license applies to `results/*.json`, `results/*.csv`, and `figures/*.json`. See `LICENSE-CODE.txt` and `LICENSE-DATA.txt` for links to the complete official license terms. The licenses do not grant rights to third-party datasets, annotations, models, or weights. The actual license terms govern, including their defined permitted uses.

This is a local review draft. It has not been published. The citation should be completed using the actual published paper or repository identifier when available. Documentation and metadata are not covered by the file-specific code and data notices above.
