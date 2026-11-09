# Research materials — long-range pedestrian evaluation

This repository provides the evaluation and plotting code and summary result tables for "Candidate absence and temporal re-scoring in long-range pedestrian detection".

## Contents

- `code/`: six historical evaluation, matching and merging scripts. They are archival material, not a validated turnkey reproduction workflow.
- `results/`: four stored summary or pooled result files; `r4_v12.csv` includes stored R4 rows for u1, rt and u2, but not every row of manuscript Table 2.
- `figures/`: three plotting scripts and two aggregated plotting inputs. Some data are hard-coded in the plotting scripts.

There is no separate configuration file in this repository. Some historical settings appear within source code; they are not a complete, independently verified run configuration. The repository does not include per-frame predictions, raw videos, official annotations, derived VIRAT reference annotations, model weights, per-frame/per-block bootstrap counts, or the P3 per-sequence curve file `p3_reconcile.json`. Aggregated P3 plotting values remain in `figures/p3_fig_data.json`. Other files, including `results/r4_v12.csv`, may include per-sequence or per-clip summaries.

## Running and limitations

Python, NumPy, SciPy and Matplotlib may be required. Several scripts embed the placeholder `/PATH/TO/PROJECT` and expect specific historical project paths and external inputs; `scommon.py` reads data on import, and evaluation/merging scripts can write to paths under the configured project root. Inspect the source and use isolated copies and output locations before running. The scripts are historical and are not a turnkey reproduction workflow: they require the external predictions and annotations described above, and this repository does not claim to reproduce all bootstrap statistics. A drawing script may generate plots from bundled inputs, but this does not verify the underlying experiments or the published figures byte for byte.

## Third-party data

The VIRAT and VisDrone-VID datasets, their official annotations, and the detector model weights are not redistributed here. Obtain them from their official channels and comply with the terms applying to the actual versions:

- VIRAT: https://viratdata.org/
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset

## Licenses and attribution

**Code:** PolyForm Noncommercial License 1.0.0 — source-available; the code is not open source in the OSI sense. See `LICENSE-CODE.txt`. For uses not permitted by the PolyForm Noncommercial License 1.0.0, please contact yangzb2557230@buaa.edu.cn for separate authorization.
**Summary and plotting data:** CC BY-NC 4.0. See `LICENSE-DATA.txt`.
**Attribution:** Please cite the accompanying paper.

The code license applies to `code/*.py` and `figures/*.py`; the data license applies to `results/*.json`, `results/*.csv`, and `figures/*.json`. The licenses do not grant rights to third-party datasets, annotations, models, or weights. The actual license terms govern, including their defined permitted uses.
