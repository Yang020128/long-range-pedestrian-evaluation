#!/usr/bin/env python3
"""TR-5N-RUN 合并：全图预测 ∪ 切片预测(逆映射回原图) → 全局 NMS IoU=0.7。
主口径 = merged；tiled_only 单列诊断。
用法: merge_tiles.py <BT-n|BT-m> <clip> [--stride S]
输出: predictions_merged/<tag>.csv (frame,x1,y1,x2,y2,conf,cls,source)
      predictions_tiled_only/<tag>.csv
逆映射(冻结公式): 原图 = (TILE*tile_i + t/2, TILE*tile_j + t/2)，
再裁到 tile 源域 ∩ 图像边界(半开区间)；中心出界或完全越界丢弃并计数。
"""
import csv, json, os, sys

BASE = '/PATH/TO/PROJECT'
FULL = {
    'BT-n': os.path.join(BASE, 'tr5l_model_compare_20261026/predictions/B1280_{c}.csv'),
    'BT-m': os.path.join(BASE, 'tr5m_stronger_baselines_20261026/predictions/D1280_{c}.csv'),
}
TILES = os.path.join(BASE, 'tr5n_run_20261026/predictions_tiles/{a}_{c}{s}.csv')
OUT_M = os.path.join(BASE, 'tr5n_run_20261026/predictions_merged')
OUT_T = os.path.join(BASE, 'tr5n_run_20261026/predictions_tiled_only')
DIMS = {'010207_04': (1280, 720), '010207_08': (1280, 720), '010110_00': (1280, 720),
        '010114_01': (1280, 720), '050202_03': (1920, 1080)}
TILE = 320
NMS_IOU = 0.7


def map_tile_box(ti, tj, t1, t2, t3, t4, W, H, stride=TILE):
    """tile 内 640 坐标 → 原图坐标，裁到 tile 源域∩图像边界(半开)。
    返回 (x1,y1,x2,y2) 或 None(应丢弃)。tile 源域按 stride 间距的 [s*k, s*k+TILE)。"""
    x1 = stride * ti + t1 / 2.0
    y1 = stride * tj + t2 / 2.0
    x2 = stride * ti + t3 / 2.0
    y2 = stride * tj + t4 / 2.0
    # 裁到 tile 源域（半开区间）
    x1 = max(x1, stride * ti); y1 = max(y1, stride * tj)
    x2 = min(x2, stride * ti + TILE); y2 = min(y2, stride * tj + TILE)
    # 裁到图像边界
    x1 = max(x1, 0.0); y1 = max(y1, 0.0)
    x2 = min(x2, float(W)); y2 = min(y2, float(H))
    if x2 <= x1 or y2 <= y1:
        return None  # 完全越界
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if cx < 0 or cx >= W or cy < 0 or cy >= H:
        return None  # 中心出界
    return (x1, y1, x2, y2)


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aw = a[2] - a[0]; ah = a[3] - a[1]
    bw = b[2] - b[0]; bh = b[3] - b[1]
    return inter / (aw * ah + bw * bh - inter)


def nms(boxes, thr=NMS_IOU):
    """boxes: list of (frame,x1,y1,x2,y2,conf,cls,source)。按帧分组，conf 降序贪心。"""
    by_frame = {}
    for b in boxes:
        by_frame.setdefault(b[0], []).append(b)
    kept = []
    for fr, bl in by_frame.items():
        bl = sorted(bl, key=lambda b: -b[5])
        sel = []
        for b in bl:
            if all(iou(b[1:5], s[1:5]) < thr for s in sel):
                sel.append(b)
        kept.extend(sel)
    return kept


def load_full(path):
    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((int(row['frame']), float(row['x1']), float(row['y1']),
                        float(row['x2']), float(row['y2']), float(row['conf']),
                        int(float(row.get('cls', 0))), 'full'))
    return out


def load_tiles(path, W, H, stride=TILE):
    out = []; n_discard = 0
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            m = map_tile_box(int(row['tile_i']), int(row['tile_j']),
                             float(row['x1']), float(row['y1']),
                             float(row['x2']), float(row['y2']), W, H, stride)
            if m is None:
                n_discard += 1
                continue
            out.append((int(row['frame']), m[0], m[1], m[2], m[3],
                        float(row['conf']), int(float(row.get('cls', 0))), 'tile'))
    return out, n_discard


def main():
    arm, clip = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[sys.argv.index('--stride') + 1]) if '--stride' in sys.argv else TILE
    W, H = DIMS[clip]
    tag = f'{arm}_{clip}' + ('_ov50' if stride != TILE else '')
    full = load_full(FULL[arm].format(c=clip))
    tpath = TILES.format(a=arm, c=clip, s='_ov50' if stride != TILE else '')
    tiled, n_discard = load_tiles(tpath, W, H, stride)
    merged = nms(full + tiled)
    merged.sort(key=lambda b: (b[0], -b[5]))
    os.makedirs(OUT_M, exist_ok=True); os.makedirs(OUT_T, exist_ok=True)
    cols = ['frame', 'x1', 'y1', 'x2', 'y2', 'conf', 'cls', 'source']
    for out, rows in ((os.path.join(OUT_M, tag + '.csv'), merged),
                      (os.path.join(OUT_T, tag + '.csv'), sorted(tiled, key=lambda b: (b[0], -b[5])))):
        with open(out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(cols)
            for b in rows:
                w.writerow([b[0]] + [round(v, 2) for v in b[1:5]] + [round(b[5], 4), b[6], b[7]])
    print(json.dumps(dict(arm=arm, clip=clip, stride=stride, n_full=len(full),
                          n_tiled=len(tiled), n_discard=n_discard, n_merged=len(merged),
                          n_tiled_kept=sum(1 for b in merged if b[7] == 'tile'))))


if __name__ == '__main__':
    main()
