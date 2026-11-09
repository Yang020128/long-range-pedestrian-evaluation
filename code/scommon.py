#!/usr/bin/env python3
"""TR-6B v1.2 共享库：参照加载（双口径）、匹配、Seq-NMS、阈值扫描、bootstrap。"""
import csv, json, os, glob
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

BASE = '/PATH/TO/PROJECT'
FIN = json.load(open(os.path.join(BASE, 'tr5i_final_freeze_20261026/tr5e_final_annotation_20261026.json')))
CLIPS = ['010207_04', '010207_08', '010110_00', '010114_01', '050202_03']
DEV = ['010207_04', '010114_01', '050202_03']
HOLDOUT = ['010207_08', '010110_00']
TAU_GRID = [round(0.01 * i, 2) for i in range(1, 61)]
BLOCK = 30

_ERR = defaultdict(list)
with open(os.path.join(BASE, 'tr5r_eval_errata_010114_01_20261026/errata_frames_77.csv'),
          encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        _ERR[int(r['frame_id'])].append(tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')))


def is_errata(f, b):
    return any(all(abs(b[k] - e[i]) < 2.0 for i, k in enumerate(('x1', 'y1', 'x2', 'y2')))
               for e in _ERR.get(f, ()))


def layer(h):
    return '<20' if h < 20 else ('20-32' if h < 32 else ('32-64' if h < 64 else '>=64'))
LAYERS = ('<20', '20-32', '32-64', '>=64')


def load_ref(clip, caliber='errata'):
    out = {}
    for f, v in FIN['windows'][clip]['frames'].items():
        f = int(f)
        bs = v['boxes']
        if caliber == 'errata' and clip == '010114_01':
            bs = [b for b in bs if not is_errata(f, b)]
        out[f] = bs
    return out


def load_pred(arm, clip, conf_col='conf'):
    """返回 dict frame -> list[dict]，conf>=0.01。conf_col 允许 'conf' 或 'conf2'（预计算 conf'）。"""
    d = defaultdict(list)
    with open(os.path.join(BASE, 'tr5n_run_20261026/predictions_merged/%s_%s.csv' % (arm, clip)),
              encoding='utf-8-sig') as f:
        for i, r in enumerate(csv.DictReader(f)):
            c = float(r[conf_col]) if conf_col in r else float(r['conf'])
            if c < 0.01:
                continue
            d[int(r['frame'])].append({'x1': float(r['x1']), 'y1': float(r['y1']),
                                       'x2': float(r['x2']), 'y2': float(r['y2']),
                                       'conf': float(r['conf']), '_i': i})
    return d


def iou(a, b):
    x1, y1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    x2, y2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    iw, ih = x2 - x1, y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a['x2']-a['x1'])*(a['y2']-a['y1']) + (b['x2']-b['x1'])*(b['y2']-b['y1']) - i)


def match_pairs(rb, pb, thr=0.5):
    """冻结匹配：一对一、IoU>=0.5、先最大匹配数再最大总 IoU。返回 [(gt_idx, pred_idx, iou)]。"""
    if not rb or not pb:
        return []
    W = np.zeros((len(rb), len(pb)))
    for i, g in enumerate(rb):
        ga = (g['x2']-g['x1']) * (g['y2']-g['y1'])
        for j, p in enumerate(pb):
            v = iou(g, p)
            W[i, j] = (1000.0 + v) if v >= thr else 0.0
    ri, cj = linear_sum_assignment(-W)
    return [(int(i), int(j), W[i, j] - 1000.0) for i, j in zip(ri, cj) if W[i, j] > 0]


def seq_nms(pools, link_thr=0.5, supp_thr=0.3):
    """简化 Seq-NMS（Han et al. 2016，RELATED_BASELINES §2 规格）。
    pools: frame -> [box(conf)]。返回 dict _i -> conf'（被抑制框无条目）。"""
    frames = sorted(pools)
    boxes = {f: pools[f] for f in frames if pools[f]}
    frames = sorted(boxes)
    # 预计算相邻帧链接（t, t+1）
    adj = {}
    for a, b in zip(frames, frames[1:]):
        if b != a + 1:
            continue
        la = []
        for i, x in enumerate(boxes[a]):
            for j, y in enumerate(boxes[b]):
                if iou(x, y) > link_thr:
                    la.append((i, j))
        adj[a] = la
    alive = {f: [True] * len(boxes[f]) for f in frames}
    out = {}
    while True:
        # 动态规划：best_end[i] = 以 (f,i) 结尾的最大分数和
        best, prev = {}, {}
        for f in frames:
            for i in range(len(boxes[f])):
                if not alive[f][i]:
                    continue
                s = boxes[f][i]['conf']
                bp, bj = 0.0, None
                if f - 1 in adj and f - 1 in boxes:
                    for (pi, ci) in adj[f - 1]:
                        if ci == i and alive[f - 1][pi] and best.get((f - 1, pi), 0) > bp:
                            bp, bj = best[(f - 1, pi)], pi
                best[(f, i)] = s + bp
                prev[(f, i)] = bj
        cands = [(v, k) for k, v in best.items() if alive[k[0]][k[1]]]
        if not cands:
            break
        _, (ef, ei) = max(cands)
        # 回溯序列
        seq = []
        f, i = ef, ei
        while i is not None:
            seq.append((f, i))
            i = prev[(f, i)]
            f = f - 1
        seq_conf = max(boxes[f][i]['conf'] for f, i in seq)
        for f, i in seq:
            out[boxes[f][i]['_i']] = seq_conf
            alive[f][i] = False
        # 抑制：同帧 IoU>0.3 于序列框
        seq_boxes = [boxes[f][i] for f, i in seq]
        for f, i in seq:
            for j in range(len(boxes[f])):
                if alive[f][j] and any(iou(boxes[f][j], sb) > supp_thr for sb in seq_boxes):
                    alive[f][j] = False
    return out


def build_blocks(clips, ref_by_clip):
    """30 帧块（不跨 clip、不跨帧范围间隙，且每满 BLOCK=30 帧即切分）。返回 (blocks, frame_list)。
    frame_list: 平铺的 (clip, frame)；块为索引区间。
    ERRATA（TR-6C CODE_FIXES）：原实现漏了 len(cur)>=BLOCK 切分，整段连续帧只成一块
    （开发集退化为 3 簇）；已修正，全部依赖 bootstrap 的产物已重导出。"""
    frame_list = []
    for clip in clips:
        fs = sorted(ref_by_clip[clip])
        for f in fs:
            frame_list.append((clip, f))
    # 块按 clip 内连续帧切，每满 BLOCK 帧即切
    blocks, cur = [], []
    prev = None
    for idx, (clip, f) in enumerate(frame_list):
        if prev is not None and (clip != prev[0] or f > prev[1] + 1 or len(cur) >= BLOCK):
            blocks.append(cur)
            cur = []
        cur.append(idx)
        prev = (clip, f)
    if cur:
        blocks.append(cur)
    return blocks, frame_list


def per_frame_tp(ref, preds, tau):
    """对单帧：返回 (gt_total, gt_by_layer, tp_by_layer, n_pred_matched, n_pred)。"""
    rb = ref
    pb = [p for p in preds if p['conf'] >= tau]
    prs = match_pairs(rb, pb)
    gtL = defaultdict(int)
    for b in rb:
        gtL[layer(max(b['y2'] - b['y1'], 0))] += 1
    tpL = defaultdict(int)
    matched_pred = set()
    for gi, pj, _ in prs:
        tpL[layer(max(rb[gi]['y2'] - rb[gi]['y1'], 0))] += 1
        matched_pred.add(pj)
    return len(rb), dict(gtL), dict(tpL), len(matched_pred), len(pb)


def sweep_matched_precision(ref_by_clip, pred_by_clip, pstar, taus=TAU_GRID):
    """对每臂（pred_by_clip 已带 conf'）扫描 τ；返回 (tau_star, row_by_tau)。
    row: dict tau -> (precision, recall_by_layer, gt_by_layer, tp_by_layer, fp_total)。仅在含 GT 帧上计 recall，
    precision 在全部预测帧上计（需要全部帧的预测，含无 GT 帧）。"""
    rows = {}
    for tau in taus:
        TP = FP = GT = 0
        tpL, gtL = defaultdict(int), defaultdict(int)
        for clip in pred_by_clip:
            ref = ref_by_clip[clip]
            # 全部帧（预测帧 ∪ GT 帧）
            frames = sorted(set(ref) | set(pred_by_clip[clip]))
            for f in frames:
                rb = ref.get(f, [])
                gt_n, gL, tL, m, n = per_frame_tp(rb, pred_by_clip[clip].get(f, []), tau)
                TP += sum(tL.values())
                FP += n - m
                GT += gt_n
                for L, v in gL.items():
                    gtL[L] += v
                for L, v in tL.items():
                    tpL[L] += v
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = {L: (tpL[L] / gtL[L] if gtL[L] else None) for L in LAYERS}
        rows[tau] = {'precision': prec, 'recall': rec, 'tpL': dict(tpL), 'gtL': dict(gtL),
                     'tp': TP, 'fp': FP}
    # 找 tau*：precision 首次 >= p* 的点（τ 升序扫描；precision 随 τ 单调不降）。
    # ERRATA（STEP1B_ERRATA.md，2026-10-26）：原实现找 <= p*，在网格左缘误命中，
    # 使"匹配精度"点实际精度远低于 p*；已修正为 >= 方向。
    star, clamped = None, None
    prev_tau = None
    for tau in taus:
        if rows[tau]['precision'] >= pstar - 1e-12:
            star = tau
            break
        prev_tau = tau
    if star is None:
        star = taus[-1]
        clamped = 'HIGH'
    elif star == taus[0]:
        clamped = 'LOW'
    # 插值 tau*
    if not clamped and prev_tau is not None:
        p1, p2 = rows[prev_tau]['precision'], rows[star]['precision']
        w = 0.0 if p2 == p1 else (pstar - p1) / (p2 - p1)
        tau_eff = prev_tau + w * (star - prev_tau)
    else:
        tau_eff = float(star)
    # tau* 行（取 star 行；插值只在 τ 维度，指标取 star 行并记录 tau_eff）
    r = dict(rows[star])
    r['tau_eff'] = round(tau_eff, 4)
    r['clamped'] = clamped or ''
    return r, rows


def paired_bootstrap_sigma(ref_by_clip, predA, predB, tauA, tauB, L, n_boot=10000, seed=20261026):
    """配对 cluster bootstrap：Δrecall_L(A−B) 的标准差。A/B 为两臂（clip->frame->boxes, conf' 已写入 conf 字段）。"""
    blocks, frame_list = build_blocks(list(ref_by_clip), ref_by_clip)
    nF = len(frame_list)
    gtA = np.zeros(nF); tpA = np.zeros(nF)
    gtB = np.zeros(nF); tpB = np.zeros(nF)
    for idx, (clip, f) in enumerate(frame_list):
        rb = ref_by_clip[clip].get(f, [])
        gA = defaultdict(int); tA = defaultdict(int)
        _, gl, tl, _, _ = per_frame_tp(rb, predA[clip].get(f, []), tauA)
        gtA[idx] = gl.get(L, 0); tpA[idx] = tl.get(L, 0)
        _, gl, tl, _, _ = per_frame_tp(rb, predB[clip].get(f, []), tauB)
        gtB[idx] = gl.get(L, 0); tpB[idx] = tl.get(L, 0)
    rng = np.random.default_rng(seed)
    nb = len(blocks)
    deltas = np.empty(n_boot)
    starts = np.cumsum([0] + [len(b) for b in blocks])
    for r in range(n_boot):
        sel = rng.integers(0, nb, size=nb)
        idx = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in sel])
        gA = gtA[idx].sum(); tA = tpA[idx].sum()
        gB = gtB[idx].sum(); tB = tpB[idx].sum()
        ra = tA / gA if gA else np.nan
        rb = tB / gB if gB else np.nan
        deltas[r] = ra - rb
    return float(np.nanstd(deltas))
