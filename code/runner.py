# P1 真实评分 runner — 纯函数（无 I/O 副作用），供 runner 与测试共用。
# 复用 visdrone_ann.py（源: tr8a_upgrade_20261026/u2/visdrone_ann.py，
# SHA256 b4fd1652512bc62f71441a5d330868e33d1cc2592e840700ded4c0fb1c28f858）
# 的解析/ignore/类别/分层规则；Seq-NMS 复用 P1-0 已测试的 DP/max 语义（H/F 只改抑制参照范围）。
import csv, json, sys, time

PERSON_CATS = (1, 2)
VEHICLE_CATS = (3, 4, 5, 6, 7, 8, 9, 10)
IGNORED_CATS = (0, 11)
LINK = 0.5
SUPP = 0.3
TB = 0.25
GRID = [round(0.01 + 0.01 * k, 2) for k in range(99)]  # 0.01..0.99 step 0.01
TOL = 0.005


def parse_gt_line(line):
    line = line.strip()
    if not line:
        return None
    p = line.split(',')
    assert len(p) == 10, 'GT 应为 10 字段: %r' % line
    fid, tid, bl, bt, bw, bh, score, cat, trunc, occ = [int(float(x)) for x in p]
    return {'frame': fid, 'id': tid, 'x1': bl, 'y1': bt, 'x2': bl + bw, 'y2': bt + bh,
            'cat': cat}


def load_annotation(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = parse_gt_line(line)
            if r is None:
                continue
            fr = out.setdefault(r['frame'], {'person': [], 'vehicle': [], 'ignored': []})
            if r['cat'] in PERSON_CATS:
                fr['person'].append(r)
            elif r['cat'] in VEHICLE_CATS:
                fr['vehicle'].append(r)
            elif r['cat'] in IGNORED_CATS:
                fr['ignored'].append(r)
    return out


def center(b):
    return ((b['x1'] + b['x2']) / 2.0, (b['y1'] + b['y2']) / 2.0)


def center_in(b, regions):
    cx, cy = center(b)
    return any(r['x1'] <= cx <= r['x2'] and r['y1'] <= cy <= r['y2'] for r in regions)


def eval_person_gt(frame_entry):
    ig = frame_entry['ignored']
    return [b for b in frame_entry['person'] if not center_in(b, ig)]


def filter_predictions(preds, frame_entry):
    ig = frame_entry['ignored']
    return [p for p in preds if not center_in(p, ig)]


def size_layer(h):
    return '<20' if h < 20 else ('20-32' if h < 32 else '>=32')


def load_candidates(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({'frame': int(r['frame']), 'x1': float(r['x1']), 'y1': float(r['y1']),
                         'x2': float(r['x2']), 'y2': float(r['y2']), 'conf': float(r['conf']),
                         'cls': int(r['cls']), '_i': i})
    return rows


def iou(a, b):
    ix1, iy1 = max(a['x1'], b['x1']), max(a['y1'], b['y1'])
    ix2, iy2 = min(a['x2'], b['x2']), min(a['y2'], b['y2'])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aw, ah = a['x2'] - a['x1'], a['y2'] - a['y1']
    bw, bh = b['x2'] - b['x1'], b['y2'] - b['y1']
    return inter / (aw * ah + bw * bh - inter)


def seq_nms(pools, link=LINK, supp=SUPP, supp_all_frames=True):
    """DP 最大分数和路径 + max 重评分；H(supp_all_frames=True)/F(False) 只改抑制参照范围。"""
    frames = sorted(pools)
    boxes = {f: pools[f] for f in frames if pools[f]}
    frames = sorted(boxes)
    adj = {}
    for a, b in zip(frames, frames[1:]):
        if b != a + 1:
            continue
        adj[a] = [(i, j) for i, x in enumerate(boxes[a]) for j, y in enumerate(boxes[b]) if iou(x, y) > link]
    alive = {f: [True] * len(boxes[f]) for f in frames}
    out = {}
    while True:
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
        seq = []
        f, i = ef, ei
        while i is not None:
            seq.append((f, i))
            i = prev[(f, i)]
            f -= 1
        seq_conf = max(boxes[f][i]['conf'] for f, i in seq)
        for f, i in seq:
            out[boxes[f][i]['_i']] = seq_conf
            alive[f][i] = False
        seq_boxes = [boxes[f][i] for f, i in seq]
        for f, i in seq:
            for j in range(len(boxes[f])):
                if alive[f][j]:
                    ref = seq_boxes if supp_all_frames else [boxes[ff][ii] for ff, ii in seq if ff == f]
                    if any(iou(boxes[f][j], sb) > supp for sb in ref):
                        alive[f][j] = False
    return out


def match_eval(gt_by_frame, cands, thr):
    """一对一 IoU>=0.5，conf 降序贪心。返回 (tp_by_layer, fp, gt_by_layer, recall_by_layer, precision)。"""
    from collections import defaultdict
    tp_l = defaultdict(int)
    gt_l = defaultdict(int)
    fp = 0
    # 先算每层 GT 数（用 eval_person_gt 过滤 ignored）
    for fr, entry in gt_by_frame.items():
        for g in eval_person_gt(entry):
            gt_l[size_layer(g['y2'] - g['y1'])] += 1
    # 分组候选并按 conf 降序
    cand_by_frame = defaultdict(list)
    for c in cands:
        if c['conf'] >= thr:
            cand_by_frame[c['frame']].append(c)
    for fr in sorted(cand_by_frame):
        entry = gt_by_frame.get(fr, {'person': [], 'ignored': []})
        preds = filter_predictions(cand_by_frame[fr], entry)
        preds = sorted(preds, key=lambda p: -p['conf'])
        gts = eval_person_gt(entry)
        matched = [False] * len(gts)
        for p in preds:
            best_j = -1
            best_iou = 0.0
            for j, g in enumerate(gts):
                if matched[j]:
                    continue
                v = iou(p, g)
                if v >= 0.5 and v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j >= 0:
                matched[best_j] = True
                tp_l[size_layer(gts[best_j]['y2'] - gts[best_j]['y1'])] += 1
            else:
                fp += 1
    tp = sum(tp_l.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_by_layer = {k: (tp_l[k] / gt_l[k] if gt_l[k] else None) for k in ('<20', '20-32', '>=32')}
    return tp_l, fp, gt_l, recall_by_layer, precision


def solve_tau(cands, gt_by_frame, target):
    """升序扫描首个非空且 precision>=target 的网格点，返回 (tau, precision, tp, fp)。不可达返回 (None,None,0,0)。"""
    for t in GRID:
        _, fp, _, _, prec = match_eval(gt_by_frame, cands, t)
        tp = None
        if prec >= target:
            # 重新取 tp
            tp_l, fpx, gt_l, rl, precx = match_eval(gt_by_frame, cands, t)
            if sum(tp_l.values()) + fpx == 0:
                continue  # 空预测不得作为达标
            return t, precx, tp_l, fpx
    return None, None, {}, 0
