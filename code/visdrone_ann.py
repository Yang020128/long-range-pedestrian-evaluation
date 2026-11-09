#!/usr/bin/env python3
"""TR-8B-2：VisDrone2019-VID 标注解析（U2_RULES_v1.1 §2 的实现）。

VID 地面真值 10 字段：frame_idx,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
score,object_category,truncation,occlusion（score 恒为 1，记录不判定）。
类别映射（冻结）：person={1,2}；vehicle={3..10}（仅参照统计）；ignored={0,11}。
ignored 处理：GT 中心落入 ignored 框 → 不参评；预测中心落入 ignored 框 → 不计 FP。
尺寸分层（冻结）：GT 框高（原始像素）h<20 → '<20'；20<=h<32 → '20-32'；h>=32 → '>=32'。
"""
import os

PERSON_CATS = (1, 2)
VEHICLE_CATS = (3, 4, 5, 6, 7, 8, 9, 10)
IGNORED_CATS = (0, 11)


def parse_line(line):
    """解析单行标注；返回 dict 或 None（空行）。"""
    line = line.strip()
    if not line:
        return None
    p = line.split(',')
    assert len(p) == 10, 'VID GT 应为 10 字段，得到 %d：%r' % (len(p), line)
    fid, tid, bl, bt, bw, bh, score, cat, trunc, occ = [int(float(x)) for x in p]
    return {'frame': fid, 'id': tid, 'x1': bl, 'y1': bt, 'x2': bl + bw, 'y2': bt + bh,
            'score': score, 'cat': cat, 'trunc': trunc, 'occ': occ}


def load_annotation(path):
    """返回 {frame: {'person': [box], 'vehicle': [box], 'ignored': [box]}}（box 含全部字段）。"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = parse_line(line)
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
    """参评 person GT：去掉中心落入 ignored 框的。"""
    ig = frame_entry['ignored']
    return [b for b in frame_entry['person'] if not center_in(b, ig)]


def filter_predictions(preds, frame_entry):
    """预测框：中心落入 ignored 框的不计 FP（返回保留列表）。"""
    ig = frame_entry['ignored']
    return [p for p in preds if not center_in(p, ig)]


def size_layer(h):
    """冻结分层：h<20 '<20'；20<=h<32 '20-32'；h>=32 '>=32'。"""
    return '<20' if h < 20 else ('20-32' if h < 32 else '>=32')
