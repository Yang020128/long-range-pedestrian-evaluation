#!/usr/bin/env python3
"""TR-8B-7 C：口径一（p* = 本次运行 T-base@0.25 精度）下 4 次 R4 运行的统一重算。

只重算 R4 主判定，不跑新推理。规则：PROTOCOL_REVISION_v1.2.md（冻结）。
τ 网格 = 0.01–0.99 步长 0.01（99 点）。判定：方向一致且 CI 不含 0 = 复现；含 0 = 部分复现；相反 = 未复现。
用法：--stage pstar ｜ --stage sweep --run u1|rt|u2 ｜ --stage boot --run u1|rt|u2
输出：R4_V12_RESULT.md、r4_v12.csv（tr8a 根目录）
"""
import csv, json, os, sys, time
from collections import defaultdict
import numpy as np

BASE = '/PATH/TO/PROJECT'
OUT6B = os.path.join(BASE, 'tr6b_stageB_run_20261026')
OUT6D = os.path.join(BASE, 'tr6d_stageB_confirm_20261026')
U2 = os.path.join(BASE, 'tr8a_upgrade_20261026/u2')
U3 = os.path.join(BASE, 'tr8a_upgrade_20261026/u3')
OUT = os.path.join(BASE, 'tr8a_upgrade_20261026')
CACHE = os.path.join(OUT, '.r4_v12_cache')
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, OUT6B); sys.path.insert(0, U2); sys.path.insert(0, U3)
from scommon import load_ref, load_pred, per_frame_tp, build_blocks, layer  # noqa
import visdrone_ann as VA  # noqa
import importlib.util
_spec = importlib.util.spec_from_file_location('merge_tiles', os.path.join(BASE, 'tr5n_run_20261026', 'merge_tiles.py'))
MT = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(MT)

TAU_GRID = [round(0.01 * i, 2) for i in range(1, 100)]
N_BOOT = 10000
LAYERS2 = ('<20', '20-32')
GE32 = ('32-64', '>=64')
VIRAT_HOLD = ['010207_08', '010110_00']
U2_SEQS = ['uav0000086_00000_v', 'uav0000117_02622_v', 'uav0000137_00458_v',
           'uav0000182_00000_v', 'uav0000268_05773_v', 'uav0000305_00000_v',
           'uav0000339_00001_v']
U2_DIMS = {'uav0000086_00000_v': (1344, 756), 'uav0000117_02622_v': (2720, 1530),
           'uav0000137_00458_v': (1920, 1080), 'uav0000182_00000_v': (1920, 1080),
           'uav0000268_05773_v': (3840, 2160), 'uav0000305_00000_v': (1904, 1071),
           'uav0000339_00001_v': (1904, 1071)}


def load_run(name):
    """返回 (clips, ref, arm_pred={'T-base':{c:frame->boxes}, 'S-temp':...}, meta)"""
    if name == 'u1':
        ref = {c: load_ref(c, 'errata') for c in VIRAT_HOLD}
        arms = {'T-base': {}, 'S-temp': {}}
        for c in VIRAT_HOLD:
            arms['T-base'][c] = load_pred('BT-m', c)
        st_map = {}
        with open(os.path.join(OUT6B, 's_temp_conf.csv'), encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                st_map[(r['clip'], int(r['frame']),
                        round(float(r['x1']), 1), round(float(r['y1']), 1),
                        round(float(r['x2']), 1), round(float(r['y2']), 1))] = float(r['conf_stemp'])
        for c in VIRAT_HOLD:
            st = defaultdict(list)
            for f, bs in arms['T-base'][c].items():
                smap = {(k[2], k[3], k[4], k[5]): v for k, v in st_map.items()
                        if k[0] == c and k[1] == f}
                for b in bs:
                    k = (round(b['x1'], 1), round(b['y1'], 1), round(b['x2'], 1), round(b['y2'], 1))
                    if k in smap:
                        st[f].append({**b, 'conf': smap[k]})
            arms['S-temp'][c] = st
        return VIRAT_HOLD, ref, arms, {'seed': 20261026, 'label': 'U1 留出集 × YOLO11m'}
    if name == 'rt':
        clips = ['010207_04', '010207_08', '010110_00', '010114_01', '050202_03']
        ref = {c: load_ref(c, 'errata') for c in clips}
        arms = {'T-base': {}, 'S-temp': {}}
        for c in clips:
            tb = defaultdict(list)
            with open(os.path.join(U3, 'pred_rtdetr_both_%s.csv' % c), encoding='utf-8-sig') as f:
                for i, r in enumerate(csv.DictReader(f)):
                    if float(r['conf']) >= 0.01:
                        tb[int(r['frame'])].append({'x1': float(r['x1']), 'y1': float(r['y1']),
                                                    'x2': float(r['x2']), 'y2': float(r['y2']),
                                                    'conf': float(r['conf']), '_i': i})
            arms['T-base'][c] = tb
            stf = defaultdict(list)
            with open(os.path.join(U3, 'st_conf_rtdetr_%s.csv' % c), encoding='utf-8-sig') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    key_i = int(r['i'])
                    fr = int(r['frame'])
                    conf = float(r['conf_stemp'])
                    # 匹配 both 池中的 _i（st_conf 的 i 即 both 文件的行号）
                    for b in tb.get(fr, []):
                        if b['_i'] == key_i:
                            stf[fr].append({**b, 'conf': conf})
                            break
            arms['S-temp'][c] = stf
        return clips, ref, arms, {'seed': 20261026, 'label': 'VIRAT 开发集 × RT-DETR-l'}
    if name == 'u2':
        ref = {}
        ann_dir = os.path.join(U2, 'VisDrone2019-VID-val', 'annotations')
        seq_dir = os.path.join(U2, 'VisDrone2019-VID-val', 'sequences')
        infer = os.path.join(U2, 'infer')
        for s in U2_SEQS:
            ann = VA.load_annotation(os.path.join(ann_dir, s + '.txt'))
            d = {}
            for f, fe in ann.items():
                ev = VA.eval_person_gt(fe)
                if ev:
                    d[f] = [{'x1': float(b['x1']), 'y1': float(b['y1']),
                             'x2': float(b['x2']), 'y2': float(b['y2'])} for b in ev]
            ref[s] = d
        arms = {'T-base': {}, 'S-temp': {}}
        for s in U2_SEQS:
            ann = VA.load_annotation(os.path.join(ann_dir, s + '.txt'))
            def ig(f):
                return ann.get(f, {'person': [], 'vehicle': [], 'ignored': []})['ignored']
            tb = defaultdict(list)
            with open(os.path.join(infer, 'pred_merged_%s.csv' % s)) as f:
                for i, r in enumerate(csv.DictReader(f)):
                    fr = int(r['frame'])
                    b = {'x1': float(r['x1']), 'y1': float(r['y1']), 'x2': float(r['x2']),
                         'y2': float(r['y2']), 'conf': float(r['conf']), '_i': i}
                    if not VA.center_in(b, ig(fr)):
                        tb[fr].append(b)
            arms['T-base'][s] = tb
            stf = defaultdict(list)
            with open(os.path.join(U2, 'st_conf_%s.csv' % s)) as f:
                for r in csv.DictReader(f):
                    fr, ii, conf = int(r['frame']), int(r['i']), float(r['conf_stemp'])
                    for b in tb.get(fr, []):
                        if b['_i'] == ii:
                            stf[fr].append({**b, 'conf': conf})
                            break
            arms['S-temp'][s] = stf
        return U2_SEQS, ref, arms, {'seed': 20261027, 'label': 'VisDrone × YOLO11m'}
    raise ValueError(name)


def pstar_of(ref, tb):
    TP = FP = 0
    for c in ref:
        for f in sorted(set(ref[c]) | set(tb[c])):
            _, _, tl, m, n = per_frame_tp(ref[c].get(f, []), tb[c].get(f, []), 0.25)
            TP += sum(tl.values()); FP += n - m
    return (TP / (TP + FP) if TP + FP else 0.0), TP, FP


def solve_tau(ref, st, pstar, tag, budget=250.0):
    rows_path = os.path.join(CACHE, 'sweep_%s.json' % tag)
    rows = json.load(open(rows_path)) if os.path.exists(rows_path) else {}
    t0 = time.time()
    for tau in TAU_GRID:
        tk = '%.2f' % tau
        if tk in rows:
            continue
        TP = FP = 0
        for c in ref:
            for f in sorted(set(ref[c]) | set(st[c])):
                _, _, tl, m, n = per_frame_tp(ref[c].get(f, []), st[c].get(f, []), tau)
                TP += sum(tl.values()); FP += n - m
        rows[tk] = {'precision': TP / (TP + FP) if TP + FP else 0.0}
        json.dump(rows, open(rows_path, 'w'))
        if time.time() - t0 > budget:
            return None  # 未跑完，调用方应重入
    star, clamped, prev = None, None, None
    for tau in TAU_GRID:
        if rows['%.2f' % tau]['precision'] >= pstar - 1e-12:
            star = tau
            break
        prev = tau
    if star is None:
        star, clamped = TAU_GRID[-1], 'HIGH'
    elif star == TAU_GRID[0]:
        clamped = 'LOW'
    if not clamped and prev is not None:
        p1, p2 = rows['%.2f' % prev]['precision'], rows['%.2f' % star]['precision']
        w = 0.0 if p2 == p1 else (pstar - p1) / (p2 - p1)
        tau_eff = prev + w * (star - prev)
    else:
        tau_eff = float(star)
    return {'tau_eff': tau_eff, 'clamped': clamped,
            'precision_at_star': rows['%.2f' % star]['precision'],
            'rows': rows}


def arm_stats_run(ref, arms, clips, tau_st, tau_tb=0.25):
    out = {}
    for arm, tau in (('T-base', tau_tb), ('S-temp', tau_st)):
        TP = FP = 0
        tpL = defaultdict(int); gtL = defaultdict(int)
        for c in clips:
            for f in sorted(set(ref[c]) | set(arms[arm][c])):
                _, gl, tl, m, n = per_frame_tp(ref[c].get(f, []), arms[arm][c].get(f, []), tau)
                TP += sum(tl.values()); FP += n - m
                for L, v in gl.items():
                    gtL[L] += v
                for L, v in tl.items():
                    tpL[L] += v
        out[arm] = {'precision': TP / (TP + FP) if TP + FP else 0.0,
                    'tp': TP, 'fp': FP, 'tpL': dict(tpL), 'gtL': dict(gtL)}
    return out


def paired_boot(ref, arms, clips, tau_st, seed, layers=LAYERS2):
    bl, fl = build_blocks(clips, ref)
    nb = len(bl)
    starts = np.cumsum([0] + [len(b) for b in bl])
    res = {}
    for L in layers:
        Ls = GE32 if L == '>=32' else (L,)
        gtA = np.zeros(len(fl)); tpA = np.zeros(len(fl))
        gtB = np.zeros(len(fl)); tpB = np.zeros(len(fl))
        for i, (c, f) in enumerate(fl):
            rb = ref[c].get(f, [])
            _, gl, tl, _, _ = per_frame_tp(rb, arms['S-temp'][c].get(f, []), tau_st)
            gtA[i] = sum(gl.get(x, 0) for x in Ls); tpA[i] = sum(tl.get(x, 0) for x in Ls)
            _, gl, tl, _, _ = per_frame_tp(rb, arms['T-base'][c].get(f, []), 0.25)
            gtB[i] = sum(gl.get(x, 0) for x in Ls); tpB[i] = sum(tl.get(x, 0) for x in Ls)
        rng = np.random.default_rng(seed)
        d = np.empty(N_BOOT)
        for r in range(N_BOOT):
            sel = rng.integers(0, nb, size=nb)
            ii = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in sel])
            gA, gB = gtA[ii].sum(), gtB[ii].sum()
            d[r] = (tpA[ii].sum() / gA if gA else np.nan) - (tpB[ii].sum() / gB if gB else np.nan)
        point = (tpA.sum() / gtA.sum() if gtA.sum() else np.nan) - \
                (tpB.sum() / gtB.sum() if gtB.sum() else np.nan)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        res[L] = {'delta': float(point), 'P(>0)': float(np.nanmean(d > 0)),
                  'ci': [float(lo), float(hi)]}
    return res


def verdict(e):
    if e['delta'] != e['delta']:
        return '无GT'
    return '复现' if (e['delta'] > 0 and not (e['ci'][0] <= 0 <= e['ci'][1])) else \
           ('部分复现' if e['delta'] > 0 else '未复现')


def main():
    stage = sys.argv[sys.argv.index('--stage') + 1]
    results = {}
    if os.path.exists(os.path.join(OUT, 'r4_v12_partial.json')):
        results = json.load(open(os.path.join(OUT, 'r4_v12_partial.json')))
    runs = sys.argv[sys.argv.index('--run') + 1].split(',') if '--run' in sys.argv else ['u1', 'rt', 'u2']
    for name in runs:
        clips, ref, arms, meta = load_run(name)
        key_p = 'pstar_%s' % name
        if stage in ('pstar', 'all') or key_p not in results:
            p, TP, FP = pstar_of(ref, arms['T-base'])
            results[key_p] = {'pstar': p, 'tp': TP, 'fp': FP}
            json.dump(results, open(os.path.join(OUT, 'r4_v12_partial.json'), 'w'))
        pstar = results[key_p]['pstar']
        key_t = 'tau_%s' % name
        if stage in ('sweep', 'all') or key_t not in results:
            star = solve_tau(ref, arms['S-temp'], pstar, name)
            if star is None:
                print(json.dumps({'run': name, 'status': 'sweep-timeout-rerun'}), flush=True)
                continue
            results[key_t] = star
            json.dump(results, open(os.path.join(OUT, 'r4_v12_partial.json'), 'w'))
        tau_st = results[key_t]['tau_eff']
        clamped = results[key_t]['clamped']
        if stage in ('boot', 'all') or ('boot_%s' % name) not in results:
            if clamped:
                results['boot_%s' % name] = {'verdict': '无法判定', 'clamped': clamped}
            else:
                scopes = [('merged', clips)]
                if name == 'u1':
                    scopes += [(c, [c]) for c in clips]
                if name == 'u2':
                    scopes += [(c, [c]) for c in clips]
                    scopes += [('LOSO-%s' % d, [c for c in clips if c != d]) for d in clips]
                layers = LAYERS2 + (('>=32',) if name == 'u2' else ())
                b = {}
                for scope, cl in scopes:
                    b[scope] = paired_boot(ref, arms, cl, tau_st, meta['seed'], layers=layers)
                stats = arm_stats_run(ref, arms, clips, tau_st)
                results['boot_%s' % name] = {'scopes': b, 'arm_stats': stats,
                                             'verdict': {L: verdict(b['merged'][L]) for L in layers}}
            json.dump(results, open(os.path.join(OUT, 'r4_v12_partial.json'), 'w'))
        print(json.dumps({'run': name, 'pstar': round(pstar, 4),
                          'tau_star': round(tau_st, 4),
                          'clamped': results[key_t]['clamped'],
                          'verdict': results['boot_%s' % name].get('verdict')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
