import os, json, sys
from collections import defaultdict
sys.path.insert(0, '/PATH/TO/PROJECT/p1_real_20261103_131548/runner')
from runner import load_annotation, load_candidates, match_eval, TB, GRID

BASE='/PATH/TO/PROJECT'
U2=f'{BASE}/tr8a_upgrade_20261026/u2'
SCORES=f'{BASE}/p1_real_v2_20261103_133318'
SEQS=['uav0000086_00000_v','uav0000117_02622_v','uav0000137_00458_v','uav0000182_00000_v','uav0000268_05773_v','uav0000305_00000_v','uav0000339_00001_v']
ARMS=['T','M_H','S_H','M_F','S_F']

def load_scores(seq):
    if seq=='uav0000117_02622_v':
        H=json.load(open(f'{SCORES}/cache/{seq}_H.ckpt'))['state']['out']
        F=json.load(open(f'{SCORES}/cache/{seq}_F.ckpt'))['state']['out']
        H={int(k):v for k,v in H.items()}; F={int(k):v for k,v in F.items()}
    else:
        s=json.load(open(f'{SCORES}/scores_{seq}.json'))
        H={int(k):v for k,v in s['H_scores'].items()}; F={int(k):v for k,v in s['F_scores'].items()}
    return H,F

def build_arms(cands, H, F):
    arms={}
    arms['T']=cands
    arms['M_H']=[c for c in cands if c['_i'] in H]  # 原始 conf
    arms['S_H']=[dict(c, conf=H[c['_i']]) for c in cands if c['_i'] in H]
    arms['M_F']=[c for c in cands if c['_i'] in F]
    arms['S_F']=[dict(c, conf=F[c['_i']]) for c in cands if c['_i'] in F]
    return arms

def curve(ann, cands):
    return [match_eval(ann, cands, t) for t in GRID]  # (tp_l, fp, gt_l, rl, prec)

def main():
    seq_curves={seq:{} for seq in SEQS}
    for seq in SEQS:
        ann=load_annotation(f'{U2}/VisDrone2019-VID-val/annotations/{seq}.txt')
        cands=load_candidates(f'{U2}/infer/pred_merged_{seq}.csv')
        H,F=load_scores(seq)
        arms=build_arms(cands,H,F)
        for arm in ARMS:
            seq_curves[seq][arm]=curve(ann, arms[arm])
        print(f'{seq}: 5 臂曲线完成', flush=True)
    # pooled：逐阈值求和
    pooled={arm:[] for arm in ARMS}
    for gi in range(len(GRID)):
        for arm in ARMS:
            tp=defaultdict(int); fp=0; gt=defaultdict(int)
            for seq in SEQS:
                tp_l, f, gt_l, rl, prec = seq_curves[seq][arm][gi]
                for L,v in tp_l.items(): tp[L]+=v
                fp+=f
                for L,v in gt_l.items(): gt[L]+=v
            ttot=sum(tp.values())
            prec=ttot/(ttot+fp) if ttot+fp else 0.0
            rl={L:(tp[L]/gt[L] if gt.get(L) else None) for L in ('<20','20-32','>=32')}
            pooled[arm].append({'tau':GRID[gi],'tp':dict(tp),'fp':fp,'gt':dict(gt),'precision':prec,'recall':rl})
    # 固定 0.25
    gi25=GRID.index(0.25)
    print('=== 固定 τ=0.25 ===')
    for arm in ARMS:
        p=pooled[arm][gi25]
        print(f'{arm}: prec={p["precision"]:.6f} TP={sum(p["tp"].values())} FP={p["fp"]} recall={ {k:(round(v,4) if v is not None else None) for k,v in p["recall"].items()} }')
    # 目标 = pooled T@0.25 precision
    tgt=pooled['T'][gi25]['precision']
    print(f'\n=== 目标 precision = pooled T@0.25 = {tgt:.6f} ===')
    def solve(arm):
        for p in pooled[arm]:
            if sum(p['tp'].values())+p['fp']>0 and p['precision']>=tgt:
                return p['tau'], p['precision'], p['precision']-tgt, p['recall']
        return None,None,None,None
    for arm in ARMS:
        tau,prec,res,rl=solve(arm)
        within = (abs(res)<=0.005) if res is not None else False
        print(f'{arm}: tau={tau} prec={prec if prec else "N/A"} residual={res if res is not None else "N/A"} within={within} recall={rl}')
    json.dump({'pooled':pooled,'T_target':tgt}, open(f'{BASE}/p2_ablation1.json','w'), indent=1)
    print('\nsaved p2_ablation1.json')

if __name__=='__main__':
    main()
