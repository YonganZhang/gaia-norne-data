from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import forecast_gen as FG
import norne_bulk as NB
from fc_mech import PosNet
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_confirm'
CACHE = OUT / '_cache'
V1 = __import__('release_paths').data_path('norne_fc')
CONF = __import__('release_paths').data_path('norne_fc_confirm')
(NPH, NT) = (2, FG.FC_N)
DAYS = FG.FC_GRID
(D_MODEL, HEADS, LAYERS, FF, N_BANDS) = (128, 4, 3, 256, 16)
(LR, EPOCHS, BATCH, WD) = (0.003, 600, 64, 0.0001)

def load_dir(d: Path, tag: str):
    CACHE.mkdir(parents=True, exist_ok=True)
    c = CACHE / f'{tag}.npz'
    if c.exists():
        z = np.load(c)
        return (z['TH'], z['Y'], z['IA'], z['FC'])
    sh = sorted((d / 'shards').glob('*.npz'))
    (TH, Y, IA, FC) = ([], [], [], [])
    for p in sh:
        z = np.load(p)
        ob = z['obs']
        TH.append(z['theta'].ravel())
        IA.append(z['inj_actual'])
        FC.append(z['field_cum'])
        Y.append(np.stack([[ob[(i * 3 + k) * NT:(i * 3 + k + 1) * NT] for k in (0, 1)] for i in range(len(NB.PRODUCERS))]))
    TH = np.stack(TH).astype(np.float32)
    Y = np.stack(Y).astype(np.float32)
    IA = np.stack(IA).astype(np.float32)
    FC = np.stack(FC).astype(np.float32)
    np.savez_compressed(c, TH=TH, Y=Y, IA=IA, FC=FC)
    print(f'  {tag}: {len(TH)} 样本  Y{Y.shape}  → 缓存 {c.name}', flush=True)
    return (TH, Y, IA, FC)

def cum_oil(A, nw):
    return np.trapezoid(A.reshape(-1, nw, NPH, NT)[:, :, 0, :], DAYS, axis=-1)

def metrics(cP, cT):
    (fP, fT) = (cP.sum(1), cT.sum(1))
    fr = np.abs(fP - fT) / np.abs(fT)
    pr = (np.abs(cP - cT) / np.maximum(np.abs(cT), 1e-09)).ravel()
    ss = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return {'field_rel': float(fr.mean()), 'field_med': float(np.median(fr)), 'field_p90': float(np.quantile(fr, 0.9)), 'field_max': float(fr.max()), 'field_r2': ss(fP, fT), 'well_rel': float(pr.mean()), 'well_med': float(np.median(pr)), 'well_p90': float(np.quantile(pr, 0.9)), 'well_max': float(pr.max()), 'well_r2': ss(cP.ravel(), cT.ravel())}

def agg(rows):
    ks = rows[0].keys()
    return {k: {'mean': float(np.mean([r[k] for r in rows])), 'sd': float(np.std([r[k] for r in rows], ddof=1)), 'raw': [float(r[k]) for r in rows]} for k in ks}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    t0 = time.time()
    print('=== 读数据 ===', flush=True)
    (TH1, Y1, IA1, FC1) = load_dir(V1, 'v1_8000')
    (THc, Yc, IAc, FCc) = load_dir(CONF, 'confirm_500')
    (n1, nc) = (len(TH1), len(THc))
    live = ~(np.abs(Y1).max(axis=(0, 3)) < 1e-09).all(1)
    live_c = ~(np.abs(Yc).max(axis=(0, 3)) < 1e-09).all(1)
    assert (live == live_c).all(), f'live 井集合不一致 v1={live.sum()} conf={live_c.sum()}'
    WELLS = [NB.PRODUCERS[i] for i in np.where(live)[0]]
    nw = len(WELLS)
    NCH = nw * NPH
    Y1 = Y1[:, live]
    Yc = Yc[:, live]
    Yf1 = Y1.reshape(n1, -1)
    Yfc = Yc.reshape(nc, -1)
    print(f'  live 井 {nw}: {WELLS}')
    old_te = np.random.default_rng(0).permutation(4000)[:400]
    h1 = {hashlib.md5(r.tobytes()).hexdigest() for r in TH1}
    dup = np.array([hashlib.md5(r.tobytes()).hexdigest() in h1 for r in THc])
    print(f'  θ 与 v1 重合的 confirm 样本: {int(dup.sum())} 条 (索引 {np.where(dup)[0].tolist()})')
    keep = ~dup
    cT_conf = cum_oil(Yfc, nw)
    cT_old = cum_oil(Yf1[old_te], nw)

    def train_eval(tr_idx, seed):
        (ym, ys) = (Yf1[tr_idx].mean(0), Yf1[tr_idx].std(0) + 1e-08)
        (xm, xs) = (TH1[tr_idx].mean(0), TH1[tr_idx].std(0) + 1e-08)
        Xtr = torch.tensor(((TH1[tr_idx] - xm) / xs).astype(np.float32), device=dev)
        Ytr = torch.tensor(((Yf1[tr_idx] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        torch.manual_seed(seed)
        net = PosNet(TH1.shape[1], NCH, NT, 'fourier', d=D_MODEL, heads=HEADS, layers=LAYERS, ff=FF, n_bands=N_BANDS, seed=seed).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            net.train()
            pm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), BATCH):
                b = pm[i:i + BATCH]
                opt.zero_grad()
                F.mse_loss(net(Xtr[b]), Ytr[b]).backward()
                opt.step()
            sch.step()
        net.eval()

        def pred(THx):
            with torch.no_grad():
                P = net(torch.tensor(((THx - xm) / xs).astype(np.float32), device=dev)).cpu().numpy().reshape(len(THx), -1)
            return cum_oil(P * ys + ym, nw)
        triv = cum_oil(np.repeat(ym[None], 1, 0), nw)[0]
        return (pred, triv, sum((p.numel() for p in net.parameters())))
    R = {}
    print(f'\n=== MAIN: 训练 v1 全部 {n1} → 评 confirm {nc} ({args.seeds} 种子) ===', flush=True)
    tr_all = np.arange(n1)
    (rows_c, rows_ck, npar) = ([], [], 0)
    for s in range(args.seeds):
        (pred, triv, npar) = train_eval(tr_all, s)
        cP = pred(THc)
        rows_c.append(metrics(cP, cT_conf))
        rows_ck.append(metrics(cP[keep], cT_conf[keep]))
        print(f"  seed {s}: confirm 场级 {rows_c[-1]['field_rel']:.4%}  每井 {rows_c[-1]['well_rel']:.4%}  R² {rows_c[-1]['field_r2']:.4f}   ({(time.time() - t0) / 60:.0f}min)", flush=True)
    R['MAIN_confirm500'] = agg(rows_c)
    R['MAIN_confirm499_dedup'] = agg(rows_ck)
    R['params'] = int(npar)
    trivP = np.repeat(triv[None], nc, 0)
    R['TRIVIAL_confirm500'] = metrics(trivP, cT_conf)
    R['TRIVIAL_oldtest400'] = metrics(np.repeat(triv[None], 400, 0), cT_old)
    tr_ctrl = np.setdiff1d(np.arange(n1), old_te)
    print(f'\n=== CTRL: 训练 {len(tr_ctrl)} (= 8000 − 旧 test 400) → 同时评 旧test400 与 confirm500 ({args.seeds} 种子) ===', flush=True)
    (rows_o, rows_c2) = ([], [])
    for s in range(args.seeds):
        (pred, _, _) = train_eval(tr_ctrl, s)
        mo = metrics(pred(TH1[old_te]), cT_old)
        mc = metrics(pred(THc), cT_conf)
        rows_o.append(mo)
        rows_c2.append(mc)
        print(f"  seed {s}: 旧test 场级 {mo['field_rel']:.4%} 每井 {mo['well_rel']:.4%} | confirm 场级 {mc['field_rel']:.4%} 每井 {mc['well_rel']:.4%}   ({(time.time() - t0) / 60:.0f}min)", flush=True)
    R['CTRL_oldtest400'] = agg(rows_o)
    R['CTRL_confirm500'] = agg(rows_c2)
    from scipy.stats import ttest_rel
    a = np.array(R['CTRL_confirm500']['field_rel']['raw'])
    b = np.array(R['CTRL_oldtest400']['field_rel']['raw'])
    pf = float(ttest_rel(a, b).pvalue)
    aw = np.array(R['CTRL_confirm500']['well_rel']['raw'])
    bw = np.array(R['CTRL_oldtest400']['well_rel']['raw'])
    pw = float(ttest_rel(aw, bw).pvalue)
    R['set_effect'] = {'field_delta_pp': float((a.mean() - b.mean()) * 100), 'field_p': pf, 'well_delta_pp': float((aw.mean() - bw.mean()) * 100), 'well_p': pw, 'note': '同一次训练(7600 样本，两个集合都不在训练里)下两个评估集的差 = 纯集合效应'}
    R['meta'] = {'seeds': args.seeds, 'epochs': args.epochs, 'n_train_main': int(n1), 'n_train_ctrl': int(len(tr_ctrl)), 'n_confirm': int(nc), 'wells': WELLS, 'n_live_wells': nw, 'arch': {'backbone': 'Transformer+傅里叶位置嵌入(PosNet fourier)', 'd': D_MODEL, 'heads': HEADS, 'layers': LAYERS, 'ff': FF, 'n_bands': N_BANDS, 'lr': LR, 'batch': BATCH, 'wd': WD}, 'confirm_dir': str(CONF), 'v1_dir': str(V1), 'historical_claim': {'field': 0.00366, 'well': 0.00447, 'r2': 0.996, 'n_train': 3400, 'eval': 'norne_fc test-400(已被污染)'}, 'dup_theta_in_confirm': int(dup.sum())}
    json.dump(R, open(OUT / 'confirm.json', 'w'), indent=1, ensure_ascii=False)
    (m, c) = (R['MAIN_confirm500'], R['CTRL_oldtest400'])
    print(f'\n=== 结果 ===')
    print(f"  MAIN 8000→confirm500  场级 {m['field_rel']['mean']:.4%} ± {m['field_rel']['sd']:.4%}  每井 {m['well_rel']['mean']:.4%} ± {m['well_rel']['sd']:.4%}  R² {m['field_r2']['mean']:.4f}  p90 {m['field_p90']['mean']:.4%}")
    print(f"  CTRL 7600→旧test400   场级 {c['field_rel']['mean']:.4%}  每井 {c['well_rel']['mean']:.4%}")
    print(f"  CTRL 7600→confirm500  场级 {R['CTRL_confirm500']['field_rel']['mean']:.4%}  每井 {R['CTRL_confirm500']['well_rel']['mean']:.4%}")
    print(f"  集合效应(场级) {R['set_effect']['field_delta_pp']:+.4f}pp  p={pf:.4f}")
    print(f"  平凡基线 confirm 场级 {R['TRIVIAL_confirm500']['field_rel']:.4%}  每井 {R['TRIVIAL_confirm500']['well_rel']:.4%}")
    print(f'\n已写入 {OUT}/confirm.json   总耗时 {(time.time() - t0) / 60:.0f}min')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
