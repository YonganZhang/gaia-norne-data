from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
import forecast_gen as FG
import norne_bulk as NB
import fc_decision as FD_data
import fc_decide as FD
import fc_agent as FA
from fc_mech import PosNet
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_rank_fidelity'
SIM = ROOT / '_pipelines' / 'fc_decide' / 'sim'
CONFIRM = __import__('release_paths').data_path('norne_fc_confirm')
(NPH, NT) = (2, FG.FC_N)
DAYS = FG.FC_GRID
(N_STAGE, INJ) = (FG.N_STAGE, FG.INJ_W)
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ])
SHORT_Y = 3.0
WT = np.gradient(DAYS).astype(float)
WT[0] *= 0.5
WT[-1] *= 0.5
MS = (DAYS - DAYS[0] <= SHORT_Y * 365.25).astype(float)

class _A:

    def __init__(self, threads):
        self.threads = threads

def load_dir(d: Path, max_n=0):
    sh = sorted((d / 'shards').glob('*.npz'))
    if max_n:
        sh = sh[:max_n]
    (TH, Y) = ([], [])
    for p in sh:
        z = np.load(p)
        ob = z['obs']
        TH.append(z['theta'].ravel())
        Y.append(np.stack([[ob[(i * 3 + k) * NT:(i * 3 + k + 1) * NT] for k in (0, 1)] for i in range(len(NB.PRODUCERS))]))
    return (np.stack(TH).astype(np.float32), np.stack(Y).astype(np.float32))

def rank_metrics(surr, true, ks=(1, 3, 5)):
    (surr, true) = (np.asarray(surr, float), np.asarray(true, float))
    n = len(surr)
    (rho, p_rho) = stats.spearmanr(surr, true)
    (tau, p_tau) = stats.kendalltau(surr, true)
    (os_, ot) = (np.argsort(-surr), np.argsort(-true))
    hit = {}
    for k in ks:
        if k <= n:
            hit[f'top{k}'] = float(len(set(os_[:k].tolist()) & set(ot[:k].tolist())) / k)
    (pick, best) = (int(os_[0]), int(ot[0]))
    return {'n': n, 'spearman_rho': float(rho), 'spearman_p': float(p_rho), 'kendall_tau': float(tau), 'kendall_p': float(p_tau), 'hit_rate': hit, 'surr_pick_idx': pick, 'true_best_idx': best, 'surr_pick_true_rank': int(np.where(ot == pick)[0][0]) + 1, 'true_best_surr_rank': int(np.where(os_ == best)[0][0]) + 1, 'top1_regret_pct': float((true[best] - true[pick]) / true[best] * 100.0)}

def rho_uncertainty(surr, true, n_perm=20000, n_boot=4000, seed=0):
    (surr, true) = (np.asarray(surr, float), np.asarray(true, float))
    rng = np.random.default_rng(seed)
    obs = stats.spearmanr(surr, true).statistic
    perm = np.array([stats.spearmanr(surr, rng.permutation(true)).statistic for _ in range(n_perm)])
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(surr), len(surr))
        if len(set(i.tolist())) > 3:
            bs.append(stats.spearmanr(surr[i], true[i]).statistic)
    (lo, hi) = np.percentile(bs, [2.5, 97.5])
    return {'rho': float(obs), 'perm_p_two_sided': float(np.mean(np.abs(perm) >= abs(obs))), 'boot_ci95': [float(lo), float(hi)], 'n_perm': n_perm, 'n_boot': len(bs)}

def bias_fit(x, y):
    (x, y) = (np.asarray(x, float), np.asarray(y, float))
    r = stats.linregress(x, y)
    b0 = float(x @ y / (x @ x)) if np.abs(x).sum() > 0 else float('nan')
    return {'slope': float(r.slope), 'intercept': float(r.intercept), 'r2': float(r.rvalue ** 2), 'slope_stderr': float(r.stderr), 'slope_p': float(r.pvalue), 'slope_through_origin': b0, 'overestimation_factor': float(1.0 / b0) if b0 not in (0.0,) else float('nan')}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--n-pin', type=int, default=8000)
    ap.add_argument('--ntrain', type=int, default=7400)
    ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--n-start', type=int, default=40)
    ap.add_argument('--n-cand', type=int, default=10)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--trust-r', type=float, default=2.5)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--calib-iter', type=int, default=6)
    ap.add_argument('--water-tol', type=float, default=0.0005)
    ap.add_argument('--dry-run', action='store_true', help='只跑到候选生成，不做模拟')
    ap.add_argument('--tag', default='', help='模拟 key 前缀后缀，避免不同配置复用同名算例')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f'cuda:{a.gpu}' if a.gpu >= 0 else 'cpu'
    torch.set_num_threads(8)
    t0 = time.time()
    (TH, Y) = load_dir(FG.OUT, a.n_pin)
    live = ~(np.abs(Y).max(axis=(0, 3)) < 1e-09).all(1)
    Y = Y[:, live]
    nw = int(live.sum())
    n = len(TH)
    idx = np.random.default_rng(0).permutation(n)
    (te, pool) = (idx[:400], idx[600:])
    tr = pool[:a.ntrain]
    Yf = Y.reshape(n, -1)
    NCH = nw * NPH
    (ym, ys) = (Yf[tr].mean(0), Yf[tr].std(0) + 1e-08)
    (xm, xs) = (TH[tr].mean(0), TH[tr].std(0) + 1e-08)
    X = ((TH - xm) / xs).astype(np.float32)
    print(f'训练代理: {len(tr)} 样本  活井 {nw}  θ {TH.shape[1]} 维  device={dev}', flush=True)
    torch.manual_seed(0)
    net = PosNet(TH.shape[1], NCH, NT, 'fourier', n_bands=16).to(dev)
    ck = OUT / f'surrogate{a.tag}.pt'
    if ck.exists():
        net.load_state_dict(torch.load(ck, map_location=dev))
        net.eval()
        print(f'载入已保存的定版代理 {ck.name}(不重训,保证与缓存算例的 θ 一致)', flush=True)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
        A = torch.tensor(X[tr], device=dev)
        B = torch.tensor(((Yf[tr] - ym) / ys).reshape(-1, NCH, NT).astype(np.float32), device=dev)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        for ep in range(a.epochs):
            pm = torch.randperm(len(A), device=dev)
            for i in range(0, len(A), 64):
                b = pm[i:i + 64]
                opt.zero_grad()
                F.mse_loss(net(A[b]), B[b]).backward()
                opt.step()
            sch.step()
        net.eval()
        torch.save(net.state_dict(), ck)
        print(f'代理训练完 {(time.time() - t0) / 60:.1f} min  → 权重存 {ck.name}', flush=True)

    def predict(THq):
        o = []
        with torch.no_grad():
            for i in range(0, len(THq), 256):
                z = torch.tensor(((THq[i:i + 256] - xm) / xs).astype(np.float32), device=dev)
                p = net(z).cpu().numpy().reshape(len(z), -1) * ys + ym
                o.append(p.reshape(len(z), nw, NPH, NT)[:, :, 0, :].sum(1))
        return np.concatenate(o)

    def obj_of(curve, which):
        m = MS if which == 'short' else 1.0
        return (curve * WT * m).sum(-1)
    ct = obj_of(Yf[te].reshape(-1, nw, NPH, NT)[:, :, 0, :].sum(1), 'long')
    cp = obj_of(predict(TH[te]), 'long')
    acc_test = float(np.mean(np.abs(cp - ct) / ct))
    (THc, Yc) = load_dir(CONFIRM)
    Yc = Yc[:, live]
    conf_true = {w: obj_of(Yc[:, :, 0, :].sum(1), w) for w in ('long', 'short')}
    Pc = predict(THc)
    conf_pred = {w: obj_of(Pc, w) for w in ('long', 'short')}
    acc_conf = float(np.mean(np.abs(conf_pred['long'] - conf_true['long']) / conf_true['long']))
    print(f'代理封存 test(400) 场级误差 {acc_test:.3%}   独立确认集({len(THc)}) {acc_conf:.3%}\n', flush=True)
    ymT = torch.tensor(ym, device=dev, dtype=torch.float32)
    ysT = torch.tensor(ys, device=dev, dtype=torch.float32)
    xmT = torch.tensor(xm, device=dev, dtype=torch.float32)
    xsT = torch.tensor(xs, device=dev, dtype=torch.float32)
    BWt = torch.tensor(BASE_W, device=dev, dtype=torch.float32)
    wT = torch.tensor(WT, device=dev, dtype=torch.float32)
    msT = torch.tensor(MS, device=dev, dtype=torch.float32)
    sig = float(TH.std())

    def curve_t(thf):
        o = net(((thf - xmT) / xsT)[None])[0].reshape(-1) * ysT + ymT
        return o.reshape(nw, NPH, NT)[:, 0, :].sum(0)

    def solve(mask, seed):
        g = torch.Generator(device='cpu').manual_seed(seed)
        z = (torch.randn(N_STAGE * len(INJ), generator=g) * 0.5).to(dev).requires_grad_(True)
        o = torch.optim.Adam([z], lr=a.lr)
        (best, bth) = (-1e+30, None)
        for _ in range(a.steps):
            o.zero_grad()
            th = a.trust_r * sig * torch.tanh(z)
            frac = torch.softmax(th.reshape(N_STAGE, len(INJ)) * np.log(10.0), 1)
            th_eff = torch.log10(frac * BWt.sum() / BWt[None] + 1e-12).reshape(-1)
            v = (curve_t(th_eff) * wT * mask).sum()
            (-v).backward()
            o.step()
            with torch.no_grad():
                if float(v) > best:
                    (best, bth) = (float(v), th_eff.detach().cpu().numpy().copy())
        return (bth, best)

    def pick_distinct(cands, k):
        cs = sorted(cands, key=lambda c: -c[1])
        for dmin in (0.6, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02, 0.0):
            sel = []
            for (th, v) in cs:
                if all((np.linalg.norm(th - s[0]) >= dmin for s in sel)):
                    sel.append((th, v))
                if len(sel) == k:
                    break
            if len(sel) == k:
                return (sel, dmin)
        return (cs[:k], 0.0)
    CAND = {}
    for (name, mask) in (('long', torch.ones(NT, device=dev)), ('short', msT)):
        raw = [solve(mask, 3000 + 17 * s) for s in range(a.n_start)]
        (sel, dmin) = pick_distinct(raw, a.n_cand)
        D = np.array([[np.linalg.norm(p[0] - q[0]) for q in sel] for p in sel])
        off = D[~np.eye(len(sel), dtype=bool)]
        CAND[name] = {'sel': sel, 'dmin': dmin, 'pair_l2': {'min': float(off.min()), 'median': float(np.median(off)), 'max': float(off.max())}}
        print(f'{name}: {a.n_start} 起点 → 去重阈值 L2≥{dmin:.2f} → {len(sel)} 个候选; 两两 L2 min/中位/max = {off.min():.3f}/{np.median(off):.3f}/{off.max():.3f}; 代理目标跨度 {sel[-1][1] / sel[0][1] - 1:+.2%}', flush=True)
    if a.dry_run:
        import hashlib
        for nm in ('long', 'short'):
            arr = np.array([c[0] for c in CAND[nm]['sel']], np.float64)
            print(f'  {nm} 候选校验和 {hashlib.sha1(arr.tobytes()).hexdigest()[:16]} sum={arr.sum():.10f}')
        print('dry-run: 候选已生成，退出')
        return 0
    base_key = 'ms_base'
    if not (SIM / f'{base_key}.npz').exists():
        FD._run_sim(base_key, np.zeros((N_STAGE, len(INJ)), np.float32), _A(a.threads))
    mb = FA._metrics(base_key)
    W0 = mb['water']
    print(f"\n基准 {base_key}: 全期 {mb['oil']:,.0f}  前3年 {mb['short']:,.0f}  实测总注水 {W0:,.0f}\n", flush=True)
    lock = __import__('threading').Lock()
    done = [0]
    total = 2 * a.n_cand
    CACHE = {}
    for f in sorted(SIM.glob('rf*_c*.npz')):
        try:
            CACHE[f.stem] = np.load(f)['theta'].astype(np.float64).ravel()
        except Exception:
            pass
    print(f'已有算例索引 {len(CACHE)} 条(按 θ 内容复用)', flush=True)

    def run_or_reuse(want, name, i):
        with lock:
            for (k, v) in CACHE.items():
                if v.shape == want.shape and np.abs(v - want).max() <= 1e-06:
                    return (k, True)
            j = 0
            while f'rf{a.tag}_{name}_{i}_c{j}' in CACHE:
                j += 1
            key = f'rf{a.tag}_{name}_{i}_c{j}'
            CACHE[key] = want.copy()
        FD._run_sim(key, want.astype(np.float32).reshape(N_STAGE, len(INJ)), _A(a.threads))
        return (key, False)

    def calibrate(name, i, th):
        (pts, it) = ([], 0)

        def ev(s):
            nonlocal it
            s = float(s)
            it += 1
            want = np.asarray(th + s, np.float32).astype(np.float64).ravel()
            (key, hit) = run_or_reuse(want, name, i)
            m = FA._metrics(key)
            d = m['water'] / W0 - 1
            pts.append((s, d, m, key, hit))
            return d
        d0 = ev(0.0)
        if abs(d0) > a.water_tol:
            d1 = ev(np.clip(-np.log10(1 + d0), -0.5, 0.5))
            g = 1
            while d1 * d0 > 0 and abs(d1) > a.water_tol and (it < a.calib_iter):
                (s_lo, s_hi) = (pts[0][0], pts[-1][0])
                step = (s_hi - s_lo) * 1.6 ** g
                g += 1
                d1 = ev(np.clip(s_hi + step, -0.8, 0.8))
            if abs(d1) > a.water_tol:
                neg = max((p for p in pts if p[1] < 0), key=lambda p: p[1], default=None)
                pos = min((p for p in pts if p[1] > 0), key=lambda p: p[1], default=None)
                if neg is not None and pos is not None:
                    (lo, flo, hi, fhi) = (neg[0], neg[1], pos[0], pos[1])
                    side = 0
                    while it < a.calib_iter and abs(hi - lo) > 1e-07:
                        sm = hi - fhi * (hi - lo) / (fhi - flo)
                        sm = float(np.clip(sm, min(lo, hi), max(lo, hi)))
                        fm = ev(sm)
                        if abs(fm) <= a.water_tol:
                            break
                        if fm < 0:
                            (lo, flo) = (sm, fm)
                            if side == -1:
                                fhi /= 2.0
                            side = -1
                        else:
                            (hi, fhi) = (sm, fm)
                            if side == 1:
                                flo /= 2.0
                            side = 1
        b = min(pts, key=lambda t: abs(t[1]))
        gap = None
        if abs(b[1]) > a.water_tol:
            sp = sorted(pts, key=lambda t: t[0])
            for (u, v) in zip(sp, sp[1:]):
                if u[1] < 0 < v[1] and (gap is None or v[0] - u[0] < gap['ds']):
                    gap = {'s_lo': float(u[0]), 's_hi': float(v[0]), 'ds': float(v[0] - u[0]), 'dev_lo': float(u[1]), 'dev_hi': float(v[1]), 'jump_pp': float((v[1] - u[1]) * 100)}
        with lock:
            done[0] += 1
            print(f"  [{done[0]}/{total}] {name}#{i}: {len(pts)} 次评估({sum((1 for p in pts if not p[4]))} 新跑), s={b[0]:+.5f}, 注水偏离 {b[1]:+.4%} {('OK' if abs(b[1]) <= a.water_tol else '⚠超容差')}, 全期 {b[2]['oil']:,.0f} 前3年 {b[2]['short']:,.0f}  ({(time.time() - t0) / 60:.0f}min)", flush=True)
        return {'i': i, 'theta_raw': th.tolist(), 's': float(b[0]), 'calib_trace': [[float(p[0]), float(p[1]), p[3], bool(p[4])] for p in pts], 'theta_cal': (th + b[0]).tolist(), 'sim_key': b[3], 'n_eval': len(pts), 'n_sim': int(sum((1 for p in pts if not p[4]))), 'water_dev': float(b[1]), 'water_ok': bool(abs(b[1]) <= a.water_tol), 'infeasibility_evidence': gap, **b[2]}
    jobs = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for name in ('long', 'short'):
            for (i, (th, v)) in enumerate(CAND[name]['sel']):
                jobs.append((name, ex.submit(calibrate, name, i, th)))
        RES = {'long': [], 'short': []}
        for (name, f) in jobs:
            RES[name].append(f.result())
    for name in RES:
        RES[name].sort(key=lambda d: d['i'])
    REPORT = {}
    for name in ('long', 'short'):
        rows = RES[name]
        th_raw = np.array([r['theta_raw'] for r in rows])
        th_cal = np.array([r['theta_cal'] for r in rows])
        surr_raw = obj_of(predict(th_raw), name)
        surr_cal = obj_of(predict(th_cal), name)
        true = np.array([r['oil'] if name == 'long' else r['short'] for r in rows])
        base_true = mb['oil'] if name == 'long' else mb['short']
        base_surr = float(obj_of(predict(np.zeros((1, 24), np.float32)), name)[0])
        for (r, sr, sc) in zip(rows, surr_raw, surr_cal):
            r['surr_relerr'] = float(abs(sc - (r['oil'] if name == 'long' else r['short'])) / (r['oil'] if name == 'long' else r['short']))
            r['surr_raw'] = float(sr)
            r['surr_cal'] = float(sc)
            r['surr_uplift'] = float(sc / base_surr - 1)
            r['true_uplift'] = float((r['oil'] if name == 'long' else r['short']) / base_true - 1)
        relerr = np.abs(surr_cal - true) / true
        (dx, dy) = ([], [])
        for r in rows:
            tp = [(FA._metrics(k)['water'], FA._metrics(k)['oil'] if name == 'long' else FA._metrics(k)['short']) for (_, _, k, _) in r['calib_trace']]
            for u in range(len(tp)):
                for v in range(u + 1, len(tp)):
                    lw = np.log(tp[v][0] / tp[u][0])
                    lo = np.log(tp[v][1] / tp[u][1])
                    if 1e-06 < abs(lw) < 0.02:
                        dx.append(lw)
                        dy.append(lo)
        (dx, dy) = (np.array(dx), np.array(dy))
        elas = float(dx @ dy / (dx @ dx)) if len(dx) > 2 else float('nan')
        R_spread_pct = float((true.max() / true.min() - 1) * 100)
        m_cal = rank_metrics(surr_cal, true)
        m_cal['uncertainty'] = rho_uncertainty(surr_cal, true)
        m_raw = rank_metrics(surr_raw, true)
        su = np.array([r['surr_uplift'] for r in rows])
        tu = np.array([r['true_uplift'] for r in rows])
        (pick, best) = (m_cal['surr_pick_idx'], m_cal['true_best_idx'])
        npv_keys = ('npv0', 'npv8', 'npv15')
        regret_musd = {k: float((rows[best][k] - rows[pick][k]) / 1000000.0) for k in npv_keys}
        gain_best = true[best] - base_true
        gain_pick = true[pick] - base_true
        REPORT[name] = {'n_candidates': len(rows), 'l2_dedup_threshold': CAND[name]['dmin'], 'pairwise_l2': CAND[name]['pair_l2'], 'all_water_within_tol': bool(all((r['water_ok'] for r in rows))), 'worst_water_dev': float(max((abs(r['water_dev']) for r in rows))), 'n_opm_evals': int(sum((r['n_eval'] for r in rows))), 'n_new_sims_this_run': int(sum((r['n_sim'] for r in rows))), 'true_spread_pct': float((true.max() / true.min() - 1) * 100), 'surr_spread_pct': float((surr_cal.max() / surr_cal.min() - 1) * 100), 'rank_fidelity_on_simulated_theta': m_cal, 'rank_fidelity_on_optimizer_raw_theta': m_raw, 'top1_regret': {'surrogate_pick': f'{name}#{pick}', 'true_best': f'{name}#{best}', 'pct_of_objective': m_cal['top1_regret_pct'], 'musd': regret_musd, 'gain_capture_ratio': float(gain_pick / gain_best) if gain_best != 0 else None, 'note': 'regret = 真最优 − 代理选中，二者均为 OPM Flow 实测；NPV 用 EIA 真实历史 Brent'}, 'signal_to_error_ratio': float(R_spread_pct / (relerr.mean() * 100)), 'point_accuracy_at_candidates': {'relerr_mean': float(relerr.mean()), 'relerr_max': float(relerr.max()), 'note': '同一代理在随机 θ 上的误差见 surrogate.confirm500_relerr_field'}, 'water_to_objective_elasticity': {'value': elas, 'n_pairs': int(len(dx)), 'note': '同一候选内不同缩放因子的点做差分回归 d ln(目标)/d ln(实测注水)', 'max_residual_water_dev': float(max((abs(r['water_dev']) for r in rows))), 'objective_shift_bound_pct': float(abs(elas) * max((abs(r['water_dev']) for r in rows)) * 100)}, 'optimizers_curse': {'err_pp': [float((r['surr_uplift'] - r['true_uplift']) * 100) for r in rows], 'overestimation_factor_per_candidate': [float(r['surr_uplift'] / r['true_uplift']) for r in rows], 'of_median': float(np.median([r['surr_uplift'] / r['true_uplift'] for r in rows])), 'of_at_surrogate_pick': float(rows[pick]['surr_uplift'] / rows[pick]['true_uplift']), 'err_pp_at_surrogate_pick': float((rows[pick]['surr_uplift'] - rows[pick]['true_uplift']) * 100), 'err_rank_of_surrogate_pick': int(np.where(np.argsort(-(su - tu)) == pick)[0][0]) + 1, 'corr_surr_vs_err': float(stats.pearsonr(su, su - tu).statistic), 'note': 'err_rank_of_surrogate_pick=1 表示代理选中的正是它自己高估最狠的候选'}, 'bias_level': bias_fit(surr_cal, true), 'bias_uplift': bias_fit(su, tu), 'candidates': rows}
        so = np.sort(true)[::-1]
        REPORT[name]['winner_margin_check'] = {'winner': f"{name}#{rows[int(np.argmax(true))]['i']}", 'margin_over_runner_up_pct': float((so[0] / so[1] - 1) * 100), 'water_bias_bound_pct': float(abs(elas) * abs(rows[int(np.argmax(true))]['water_dev']) * 100), 'safety_factor': float((so[0] / so[1] - 1) * 100 / max(abs(elas) * abs(rows[int(np.argmax(true))]['water_dev']) * 100, 1e-12)), 'note': 'safety_factor >> 1 表示冠军不是靠多注水赢的'}
        ok = [j for (j, r) in enumerate(rows) if r['water_ok']]
        REPORT[name]['strict_subset'] = {'n': len(ok), 'excluded': [f"{name}#{rows[j]['i']}" for j in range(len(rows)) if j not in ok], 'excluded_reason': '标量缩放无法把实测注水量校进 ±0.05%：dev(s) 是阶梯函数，容差落在台阶之间（见 infeasibility_evidence）', **rank_metrics(surr_cal[ok], true[ok]), 'bias_uplift': bias_fit(su[ok], tu[ok])} if 2 < len(ok) < len(rows) else {'n': len(ok), 'note': '全部候选均在容差内，与主分析相同'}
        (ct_, cp_) = (conf_true[name], conf_pred[name])
        ord_t = np.argsort(-ct_)[:a.n_cand]
        REPORT[name]['control_random_theta_full'] = {**rank_metrics(cp_, ct_), 'true_spread_pct': float((ct_.max() / ct_.min() - 1) * 100), 'bias_level': bias_fit(cp_, ct_), 'bias_uplift': bias_fit(cp_ / base_surr - 1, ct_ / base_true - 1), 'relerr_mean': float(np.mean(np.abs(cp_ - ct_) / ct_)), 'signal_to_error_ratio': float((ct_.max() / ct_.min() - 1) * 100 / (np.mean(np.abs(cp_ - ct_) / ct_) * 100))}
        REPORT[name]['control_random_theta_true_topk'] = {**rank_metrics(cp_[ord_t], ct_[ord_t]), 'true_spread_pct': float((ct_[ord_t].max() / ct_[ord_t].min() - 1) * 100), 'bias_level': bias_fit(cp_[ord_t], ct_[ord_t]), 'note': '按真值选前 k 再看排名 —— 条件在 y 上，相关系数被结构性压低，只作跨度参照'}
        ord_p = np.argsort(-cp_)[:a.n_cand]
        mp = rank_metrics(cp_[ord_p], ct_[ord_p])
        REPORT[name]['control_random_theta_surr_topk'] = {**mp, 'true_spread_pct': float((ct_[ord_p].max() / ct_[ord_p].min() - 1) * 100), 'surr_spread_pct': float((cp_[ord_p].max() / cp_[ord_p].min() - 1) * 100), 'bias_level': bias_fit(cp_[ord_p], ct_[ord_p]), 'bias_uplift': bias_fit(cp_[ord_p] / base_surr - 1, ct_[ord_p] / base_true - 1), 'relerr_mean': float(np.mean(np.abs(cp_[ord_p] - ct_[ord_p]) / ct_[ord_p])), 'signal_to_error_ratio': float((ct_[ord_p].max() / ct_[ord_p].min() - 1) * 100 / (np.mean(np.abs(cp_[ord_p] - ct_[ord_p]) / ct_[ord_p]) * 100)), 'note': '随机 θ 池里按代理选前 k、由模拟器裁判 —— 与优化区域用的是同一决策程序，唯一差别是候选来自随机采样而非优化器'}
        REPORT[name]['baseline'] = {'true': base_true, 'surrogate': base_surr, 'sim_key': base_key}
    OUT_J = {'what': '代理模型在优化区域的排名保真度（裁判=OPM Flow）', 'surrogate': {'arch': 'PosNet + fourier pos-emb (n_bands=16)', 'n_train': len(tr), 'n_pin': a.n_pin, 'epochs': a.epochs, 'heldout_test_relerr_field': acc_test, 'confirm500_relerr_field': acc_conf, 'confirm_set': str(CONFIRM), 'confirm_n': int(len(THc))}, 'optimizer': {'n_start': a.n_start, 'n_cand': a.n_cand, 'steps': a.steps, 'trust_r': a.trust_r, 'lr': a.lr}, 'water_constraint': {'tol': a.water_tol, 'basis': 'inj_actual (实测)', 'baseline_water': W0}, 'baseline_metrics': mb, 'results': REPORT, 'caveats': ['候选来自同一个代理模型的多起点优化，故它们的 θ 集中在代理认为好的区域；这正是本检验的目的（量化优化区域的排名失真），不是抽样偏差。', 'top-1 regret 的 NPV 只算收入侧（EIA Brent 年均价），口径同 fc_npv.py。', '地质维度失效、F-4H 假旋钮、预测段真实工况是 WAG —— 三条主线限制在此同样成立。', 'n=10 的 Spearman/Kendall 统计功效有限，p 值一并给出，不做过强推断。']}
    (OUT / 'rank.json').write_text(json.dumps(OUT_J, indent=1, ensure_ascii=False))
    print(f"\n{'=' * 84}\n=== 🔴 排名保真度 ===\n")
    for name in ('long', 'short'):
        R = REPORT[name]
        m = R['rank_fidelity_on_simulated_theta']
        c = R['control_random_theta_full']
        c10 = R['control_random_theta_surr_topk']
        print(f"[{name}] n={R['n_candidates']}  真值跨度 {R['true_spread_pct']:.2f}%  注水最差偏离 {R['worst_water_dev']:.4%}  OPM 评估 {R['n_opm_evals']} 次")
        u = m['uncertainty']
        print(f"  信噪比 SNR(真值跨度/代理点误差) = {R['signal_to_error_ratio']:.2f}   [对照: 随机θ前{c10['n']} {c10['signal_to_error_ratio']:.2f}, 全500 {c['signal_to_error_ratio']:.1f}]")
        print(f"  优化区域   ρ={m['spearman_rho']:+.3f}(置换p={u['perm_p_two_sided']:.4f}, 95%CI[{u['boot_ci95'][0]:+.2f},{u['boot_ci95'][1]:+.2f}])  τ={m['kendall_tau']:+.3f}  hit={m['hit_rate']}")
        print(f"  随机θ(500) ρ={c['spearman_rho']:+.3f}  τ={c['kendall_tau']:+.3f}  跨度 {c['true_spread_pct']:.1f}%")
        print(f"  随机θ代理前{c10['n']}  ρ={c10['spearman_rho']:+.3f}  τ={c10['kendall_tau']:+.3f}  hit={c10['hit_rate']}  regret {c10['top1_regret_pct']:.3f}%  跨度 {c10['true_spread_pct']:.2f}%")
        pa = R['point_accuracy_at_candidates']
        print(f"  点精度: 候选上 {pa['relerr_mean']:.3%}(max {pa['relerr_max']:.3%}) vs 随机θ确认集 {acc_conf:.3%}")
        t1 = R['top1_regret']
        print(f"  top-1 regret {t1['pct_of_objective']:.3f}%  NPV0 {t1['musd']['npv0']:+.2f}M$  NPV8 {t1['musd']['npv8']:+.2f}M$  NPV15 {t1['musd']['npv15']:+.2f}M$")
        print(f"  代理第1名真实排第 {m['surr_pick_true_rank']}/{m['n']}；真第1名代理排第 {m['true_best_surr_rank']}/{m['n']}")
        ss = R['strict_subset']
        if ss.get('spearman_rho') is not None:
            print(f"  严格子集(n={ss['n']}, 剔 {ss['excluded']}) ρ={ss['spearman_rho']:+.3f} τ={ss['kendall_tau']:+.3f} regret {ss['top1_regret_pct']:.3f}%")
        el = R['water_to_objective_elasticity']
        print(f"  注水弹性 {el['value']:.2f} × 残余偏差 {el['max_residual_water_dev']:.4%} → 目标最多被挪动 {el['objective_shift_bound_pct']:.4f}%")
        oc = R['optimizers_curse']
        print(f"  优化者诅咒: 代理选中的候选正是它高估第 {oc['err_rank_of_surrogate_pick']}/{m['n']} 狠的(高估 {oc['err_pp_at_surrogate_pick']:+.2f}pp = {oc['of_at_surrogate_pick']:.1f}×; 全体中位 {oc['of_median']:.2f}×);corr(代理预测, 高估量)={oc['corr_surr_vs_err']:+.3f}")
        bu = R['bias_uplift']
        print(f"  增益偏差: true = {bu['intercept']:+.5f} + {bu['slope']:.4f}·surr  (R²={bu['r2']:.3f});  过原点斜率 {bu['slope_through_origin']:.4f} → 代理高估 {bu['overestimation_factor']:.2f}×\n")
    print(f'已写入 {OUT}/rank.json   总耗时 {(time.time() - t0) / 60:.0f} min')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
