from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import norne_bulk as NB
OUT = __import__('release_paths').data_path('norne_fc')
HIST = OUT / '_history'
OFFICIAL = __import__('release_paths').ROOT / 'vendor' / 'opm-tests' / 'norne'
PRED_DATES = [(1, 'JAN', 2007), (1, 'JUL', 2007), (1, 'JAN', 2008), (1, 'JUL', 2008), (1, 'JAN', 2009), (1, 'JUL', 2009), (1, 'JAN', 2010), (1, 'JAN', 2011), (1, 'JAN', 2012), (1, 'JAN', 2013), (1, 'JAN', 2014), (1, 'JAN', 2015), (1, 'JAN', 2016), (1, 'JAN', 2017), (1, 'JAN', 2018), (1, 'JAN', 2019), (1, 'JAN', 2020)]
STAGE_AT = [2007, 2008, 2010, 2012, 2014, 2017]
INJ_W = ['F-1H', 'F-2H', 'F-3H', 'F-4H']
BASE_WINJ = {'F-1H': 8554.2, 'F-2H': 3676.3, 'F-3H': 6865.3, 'F-4H': 2071.0}
PROD_W = ['B-1BH', 'B-2H', 'B-4DH', 'D-1CH', 'D-2H', 'D-3BH', 'E-1H', 'E-3CH']
PROD_LIM = '5000.0    1*  4.0E6    8000.0    1*    60.0'
BASE_QLIQ = 8000.0
N_CTRL_V2 = len(INJ_W) + len(PROD_W)
N_STAGE = len(STAGE_AT)

def write_pred_sch(theta_stage: np.ndarray, path: Path) -> None:
    L = ['-- 预测段:由 forecast_gen.py 生成', '-- 生产井控制参数抄 Equinor 官方 4A deck', '-- (opm-tests/norne/INCLUDE/PRED_WINJ_MIN_THP_4A_STDW.SCH, ODbL)', '']
    L += ['WCONPROD', '-- 井  状态  控制  Qo  Qw  Qg  Qliq  Qresv  bhp(最低)']
    for w in PROD_W:
        L.append(f" '{w}'  'OPEN'  'GRUP'  {PROD_LIM}  /")
    L += ['/', '']
    L += ['WECON', " 'B-*'   3*   20000  1*  'WELL' /", " 'D-*'   3*   20000  1*  'WELL' /", " 'E-*'   3*   20000  1*  'WELL' /", '/', '']
    n_inj = len(INJ_W)
    v2 = theta_stage.shape[1] > n_inj
    si = 0
    for (d, mon, yr) in PRED_DATES:
        if si < N_STAGE - 1 and yr >= STAGE_AT[si + 1]:
            si += 1
        m = 10.0 ** theta_stage[si]
        L += ['WCONINJE']
        for (k, w) in enumerate(INJ_W):
            L.append(f" '{w}'  WATER  'OPEN'  'RATE'  {BASE_WINJ[w] * m[k]:.1f}  1*  500.0  2*  /")
        L += ['/', '']
        if v2:
            L += ['WCONPROD', '-- 井  状态  控制  Qo  Qw  Qg  Qliq  Qresv  bhp(最低)']
            for (k, w) in enumerate(PROD_W):
                q = BASE_QLIQ * m[n_inj + k]
                L.append(f" '{w}'  'OPEN'  'GRUP'  5000.0    1*  4.0E6    {q:.1f}    1*    60.0  /")
            L += ['/', '']
        L += ['DATES', f" {d} '{mon}' {yr} /", '/', '']
    path.write_text('\n'.join(L), encoding='utf-8')

def build_restart_deck(work: Path, theta_stage: np.ndarray, hist_rel: str, step: int) -> None:
    src = NB.BASE
    shutil.copytree(src, work, ignore=shutil.ignore_patterns('out', '*.UNRST', '*.UNSMRY', '*.EGRID', '*.INIT', '*.PRT', '*.DBG', '*.ESMRY'))
    deck = work / NB.DECK
    L = deck.read_text(errors='ignore').split('\n')
    i_sol = next((k for (k, l) in enumerate(L) if l.strip().upper() == 'SOLUTION'))
    i_sum = next((k for (k, l) in enumerate(L) if l.strip().upper() == 'SUMMARY'))
    sol = ['SOLUTION', '', 'RESTART', f"  '{hist_rel}' {step} /", '', 'RPTRST', 'BASIC=2 /', '']
    KEEP_KW = {'THPRES', 'INCLUDE'}
    (keep, cur) = ([], None)
    for l in L[i_sol + 1:i_sum]:
        u = l.split('--')[0].strip().upper()
        if u and u.split()[0].isalpha() and u.replace('=', ' ').split()[0].isupper():
            cur = u.split()[0]
        if cur in KEEP_KW:
            keep.append(l)
    L = L[:i_sol] + sol + keep + [''] + L[i_sum:]
    i_sch = next((k for (k, l) in enumerate(L) if l.strip().upper() == 'SCHEDULE'))
    L.insert(i_sch + 1, 'SKIPREST')
    write_pred_sch(theta_stage, work / 'INCLUDE' / 'PRED.SCH')
    hs = work / NB.SCH
    HL = hs.read_text(errors='ignore').split('\n')
    j_end = next((k for (k, l) in enumerate(HL) if l.split('--')[0].strip().upper() == 'END'), None)
    ins = ['', '-- 预测段(forecast_gen.py 生成)', 'INCLUDE', " './INCLUDE/PRED.SCH' /", '']
    HL = HL[:j_end] + ins + HL[j_end:] if j_end is not None else HL + ins
    hs.write_text('\n'.join(HL))
    deck.write_text('\n'.join(L))
    (work / 'out').mkdir(exist_ok=True)
(FC_T0, FC_T1, FC_N) = (3312.0, 8091.0, 40)
FC_GRID = np.linspace(FC_T0, FC_T1, FC_N)
FC_SNAP_N = 8
FC_SNAP_DAYS = np.linspace(FC_T0, FC_T1, FC_SNAP_N)

def harvest_fc(work: Path) -> dict | None:
    from resdata.resfile import ResdataFile
    from resdata.summary import Summary
    smry = work / 'out' / 'NORNE_ATW2013.UNSMRY'
    unrst = work / 'out' / 'NORNE_ATW2013.UNRST'
    if not smry.exists():
        return None
    s = Summary(str(smry))
    days = np.asarray(s.numpy_vector('TIME'), float)
    if days.max() < FC_T1 - 1.0:
        return None

    def series(key):
        try:
            v = np.asarray(s.numpy_vector(key), float)
        except Exception:
            return np.full(FC_N, np.nan)
        mk = np.isfinite(days) & np.isfinite(v)
        return np.interp(FC_GRID, days[mk], v[mk]) if mk.sum() >= 2 else np.full(FC_N, np.nan)
    cols = [series(f'{k}:{w}') for w in NB.PRODUCERS for k in NB.OBS_KEYS]
    inj = np.stack([series(f'WWIR:{w}') for w in INJ_W])
    fld = np.stack([series(k) for k in ('FOPT', 'FWIT', 'FWPT', 'FPR')])
    fields = None
    if unrst.exists():
        try:
            f = ResdataFile(str(unrst))
            n = f.num_named_kw(NB.FIELDS[0])
            rd = np.asarray([(d - f.dates[0]).total_seconds() / 86400.0 for d in f.dates], float)
            if len(rd) == n and n >= 2:
                j = np.clip(np.searchsorted(rd, FC_SNAP_DAYS - rd[0], side='right') - 1, 0, n - 2)
                w_ = np.clip((FC_SNAP_DAYS - rd[0] - rd[j]) / np.where(rd[j + 1] - rd[j] > 0, rd[j + 1] - rd[j], 1.0), 0.0, 1.0)[:, None]
                need = sorted(set(j.tolist()) | set((j + 1).tolist()))
                pos = {g: r for (r, g) in enumerate(need)}
                lo = np.asarray([pos[g] for g in j])
                hi = np.asarray([pos[g] for g in j + 1])
                st = []
                for k in NB.FIELDS:
                    cube = np.stack([np.asarray(f[k][i], np.float32) for i in need])
                    st.append((cube[lo] * (1.0 - w_) + cube[hi] * w_).astype(np.float32))
                fields = np.stack(st, axis=1)
        except Exception:
            fields = None
    out = {'obs': np.concatenate(cols).astype(np.float32), 'inj_actual': inj.astype(np.float32), 'field_cum': fld.astype(np.float32), 'grid_days': FC_GRID.copy(), 'snap_days': FC_SNAP_DAYS.copy(), 'has_fields': np.int32(fields is not None), 'sim_days_end': np.float64(days.max())}
    if fields is not None:
        out['fields'] = fields
    return out

def run_history(threads: int) -> int:
    HIST.mkdir(parents=True, exist_ok=True)
    if (HIST / 'out' / 'NORNE_ATW2013.UNRST').exists():
        print('历史档已存在, 跳过')
        return 0
    w = HIST / 'case'
    shutil.rmtree(w, ignore_errors=True)
    shutil.copytree(NB.BASE, w, ignore=shutil.ignore_patterns('out', '*.UNRST', '*.UNSMRY', '*.EGRID', '*.INIT', '*.PRT'))
    (w / 'out').mkdir(exist_ok=True)
    t0 = time.time()
    cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{w}:/data', '-w', '/data', NB.IMAGE, 'flow', NB.DECK, '--output-dir=/data/out', f'--threads-per-process={threads}']
    rc = subprocess.run(cmd, capture_output=True, timeout=7200).returncode
    print(f'历史算例 rc={rc}  耗时 {(time.time() - t0) / 60:.1f} min')
    if rc != 0:
        return 1
    shutil.move(str(w / 'out'), str(HIST / 'out'))
    shutil.rmtree(w, ignore_errors=True)
    from resdata.resfile import ResdataFile
    f = ResdataFile(str(HIST / 'out' / 'NORNE_ATW2013.UNRST'))
    n = f.num_named_kw('PRESSURE')
    seq = [int(np.asarray(f['SEQNUM'][i])[0]) for i in range(f.num_named_kw('SEQNUM'))]
    (HIST / 'meta.json').write_text(json.dumps({'n_restart': n, 'last_step': seq[-1], 'seqnum_last': seq[-1], 'T0': str(f.dates[-1].date())}, indent=1))
    print(f'重启档 {n} 个状态, 末帧报告步 SEQNUM={seq[-1]}, T0 = {f.dates[-1].date()}')
    return 0

def main() -> int:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['history', 'run', 'probe'])
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--jobs', type=int, default=24)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--schema', choices=['v1', 'v2'], default='v1', help='v1=只调 4 口注水井(θ 24 维);v2=加 8 口生产井采液上限(θ 72 维)')
    ap.add_argument('--out-suffix', default='', help='输出目录后缀,避免覆盖 v1 数据')
    ap.add_argument('--seed', type=int, default=20260821)
    args = ap.parse_args()
    if args.out_suffix:
        OUT = OUT.parent / (OUT.name + args.out_suffix)
        OUT.mkdir(parents=True, exist_ok=True)
        print(f'🔴 输出目录 {OUT}(不覆盖 v1 数据)')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'history':
        return run_history(args.threads)
    meta_f = HIST / 'meta.json'
    if not meta_f.exists():
        print('先跑 history')
        return 1
    meta = json.load(open(meta_f))
    hist_rel = str((HIST / 'out' / 'NORNE_ATW2013').resolve())
    n = 2 if args.cmd == 'probe' else args.n
    rng = np.random.default_rng(args.seed)
    n_ctrl = N_CTRL_V2 if args.schema == 'v2' else len(INJ_W)
    TH = rng.normal(0, args.sigma, (n, N_STAGE, n_ctrl)).astype(np.float32)
    TH[0] = 0.0
    (OUT / 'shards').mkdir(exist_ok=True)
    np.save(OUT / 'theta_all.npy', TH)
    print(f'预测段 {n} 个样本  schema={args.schema}  θ {TH.shape} = {N_STAGE} 段 × {n_ctrl} 个控制' + (f'({len(INJ_W)} 注水 + {len(PROD_W)} 生产)' if args.schema == 'v2' else '(注水井)'))
    print(f"T0 = {meta['T0']}  重启步 {meta['last_step']}  → 2020-01-01")

    def one(j):
        sh = OUT / 'shards' / f'{j:05d}.npz'
        if sh.exists():
            return {'j': j, 'status': 'cached'}
        w = OUT / '_work' / f'w{j:05d}'
        shutil.rmtree(w, ignore_errors=True)
        t0 = time.time()
        try:
            build_restart_deck(w, TH[j], hist_rel, meta['last_step'])
            cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{w}:/data', '-v', f"{HIST / 'out'}:{HIST / 'out'}:ro", '-w', '/data', NB.IMAGE, 'flow', NB.DECK, '--output-dir=/data/out', f'--threads-per-process={args.threads}', '--parsing-strictness=low']
            with open(w / 'run.log', 'w') as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200).returncode
            if rc != 0:
                return {'j': j, 'status': 'failed', 'rc': rc, 'tail': (w / 'run.log').read_text(errors='ignore')[-400:]}
            g = harvest_fc(w)
            if g is None:
                return {'j': j, 'status': 'failed', 'why': 'harvest'}
            np.savez_compressed(sh, theta=TH[j], rc=np.int32(rc), **g)
            return {'j': j, 'status': 'ok', 'seconds': round(time.time() - t0, 1)}
        except Exception as e:
            return {'j': j, 'status': 'error', 'err': f'{type(e).__name__}: {e}'[:300]}
        finally:
            shutil.rmtree(w, ignore_errors=True)
    (t0, bad) = (time.time(), [])
    with ThreadPoolExecutor(max_workers=min(args.jobs, n)) as ex:
        futs = [ex.submit(one, j) for j in range(n)]
        for (i, f) in enumerate(as_completed(futs), 1):
            r = f.result()
            if r['status'] not in ('ok', 'cached'):
                bad.append(r)
            if i % 25 == 0 or i == n or args.cmd == 'probe':
                print(f'  [{i}/{n}] 失败 {len(bad)}  已用 {(time.time() - t0) / 3600:.2f}h', flush=True)
    if bad:
        (OUT / 'failures.json').write_text(json.dumps(bad[:20], indent=1, ensure_ascii=False))
        print(f'失败 {len(bad)}, 前几条:')
        for r in bad[:2]:
            print('   ', json.dumps(r, ensure_ascii=False)[:400])
    return 0 if not bad else 1
if __name__ == '__main__':
    raise SystemExit(main())
