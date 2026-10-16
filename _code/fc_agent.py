from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import forecast_gen as FG
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_agent'
KB = __import__('release_paths').ROOT / 'knowledge'
SPEC = {'theta': {'shape': [FG.N_STAGE, len(FG.INJ_W)], 'meaning': '6 个决策时点 × 4 口注水井的 log10 注水率乘子', 'stages': FG.STAGE_AT, 'wells': FG.INJ_W, 'base_rate_sm3_per_day': {w: FG.BASE_WINJ[w] for w in FG.INJ_W}, 'range': [-0.9, 0.9], 'note': '实际注水率 = 基准率 × 10^theta。θ=0 即维持基准。'}, 'hard_constraint': {'total_water': '全期**实测**总注水量必须与基准一致（容差 1%）。🔴 注意：目标注水率守恒 ≠ 实测守恒，因为 F-4H 被 BHP 500 bar 截断。'}, 'objective': '必须二选一并写明：(a) 最大化前 3 年累计产油；(b) 最大化 13 年累计产油。', 'required_fields': ['theta', 'objective', 'rationale', 'expected_tradeoff'], 'termination': '四个字段全部填满且 theta 通过 shape/range 校验，即为完成。预算上限（OPM 调用次数）只是安全网，不是终止判据。'}
_L2_FACTS = '以下是本油藏在**本项目前期诊断实验**中实测到的事实。\n这些是对油藏与井况的观察，**不包含任何已优化的方案或其产量结果**。\n\n1. **F-4H 是"假旋钮"**：它的目标注水率被井底压力 500 bar 上限截断，\n   实测注水率只达到目标的 27.6%。因此调高它的目标率基本不改变实际注入量；\n   反过来，把水"从它那里挪走"也挪不动。\n   → 推论：优化时给 F-4H 的目标率变化，对实测总注水量的影响远小于其它三口井。\n\n2. **井间连通性（流动诊断实测的注入井→生产井分配因子，越大表示注入水越多流向该井）**：\n   F-1H → B-1BH、B-3H、E-2AH 方向为主\n   F-2H → E-1H 方向为主\n   F-3H → D-3BH、D-2H 方向为主\n   F-4H → 对全部生产井的分配因子接近 0（与第 1 条一致）\n\n3. **22 口生产井中有 11 口在预测段全程产量为 0**（已关停）。\n   实际有产量的是：B-1BH, B-2H, B-4DH, D-1CH, D-2H, D-3BH, E-1H, E-2AH, E-3CH, K-3H, B-3H\n\n4. **目标注水量守恒 ≠ 实测注水量守恒**（由第 1 条导致）。\n   任何"同样多的水换分法"的方案，必须用**实测**注水量校验，\n   且通常需要一个整体缩放因子来把实测量校回基准。\n\n5. **响应面在最优解附近存在陡峭方向**：θ 加 1e-3 量级的扰动，\n   累计产油可能变化 0.65%（与注水量变化无关）。\n   → 推论：不要把 θ 推到极端值；同等产量下应优先选局部平坦的方案。\n\n6. **模拟器是逐位确定性的**：同一套 θ 重复跑结果完全一致，无随机噪声。\n'
BRIEF = '你是一名油藏工程师，需要为 Norne 油田设计一套注水策略。\n\n【油藏与时间】\nNorne 油田（挪威海），网格 46×112×22，44,431 个活动网格。\n历史段 1997-11 → 2006-12 已经跑完，你的决策从 2006-12 开始，到 2020-01 结束（13 年）。\n\n【你能调什么】\n6 个决策时点：{stages}\n4 口注水井及其当前注水率（sm³/天）：\n{wells}\n对每个时点、每口井，你给一个 log10 乘子 θ，实际注水率 = 基准率 × 10^θ。\nθ 的合理范围是 [-0.9, +0.9]（即 0.126 倍 ~ 7.9 倍）。\n\n【硬约束】\n全期**实测**总注水量必须与基准一致（±1%）。\n注意：这是**实测**量，不是目标量——如果某口井的目标率被井底压力上限截断，\n它实际注不进去，你把水"从它那里挪走"是挪不动的。\n\n【8 口生产井】\n{prods}\n它们按组控制自由生产，采液上限 8000 sm³/天，最低井底压力 60 bar，\n含水率超过阈值会按经济极限自动关井。\n\n【你的任务】\n选定一个目标（二选一）：\n  (a) 最大化**前 3 年**累计产油（急需现金流）\n  (b) 最大化**全 13 年**累计产油（追求采收率）\n然后给出 6×4 的 θ 矩阵。\n\n【输出格式】严格的 JSON，不要有别的内容：\n{{"objective": "short" 或 "long",\n  "theta": [[6 行 × 4 列的数字]],\n  "rationale": "你的推理（中文，200 字以内）",\n  "expected_tradeoff": "你预期这个方案在另一个目标上会损失多少（中文，一句话）"}}\n'

def brief(arm: str) -> str:
    wells = '\n'.join((f'  {w}: {FG.BASE_WINJ[w]:.1f}' for w in FG.INJ_W))
    prods = '  ' + ', '.join(FG.PROD_W)
    txt = BRIEF.format(stages=FG.STAGE_AT, wells=wells, prods=prods)
    if arm == 'G0':
        return txt
    if arm in ('G1', 'G2'):
        files = sorted(KB.glob('*.md'))
        kb = '\n\n'.join((f'### {f.name}\n{f.read_text()[:4000]}' for f in files))
        txt = f"【领域知识库】\n{kb}\n\n{'=' * 60}\n\n{txt}"
    if arm == 'G2':
        txt = f"【本油藏的诊断事实（本项目实测）】\n{_L2_FACTS}\n\n{'=' * 60}\n\n{txt}"
    if arm == 'G3':
        import random
        files = sorted(KB.glob('*.md'))
        chunks = []
        for f in files:
            ws = f.read_text().split()
            random.Random(42).shuffle(ws)
            chunks.append(' '.join(ws[:600]))
        txt = '【领域知识库】\n' + '\n\n'.join(chunks) + f"\n\n{'=' * 60}\n\n{txt}"
    return txt

def _metrics(key):
    import datetime as dt
    import norne_bulk as NB
    from fc_npv import BRENT, BBL_PER_M3, T_START
    d = np.load(OUT.parent / 'fc_decide' / 'sim' / f'{key}.npz')
    ob = d['obs']
    n = len(NB.PRODUCERS)
    NT = FG.FC_N
    DAYS = FG.FC_GRID
    oil = np.stack([ob[i * 3 * NT:(i * 3 + 1) * NT] for i in range(n)]).sum(0)
    w = np.gradient(DAYS).astype(float)
    w[0] *= 0.5
    w[-1] *= 0.5
    yrs = np.array([(T_START + dt.timedelta(days=float(x))).year for x in DAYS])
    price = np.array([BRENT[y] for y in yrs])
    t = (DAYS - DAYS[0]) / 365.25
    rev = oil * w * BBL_PER_M3 * price
    ms = DAYS - DAYS[0] <= 3.0 * 365.25
    return {'oil': float((oil * w).sum()), 'short': float((oil * w * ms).sum()), 'water': float(d['field_cum'][1][-1] - d['field_cum'][1][0]), 'npv8': float((rev / 1.08 ** t).sum()), 'npv0': float(rev.sum()), 'npv15': float((rev / 1.15 ** t).sum())}

def evaluate(arm, theta, args):
    import fc_decide as FD

    class _A:
        threads = args.threads
    W0 = None
    base_key = 'ms_base'
    if not (OUT.parent / 'fc_decide' / 'sim' / f'{base_key}.npz').exists():
        FD._run_sim(base_key, np.zeros((FG.N_STAGE, len(FG.INJ_W)), np.float32), _A())
    W0 = _metrics(base_key)['water']
    th = np.asarray(theta, dtype=np.float64).reshape(FG.N_STAGE, len(FG.INJ_W))
    (pts, calls) = ([], 0)
    for it in range(args.calib_iter):
        ls = 0.0 if it == 0 else -np.log10(1 + pts[0][1]) if it == 1 else float(np.clip(pts[-1][0] - pts[-1][1] * (pts[-1][0] - pts[-2][0]) / (pts[-1][1] - pts[-2][1] + 1e-12), -0.5, 0.5))
        key = f'agent_{arm}_c{it}'
        if not (OUT.parent / 'fc_decide' / 'sim' / f'{key}.npz').exists():
            FD._run_sim(key, (th + ls).astype(np.float32), _A())
            calls += 1
        m = _metrics(key)
        dev = m['water'] / W0 - 1
        pts.append((ls, dev, m))
        print(f'    校正 iter{it}: s=10^{ls:+.4f}  实测注水偏离 {dev:+.2%}', flush=True)
        if abs(dev) <= args.water_tol:
            break
    best = min(pts, key=lambda t: abs(t[1]))
    return ({**best[2], 'water_dev': best[1], 's': best[0]}, calls)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['spec', 'brief', 'eval', 'report'])
    ap.add_argument('--theta-json', help='包含 {objective, theta, ...} 的 JSON 文件')
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--calib-iter', type=int, default=3)
    ap.add_argument('--water-tol', type=float, default=0.01)
    ap.add_argument('--arm', default='G0', choices=['G0', 'G1', 'G2', 'G3'])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'spec':
        print(json.dumps(SPEC, indent=1, ensure_ascii=False))
        return 0
    if a.cmd == 'eval':
        j = json.loads(Path(a.theta_json).read_text())
        print(f"=== {a.arm}  目标={j.get('objective')} ===")
        (m, calls) = evaluate(a.arm, j['theta'], a)
        rec = {'arm': a.arm, 'objective': j.get('objective'), 'rationale': j.get('rationale'), 'opm_calls': calls, **m}
        (OUT / f'eval_{a.arm}.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        print(f"  累计产油 {m['oil']:,.0f}  前3年 {m['short']:,.0f}  NPV@8% {m['npv8'] / 1000000.0:,.1f}M$  注水偏离 {m['water_dev']:+.2%}  OPM 调用 {calls} 次")
        return 0
    if a.cmd == 'report':
        base = _metrics('ms_base')
        rows = []
        for f in sorted(OUT.glob('eval_*.json')):
            rows.append(json.loads(f.read_text()))
        if not rows:
            print('还没有评测结果')
            return 1
        print(f"  {'臂':6s}{'目标':7s}{'累计产油':>13s}{'vs基准':>9s}{'前3年':>13s}{'vs基准':>9s}{'NPV@8%':>12s}{'vs基准':>11s}{'注水偏离':>10s}{'OPM':>6s}")
        print(f"  {'基准':6s}{'-':7s}{base['oil']:>13,.0f}{0:>+8.2f}%{base['short']:>13,.0f}{0:>+8.2f}%{base['npv8'] / 1000000.0:>10,.1f}M${0:>+9.1f}M${0:>+9.2f}%{'-':>6s}")
        for r in sorted(rows, key=lambda r: -r['npv8']):
            print(f"  {r['arm']:6s}{r['objective'] or '-':7s}{r['oil']:>13,.0f}{(r['oil'] / base['oil'] - 1) * 100:>+8.2f}%{r['short']:>13,.0f}{(r['short'] / base['short'] - 1) * 100:>+8.2f}%{r['npv8'] / 1000000.0:>10,.1f}M${(r['npv8'] - base['npv8']) / 1000000.0:>+9.1f}M${r['water_dev'] * 100:>+9.2f}%{r['opm_calls']:>6d}")
        return 0
    t = brief(a.arm)
    f = OUT / f'brief_{a.arm}.txt'
    f.write_text(t, encoding='utf-8')
    print(f'{a.arm} 提示词 {len(t):,} 字符 → {f}')
    print(f"  知识库部分 {len(t) - len(brief('G0')):,} 字符")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
