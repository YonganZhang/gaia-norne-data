from __future__ import annotations
import argparse, hashlib, json, random, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import forecast_gen as FG
import norne_bulk as NB
from fc_team import ROLES, Flag, Message, Provenance, OUT, sha256_arr, sha256_file, validate
from fc_team_proxy import Proxy
import fc_decide as FD
from fc_water_econ import econ
ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / '_pipelines' / 'fc_decide' / 'sim'
(INJ, N_STAGE) = (FG.INJ_W, FG.N_STAGE)
BASE_W = np.array([FG.BASE_WINJ[w] for w in INJ], float)
DAYS = np.asarray(FG.FC_GRID, float)
TW = np.gradient(DAYS)
TW[0] *= 0.5
TW[-1] *= 0.5
_yr = (DAYS - DAYS[0]) / 365.25 + FG.STAGE_AT[0]
SI = np.clip(np.searchsorted(np.asarray(FG.STAGE_AT, float), _yr, 'right') - 1, 0, N_STAGE - 1)
_CAPF = ROOT / '_pipelines' / 'fc_team' / 'cap_refit.npy'
CAP = np.load(_CAPF)[:, None] * np.ones((1, len(DAYS))) if _CAPF.exists() else np.repeat(BASE_W[:, None] * 2.5, len(DAYS), 1)
BBL = 6.2898
PRICE_AVG = 74.0
from fc_truth import baseline as _truth_baseline
BASE_WATER = _truth_baseline()['winj']
(C_INJ, C_PROD) = (2.0, 1.0)
SHARED_BG = ''
_WMODEL = ROOT / '_pipelines' / 'fc_team' / 'water_model.npy'
_WB = np.load(_WMODEL) if _WMODEL.exists() else None

def inj_water(th: np.ndarray) -> float:
    t = np.asarray(th, float).ravel()
    if _WB is not None:
        tc = np.clip(t, -1.0, 1.0)
        dev = float(np.r_[tc, tc ** 2] @ _WB) / 100.0
        return BASE_WATER * (1.0 + dev)
    tgt = BASE_W[:, None] * 10.0 ** t.reshape(N_STAGE, len(INJ))[SI, :].T
    return float((np.minimum(tgt, CAP) * TW).sum())

def hard_check(th: np.ndarray, w_max_dev: float) -> list[Flag]:
    (f, thm) = ([], th.reshape(N_STAGE, len(INJ)))
    dev = inj_water(th) / BASE_WATER - 1.0
    if abs(dev) > w_max_dev * 2.5:
        f.append(Flag('high', 'water_budget_extreme', f'预测注水偏离 {dev * 100:+.1f}%，远超 ±{w_max_dev * 100:.0f}% (预测模型 MAE 1.5pp、高 θ 区最大 29pp，故只在极端偏离时否决)'))
    elif abs(dev) > w_max_dev:
        f.append(Flag('medium', 'water_budget_predicted', f'预测注水偏离 {dev * 100:+.1f}% 超出 ±{w_max_dev * 100:.0f}%，但预测在高 θ 区不可靠，交由模拟器实测裁定'))
    if np.abs(thm).max() > 1.0:
        f.append(Flag('high', 'theta_out_of_box', f'θ 绝对值最大 {np.abs(thm).max():.2f} > 1.0，超出代理训练分布'))
    tgt = BASE_W[:, None] * 10.0 ** thm[SI, :].T
    over = (tgt > CAP * 1.05).mean()
    if over > 0.5:
        f.append(Flag('medium', 'capacity_saturated', f'{over * 100:.0f}% 的时段目标率超过实测注入能力上限，该部分 θ 无效'))
    return f

class _A:

    def __init__(self, threads):
        self.threads = threads

def _theta_key(th) -> str:
    return 'sol-' + hashlib.sha256(np.ascontiguousarray(np.asarray(th, np.float32).ravel()).tobytes()).hexdigest()[:8]

def adjudicate(th: np.ndarray, key: str, threads: int, c_inj: float=1.0, c_prod: float=0.5, w_max: float=0.25) -> dict | None:
    _fp = hashlib.sha256(np.ascontiguousarray(np.asarray(th, np.float32).ravel()).tobytes()).hexdigest()[:10]
    key = f'{key}_{_fp}'
    f = SIM / f'{key}.npz'
    if not f.exists():
        try:
            FD._run_sim(key, th.astype(np.float32).reshape(N_STAGE, len(INJ)), _A(threads))
        except (Exception, SystemExit) as e:
            print(f'    🔴 模拟失败 {key}: {str(e)[:90]}')
            return None
    if not f.exists():
        return None
    m = econ(f, c_inj=c_inj, c_prod=c_prod)
    m['simulation_file'] = f.name
    d = np.load(f)
    m['water_inj'] = float(d['field_cum'][1][-1] - d['field_cum'][1][0])
    m['water_dev'] = m['water_inj'] / BASE_WATER - 1.0
    m['budget_ok'] = abs(m['water_dev']) <= w_max
    return m
PERSPECTIVE = {'reservoir_engineer': '你盯的是地层能量与波及:压力剖面、含水演化、注采是否平衡。压力和波及支持增注你就说增，支持收敛你就说收敛 —— 方向从数据读，不预设。', 'connectivity_analyst': "你盯的是井间通道:哪口注水井的水真正到达了哪口生产井，哪些是无效或负贡献。你对'水打给谁'负责，不对'总量多少'预设立场。", 'production_surveillance': '你盯的是单井动态:谁在产油、谁只在产水、谁已水淹、谁还有余量。你按每口井的实际油水表现给增减建议，不搞一刀切。', 'economics_analyst': '你盯的是扣除注水与采出水成本后的 NPV。它支持增注你就支持增注，支持收缩你就支持收缩 —— 你对钱负责，不对某个方向负责。', 'geomechanics_expert': "你盯的是注入强度的物理边界:压力、压实、诱导裂缝风险。你的产出是**可行区间的上限**，不是'该增还是该减'的主张。🔴 你没有本油藏的任何实测应力数据，只有同类砂岩的类比经验。因此:给上限时必须说明它是类比估计、不确定性有多大;**不要因为不确定就把上限压低** —— 那是把无知当成保守，会让团队错过实测数据支持的方向。若实测数据显示某方向有效而你无证据反对，请明确说'本角色无证据反对'。", 'seismic_4d_analyst': "你盯的是水驱前缘的不确定性:哪些方案对前缘假设最敏感、判断错了代价最大。你的产出是**风险敞口排序**，不是方向建议。🔴 你只有文献标题与 DOI，没有本油藏的前缘实测。因此只排序风险、不否决方向;无证据时明确说'本角色无证据反对'。", 'constraint_auditor': '你盯的是可执行性:目标注水率能否兑现、哪口井的 theta 是空转、方案是否越界。你的产出是**可行/不可行的判定**，与增减方向无关。'}

def _econ_limit_table() -> str:
    b = np.load(SIM / 'baseline.npz')
    ob = b['obs'].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    (o, w) = (ob[:, 0, :].sum(0), ob[:, 1, :].sum(0))
    wc = w / np.maximum(o + w, 1e-09)
    import datetime as _dt
    from fc_water_econ import BRENT as _BR, T_START as _TS
    yrs = np.array([(_TS + _dt.timedelta(days=float(x))).year for x in DAYS])
    px = np.array([_BR[min(max(y, min(_BR)), max(_BR))] for y in yrs])
    lim = px / (px + 1.0)
    t = (DAYS - DAYS[0]) / 365.25
    rows = '\n'.join((f'    第{t[i]:4.1f}年  实际含水 {100 * wc[i]:5.1f}%   经济极限 {100 * lim[i]:5.1f}%   余量 {100 * (lim[i] - wc[i]):+5.1f} pp' for i in range(0, len(DAYS), 5)))
    return '\n  基准情形的含水率 vs 经济极限含水率(fw* = p_oil/(p_oil+c_prod)):\n' + rows + '\n  → 余量越小，该时段每多采一方液体的净收益越低。'
_TILT_BINS = ROOT / '_pipelines' / 'fc_team' / 'tilt_bins.json'

def tilt_prior() -> str:
    if not _TILT_BINS.exists():
        return ''
    d = json.loads(_TILT_BINS.read_text())
    rows = '\n'.join((f'    tilt [{lo:+.2f}, {hi:+.2f})   n={n:4d}   平均 ΔNPV {mu:+7.1f} M$   p90 {p90:+7.1f} M$' for (lo, hi, n, mu, p90) in d['bins']))
    return f"\n\n【注水时程倾斜度(tilt)与收益的统计关系 —— {d['n']} 个算例】\n  样本范围:{d.get('source', '')}\n  定义:tilt = (第1、2段的平均 θ) − (第5、6段的平均 θ)，即「早期相对后期多注多少」。\n  实测 spearman(tilt, ΔNPV) = {d['spearman']:+.3f}。\n  分箱统计:\n" + rows + '\n  → 这是本问题**最主要的收益轴**。井间分配(哪口井打多少水)在控制 tilt 后\n     只额外解释约 12 个百分点。\n  🔴 本表给的是方向与量级，不含任何具体方案。你仍需自己决定\n     各段各井的具体数值、并满足注水总量硬约束。'

def domain_rules() -> str:
    return '\n\n【水驱井控的通用领域规则(来自油藏工程文献，非本案例的解)】\n  1. 波及效率随含水率上升而下降:在综合含水率仍低时注入的水，驱油效率最高;\n     含水率越过经济极限后，注入的水基本原样采出，只增加处理成本而不增油。\n  2. 存在长期采收率与短期现金流的冲突。折现率越高，越倾向前置产量;\n     折现率接近零时，最大化最终采收率才是最优。(Van Essen 2011)\n  3. Reactive control 是行业参照策略:生产井含水超过阈值即关停或减注上游注水。\n     在高含水油田它经常出人意料地强。(Jansen 2008)\n  4. 注水井受最低井底流压约束时，提高目标注水率**不会**提高实际注入量。\n     对这类井做速率调整是无效控制,应先判断哪些井的控制是真正有效的。\n  5. 井间连通性决定注水去向。注入到与高含水生产井强连通通道的水，\n     边际增油低;应优先支持仍有产油潜力的连通方向。\n  6. 在总注入量受约束时，价值来自**重新分配**而非增加总量:\n     把水从低效时段/井挪到高效时段/井。\n'

def _dyn(last: Path | None, role_id: str) -> str:
    if last is None:
        return '\n\n【实测反馈】尚无可读取的实测算例。'
    if not last.is_file():
        raise FileNotFoundError(f'实测反馈算例不存在: {last}')
    (b, c) = (np.load(SIM / 'baseline.npz'), np.load(last))
    (fb, fcc) = (b['field_cum'], c['field_cum'])
    ob = b['obs'].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    oc = c['obs'].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    d_oil = (fcc[0][-1] - fcc[0][0]) / (fb[0][-1] - fb[0][0]) - 1
    d_wat = (fcc[1][-1] - fcc[1][0]) / (fb[1][-1] - fb[1][0]) - 1
    head = f'\n\n【上一轮实测算例 {last.stem}(模拟器裁定，非代理预测)】\n  全期产油 {100 * d_oil:+.2f}% vs 基准，实测注水 {100 * d_wat:+.2f}% vs 基准'
    if role_id == 'reservoir_engineer':
        return head + f'\n  你的数据域:油藏压力 FPR 末值 {fcc[3][-1]:.1f} bar(基准 {fb[3][-1]:.1f})，中期 {fcc[3][len(DAYS) // 2]:.1f}(基准 {fb[3][len(DAYS) // 2]:.1f})。\n  → 这套注水实际把压力推到了哪里?与你上一轮的预期一致吗?'
    if role_id == 'production_surveillance':
        (wo, ww) = (oc[:, 0, :].sum(1), oc[:, 1, :].sum(1))
        (bo, bw) = (ob[:, 0, :].sum(1), ob[:, 1, :].sum(1))
        m = np.argsort(-bo)[:5]
        rows = '\n'.join((f'    {NB.PRODUCERS[i]:>7s} 油 {100 * (wo[i] / max(bo[i], 1e-09) - 1):+6.1f}%  水 {100 * (ww[i] / max(bw[i], 1e-09) - 1):+6.1f}%' for i in m))
        return head + '\n  你的数据域:主力生产井相对基准的变化\n' + rows + '\n  → 哪些井真的多产了油?哪些只是多产了水?'
    if role_id == 'constraint_auditor':
        (ia, ib) = (c['inj_actual'], b['inj_actual'])
        rows = '\n'.join((f'    {w:>6s} 实测峰值 {ia[k].max():8,.0f}(基准 {ib[k].max():8,.0f})' for (k, w) in enumerate(INJ)))
        return head + '\n  你的数据域:各注水井实测峰值\n' + rows + '\n  → 目标值兑现了吗?哪口井的 theta 又是空转?'
        return head + '\n  你的数据域:各注水井实测峰值\n' + rows + '\n  → 目标值兑现了吗?哪口井的 theta 又是空转?'
    if role_id == 'connectivity_analyst':
        wo = oc[:, 0, :].sum(1)
        bo = ob[:, 0, :].sum(1)
        m = np.argsort(-bo)[:5]
        rows = ', '.join((f'{NB.PRODUCERS[i]} {100 * (wo[i] / max(bo[i], 1e-09) - 1):+.1f}%' for i in m))
        return head + f'\n  你的数据域:主力生产井产油变化 {rows}' + '\n  → 实际响应的井，和你按 beta 预测的井是同一批吗?'
    return head + '\n  → 这个结果与你的判断是否一致?'

def slice_for(role_id: str, carto: dict, last: Path | None=None) -> tuple[str, Provenance]:
    b = np.load(SIM / 'baseline.npz')
    fc = b['field_cum']
    ob = b['obs'].reshape(len(NB.PRODUCERS), 3, len(DAYS))
    yrs = (DAYS - DAYS[0]) / 365.25
    if role_id == 'reservoir_engineer':
        fpr = fc[3]
        wc = ob[:, 1, :].sum(0) / np.maximum(ob[:, 0, :].sum(0) + ob[:, 1, :].sum(0), 1e-09)
        txt = '基准情形油藏压力 FPR(bar) 与综合含水率随时间:\n' + '\n'.join((f'  第{y:4.1f}年  FPR={p:7.1f}  含水={w * 100:5.1f}%' for (y, p, w) in zip(yrs[::5], fpr[::5], wc[::5])))
        return (txt + _dyn(last, role_id), Provenance('sim/baseline.npz', 'field_cum[FPR] + obs[WWPR,WOPR]', 'simulation', sha256_file(SIM / 'baseline.npz')))
    if role_id == 'connectivity_analyst':
        B = np.array(carto['payload']['beta'])
        prods = carto['payload']['producers']
        rows = [(p, B[:, j]) for (j, p) in enumerate(prods) if np.abs(B[:, j]).max() >= 0.05]
        txt = f"注水井→生产井 标准化偏回归系数(由 283 个已模拟算例反推，条件数 3.09 可归因):\n  {'生产井':>8s} " + ' '.join((f'{w:>8s}' for w in INJ)) + '\n' + '\n'.join((f'  {p:>8s} ' + ' '.join((f'{v:+8.3f}' for v in c)) for (p, c) in rows)) + f'\n各注水井总影响力(|β|行和): ' + ', '.join((f'{w}={np.abs(B[k]).sum():.2f}' for (k, w) in enumerate(INJ)))
        return (txt + _dyn(last, role_id), Provenance('_pipelines/fc_team/cartography.json', 'payload.beta', 'simulation', carto['provenance']['source_sha256']))
    if role_id == 'production_surveillance':
        oil = ob[:, 0, :].sum(1)
        wat = ob[:, 1, :].sum(1)
        wc = wat / np.maximum(oil + wat, 1e-09)
        order = np.argsort(-oil)
        txt = '基准情形各生产井(按累计产油排序，含水率高者优先考虑减注上游水):\n' + '\n'.join((f'  {NB.PRODUCERS[i]:>7s}  累计油={oil[i]:10,.0f}  含水={wc[i] * 100:5.1f}%' for i in order[:12] if oil[i] > 0))
        return (txt + _dyn(last, role_id), Provenance('sim/baseline.npz', 'obs[:, WOPR/WWPR, :]', 'simulation', sha256_file(SIM / 'baseline.npz')))
    if role_id == 'economics_analyst':
        import datetime as _dt
        from fc_water_econ import BRENT as _BR, T_START as _TS, BBL as _BBL
        (_CI, _CP) = (C_INJ, C_PROD)
        _yrs = sorted({(_TS + _dt.timedelta(days=float(x))).year for x in DAYS})
        _px = '\n'.join((f'    {y}  {_BR[y]:6.2f} USD/bbl' for y in _yrs if y in _BR))
        txt = '经济口径(市场与成本事实，不含任何方案评价结果):\n  预测期各年 EIA Brent 年均现货价:\n' + _px + f'\n  注水成本 {_CI:.1f} USD/bbl = {_CI * _BBL:.2f} USD/Sm3\n  采出水处理 {_CP:.1f} USD/bbl = {_CP * _BBL:.2f} USD/Sm3\n  折现率 8%/年;体积换算 1 Sm3 = {_BBL:.4f} bbl\n  目标 = 折现后的(油收入 - 注水成本 - 采出水处理成本)\n' + _econ_limit_table()
        return (txt + _dyn(last, role_id), Provenance('EIA-RBRTE + cost assumptions', 'annual Brent price series + unit costs', 'measurement', sha256_arr(np.array(sorted(_BR), dtype='U'))))
    if role_id == 'constraint_auditor':
        ia = b['inj_actual']
        txt = '目标注水率 vs 实测(基准情形峰值，Sm3/day):\n' + '\n'.join((f'  {w:>6s}  基准率={BASE_W[k]:8,.0f}  实测峰值={ia[k].max():8,.0f}  达成率={100 * ia[k].max() / BASE_W[k]:5.1f}%' for (k, w) in enumerate(INJ))) + '\n🔴 F-4H 受最低井底流压上限截断，目标守恒 ≠ 实测守恒。任何方案都要复核实测注水量。'
        return (txt + _dyn(last, role_id), Provenance('sim/baseline.npz', 'inj_actual', 'simulation', sha256_file(SIM / 'baseline.npz')))
    if role_id == 'geomechanics_expert':
        txt = '🔴 本油藏 deck 不含 GEOMECH/STRESS/YOUNGMOD/POISSON 关键字，无 Norne 实测岩石力学数据。你只能基于北海同类砂岩油藏的**类比**经验，给出注入压力上限与压实/破裂风险的**定性**边界。严禁声称掌握 Norne 的实测应力数据。'
        return (txt + _dyn(last, role_id), Provenance('north-sea-sandstone-analogue', 'qualitative-prior', 'literature', sha256_arr(np.array([role_id], dtype='U'))))
    if role_id == 'seismic_4d_analyst':
        txt = 'Norne 有丰富的 4D 时移地震文献(检索所得，示例):\n  doi:10.4043/19049-ms  Estimating 4D Velocity Changes and Contact Movement, Norne\n  doi:10.1190/segam2019-3216321.1  Time-lapse seismic inversion, Norne Field\n  doi:10.2523/iptc-10894-ms  Integrated Reservoir Management: Time-Lapse Acquisition\n据此给出水驱前缘推进与流体接触面移动的**定性**先验。你只有文献标题与 DOI，没有其中的数值，不得编造具体数字。'
        return (txt + _dyn(last, role_id), Provenance('crossref:norne-4d-seismic', 'titles+DOIs', 'literature', sha256_arr(np.array(['4d'], dtype='U'))))
    return ('(无专属数据，仅综合他人消息)' + _dyn(last, role_id), Provenance('internal', 'messages', 'reasoning', sha256_arr(np.array([role_id], dtype='U'))))
LLM_CALLS = {'n': 0, 'chars_in': 0}

def ask(prompt: str, model: str, tmo: int, retry: int=1) -> tuple[str, dict]:
    LLM_CALLS['n'] += 1
    LLM_CALLS['chars_in'] += len(prompt)
    r = None
    for k in range(4):
        r = subprocess.run(['claude', '-p', '--model', model, '--tools', '', '--setting-sources', 'project,local', '--strict-mcp-config'], input=prompt, cwd='/tmp', capture_output=True, text=True, timeout=tmo)
        if r.returncode == 0:
            break
        time.sleep(5 * 2 ** k + random.uniform(0, 3))
    if r.returncode != 0:
        raise RuntimeError(f'LLM rc={r.returncode} (重试 4 次仍失败): {r.stderr[:400]}')
    s = r.stdout.strip()
    if '```' in s:
        s = s.split('```')[1]
        s = s[s.find('{'):] if s.lstrip().startswith('json') else s
    try:
        return (r.stdout, json.loads(s[s.find('{'):s.rfind('}') + 1]))
    except json.JSONDecodeError:
        if retry <= 0:
            raise
        return ask(prompt + '\n\n🔴 上次输出不是合法 JSON。只输出一个 JSON 对象，字符串内不得出现未转义的引号或换行。', model, tmo, retry - 1)
FIXED = '你是一个油藏注水决策团队中的角色:**{role}**。\n职责:{duty}\n\n【固定背景 —— Norne 油田】\n北海砂岩油藏，46x112x22 网格，44431 活网格。\n决策变量 theta 是 {ns} 个时段 x {ni} 口注水井 = 24 个数，\n每个数是该井该时段注水率的 log10 乘子(0 表示保持基准，+0.3 约为 2 倍，-0.3 约为一半)。\n注水井 {inj}，基准率(Sm3/day) {basew}。\n时段起始年 {stages}。预测期约 13 年。\n【本次优化的经济口径】\n注水成本 {cinj} USD/bbl，采出水处理 {cprod} USD/bbl，油价用 2006-2019 真实 Brent 年均价。\n水不是免费的:多注水通常多产油，但目标是**扣除水成本后的 NPV**，不是产油量。\n最优注水总量该往哪个方向走，由你从下面的数据自己判断 —— 不要假设答案。\n\n【共享背景:全部角色一致(不区分角色)】\n{shared}\n\n【你的专业视角】\n{stance}\n🔴 你和其他角色读的是**不同的数据**，所以出现分歧是正常的、也是团队价值所在。\n   请如实报告你的数据支持什么，即使它与别人相反；\n   不要为了显得稳妥而向中庸靠拢，也不要为了制造分歧而故意唱反调。\n\n🔴 你只能基于下面给你的数据发言。数据支持到什么程度，confidence 就给到什么程度 ——\n   有硬数据支撑就大胆给高分，纯属类比或猜测就如实给低分。不要一律往低报。\n\n【你能读到的数据】\n{data}\n'
DYN = '\n【本轮动态信息 —— 第 {r} 轮】\n{hist}\n'
ASK_READER = '\n请给出你的专业判断。只输出 JSON:\n{"assessment": "<=120字的判断", "recommendation": "<=100字，具体说哪些井哪些时段该增/减注水",\n "confidence": 0.0~1.0, "concerns": ["<=3条风险"],\n "evidence": "<=400字。从你上面读到的数据里**逐字摘录**你认为总工必须亲眼看到的部分(表格照抄即可)。总工看不到你的原始数据，只能通过这个字段拿到证据。不要在这里写你的观点。"}\n'
ASK_CRITIQUE = '\n【第二轮 —— 你现在能看到其他角色的意见了】\n{peers}\n\n请以你的立场审视上面的分歧。只输出 JSON:\n{{"challenge": "<=120字，指名道姓反驳你最不同意的那条意见，说清理由>",\n  "revision": "<=100字，你自己的建议要不要改；不改就写\'维持\'并说明为什么其他人没说服你>",\n  "confidence": 0.0~1.0}}\n🔴 如果其他人的数据说服了你，就明说改变判断 —— 被证据说服不是让步。\n   如果没被说服，说清楚你的数据为什么更有说服力。\n'
ASK_CHIEF = '\n你是总工。\n\n{incumbent}\n【第零部分:模拟器实测(全物理裁定，非任何模型的预测)】\n{truth}\n\n【第一部分:各角色**原文转呈**的关键证据(未经他们概括，逐字转发)】\n{facts}\n\n【第二部分:各角色的判断与互相质疑】\n{msgs}\n\n🔴 三部分的优先级:第零部分是**唯一裁判给出的事实**，优先级最高;\n   第一部分是角色转呈的原始证据;第二部分是他们对证据的**解读**。\n   三者冲突时，一律以第零部分为准。\n   两者冲突时以证据为准;各角色都没提到的证据要点，你仍应自行采纳。\n   🔴 若第一部分为空，说明本臂没有配备领域角色，你只能依据任务描述与\n      模拟器反馈自行判断 —— 这是有意的对照设置，不是遗漏。\n\n🔴 各角色读的是**不同的数据**，出现分歧是正常的。你的职责不是取平均，\n   而是判断在当前证据下谁更有道理。\n\n🔴 **按证据强度裁决，不按人头。** 每条意见都标了 `依据类型`:\n     simulation / measurement = 本油藏的实测或模拟数据 —— 权重最高;\n     literature               = 同类油藏的类比文献，**不含本油藏任何实测** —— 仅作定性边界;\n     reasoning                = 无专属数据，只是综合他人意见 —— 不独立构成证据。\n   一条 literature 类的顾虑**不足以否决**一条 simulation 类的正向证据;\n   若二者冲突，应采纳实测方向并在方案里为该风险留出余量，而不是直接放弃该方向。\n   同理:多个角色都表达同一顾虑，若它们的依据类型都是 literature，\n   那仍然只是**一份**类比证据，不因人多而变强。\n   🔴 注意:角色按油藏工程学科分工(压力/连通性/生产/经济/岩石力学/地震/约束)，\n      这个分工**不等于**目标函数的敏感方向。若多数角色的关注点都落在\n      对 NPV 影响很小的维度上，你不应因为"人多"就跟随;\n      请以第零部分的实测数字判断哪个方向真正改变了 NPV。\n\n请提出 {n} 个互不相同的候选方案。只输出 JSON:\n{{"candidates": [{{"theta": [[{ni}个数] x {ns}段], "rationale": "<=80字，注明采纳了哪些角色的意见"}}],\n  "confidence": 0.0~1.0}}\n每个 theta 是 **{ns} 行 × {ni} 列**的二维数组:**第一维是时段(共 {ns} 段)，第二维是注水井(共 {ni} 口)**。\n即 theta[段][井]。不要写成 {ni} 行 × {ns} 列 —— 两者元素个数相同，写反了方案含义完全不同。\n元素取值建议在 -0.8 ~ +0.8 之间。\n\n🔴 **硬约束(违反者会被程序直接否决，不进模拟器)**:\n   1. 全期实测注水总量必须落在基准的 ±{wdev:.0f}% 以内。\n   2. 每个 theta 分量的绝对值不得超过 1.0。\n\n🔴 **不要因为怕超预算就把幅度压小。** 预算是否超标由**模拟器实测**判定，\n   不由任何近似公式判定。预测超标只给一条警告，方案照样送模拟器;\n   只有预测极端偏离(超过容差 2.5 倍)才会被直接否决。\n   幅度不足是常见的失败原因。\n'
ASK_AUDIT = '\n你是红队审计员。下面是总工的候选方案和代理模型的快速评估:\n{cands}\n\n各角色的意见与置信度:\n{msgs}\n\n请挑毛病。只输出 JSON:\n{{"flags": [{{"severity": "high|medium|low", "code": "<短标识>", "detail": "<=80字",\n             "candidate": <候选编号，若是全局问题则填 -1>}}],\n  "verdict": "pass|revise", "confidence": 0.0~1.0}}\n🔴 被你标为 high 的候选会被**直接否决**，不进入模拟器，也不会被采纳。请慎用 high。\n🔴 下面已列出程序化硬约束检查的结果；那些是数字判定，你不必重复，请找它们之外的问题。\n'

def run(a) -> int:
    if a.tag.startswith(('_F2', '_G2')):
        raise ValueError('F2/G2 属于反馈路径修复前批次；请为修复后的实验使用新批次标签')
    if (OUT / f'loop{a.tag}.json').exists():
        raise FileExistsError(f'实验结果已存在，拒绝覆盖: loop{a.tag}.json')
    OUT.mkdir(parents=True, exist_ok=True)
    strict = getattr(a, 'strict_run', False)
    LLM_CALLS.update(n=0, chars_in=0)
    carto = json.loads((OUT / 'cartography.json').read_text())
    proxy = Proxy(a.device)
    active = [r for r in ROLES if r['id'] not in a.drop and r['id'] != 'chief_engineer']
    print(f'=== Gaia 团队环路 ===\n角色 {len(active)} 个' + (f'(消融剔除 {a.drop})' if a.drop else '') + f'  模型={a.model}\n')
    (hist, all_msgs, clean_streak, rounds) = ('首轮，尚无历史反馈。', [], 0, [])
    last_case: Path | None = None
    global C_INJ, C_PROD, SHARED_BG
    (C_INJ, C_PROD) = (a.c_inj, a.c_prod)
    SHARED_BG = (tilt_prior() + domain_rules()).strip()
    base_m = econ(SIM / 'baseline.npz', c_inj=a.c_inj, c_prod=a.c_prod)
    prev_truth = ''
    incumbent: dict | None = None
    incumbent_th: np.ndarray | None = None
    eligible = {}
    disqualified = set()
    stop_reason = 'max_rounds'
    for rnd in range(1, a.max_rounds + 1):
        print(f'--- 第 {rnd} 轮 ---')

        def one(role):
            (data, prov) = slice_for(role['id'], carto, last_case)
            p = FIXED.format(role=role['id'], duty=role['duty'], ns=N_STAGE, ni=len(INJ), inj=INJ, basew=BASE_W.round(0).tolist(), stages=FG.STAGE_AT, data=data, cinj=a.c_inj, cprod=a.c_prod, stance=PERSPECTIVE.get(role['id'], '按你能读到的数据说话。'), shared=SHARED_BG) + DYN.format(r=rnd, hist=hist) + ASK_READER
            if getattr(a, 'trace_inputs', False):
                trace = OUT / f'trace{a.tag}'
                trace.mkdir(parents=True, exist_ok=True)
                (trace / f"round{rnd}_{role['id']}.json").write_text(json.dumps({'round': rnd, 'role': role['id'], 'prompt': p, 'feedback_file': None if last_case is None else last_case.name, 'feedback_sha256': None if last_case is None else sha256_file(last_case)}, ensure_ascii=False, indent=1))
            try:
                (_, ans) = ask(p, a.model, a.timeout)
            except Exception as e:
                return (role['id'], None, f'{type(e).__name__}: {e}')
            m = Message(role['id'], 'reading', ans, float(np.clip(ans.get('confidence', 0.5), 0, 1)), prov)
            return (role['id'], m, validate(m))
        res = []
        if active:
            with ThreadPoolExecutor(max_workers=min(4, len(active))) as ex:
                res = list(ex.map(one, active))
        msgs = []
        for (rid, m, err) in res:
            if m is None:
                print(f'  🔴 {rid:<24s} 调用失败: {err}')
                continue
            ok = '✅' if not err else f'🔴 {err}'
            print(f"  {rid:<24s} conf={m.confidence:.2f} {ok}  {m.payload.get('assessment', '')[:52]}")
            if not err:
                msgs.append(m)
                all_msgs.append(m)
        if not msgs and active:
            print('  🔴 本轮全部角色失败，中止')
            return 1
        if strict and len(msgs) != len(active):
            print('  🔴 角色未到齐，运行无效')
            return 2
        crit = {}
        if a.critique and len(msgs) > 1:
            peers = '\n'.join((f"[{m.agent_id}] {m.payload.get('assessment', '')} 建议: {m.payload.get('recommendation', '')}" for m in msgs))

            def one_crit(m):
                role = next((r for r in ROLES if r['id'] == m.agent_id))
                (data, _) = slice_for(m.agent_id, carto, last_case)
                others = '\n'.join((l for l in peers.split('\n') if not l.startswith(f'[{m.agent_id}]')))
                p = FIXED.format(role=role['id'], duty=role['duty'], ns=N_STAGE, ni=len(INJ), inj=INJ, basew=BASE_W.round(0).tolist(), stages=FG.STAGE_AT, data=data, cinj=a.c_inj, cprod=a.c_prod, stance=PERSPECTIVE.get(role['id'], ''), shared=SHARED_BG) + ASK_CRITIQUE.format(peers=others)
                try:
                    (_, ans) = ask(p, a.model, a.timeout)
                    return (m.agent_id, ans)
                except Exception:
                    return (m.agent_id, None)
            with ThreadPoolExecutor(max_workers=min(4, len(msgs))) as ex:
                for (rid, ans) in ex.map(one_crit, msgs):
                    if ans:
                        crit[rid] = ans
            if strict and len(crit) != len(msgs):
                print('  🔴 质疑轮角色未到齐，运行无效')
                return 2
            n_rev = sum((1 for v in crit.values() if '维持' not in str(v.get('revision', ''))))
            print(f'  质疑轮: {len(crit)}/{len(msgs)} 回应，{n_rev} 个改了主意')
            for (rid, v) in list(crit.items())[:3]:
                print(f"    {rid:<24s} 反驳: {str(v.get('challenge', ''))[:56]}")
        evidence_txt = '\n\n'.join((f"[{m.agent_id} 转呈]\n{m.payload.get('evidence', '')}" for m in msgs if str(m.payload.get('evidence', '')).strip())) or '(本臂未配备领域角色，无证据转呈)'
        _RANK = {'simulation': 0, 'measurement': 0, 'deck': 0, 'literature': 1, 'reasoning': 2}
        msgs = sorted(msgs, key=lambda m: (_RANK.get(m.provenance.source_type, 2), -m.confidence))
        msg_txt = '(本臂无分析师角色，总工独立决策)' if not active else '\n'.join((f"[{m.agent_id} | conf={m.confidence:.2f} | 依据类型={m.provenance.source_type}] {m.payload.get('assessment', '')} 建议: {m.payload.get('recommendation', '')}" + (f"\n    ↳ 质疑他人: {crit[m.agent_id].get('challenge', '')}\n    ↳ 二轮结论: {crit[m.agent_id].get('revision', '')}" if m.agent_id in crit else '') for m in msgs))
        try:
            inc_txt = ''
            if incumbent is not None and incumbent_th is not None:
                _t = incumbent_th.reshape(N_STAGE, len(INJ))
                inc_txt = f"【迄今已被模拟器验证的最优方案(现任冠军)】\n  ΔNPV@8% = {(incumbent['npv8'] - base_m['npv8']) / 1000000.0:+.1f} M$   实测注水 {incumbent['water_dev'] * 100:+.2f}%\n  它的 theta[段][井]:\n" + '\n'.join(('    ' + ' '.join((f'{_t[_s, _j]:+6.2f}' for _j in range(len(INJ)))) for _s in range(N_STAGE))) + '\n  🔴 这是**本轮之前已经真跑出来的成绩**，不是猜测。\n     你的新候选里必须至少有一个是在它基础上的小幅改进(而不是推倒重来);\n     若某个角色的意见会让你明显偏离它，请先说明为什么值得冒这个险。\n     角色每轮都会提出新顾虑，但**已被验证的成绩不应被未验证的顾虑推翻**。\n\n'
            truth_txt = prev_truth + '\n' + hist if prev_truth else hist
            (_, syn) = ask(ASK_CHIEF.format(msgs=msg_txt, facts=evidence_txt, incumbent=inc_txt, truth=truth_txt, n=a.n_cand, ni=len(INJ), ns=N_STAGE, wdev=a.w_max_dev * 100), a.model, a.timeout)
        except Exception as e:
            print(f'  🔴 总工失败: {e}')
            return 1
        cands = []
        n_transposed = 0
        for c in syn.get('candidates', []):
            try:
                th = np.asarray(c['theta'], float)
            except Exception:
                continue
            if th.shape == (len(INJ), N_STAGE):
                th = th.T
                n_transposed += 1
            elif th.shape != (N_STAGE, len(INJ)):
                if th.size != N_STAGE * len(INJ):
                    continue
                th = th.reshape(N_STAGE, len(INJ))
            cands.append((th, c.get('rationale', '')))
        if n_transposed:
            print(f'  ⚠ 自动转置 {n_transposed} 个候选(LLM 返回了 井×时段 而非 时段×井)')
        if not cands:
            print('  🔴 总工未给出合法候选')
            return 1
        if strict and len(cands) > a.n_cand:
            print('  🔴 候选数超过固定预算，运行无效')
            return 2
        inc_idx = None
        if incumbent_th is not None:
            inc_idx = len(cands)
            cands.append((incumbent_th.reshape(N_STAGE, len(INJ)), '[现任冠军·精英保留]'))
        TH = np.stack([t.ravel() for (t, _) in cands]).astype(np.float32)
        candidate_keys = [_theta_key(t) for t in TH]
        t0 = time.time()
        ev = proxy.evaluate(TH)
        dt = time.time() - t0
        hard = {i: hard_check(TH[i], a.w_max_dev) for i in range(len(TH))}
        vetoed = {i for (i, fs) in hard.items() if any((x.severity == 'high' for x in fs))}
        vetoed |= {i for (i, key) in enumerate(candidate_keys) if key in disqualified}
        wi = np.array([inj_water(t) for t in TH])
        ev['score'] = ev['oil'] * BBL * PRICE_AVG - wi * BBL * a.c_inj - ev['water_prod'] * BBL * a.c_prod
        ev['inj_water'] = wi
        order = [int(i) for i in np.argsort(-ev['score']) if i not in vetoed]
        if vetoed:
            print(f'  🔴 硬约束否决 {len(vetoed)}/{len(TH)} 个候选:')
            for i in sorted(vetoed):
                print(f'     #{i}: ' + ('; '.join((f'{x.code}({x.detail[:40]})' for x in hard[i] if x.severity == 'high')) or '此前已否决的同一方案'))
        if not order:
            disqualified.update((candidate_keys[i] for i in vetoed))
            print('  🔴 全部候选被硬约束否决，要求总工重来')
            hist = '上一轮你给的**全部**候选都因违反硬约束被否决:' + '; '.join(sorted({x.code for fs in hard.values() for x in fs if x.severity == 'high'})) + f'。注水量必须落在基准的 ±{a.w_max_dev * 100:.0f}% 内，且 θ 每个分量绝对值不得超过 1.0。请重新给方案。'
            rounds.append({'round': rnd, 'all_vetoed': True, 'n_msg': len(msgs), 'n_critique': len(crit), 'n_cand': len(TH), 'n_veto': len(vetoed), 'candidate_keys': candidate_keys, 'vetoed': sorted(vetoed), 'adjudicated': {}, 'simulation_results': {}, 'sim_attempts': 0})
            clean_streak = 0
            continue
        print(f'  总工给出 {len(cands)} 个候选 → 代理评估耗时 {1000.0 * dt:.1f} ms ({1000000.0 * dt / len(cands):.0f} µs/方案)')
        for i in order[:3]:
            print(f"    #{i} 代理:产油 {ev['oil'][i]:,.0f}  注水 {100 * (ev['inj_water'][i] / BASE_WATER - 1):+6.1f}%  经济分 {ev['score'][i] / 1000000.0:,.0f}M$  {cands[i][1][:38]}")
        (adj, errs_pp, sim_keys, simulation_results) = ({}, [], {}, {})
        sim_attempts = 0
        for i in order[:a.n_sim]:
            if i == inc_idx:
                adj[int(i)] = incumbent
                sim_keys[int(i)] = _theta_key(TH[i])
                print(f'    候选 #{i} 是现任冠军，复用已有实测(省一次模拟)')
                continue
            key = f'team{a.tag}_r{rnd}_c{i}'
            print(f'    模拟裁定 #{i} …', flush=True)
            sim_attempts += 1
            m = adjudicate(TH[i], key, a.threads, a.c_inj, a.c_prod, a.w_max_dev)
            sim_keys[int(i)] = _theta_key(TH[i])
            if m is None:
                if strict:
                    print('  🔴 模拟器失败，运行无效')
                    return 2
                continue
            simulation_results[int(i)] = m
            if not m.get('budget_ok', True):
                vetoed.add(int(i))
                print(f"      🔴 实测注水 {m['water_dev'] * 100:+.1f}% 超出预算，该算例不计入")
                continue
            adj[int(i)] = m
            e = 100.0 * (ev['oil'][i] / m['oil'] - 1.0)
            errs_pp.append(e)
            print(f"      真值产油 {m['oil']:,.0f} (基准 {base_m['oil']:,.0f}, {100 * (m['oil'] / base_m['oil'] - 1):+.2f}%)  注水 {100 * (m['water_inj'] / base_m['winj'] - 1):+.2f}%  ΔNPV8={(m['npv8'] - base_m['npv8']) / 1000000.0:+,.1f}M$  代理误差 {e:+.2f}%")
        sim_txt = '\n【模拟器(唯一裁判)对上述候选的实测结果】\n' + '\n'.join((f"  候选#{i}: 实测累计产油 {m['oil']:,.0f} ({100 * (m['oil'] / base_m['oil'] - 1):+.2f}% vs 基准)，实测注水 {100 * (m['water_inj'] / base_m['winj'] - 1):+.2f}%，ΔNPV@8%={(m['npv8'] - base_m['npv8']) / 1000000.0:+,.1f}M$，代理对它高估了 {100 * (ev['oil'][i] / m['oil'] - 1):+.2f}%" for (i, m) in adj.items())) if adj else '\n(本轮无模拟结果)'
        cand_txt = '\n'.join((f"  候选#{i}: 代理预测累计产油 {ev['oil'][i]:,.0f}(基准 {base_m['oil']:,.0f})，注水量为基准的 {100 * ev['inj_water'][i] / BASE_WATER:.1f}%，扣水成本后经济分 {ev['score'][i] / 1000000.0:,.0f}M$，理由: {cands[i][1]}" for i in order[:a.n_cand]))
        try:
            hard_txt = '\n【程序化硬约束检查(数字判定，已生效)】\n' + ('\n'.join((f'  候选#{i}: ' + '; '.join((f'[{x.severity}]{x.code} {x.detail}' for x in fs)) for (i, fs) in hard.items() if fs)) or '  全部候选通过硬约束')
            (_, aud) = ask(ASK_AUDIT.format(cands=cand_txt + sim_txt + hard_txt, msgs=msg_txt), a.model, a.timeout)
        except Exception as e:
            print(f'  🔴 审计失败: {e}')
            return 1
        aud_flags = aud.get('flags', [])
        flags = [Flag(f.get('severity', 'low'), f.get('code', '?'), f.get('detail', '')) for f in aud_flags]
        flags += [x for fs in hard.values() for x in fs]
        aud_veto = {int(f['candidate']) for f in aud_flags if f.get('severity') == 'high' and isinstance(f.get('candidate'), (int, float)) and (0 <= int(f['candidate']) < len(TH))}
        if aud_veto:
            print(f'  🔴 审计员否决候选 {sorted(aud_veto)}')
        vetoed |= aud_veto
        hi = [f for f in flags if f.severity in ('high', 'medium')]
        print(f"  审计: verdict={aud.get('verdict')} flags={len(flags)} (high/medium {len(hi)})")
        for f in flags[:4]:
            print(f'    [{f.severity}] {f.code}: {f.detail[:64]}')
        rounds.append({'round': rnd, 'n_msg': len(msgs), 'n_cand': len(cands), 'n_critique': len(crit), 'candidate_keys': candidate_keys, 'sim_attempts': sim_attempts, 'simulation_results': {str(k): v for (k, v) in simulation_results.items()}, 'proxy_best_oil': float(ev['oil'][order[0]]), 'flags': [asdict(f) for f in flags], 'verdict': aud.get('verdict'), 'adjudicated': {str(k): v for (k, v) in adj.items()}, 'vetoed': sorted(vetoed), 'hard_flags': {str(i): [asdict(x) for x in fs] for (i, fs) in hard.items() if fs}, 'proxy_err_pp': errs_pp, 'best_theta': TH[order[0]].tolist()})
        disqualified.update((candidate_keys[i] for i in vetoed))
        for key in disqualified:
            eligible.pop(key, None)
        for (_i, _m) in adj.items():
            if candidate_keys[_i] not in disqualified:
                eligible[candidate_keys[_i]] = (_m, TH[_i].copy())
        (incumbent, incumbent_th) = (None, None)
        for (_m, _th) in eligible.values():
            if incumbent is None or _m['npv8'] > incumbent['npv8']:
                (incumbent, incumbent_th) = (_m, _th.copy())
        if adj:
            prev_truth += ('' if not prev_truth else '\n') + '\n'.join((f"  [第{rnd}轮·{sim_keys.get(int(i), '?')}] 实测 ΔNPV@8% {(m['npv8'] - base_m['npv8']) / 1000000.0:+.1f} M$   产油 {100 * (m['oil'] / base_m['oil'] - 1):+.2f}%   实测注水 {m['water_dev'] * 100:+.2f}%   θ各段均值 " + ' '.join((f'{x:+.2f}' for x in TH[i].reshape(N_STAGE, len(INJ)).mean(1))) for (i, m) in adj.items()))
        clean_streak = clean_streak + 1 if not hi else 0
        print(f'  连续无高/中风险轮数 = {clean_streak}/2')
        if clean_streak >= 2:
            stop_reason = 'clean_streak'
            print('\n✅ 达到终止条件(连续两轮零 high/medium 风险)')
            break
        best_sim = max(adj.items(), key=lambda kv: kv[1]['npv8'], default=None)
        if best_sim is not None:
            last_case = SIM / best_sim[1]['simulation_file']
        hist = (f"上一轮(第{rnd}轮)模拟器实测最佳:{sim_keys.get(best_sim[0], '?')}  产油 {best_sim[1]['oil']:,.0f}({100 * (best_sim[1]['oil'] / base_m['oil'] - 1):+.2f}% vs 基准)，实测注水 {100 * (best_sim[1]['water_inj'] / base_m['winj'] - 1):+.2f}%，ΔNPV@8%={(best_sim[1]['npv8'] - base_m['npv8']) / 1000000.0:+,.1f}M$。" if best_sim else '上一轮无有效模拟结果。') + (f' 🔴 代理在这些候选上平均高估 {np.mean(errs_pp):+.2f}%，因此**不要相信代理的绝对数值**，只用它排序。' if errs_pp else '') + f' 审计给出 {len(hi)} 条高/中风险:' + '; '.join((f'{f.code}—{f.detail}' for f in hi[:3]))
    res = {'feedback_protocol': 'artifact-path-v1', 'run_protocol': 'strict-batch-v1', 'run_status': 'completed' if strict else 'unvalidated', 'outcome': 'accepted' if incumbent is not None else 'all_vetoed', 'stop_reason': stop_reason, 'config': {k: getattr(a, k) for k in ('model', 'max_rounds', 'n_cand', 'n_sim', 'threads', 'device', 'c_inj', 'c_prod', 'w_max_dev', 'critique', 'timeout')}, 'strict_run': strict, 'disqualified_keys': sorted(disqualified), 'incumbent_npv8': None if incumbent is None else incumbent['npv8'], 'incumbent_theta': None if incumbent_th is None else incumbent_th.tolist(), 'rounds': rounds, 'roles': [r['id'] for r in active], 'dropped': a.drop, 'model': a.model, 'llm_calls': LLM_CALLS['n'], 'llm_prompt_chars': LLM_CALLS['chars_in'], 'note': '代理仅用于筛选；最终取值须由 OPM Flow 的 field_cum 裁定。'}
    (OUT / f'loop{a.tag}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n→ {OUT / f'loop{a.tag}.json'}")
    return 0
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='sonnet')
    ap.add_argument('--max-rounds', type=int, default=4)
    ap.add_argument('--n-cand', type=int, default=6)
    ap.add_argument('--timeout', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--drop', nargs='*', default=[], help='消融:剔除的角色 id')
    ap.add_argument('--critique', action='store_true', default=True, help='启用角色互看质疑轮(P4.9)')
    ap.add_argument('--no-critique', dest='critique', action='store_false')
    ap.add_argument('--w-max-dev', type=float, default=0.25, help='允许的注水量偏离基准的上限(硬约束)')
    ap.add_argument('--c-inj', type=float, default=1.0, help='注水成本 USD/bbl')
    ap.add_argument('--c-prod', type=float, default=0.5, help='采出水处理 USD/bbl')
    ap.add_argument('--n-sim', type=int, default=2, help='每轮交模拟器裁定的候选数')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--tag', default='')
    ap.add_argument('--trace-inputs', action='store_true', help='保存各轮专家实际输入及反馈文件指纹，用于闭环验收')
    ap.add_argument('--strict-run', action='store_true', help='正式批次：角色/质疑未到齐或模拟失败即中止，不生成完成结果')
    sys.exit(run(ap.parse_args()))
