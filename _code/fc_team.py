from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '_pipelines' / 'fc_team'
STAGES = ('cartography', 'reading', 'audit', 'synthesis')
SOURCE_TYPES = ('deck', 'simulation', 'measurement', 'literature', 'reasoning')
SEVERITIES = ('high', 'medium', 'low')
TRANSITIONS = {'reading': {'audited', 'flagged'}, 'flagged': {'reverified', 'discarded', 'caveated'}}

@dataclass
class Provenance:
    source_id: str
    source_region: str
    source_type: str
    source_sha256: str

@dataclass
class Flag:
    severity: str
    code: str
    detail: str

@dataclass
class Message:
    agent_id: str
    stage: str
    payload: dict
    confidence: float
    provenance: Provenance
    flags: list[Flag] = field(default_factory=list)

def validate(m: Message) -> list[str]:
    e = []
    if not m.agent_id.strip():
        e.append('agent_id 为空')
    if m.stage not in STAGES:
        e.append(f'stage 非法: {m.stage}')
    if not 0.0 <= m.confidence <= 1.0:
        e.append('confidence 不在 [0,1]')
    p = m.provenance
    if not p.source_id.strip():
        e.append('provenance.source_id 为空')
    if not p.source_region.strip():
        e.append('provenance.source_region 为空')
    if p.source_type not in SOURCE_TYPES:
        e.append(f'source_type 非法: {p.source_type}')
    if len(p.source_sha256) != 64:
        e.append('source_sha256 不是 SHA-256')
    for f in m.flags:
        if f.severity not in SEVERITIES:
            e.append(f'severity 非法: {f.severity}')
    return e

def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def cartography(sim_dir: Path, n_prod: int=22, n_key: int=3, nt: int=40) -> dict:
    import norne_bulk as NB
    prods = list(NB.PRODUCERS)
    files = sorted(sim_dir.glob('*.npz'))
    (X, Y, keys) = ([], [], [])
    for f in files:
        d = np.load(f)
        if 'inj_actual' not in d or 'obs' not in d:
            continue
        inj = d['inj_actual']
        ob = d['obs'].reshape(n_prod, n_key, nt)
        if ob.shape != (n_prod, n_key, nt):
            continue
        X.append(inj.sum(1))
        Y.append(np.nan_to_num(ob[:, 0, :]).sum(1))
        keys.append(f.stem)
    if len(X) < 20:
        raise SystemExit(f'🔴 算例太少({len(X)}),无法建连通性')
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.c_[Xz, np.ones(len(Xz))]
    (beta, *_) = np.linalg.lstsq(A, (Y - Y.mean(0)) / (Y.std(0) + 1e-12), rcond=None)
    B = beta[:4]
    sv = np.linalg.svd(Xz, compute_uv=False)
    cond = float(sv.max() / max(sv.min(), 1e-12))
    diag = {'cond': cond, 'inj_cv': (X.std(0) / (np.abs(X.mean(0)) + 1e-12)).tolist()}
    return {'n_case': len(X), 'producers': prods, 'diag': diag, 'injectors': list(__import__('forecast_gen').INJ_W), 'beta': B.tolist(), 'case_keys_sha': sha256_arr(np.array(sorted(keys), dtype='U'))}
ROLES = [dict(id='reservoir_engineer', type='simulation', reads='field_cum[FPR] 压力 + obs[WWPR] 含水 + PERMX/PORO/NTG', duty='波及效率、压力保持、水往哪儿走'), dict(id='connectivity_analyst', type='simulation', reads='cartography.beta 影响矩阵 + deck FAULTS/MULTFLT', duty='哪口注水井真正影响哪口生产井，谁是小旋钮'), dict(id='production_surveillance', type='simulation', reads='obs[22 井 × WOPR/WWPR/WBHP × 40 步]', duty='谁水淹了、谁在衰减、何时该减注'), dict(id='economics_analyst', type='measurement', reads='EIA Brent 年均价 + 折现率档 + 注水/采出水成本', duty='短期 vs 长期、注水的边际经济性'), dict(id='geomechanics_expert', type='literature', reads='北海同类砂岩油藏岩石力学文献(类比，非 Norne 实测)', duty='注入压力上限、储层压实与破裂风险的定性边界', caveat='🔴 Norne deck 无 GEOMECH/STRESS/YOUNGMOD/POISSON。本角色只能类比推理，不得声称拥有 Norne 实测应力数据。'), dict(id='seismic_4d_analyst', type='literature', reads='Norne 4D 时移地震文献(已检索到 doi:10.4043/19049-ms 等)', duty='水驱前缘推进与流体接触面移动的先验'), dict(id='constraint_auditor', type='simulation', reads='inj_actual vs 目标、WBHP 上限、泵能力', duty='红队：抓不可行方案、抓目标守恒≠实测守恒'), dict(id='chief_engineer', type='reasoning', reads='以上全部消息', duty='合成 θ；对每条采纳的建议标注来源 agent_id')]

def print_roles() -> None:
    print(f'\n=== 角色注册表({len(ROLES)} 个)===')
    for r in ROLES:
        print(f"  {r['id']:<24s} [{r['type']:<11s}] {r['duty']}")
        print(f"  {'':<24s}  读: {r['reads']}")
        if r.get('caveat'):
            print(f"  {'':<24s}  {r['caveat']}")

def main() -> None:
    import forecast_gen as FG
    OUT.mkdir(parents=True, exist_ok=True)
    sim = ROOT / '_pipelines' / 'fc_decide' / 'sim'
    carto = cartography(sim)
    B = np.array(carto['beta'])
    inj = carto['injectors']
    prods = carto['producers']
    print(f"=== Cartographer:注水井 → 生产井 影响矩阵(n={carto['n_case']} 算例)===\n")
    print(f"{'生产井':>8s} " + ' '.join((f'{w:>8s}' for w in inj)) + '   主控注水井')
    print('-' * 62)
    rows = []
    for (j, p) in enumerate(prods):
        col = B[:, j]
        if np.abs(col).max() < 0.05:
            continue
        k = int(np.argmax(np.abs(col)))
        print(f'{p:>8s} ' + ' '.join((f'{v:+8.3f}' for v in col)) + f'   {inj[k]}')
        rows.append((p, col.tolist(), inj[k]))
    print(f'\n显著响应生产井 {len(rows)}/{len(prods)}')
    infl = np.abs(B).sum(1)
    print('\n=== 各注水井总影响力(|β| 行和)===')
    for (k, w) in enumerate(inj):
        bar = '█' * int(round(infl[k] / max(infl.max(), 1e-09) * 40))
        print(f'  {w:>6s} {infl[k]:7.3f}  {bar}')
    cond = carto['diag']['cond']
    print(f"\n自查:设计矩阵条件数 = {cond:.2f}  ({('✅ 可归因' if cond < 30 else '🔴 共线,归因不可信')})")
    fl = [] if cond < 30 else [Flag('high', 'collinear_design', f'注水井设计矩阵条件数 {cond:.1f} ≥ 30，井间归因不可信')]
    msg = Message(agent_id='cartographer', stage='cartography', flags=fl, payload={'beta': carto['beta'], 'injectors': inj, 'producers': prods, 'n_case': carto['n_case'], 'diag': carto['diag']}, confidence=min(0.95, 0.5 + carto['n_case'] / 600), provenance=Provenance(source_id='_pipelines/fc_decide/sim/*.npz', source_region=f"inj_actual + obs[:, WOPR, :]  ({carto['n_case']} cases)", source_type='simulation', source_sha256=carto['case_keys_sha']))
    errs = validate(msg)
    print(f"\n消息校验: {('✅ 合法' if not errs else '🔴 ' + '; '.join(errs))}")
    d = asdict(msg)
    d['provenance'] = asdict(msg.provenance)
    (OUT / 'cartography.json').write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"→ {OUT / 'cartography.json'}")
    print_roles()
if __name__ == '__main__':
    main()
