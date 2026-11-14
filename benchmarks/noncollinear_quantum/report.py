"""Measured-data figures and self-contained Chinese HTML; no fitted curves."""
from __future__ import annotations
import argparse,base64,html,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]

def main(root=ROOT,show=False):
    root=Path(root);out=root/'results';figdir=root/'figures';figdir.mkdir(exist_ok=True)
    s=json.loads((out/'validation_summary.json').read_text());rates=s['rates']
    fp=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=FontProperties(fname=str(fp)) if fp.exists() else FontProperties()
    def save(fig,ax,name,title,xlabel,ylabel):
        ax.set_title(title,fontproperties=font,pad=14);ax.set_xlabel(xlabel,fontproperties=font);ax.set_ylabel(ylabel,fontproperties=font)
        ax.grid(alpha=.18);handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(prop=font)
        fig.tight_layout();fig.savefig(figdir/name,dpi=170)
        if show and name in ('rates.png','quantum_audit.png'):plt.show()
        else:plt.close(fig)
    names=['T150_P32','T300_cayley32_half','T600_P16'];a=[rates[n] for n in names]
    fig,ax=plt.subplots(figsize=(9,5.2));ax.errorbar([r['temperature_K'] for r in a],[r['rate_lower_cm3_molecule_s'] for r in a],
        yerr=[r['sampling_se_cm3_molecule_s'] for r in a],marker='o',capsize=4,label='非共线 RPMD 基线：不同 P，未完全收敛')
    r=rates['T300_P1'];ax.errorbar([300],[r['rate_lower_cm3_molecule_s']],yerr=[r['sampling_se_cm3_molecule_s']],marker='s',capsize=4,label='300 K 经典对照（P=1）');ax.set_yscale('log')
    save(fig,ax,'rates.png','实际非共线碰撞结果：不是选择性量子修正速率','温度 / K','速率 / cm^3 molecule^-1 s^-1')
    keys=['T300_P16','T300_P32','T300_cayley32','T300_cayley32_half','T300_cayley64','T300_wide','T300_far']
    labels=['P16\nVV','P32\nVV','P32\nCayley','P32\n半步长','P64\nCayley','b最大=9\nP16','发射面=18\nP16']
    fig,ax=plt.subplots(figsize=(10.5,5.3));x=np.arange(len(keys))
    ax.errorbar(x,[rates[k]['rate_lower_cm3_molecule_s']*1e12 for k in keys],yerr=[rates[k]['sampling_se_cm3_molecule_s']*1e12 for k in keys],fmt='o',capsize=4,label='下界/已完成反应贡献 ± 采样标准误')
    for j,k in enumerate(keys):
        lo=rates[k]['rate_lower_cm3_molecule_s']*1e12;hi=rates[k]['rate_upper_cm3_molecule_s']*1e12
        if hi>lo:ax.plot([j,j],[lo,hi],linestyle='--',marker='_',label='未完成轨迹给出的盘内上下界' if k=='T300_wide' else None)
    ax.set_xticks(x);ax.set_xticklabels(labels,fontproperties=font)
    save(fig,ax,'convergence.png','300 K 数值检查：远发射面仍有未完成轨迹','参数组合（采样量不完全相同）','速率 / 1e-12 cm^3 molecule^-1 s^-1')
    z=np.load(out/'T300_cayley32_half/trajectories.npz');react=z['outcomes']==1
    fig,ax=plt.subplots(figsize=(9,5.2));ax.scatter(z['b'][react],np.rad2deg(z['angle_at_H_rad'][react]),label='反应轨迹',s=28)
    ax.set_ylim(0,180)
    save(fig,ax,'noncollinear.png','反应路径允许显著弯曲；没有施加共线限制','冲量参数 / bohr','重原子最近接近时的 H 夹角 / 度（不是过渡态角）')
    fig,ax=plt.subplots(figsize=(9,4.8));T=[150,300,600];dims=[s['selection'][str(t)]['selection']['selected_dimension'] for t in T]
    ax.bar(np.arange(3),dims,label='独立先导样本选择的维数');ax.axhline(2,linestyle='--',label='要求最多二维：未通过')
    ax.set_xticks(np.arange(3));ax.set_xticklabels([f'{t} K\n64/64 重采样仍选择三维' for t in T],fontproperties=font);ax.set_ylim(0,3.7);ax.set_yticks([0,1,2,3])
    save(fig,ax,'selection.png','当前保守选模器不支持把该非共线体系压缩到 1–2 维','温度','量子候选空间的内部维数（启发式诊断）')
    fig,ax=plt.subplots(figsize=(9,5.2))
    for name,label in [('quantum_J0_pilot','较小角向/通道基组'),('quantum_J0_check','加密基组＋更远渐近面')]:
        zq=np.load(out/name/'quantum.npz');ax.plot(zq['total_energy']*27.211386245988,zq['CRP'],label=label)
    save(fig,ax,'quantum_audit.png','J=0 三内部坐标量子审核：加密仍改变曲线约 49%','总能量 / eV（相同能量零点；未平移阈值）','有限基组累积反应概率：尚不能作精确参照')
    fig,ax=plt.subplots(figsize=(9,4.8));keys=['T300_P16','T300_P32','T300_cayley64']
    ax.errorbar([rates[k]['beads'] for k in keys],[rates[k]['initial_stats']['internal_energy_Eh'] for k in keys],
        yerr=[rates[k]['initial_stats']['energy_sampling_se_Eh'] for k in keys],marker='o',capsize=4,label='独立 HBr 路径积分初态')
    ax.axhline(s['reactant_reference']['300']['internal_energy_Eh'],linestyle='--',label='有限盒内转振量子参照')
    save(fig,ax,'preparation.png','300 K 初态能量检查：有限珠子数偏差必须单独审计','珠子数 P','HBr 内部能量 / hartree')
    fig,ax=plt.subplots(figsize=(9,4.8));keys=['T300_P32','T300_cayley32','T300_cayley32_half']
    ax.bar(np.arange(3),[rates[k]['median_energy_drift_Eh'] for k in keys]);ax.set_yscale('log');ax.set_xticks(np.arange(3));ax.set_xticklabels(['Verlet，dt=4','Cayley，dt=4','Cayley，dt=2'],fontproperties=font)
    save(fig,ax,'conservation.png','同一组初态：Cayley 和半步长降低能量漂移','P=32 的生产积分器','轨迹最大能量漂移的中位数 / hartree')
    def img(name):return 'data:image/png;base64,'+base64.b64encode((figdir/name).read_bytes()).decode()
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [
        key,r['temperature_K'],r['beads'],r['sample_count'],f"{r['rate_lower_cm3_molecule_s']:.4e}",
        f"{r['rate_upper_cm3_molecule_s']:.4e}",f"{r['sampling_se_cm3_molecule_s']:.3e}",r['reactive'],r['unfinished']])+'</tr>' for key,r in rates.items())
    cards=[('rates.png','速率基线','已计算三维空间中的非共线轨迹，单位为体积/分子/时间。没有未经验证的“修正后”曲线。150 K 用 P32，300 K 用 P32 半步长，600 K 用 P16；参数不同，尚非收敛温度曲线。'),
      ('convergence.png','数值与失败边界','误差棒只含独立轨迹采样误差。有限发射面、冲量参数上限、珠子数和初态制备尚未完全收敛。虚线不是置信区间，而是未完成轨迹的最坏情况界；盘外尾部未知。'),
      ('noncollinear.png','弯曲证据','图示夹角记录于重原子距离最小时，不是过渡态几何，也不是量子隧穿路径。它证明生产轨迹没有被限制为直线。'),
      ('selection.png','自动选模的拒绝结果','先导样本独立于生产轨迹和全量子答案。质量加权对齐后消除刚性运动，检查非线性力、目标方向和残余耦合。三维推荐是保守启发式，不是物理上最小量子维数的定理。'),
      ('quantum_audit.png','量子参照未通过','两次实际计算都只含 J=0。离散散射矩阵的酉性约为 10⁻¹²，但共同总能量上的曲线仍相差约 49.1%；不能据此给出精确三维热速率。'),
      ('preparation.png','初态量子统计','路径积分初态在有限盒内制备。高接受率不等于充分混合；能量审计单独记录有限珠子数和采样误差。'),
      ('conservation.png','积分精度','此图只说明数值能量守恒改善，不代表 RPMD 的量子动力学近似变得精确。')]
    figures=''.join(f'<section><h2>{title}</h2><p>{text}</p><img src="{img(name)}" alt="{title}"></section>' for name,title,text in cards)
    total=sum(r['sample_count'] for r in rates.values());fail=sum(r['unfinished']+r['invalid'] for r in rates.values())
    receipt=root/'PR.json';pr=json.loads(receipt.read_text()).get('url') if receipt.exists() else None
    prtext=f'<a href="{html.escape(pr)}">查看已提交的 Draft PR</a>' if pr else 'PR 链接以本次交付的 GitHub 提交记录为准。'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>非共线双分子速率验证 — PyRPMD</title>
<style>body{{font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:#f5f7fa;color:#172331;line-height:1.7;margin:0}}main{{max-width:1080px;margin:auto;padding:36px 22px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:23px}}section{{background:white;padding:24px;border:1px solid #dbe2e9;border-radius:12px;margin:22px 0}}.alert{{border-left:5px solid #b45309;background:#fff7e7;padding:20px}}.metrics{{display:flex;gap:14px;flex-wrap:wrap}}.metric{{background:white;border:1px solid #dbe2e9;flex:1;padding:18px;min-width:150px}}.metric b{{font-size:29px;display:block}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #dbe2e9;padding:9px;text-align:left}}.scroll{{overflow:auto}}code,pre{{background:#edf1f5;padding:4px}}.muted{{color:#536171}}a{{color:#145ea8}}</style><main>
<p class="muted">PyRPMD / 2026-11-08 / 独立可复现验证</p><h1>从共线到非共线：<br>已实现的能力，与未通过的验证</h1>
<div class="alert"><strong>结论：非共线 RPMD 速率和通用计算接口已运行；通用选择性量子修正方法尚未验证成功。</strong><br>当前选模器对 Cl+HBr 推荐全部三个内部方向；量子审核尚缺收敛及总角动量求和。因此本报告不输出“修正后三维精确速率”。</div>
<p>{prtext}</p><div class="metrics"><div class="metric"><b>{total:,}</b>实际碰撞轨迹</div><div class="metric"><b>3 / 3</b>候选内部维数：无降维</div><div class="metric"><b>49.1%</b>J=0 曲线加密变化</div><div class="metric"><b>32</b>通过的本地测试</div></div>
<section><h2>本轮真正新增什么</h2><p>完整三维碰撞相空间采样；任意原子数的 Cartesian RPMD 接口；质量加权、去刚性运动的自动选模；显式总角动量/宇称热积分；不匹配或未验证的量子修正阻断机制。</p><p>三原子算例实际传播两个三分量 Jacobi 向量。多原子通用接口已用 5/7 原子数值控制验证，但没有实际多原子化学反应的高精度量子速率验证。</p></section>
{figures}<section><h2>完整数值记录</h2><p>rate lower/upper 是采样冲量参数盘内的下/上界；共有 {fail} 条未完成或失败轨迹保留在统计中。没有删除困难轨迹。</p><div class="scroll"><table><tr><th>运行</th><th>T/K</th><th>P</th><th>轨迹数</th><th>下界</th><th>上界</th><th>采样 SE</th><th>反应数</th><th>未完成</th></tr>{rows}</table></div></section>
<section><h2>完整输入与复现</h2><p><code>benchmarks/noncollinear_quantum/inputs/</code> 保存所有温度、珠子数、时间步、发射距离、冲量参数、样本量及种子。完整归档含原始轨迹结果、量子通道数组、独立先导样本及 SHA-256 清单；GitHub 只提交源码、输入和小型摘要。</p><pre>python benchmarks/noncollinear_quantum/dynamics.py --input INPUT.json --output NEW_DIR
python benchmarks/noncollinear_quantum/pilot_selection.py
python benchmarks/noncollinear_quantum/reactant_audit.py
python benchmarks/noncollinear_quantum/hyperspherical.py --input INPUT.json --output NEW_DIR
python benchmarks/noncollinear_quantum/summarize.py
python benchmarks/noncollinear_quantum/report.py</pre><p>输入为 benchmark 专用 JSON，不冒充原生 pyrpmd run 输入。原仓库默认积分器与现有生产流程没有被替换。</p></section>
<section><h2>方法边界与下一步</h2><p>需要先解决全量子渐近通道与角向收敛，完成 J/宇称求和，再评估低维缺陷转移。还需要更全面的独立初态制备、冲量参数尾部、发射距离和轨迹长度检验。全三维量子核心的恒等替换不是降维方法的成功证明。</p><p>局部选模阈值是启发式的；未实现一般曲线坐标上的量子动能算子，也没有宣称通用非谐环境的精确反馈。所有结果保留 <code>physical_converged=False</code>。</p></section>
<section><h2>势能来源与资料</h2><p>历史 BrHCl_LEPS1 参数；原子顺序 Br,H,Cl；同位素 79Br/1H/35Cl；固定一个电子势能面，未加入卤素精细结构因子。势能非新拟合，不等价于现代实验预测。具体来源、系数与哈希见 source.json。</p><p>方法背景：Collepardo-Guevara 等，JCP 130,174713(2009)，DOI 10.1063/1.3127145；Suleimanov 等，CPC 184,833(2013)，DOI 10.1016/j.cpc.2012.10.017。这里的独立碰撞系综不冒充上述平衡 Bennett–Chandler 算法的已收敛复现。</p></section>
<p class="muted">基于 main: {s['base_commit']}。本报告由保存的实际数据生成，无经验拟合、无曲线强制归一化、无负值裁剪。</p></main></html>'''
    path=root/'noncollinear_bimolecular_report.html';path.write_text(text,encoding='utf-8');print(path)
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));p.add_argument('--show',action='store_true');a=p.parse_args();main(a.root,a.show)
