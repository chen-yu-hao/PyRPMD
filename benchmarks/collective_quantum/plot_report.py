"""Visualize measured selection and kinetic checks. No invented rate curves."""
from pathlib import Path
import json,base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2]

def main():
    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font=FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    s=json.loads((ROOT/'results/summary.json').read_text())
    m=json.loads((ROOT/'results/molecular_main/molecular_selection.json').read_text())
    b=json.loads((ROOT/'results/molecular_main/stability.json').read_text())
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    T=[150,300,600]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for frac,label in [('0.95','保留 95%'),('0.99','默认：保留 99%'),('0.995','更严格：保留 99.5%')]:
        ax.plot(T,[b[str(t)]['retained_fraction_sensitivity'][frac]['dimension'] for t in T],marker='o',label=label)
    ax.set_xticks(T);ax.set_yticks([1,2,3]);ax.set_ylim(.8,3.2)
    ax.set_title('Cl+HBr：非线性键转移坐标的候选维数',fontproperties=font)
    ax.set_xlabel('温度 / K',fontproperties=font);ax.set_ylabel('候选维数（尚未批准用于速率）',fontproperties=font)
    ax.legend(prop=font);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'molecular_candidates.png',dpi=180);plt.close(fig)
    checks=s['metric_checks'];fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot([r['nodes_per_axis'] for r in checks],[100*r['relative_ground_error_correct'] for r in checks],marker='o',label='保留完整动能度量')
    ax.plot([r['nodes_per_axis'] for r in checks],[100*r['relative_ground_error_wrong'] for r in checks],marker='s',label='错误对照：忽略质量度量与交叉项')
    ax.set_title('同一物理体系：非线性坐标下的量子动能检验',fontproperties=font)
    ax.set_xlabel('每方向网格节点数',fontproperties=font);ax.set_ylabel('基态能量相对误差 / %',fontproperties=font)
    ax.legend(prop=font);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'kinetic_metric_check.png',dpi=180);plt.close(fig)
    rows=''.join(f'<tr><td>{t} K</td><td>{m[str(t)]["required_dimension"]}</td><td>{100*m[str(t)]["diagnostics"]["holdout_relative_force_error"]:.2f}%</td><td>{b[str(t)]["group_bootstrap_dimension_counts"]}</td><td>未批准</td></tr>' for t in T)
    def image(name):return 'data:image/png;base64,'+base64.b64encode((figdir/name).read_bytes()).decode()
    pilot_text='未提供归档先导数据；本次报告不重复 Morse/Fermi/Eckart 筛选结果。'
    if s.get('pilots'):
        pp=s['pilots'];pilot_text=f"Morse 单坐标响应候选为 {pp['morse']['required_dimension']} 维；Fermi x 响应为 {pp['fermi']['required_dimension']} 维；Eckart 候选为 {pp['eckart']['required_dimension']} 维、留出力误差 {100*pp['eckart']['diagnostics']['holdout_relative_force_error']:.2f}%。它们仍只是筛选证据。"
    last=checks[-1]
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>目标导向与非线性量子坐标</title><style>body{{font-family:system-ui,"Noto Sans CJK SC",sans-serif;max-width:1040px;margin:auto;padding:36px;color:#172b3a;background:#f8fafb;line-height:1.8}}h1,h2{{line-height:1.35}}header{{border-bottom:4px solid #2b5964;padding-bottom:20px}}section{{background:white;padding:24px;margin:22px 0;border:1px solid #dde5e8;border-radius:10px}}.tag{{font-size:13px;letter-spacing:2px}}.warn{{background:#fff5dc;border-left:4px solid #be8c20;padding:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%;height:auto}}code{{background:#eef2f4;padding:2px 5px}}small{{color:#536774}}</style><header><div class="tag">PyRPMD / 研究检查点 / 2026-11-08</div><h1>不再把所有非线性自由度<br>都纳入量子核心</h1><p>目标导向的集体坐标 + 可逆非线性坐标 + 完整动能度量 + 升维证据检查。</p></header><section><h2>最重要的结果</h2><p>新的非共线 Cl+HBr 先导数据给出一维或二维<strong>候选</strong>，不是已批准的量子传播模型。较严格阈值会改变维数；环境耦合、变质量动能以及有限 J 验证尚未完成。</p><div class="warn">本轮没有计算新的双分子反应速率，没有将位置/键转移响应验证替代通量—侧函数验证，也没有用全量子答案挑选维度。</div><table><tr><th>温度</th><th>默认候选维数</th><th>留出集目标力误差</th><th>32 次分组重抽样</th><th>速率传播许可</th></tr>{rows}</table><img src="{image('molecular_candidates.png')}"><p><small>每温度 256 条独立初始化的 32 珠子受限 PIMD 链，800 次 HMC。目标为 log(r_BrH/r_HCl)。这里只检验局部目标力结构，未证实受限系综已平衡。</small></p></section><section><h2>什么改变了？</h2><p>旧筛选器把所有方向的非线性力合并。新筛选器从指定目标出发，查找能影响该目标的线性、平方及乘积耦合；对弯曲势使用分组留出检验的灵活拟合。候选空间可旋转成集体坐标，也可使用可逆的非线性坐标。</p><p>{pilot_text}</p><p>三个模型的先导数据复用 PR #134 归档；Cl+HBr 先导是本轮新运行。各组样本哈希、种子和设置均已保存。</p></section><section><h2>非线性坐标必须同时改变动能</h2><p>使用完整的 G(u) 和体积元 ρ(u)，求解 H = −ℏ²/(2ρ) ∂ₐ(ρGᵃᵇ∂ᵦ) + V。不能把 y² 这类不可逆特征直接当作坐标，也不能把变换后的势能放进平直质量的传播器。</p><img src="{image('kinetic_metric_check.png')}"><p>在 {last["nodes_per_axis"]}×{last["nodes_per_axis"]} 节点下，正确动能的基态误差为 {100*last["relative_ground_error_correct"]:.3f}%；省略动能度量的错误对照仍有 {100*last["relative_ground_error_wrong"]:.3f}%。这是坐标变换一致性检验，不是反应速率精度。</p><p>学到的弯曲路径在几何上只需一个进度变量，而动能核心—环境交叉指标达到 0.899。因此，路径能压缩成一维并不证明环境量子耦合可忽略。</p></section><section><h2>哪些已完成，哪些未完成？</h2><p>34 项新增测试通过，包括旋转/单位一致性、偶对称共振伙伴、旁观模排除、非线性映射可逆性、三原子 J=0 质量度量、1–3 维全坐标有限元量子核与保范数的有限本征基传播，以及缺少升维证据时拒绝批准。</p><p>尚未完成：真实反应的核心/浴量子传播耦合、非线性坐标下有限 J 旋转与 Coriolis 项、完整量子速率参照、非共线速率精度提升验证。任何候选仍标记 physical_converged=False。</p></section></html>'''
    (ROOT/'collective_quantum_report.html').write_text(html,encoding='utf-8')
if __name__=='__main__':main()
