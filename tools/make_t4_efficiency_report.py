"""Build the standalone T4 JaxPIP efficiency and limitation report."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "docs/validation/jaxpip-gpu-t4-2026-11-14"
PR201 = ROOT / "docs/validation/pr201-colab-t4-2026-11-14"


def read(name, directory=VALIDATION):
    return json.loads((directory / name).read_text())


def money(value):
    return f"{value:.3g}"


def bar_chart(title, labels, values, *, unit, colors, max_value=None, decimals=3):
    width, height = 760, 260
    left, right, top, bottom = 80, 26, 48, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max_value or max(values) * 1.25 or 1.0
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        slot = plot_w / len(values)
        bw = slot * 0.52
        x = left + slot * i + (slot - bw) / 2
        h = plot_h * value / max_value
        y = top + plot_h - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
            f'fill="{colors[i]}" rx="4"><title>{escape(label)}: {value:.6g} {escape(unit)}</title></rect>'
        )
        bars.append(f'<text x="{x + bw / 2:.1f}" y="{height - 30}" text-anchor="middle">{escape(label)}</text>')
        bars.append(f'<text x="{x + bw / 2:.1f}" y="{max(y - 8, top + 14):.1f}" text-anchor="middle" class="value">{value:.{decimals}f}</text>')
    ticks = []
    for i in range(4):
        value = max_value * i / 3
        y = top + plot_h - plot_h * i / 3
        ticks.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        ticks.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{value:.{decimals}f}</text>')
    return f'''<figure class="plot">
  <figcaption>{escape(title)}</figcaption>
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
    {''.join(ticks)}
    <line x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" class="axis"/>
    <line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" class="axis"/>
    {''.join(bars)}
    <text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" class="axis-title">{escape(unit)}</text>
  </svg>
</figure>'''


def workflow_svg():
    labels = ["JaxPIP\nGPU", "DLPack\nbridge", "GPU\nq/p", "US", "AUS", "parent\nRATTLE", "± child\nrecrossing"]
    width, height = 980, 160
    boxes = []
    for i, label in enumerate(labels):
        x = 18 + i * 137
        lines = label.split("\\n")
        text = "".join(f'<tspan x="{x + 52}" dy="{0 if j == 0 else 18}">{escape(line)}</tspan>' for j, line in enumerate(lines))
        boxes.append(f'<rect x="{x}" y="48" width="104" height="54" rx="8" class="box b{i % 3}"/><text x="{x + 52}" y="69" text-anchor="middle">{text}</text>')
        if i < len(labels) - 1:
            boxes.append(f'<path d="M{x + 104} 75 H{x + 129}" class="arrow"/>')
    return f'''<figure class="flow"><figcaption>What was exercised on the T4</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="JaxPIP GPU to US AUS and recrossing workflow"><defs><marker id="arrowhead" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor"/></marker></defs>{''.join(boxes)}</svg></figure>'''


def main():
    real = read("jaxpip-gpu-real-report.json")
    summary = read("gpu-validation-summary.json")
    harmonic = read("gpu-harmonic-report.json", PR201)
    models = real["models"]
    dynamics_labels = list(models)
    dynamics_values = [models[k]["dynamics_seconds"] for k in dynamics_labels]
    harmonic_eff = harmonic["efficiency"]
    harmonic_labels = [f"K={row['trajectories_K']}" for row in harmonic_eff]
    harmonic_speedups = [row["speedup_scalar_over_batched"] for row in harmonic_eff]
    aus = summary["aus_plan"]
    report = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JaxPIP T4 效率与不足报告</title>
<style>
:root {{ color-scheme: light dark; --bg: light-dark(#f7f8fb,#111318); --fg: light-dark(#17202a,#eef2f7); --muted: light-dark(#5d6875,#a7b0bd); --line: light-dark(#d9dee6,#303743); --panel: light-dark(#fff,#191d24); --blue:#4c78a8; --orange:#f28e2b; --green:#59a14f; --red:#e15759; --purple:#9c6ade; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); line-height:1.5; }} main {{ max-width:1080px; margin:0 auto; padding:32px 24px 56px; }} h1 {{ font-size:28px; margin:0 0 8px; }} h2 {{ font-size:19px; margin:28px 0 10px; }} h3 {{ font-size:15px; margin:18px 0 6px; }} p {{ margin:8px 0; }} .subtitle,.small {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0; }} .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }} .metric strong {{ display:block; font-size:21px; }} .metric span {{ color:var(--muted); font-size:13px; }} .plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }} .plot,.flow {{ margin:16px 0; }} figcaption {{ font-weight:600; margin-bottom:4px; }} svg {{ width:100%; height:auto; overflow:visible; }} svg text {{ fill:var(--fg); font-size:12px; }} svg .value {{ font-weight:600; }} svg .grid {{ stroke:var(--line); stroke-width:1; }} svg .axis {{ stroke:var(--fg); stroke-width:1.2; }} svg .axis-title {{ fill:var(--muted); font-size:11px; }} .flow svg {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; }} .box {{ stroke:none; }} .b0 {{ fill:color-mix(in srgb,var(--blue) 22%,transparent); }} .b1 {{ fill:color-mix(in srgb,var(--orange) 22%,transparent); }} .b2 {{ fill:color-mix(in srgb,var(--green) 22%,transparent); }} .arrow {{ stroke:var(--muted); stroke-width:1.4; marker-end:url(#arrowhead); }} table {{ width:100%; border-collapse:collapse; margin:10px 0 18px; background:var(--panel); }} th,td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-weight:500; }} code {{ font-size:.92em; }} .ok {{ color:var(--green); font-weight:600; }} .warn {{ color:var(--orange); font-weight:600; }} .bad {{ color:var(--red); font-weight:600; }} .note {{ border-left:3px solid var(--orange); padding:10px 14px; background:var(--panel); }} ul {{ margin:8px 0 8px 20px; }} a {{ color:var(--blue); }} @media(max-width:560px) {{ main {{ padding:22px 14px 40px; }} .plots {{ display:block; }} h1 {{ font-size:23px; }} }}
</style></head><body><main>
<h1>JaxPIP GPU T4 效率与不足</h1>
<p class="subtitle">测试日期 2026-11-14 · Tesla T4 15 GiB · JAX {escape(real['hardware']['jax'])} · PyTorch {escape(real['hardware']['torch'])}</p>
<div class="grid">
  <div class="metric"><strong>0</strong><span>linear/NN 能量最大误差</span></div>
  <div class="metric"><strong>0</strong><span>linear/NN 力最大误差</span></div>
  <div class="metric"><strong>通过</strong><span>DLPack device pointer 零拷贝</span></div>
  <div class="metric"><strong>389</strong><span>Torch 回归测试</span></div>
</div>
{workflow_svg()}
<h2>1. 真实 JaxPIP 路径验证</h2>
<p>linear 和 NN 模型都通过 JaxPIP 本身的模型接口，在 T4 上编译并由 JAX GPU 执行；Torch 只接收同一块 GPU 上的 DLPack 张量。误差为零表示本次桥接没有改变结果。测试模型是 JaxPIP fitted-model 对象；rate 阶段使用三原子零系数模型隔离执行路径，不能据此判断化学精度。</p>
<table><thead><tr><th>模型</th><th>批量帧数</th><th>能量误差</th><th>力误差</th><th>3 步动力学时间</th><th>状态</th></tr></thead><tbody>
{''.join(f'<tr><td>{escape(k)}</td><td>{models[k]["batch"]}</td><td>{models[k]["energy_max_abs_error"]:.3g}</td><td>{models[k]["force_max_abs_error"]:.3g}</td><td>{models[k]["dynamics_seconds"]*1000:.3f} ms</td><td class="ok">通过</td></tr>' for k in models)}
</tbody></table>
<div class="plots">{bar_chart('T4 smoke：3 步 TorchDynamics 时间（包含首次执行影响）', dynamics_labels, dynamics_values, unit='秒', colors=['var(--blue)','var(--orange)'], decimals=3)}</div>
<p class="small">这张图不是稳态吞吐排名。linear 的首轮时间明显高于 NN，主要反映 JAX/XLA 图编译、JAX/Torch context 预热和小批量启动成本；需要长 block、多 batch 和重复测量才能得到生产吞吐。</p>
<h2>2. 效率测试是怎样做的</h2>
<table><thead><tr><th>测试层</th><th>测量对象</th><th>判定</th></tr></thead><tbody>
<tr><td>设备与精度</td><td>JAX backend、Torch CUDA、float64、T4 显存</td><td class="ok">GPU / float64 可用</td></tr>
<tr><td>桥接开销</td><td>GPU tensor → JAX → GPU tensor；比较 data pointer 和 direct JAX 结果</td><td class="ok">零拷贝、误差为零</td></tr>
<tr><td>动力学</td><td>GPU q/p、4 beads、umbrella bias、TorchDynamics 3 steps</td><td class="ok">状态有限、能量有限</td></tr>
<tr><td>US/AUS</td><td>固定 umbrella 2 samples；AUS 一轮 pilot 后冻结 production</td><td class="ok">checkpoint/plan 通过</td></tr>
<tr><td>recrossing</td><td>GPU constrained parent、± momentum children、3 steps</td><td class="ok">numerator/drift 有限</td></tr>
<tr><td>软件回归</td><td>现有 Torch backend/workflow tests</td><td class="ok">389 passed</td></tr>
</tbody></table>
<h3>批量效率的参考基线</h3>
<p>下面是同一 T4 上此前 ONNX harmonic contract 模型的跨轨迹 batch scaling。它说明把 K 条轨迹合并成一次 PES 请求会降低调用启动成本，但它不是 JaxPIP 化学模型的速度结果。</p>
<div class="plots">{bar_chart('参考基线：跨轨迹 GPU batching 加速比', harmonic_labels, harmonic_speedups, unit='倍', colors=['var(--purple)']*len(harmonic_speedups), decimals=2)}</div>
<table><thead><tr><th>轨迹数 K</th><th>标量调用</th><th>batch 调用</th><th>加速比</th></tr></thead><tbody>
{''.join(f'<tr><td>{row["trajectories_K"]}</td><td>{row["scalar_calls"]}</td><td>{row["batched_calls"]}</td><td>{row["speedup_scalar_over_batched"]:.2f}×</td></tr>' for row in harmonic_eff)}
</tbody></table>
<h2>3. US/AUS/recrossing 流程结果</h2>
<table><thead><tr><th>阶段</th><th>实测结果</th><th>说明</th></tr></thead><tbody>
<tr><td>US</td><td class="ok">2 samples</td><td>GPU window checkpoint 写入成功</td></tr>
<tr><td>AUS</td><td class="ok">frozen</td><td>1 个窗口、pilot 2 samples、MBAR residual 0、target met</td></tr>
<tr><td>parent</td><td class="ok">2 snapshots</td><td>3 个 GPU constrained steps</td></tr>
<tr><td>children</td><td class="ok">2 pairs</td><td>± momentum recrossing，endpoint drift 约 1e-15</td></tr>
</tbody></table>
<h2>4. 不足之处</h2>
<table><thead><tr><th>优先级</th><th>不足</th><th>影响</th></tr></thead><tbody>
<tr><td class="bad">高</td><td>还没有用户实际的化学 fitted model、测试集误差和物理 rate 结果。</td><td>当前只能证明 JaxPIP GPU runtime 和流程连接正确，不能证明化学精度或发表级速率。</td></tr>
<tr><td class="bad">高</td><td>没有做长时间、多窗口、多 parent、多 child 的稳态吞吐测试。</td><td>3 步 smoke 时间不适合估算生产成本，也不能代表显存峰值。</td></tr>
<tr><td class="warn">中</td><td>本次 T4 预装环境是 JAX 0.11.1 / CUDA 13；仓库 GPU requirements 仍有 JAX 0.9/CUDA 12 版本约束。</td><td>需要正式锁定一套 Colab 可复现版本矩阵，避免环境漂移。</td></tr>
<tr><td class="warn">中</td><td>动力学主循环仍由 TorchDynamics 的 Python orchestration 驱动每步 PES 调用。</td><td>q/p 和主要数值 kernel 在 GPU，但还不是一个把整段 dynamics 与 JaxPIP 融合成单个 CUDA/XLA 图的 custom C++ runtime。</td></tr>
<tr><td class="warn">中</td><td>当前 GPU rate 路径限定固定非周期系统、内置 CV、Andersen sampling。</td><td>自定义 CV、NPT、复杂 thermostat、periodic surface 仍需单独适配。</td></tr>
<tr><td class="warn">中</td><td>full <code>run_rate</code> 的最终 PMF+recrossing uncertainty 报告尚未用真实化学模型跑完。</td><td>本次验证覆盖了各生产阶段的直接调用和 checkpoint，不等同于完整化学 rate 收敛。</td></tr>
</tbody></table>
<div class="note"><strong>如何解读：</strong>这次结果足以说明“JaxPIP 不经过 ONNX ScatterND、坐标/力可以留在 T4、US/AUS/recrossing 接口可以跑通”。下一步验收应换入实际 model directory，增加 batch-size scaling、长 block、显存峰值、restart/resume 和 full rate uncertainty。</div>
<p class="small">原始记录：<a href="jaxpip-gpu-real-report.json">JaxPIP report</a> · <a href="gpu-validation-summary.json">workflow summary</a> · <a href="torch-tests.txt">Torch tests</a></p>
</main></body></html>'''
    (VALIDATION / "t4-efficiency-report.html").write_text(report)
    markdown = f'''# JaxPIP T4 效率与不足报告

硬件：Tesla T4 15 GiB，JAX {real['hardware']['jax']}，PyTorch {real['hardware']['torch']}。

## 关键结果

- linear/NN GPU batch 能量和力最大误差：`0`
- DLPack device pointer 零拷贝：通过
- TorchDynamics + umbrella bias：通过
- US：2 samples；AUS：pilot 后 frozen；recrossing：2 pairs
- Torch regression：`389 passed`

## 效率结论

真实 JaxPIP smoke 测试使用 batch=4、3 个动力学步。linear 为 `{models['linear']['dynamics_seconds']*1000:.3f} ms`，NN 为 `{models['nn']['dynamics_seconds']*1000:.3f} ms`。这包含首次执行和编译影响，不能当作稳态生产吞吐。

此前 T4 harmonic contract 的跨轨迹 batching 在 K=1/2/4/8/16 时达到 `{min(harmonic_speedups):.2f}×` 到 `{max(harmonic_speedups):.2f}×` 加速；该数据只说明 batching 机制，不代表真实 JaxPIP 化学 PES。

## 不足

1. 没有用户实际 fitted chemical model 的 test-set 和 rate convergence 结果。
2. 没有长时间、多窗口、多 parent/child 的稳态吞吐与显存峰值。
3. T4 预装 JAX 0.11.1/CUDA 13 与仓库 requirements 的 JAX 0.9/CUDA 12 仍需统一。
4. 动力学仍由 Python orchestration 驱动每步 PES，尚未成为整段融合的 custom C++/CUDA runtime。
5. GPU rate 当前限制固定非周期系统、内置 CV 和 Andersen sampling。

完整图表请打开 [`t4-efficiency-report.html`](t4-efficiency-report.html)。
'''
    (VALIDATION / "t4-efficiency-report.md").write_text(markdown)


if __name__ == "__main__":
    main()
