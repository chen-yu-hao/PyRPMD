# 电极 PCET 模型示例

这些例子演示如何把电子库、电子数相关 PES 和表面反应网络接入 PyRPMD。
解析势及速率参数只用于验证接口与统计约定，没有拟合真实电极材料。
物理定义、公式和开源参考见 [电极模块说明](../../docs/electrochemistry.md)。

`fes2d.inp` 的 Pt 标签仅标记固定基底的参考几何和显式质量；实际势由解析
周期余弦项及高度–电子数二次型组成，不含真实 Pt–水相互作用或成断键通道。
两个 FES 坐标是横向位移与高度，这个短 pilot 不代表 Pt(111) PCET 反应自由能。

| 文件 | 演示内容 |
| --- | --- |
| `toy_electrode.py` | 具有解析电子数导数和核力的充电模型，用于验证绝热恒电位求解 |
| `fes2d.inp` | 复用已有二维 umbrella/FES 工作流的恒电位有效势示例 |
| `sample_grand_canonical.py` | 共享电子数的巨正则平衡采样与统计诊断 |
| `her_scan.py` | Volmer–Heyrovsky–Tafel 模型中覆盖度、H₂ 生成及电流的后处理 |

在仓库根目录运行：

```bash
pyrpmd validate examples/electrochemistry/fes2d.inp --check-pes
pyrpmd run examples/electrochemistry/fes2d.inp --output runs/electrode-fes
python examples/electrochemistry/sample_grand_canonical.py
python examples/electrochemistry/her_scan.py
```

`fes2d.inp` 使用每颗珠子独立电子最小化的绝热有效势
\(\Omega(\mathbf R)=\min_N[E(\mathbf R,N)-\mu N]\)。
`sample_grand_canonical.py` 则采用一条环聚合物共享一个电子数的平衡模型，
其电子数权重含平均珠子能量。两者不是同一个近似，不能将它们的统计量或
初态静默混用。MC sweep 不是物理时间；这个示例不计算电子转移等待时间。

巨正则示例默认舍弃 500 sweeps，再记录 2000 sweeps；输出 JSON 对照质心高度
与电子数的样本均值、协方差和解析无界 Gaussian 参考。有限电子数边界远离
这个模型的主要概率区域，但样本仍有相关性，默认长度不证明统计收敛。
`--sweeps`、`--burn-in`、`--nuclear-steps` 和 `--seed` 可显式设置。

`her_scan.py` 处理已给定的局域反应常数及活度。它不从解析 FES 自动提取真实
Volmer、Heyrovsky 或 Tafel 速率。与实验电流比较还需要匹配实际位点面密度、
反应物活度和微观速率常数；这里不包含传质或双电层模型。
CSV 中电位使用教学模型参考；电流采用阳极为正、阴极为负的约定。

若研究弱电子耦合 PCET 的振电速率和 H/D 同位素效应，使用已有示例：

```bash
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic
```

它调用已验证的局域 golden-rule 模型，不调用本目录的绝热 PES。
二维质子键差自由能可以沿用 FES；现有 `surface_rate` 仅支持线性质心坐标，
不能直接把非线性质子键差交给表面速率入口。
