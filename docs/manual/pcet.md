# 电化学 PCET：局域 golden-rule 模型

`task="pcet"` 计算一个已经形成的局域电极–反应物复合物的质子耦合电子转移（PCET）。它求解两个电子态中的一维量子质子谱，再结合振动态重叠、热布居、环境重组和电极 Fermi 分布，得到还原与氧化一级速率。

这是独立的弱耦合模型工作流。它没有环聚合物珠子或实时传播，不使用气相 RPMD 的远区前因子，也不把模型结果直接解释成电流密度。理解这一归一化是选择输入和比较实验的起点。

## 运行教学扫描

从仓库根目录运行：

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/manual-pcet
pyrpmd analyze runs/manual-pcet
pyrpmd resume runs/manual-pcet
```

[完整输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/pcet/harmonic_scan.inp)使用两个位移、曲率不同的谐振子，扫描 250/300/350 K、三个电势和 H/D 两种同位素，总计 18 个条件。它没有对应的真实分子或电极材料，适合学习模型定义、输出和数值检查。

运行完成后，可以直接读取每个扫描点：

```python
import json
from pathlib import Path

summary = json.loads(Path("runs/manual-pcet/results/summary.json").read_text())
print(summary["completed_scan_points"])
print(summary["numerical_convergence"])

for row in summary["rates"]:
    print(row["temperature_K"], row["potential_V"], row["isotope"],
          row["reduction_rate_s"], row["oxidation_rate_s"])

for entry in summary["kinetic_isotope_effects"]:
    print(entry["temperature_K"], entry["potential_V"],
          entry["reduction_KIE"], entry["oxidation_KIE"])
```

`resume` 校验运行身份与每个已完成扫描点的 checksum，然后复用结果并继续剩余条件。它不是延长动力学轨迹。改变物理参数、势曲线或数值设置时应新建运行目录。`analyze` 读取保存摘要；PCET 不支持 FES 的 `--recompute` 去偏置分析。

## 模型包含什么

| 成分 | 当前模型 |
| --- | --- |
| 电子态 | 一对 `oxidized` / `reduced` 局域态 |
| 量子核 | 一个转移核，共同一维坐标中的 H、D 或 T |
| 质子势 | 各电子态中的谐振子或表格势 |
| 耦合 | 与质子坐标及电子能量无关的常数电子耦合 |
| 环境 | 单一正重组能，经典 Gaussian 环境 |
| 电极 | 平衡 Fermi 占据、常数态密度的理想宽带金属储库 |
| 归一化 | `local_first_order`，速率单位 `s^-1` |

温度、电势和同位素列表按笛卡尔积扫描。H/D/T 使用同一对势曲线，仅改变转移核质量后重新计算谱、布居和重叠；不预设一个质量比例来修正速率。自动 KIE 要求同一条件下包含 H 和 D，定义为 $k_H/k_D$，还原与氧化分别计算。

该同位素替换不自动改变溶剂、环境重组能或酸碱平衡。因此它不包含 H₂O/D₂O 替换可能带来的全部实验同位素效应。

## 输入模型：单位和字段

PCET 不需要 `[system]`、`[potential]`、`beads`、`[dynamics]` 或 `[sampling]`，也不调用分子 PES 插件。它使用顶层温度与独立的 `[pcet]` 模型：

| 字段 | 含义 |
| --- | --- |
| `temperature` | 单个温度或温度列表，如 `["250 K", "300 K"]` |
| `pcet.method` | `"golden_rule"` |
| `pcet.normalization` | `"local_first_order"` |
| `pcet.isotopes` | 如 `["H", "D"]` |
| `pcet.driving_force` | 参考电势下 reduced−oxidized 的势阱最低点自由能差 |
| `pcet.reorganization_energy` | 正的环境重组能，如 `"0.7 eV"` |
| `pcet.electronic_coupling` | 常电子耦合，如 `"0.001 eV"` |
| `pcet.electrode.model` | `"wide_band"` |
| `pcet.electrode.density_of_states` | 电子态密度，如 `"1 eV^-1"` |
| `pcet.electrode.potential_reference` | 必需的电势标尺标签，如 `"SHE"` |
| `reference_potential_V` | driving force 对应的参考电势，数值按 V |
| `potentials_V` | 扫描电势列表，数值按 V |
| `pcet.proton.grid_min` / `grid_max` | 两电子态共同的一维质子坐标域 |
| `pcet.proton.grid_points` | 包含两端点的均匀网格点数 |
| `pcet.proton.states` | 每态保留的振动本征态数 |

上表的最后两个电势字段属于 `pcet.electrode` 子表。态密度应包含模型实际计入的电子态，程序不会再加入一个隐含自旋简并因子。

### 谐振子与表格势

以下子表定义 $V(x)=\tfrac12k(x-x_0)^2$：

```toml
[pcet.proton.oxidized]
kind = "harmonic"
center = "-0.2 angstrom"
force_constant = "4 eV/angstrom^2"

[pcet.proton.reduced]
kind = "harmonic"
center = "0.2 angstrom"
force_constant = "5 eV/angstrom^2"
```

两个势必须使用同一个物理坐标和原点。各自重新移动坐标后得到的两个势，不能在未恢复相对位置的情况下计算 Franck–Condon 重叠。

若已有两态的一维能量扫描，可使用表格势：

```toml
[pcet.proton.oxidized]
kind = "tabulated"
file = "oxidized.csv"
coordinate_unit = "angstrom"
energy_unit = "eV"
```

另一电子态用相同结构的 `reduced` 子表。文件至少五行、两列（坐标和能量），坐标严格递增；CSV 用逗号，其他后缀用空白分隔，可用 `#` 写注释。文件路径相对于输入文件。两个表都必须覆盖完整求解域，程序线性插值而不外推；势阱最低点不能落在网格边界。

## 势阱零点与电势符号

`driving_force` 定义为参考电势下势阱最低点的环境自由能差

$$
\Delta G_{\min,\mathrm{ref}}=G_{\min,R}-G_{\min,O}.
$$

它不含转移质子的零点能或热振动自由能。求解器从每条质子势中去除其最低值，本征能级在相应最低点之上计量，包含量子零点能。若外部计算给的是已经含此质子振动贡献的总反应自由能，不能直接再填入 `driving_force`，否则会重复计入。

令 e 为电子电荷的正绝对值，电势修正为

$$
\Delta G_{\min}(U)=\Delta G_{\min,\mathrm{ref}}+e(U-U_{\mathrm{ref}}).
$$

电势变正会提高本模型的还原自由能变化，使还原更不利。不要在 driving force 中再次加入相同电势修正。表格势的任意整体能量常数也不能替代两电子态最低点差。

`potential_reference` 仅记录标尺名称，不自动做 SHE/RHE、pH 或真空能级换算。若改变标尺，需要同时完成相应物理数值转换。当前输入没有自动 pH、质子供体活度或酸碱物种分布模型。

## 从振动态到速率

共同坐标域中，求解器采用 Dirichlet 边界求解

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V_a(x)\right]\chi_n^a(x)
=E_n^a\chi_n^a(x),\qquad a\in\{O,R\}.
$$

两态的重叠 $S_{\mu\nu}=\langle\chi^O_\mu|\chi^R_\nu\rangle$ 给出振电耦合 $V_{\mu\nu}=V_{\mathrm{el}}S_{\mu\nu}$。计算对所有保留的初末态对求和，用初态热布居加权，并与电极电子能量核积分。速率与 $\rho|V_{\mathrm{el}}|^2$ 成正比，这是弱耦合模型的一项可检查标度。

同一模型满足

$$
\frac{k_{\mathrm{red}}}{k_{\mathrm{ox}}}
=e^{-\beta\Delta G_{\min}(U)}\frac{Z_R}{Z_O},
\qquad Z_a=\sum_n e^{-\beta E_n^a}.
$$

两阱曲率不同会改变质子配分函数比，因此不能仅拿 $e^{-\beta\Delta G_{\min}}$ 比较正反向速率。实现使用配对核的详细平衡关系稳定计算极小的逆向速率；其详细平衡残差是内部一致性检查，不是独立的物理模型验证。

## 输出文件与数值收敛

| 文件 | 内容 |
| --- | --- |
| `input/resolved.json`、`input/model.json` | 规范输入、网格与两态势；不生成虚构分子 System |
| `rates.csv` | 温度、电势、标尺、同位素、驱动力、正反向速率及 log 速率 |
| `proton_spectra.json` | 质量、能级、能量原点与边界概率诊断 |
| `proton_wavefunctions.npz` | 各同位素/电子态的网格、势、能级与离散波函数 |
| `pcet_details.json` | 各点的布居、振动态对贡献和详细数值检查 |
| `results/summary.json` | 扫描摘要、`rates` 和 `kinetic_isotope_effects` |
| `checkpoints/pcet-*.json` | 各完成扫描点及身份、完整性信息 |

波函数列按 `sum(abs(psi)**2)=1` 离散归一化。若转换成连续波函数密度，需要考虑网格间距，不能直接把文件数组平方解释为每单位长度的密度。

`reduction_rate_s` / `oxidation_rate_s` 单位为 `s^-1`。对应 `log_reduction_rate_s` / `log_oxidation_rate_s` 是该速率数值的自然对数。当普通浮点值下溢为零时，应查看有限 log 值；这与精确零耦合造成的零速率不同。精确零耦合下 H/D 比值没有定义。

`pcet.numerics.convergence_check=true` 比较原网格 M 与加密网格 `2*M-1`，并增加保留态数；还检查初态热布居及实际反应通道加权的边界/高态贡献。非零速率使用 log 速率差检查相对变化，不能用一个较大的绝对容差掩盖微小速率的巨大相对误差。

| 数值字段 | 意义 |
| --- | --- |
| `quadrature_rtol` / `quadrature_atol` | 电极 log-scaled 自适应积分容差，缺省 `1e-9` / `1e-11` |
| `relative_tolerance` | 网格和态数对照的相对误差门槛 |
| `absolute_tolerance_s` | 精确零速率的绝对比较容差，不替代非零速率相对检查 |
| `numerical_convergence` | 已执行的数值检查是否通过 |
| `physical_applicability_unverified` | 保留为 true；数值通过没有验证模型的物理适用性 |

仍应独立扩大坐标域、加密网格、增加态数、收紧积分容差，并分别检查各温度、电势和同位素。仅加密有限盒子中的网格，不能消除波函数被盒子截断的误差；初态高能态布居很低，也不能证明高能末态对反应无贡献。

## 与真实电化学问题比较

输出归一化到一个局域复合物。要计算电流、异相速率或面积归一化量，还需要明确复合物浓度/覆盖度、界面面积、反应网络及传质。当前模型没有显式电极/溶剂采样、恒电势电子数采样、双电层、供受体距离分布或强耦合溶剂控制动力学。

完整方程和方法来源见 [PCET 专题](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/pcet.md)。固定教学模型与作者公开 pyPCET 的独立对照，记录在[模型验证档案](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)；该对照验证指定模型的实现，不构成真实电极实验的复现。
