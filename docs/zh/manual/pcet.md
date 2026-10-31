# 电化学 PCET：局域 golden-rule 模型

`task="pcet"` 计算局域电极–反应物复合物的质子耦合电子转移（PCET）速率。它求解两个电子态中的一维量子质子谱，
再结合振动态重叠、热布居、环境重组和电极 Fermi 分布，得到一级还原与氧化速率。

PCET 是独立的模型工作流，不做环聚合物传播。其速率是单个局域复合物的一级速率，单位 `s^-1`。

## 运行示例扫描

在仓库根目录执行：

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/manual-pcet
pyrpmd analyze runs/manual-pcet
pyrpmd resume runs/manual-pcet
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/pcet/harmonic_scan.inp)定义两个位置和曲率不同的谐振子势阱，
扫描 250/300/350 K、三个电极电势和 H、D 两种同位素：共 18 个条件，约 2 秒完成。

从摘要读取每个扫描点：

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

`resume` 校验运行身份和每个已完成扫描点的 checksum，复用这些结果并计算其余条件。更改模型或数值设置时使用新的运行目录。
`analyze` 读取保存的摘要。

## 模型

| 组成 | 模型 |
| --- | --- |
| 电子态 | 一对局域 `oxidized` / `reduced` 态 |
| 量子核 | 共同一维坐标上的一个转移 H、D 或 T |
| 质子势 | 每个电子态一条谐振子或表格势 |
| 耦合 | 常数电子耦合 |
| 环境 | 单一重组能的经典 Gaussian 环境 |
| 电极 | 平衡 Fermi 占据、常数态密度的宽带金属 |
| 归一化 | `local_first_order`，速率单位 `s^-1` |

温度、电势和同位素按笛卡尔积扫描。所有同位素共用同一对势曲线；PyRPMD 对每个质量重新计算谱、布居和重叠。
同一条件下同时含 H 和 D 时，分别对还原和氧化计算 KIE $k_H/k_D$。溶剂、重组能和酸碱平衡对所有同位素相同。

## 输入字段与单位

PCET 使用顶层温度和 `[pcet]` 模型，不使用 `[system]`、`[potential]`、`beads`、`[dynamics]` 或 `[sampling]`。

| 字段 | 含义 |
| --- | --- |
| `temperature` | 单个温度或列表，如 `["250 K", "300 K"]` |
| `pcet.method` | `"golden_rule"` |
| `pcet.normalization` | `"local_first_order"` |
| `pcet.isotopes` | 如 `["H", "D"]` |
| `pcet.driving_force` | 参考电势下两势阱最低点的自由能差，reduced − oxidized |
| `pcet.reorganization_energy` | 正的环境重组能，如 `"0.7 eV"` |
| `pcet.electronic_coupling` | 电子耦合，如 `"0.001 eV"` |
| `pcet.electrode.model` | `"wide_band"` |
| `pcet.electrode.density_of_states` | 电子态密度，如 `"1 eV^-1"` |
| `pcet.electrode.potential_reference` | 电势标尺标签，如 `"SHE"`（必需） |
| `reference_potential_V` | `driving_force` 对应的电势，单位 V |
| `potentials_V` | 扫描的电势，单位 V |
| `pcet.proton.grid_min` / `grid_max` | 共同的一维质子坐标域 |
| `pcet.proton.grid_points` | 含两端点的均匀网格点数 |
| `pcet.proton.states` | 每个电子态保留的振动本征态数 |

上表中的两个电势字段属于 `pcet.electrode` 子表。态密度包含模型计入的全部电子态，程序不另加自旋因子。

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

两条势共用一个坐标和原点，因为 Franck–Condon 重叠取决于它们的相对位置。

已有两态的一维能量扫描时，使用表格势：

```toml
[pcet.proton.oxidized]
kind = "tabulated"
file = "oxidized.csv"
coordinate_unit = "angstrom"
energy_unit = "eV"
```

`reduced` 子表结构相同。表格至少五行，每行为坐标和能量，坐标严格递增；`.csv` 文件以逗号分隔，其他文件以空白分隔，
`#` 开始注释。路径相对于输入文件。每个表格覆盖整个求解域，线性插值，最低点位于网格内部。

## 能量零点与电势符号

`driving_force` 是参考电势下两势阱最低点之间的环境自由能差：

$$
\Delta G_{\min,\mathrm{ref}}=G_{\min,R}-G_{\min,O}.
$$

求解器从每条势中减去其最低值，因此本征值从各自势阱底部计量，包含质子零点能。这里填写最低点之间的差；
若填入已含质子振动自由能的总反应自由能，会重复计入。表格势的绝对能量偏移不影响结果。

记 e 为元电荷，电势依赖为

$$
\Delta G_{\min}(U)=\Delta G_{\min,\mathrm{ref}}+e(U-U_{\mathrm{ref}}).
$$

电势越正，还原越不利。PyRPMD 自动施加这一偏移，因此 `driving_force` 应对应 `reference_potential_V`。
`potential_reference` 是标签；在 SHE、RHE 或真空能级等标尺之间换算后再填入电势。

## 从振动态到速率

在共同坐标域上，求解器使用 Dirichlet 边界求解

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V_a(x)\right]\chi_n^a(x)
=E_n^a\chi_n^a(x),\qquad a\in\{O,R\}.
$$

重叠 $S_{\mu\nu}=\langle\chi^O_\mu|\chi^R_\nu\rangle$ 给出振电耦合 $V_{\mu\nu}=V_{\mathrm{el}}S_{\mu\nu}$。速率对所有保留的
初末态求和，以初态热布居加权，并对电极电子能量积分。速率与 $\rho|V_{\mathrm{el}}|^2$ 成正比。

正反向速率满足

$$
\frac{k_{\mathrm{red}}}{k_{\mathrm{ox}}}
=e^{-\beta\Delta G_{\min}(U)}\frac{Z_R}{Z_O},
\qquad Z_a=\sum_n e^{-\beta E_n^a},
$$

因此两阱曲率的差异通过质子配分函数进入结果。PyRPMD 用配对的积分核稳定地计算极小的逆向速率，并把详细平衡残差
作为一致性检查报告。

## 输出与数值收敛

| 文件 | 内容 |
| --- | --- |
| `input/resolved.json`、`input/model.json` | 解析后的输入、网格与两条势 |
| `rates.csv` | 温度、电势、标尺、同位素、驱动力、正反向速率与 log 速率 |
| `proton_spectra.json` | 质量、能级、能量原点与边界概率诊断 |
| `proton_wavefunctions.npz` | 各同位素和电子态的网格、势、能级与离散波函数 |
| `pcet_details.json` | 各点的布居、振动态对贡献与数值检查 |
| `results/summary.json` | 扫描摘要、`rates` 与 `kinetic_isotope_effects` |
| `checkpoints/pcet-*.json` | 每个已完成扫描点一个，含身份与完整性数据 |

波函数列按 `sum(abs(psi)**2)=1` 归一化；除以网格间距得到单位长度上的密度。

`reduction_rate_s` 和 `oxidation_rate_s` 的单位为 `s^-1`；`log_reduction_rate_s` 和 `log_oxidation_rate_s` 是其自然对数。
小于浮点下限的速率显示为 0，同时给出有限的 log 速率。耦合恰为零时速率为零，KIE 无定义。

`pcet.numerics.convergence_check=true` 将网格 M 与加密网格 `2*M-1` 比较，增加态数，并检查初态热布居以及反应通道中
边界和高能态的贡献。非零速率通过 log 速率比较，因此极小速率与较大速率使用相同的相对容差。

| 数值字段 | 含义 |
| --- | --- |
| `quadrature_rtol` / `quadrature_atol` | 电极 log-scaled 积分的容差，缺省 `1e-9` / `1e-11` |
| `relative_tolerance` | 网格与态数比较的相对容差 |
| `absolute_tolerance_s` | 速率恰为零时的容差 |
| `numerical_convergence` | 数值检查是否通过 |
| `physical_applicability_unverified` | 始终为 `true`；模型对真实体系的适用性在 PyRPMD 之外评估 |

对每个温度、电势和同位素，通过扩大坐标域、加密网格、增加态数和收紧积分容差来收敛。坐标域与网格分开检查，
因为过小的盒子会截断波函数。

## 从局域速率到测量量

速率按单个局域复合物归一化。电流、异相速率常数和面积归一化的量，由这些速率结合复合物覆盖度、界面面积、
反应网络和传质得到。

方程与方法来源见 [PCET 专题](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/pcet.md)。在谐振子示例模型上，
PyRPMD 与 Hammes-Schiffer 课题组的 pyPCET 在全部 12 个正反向速率上的偏差不超过 0.062%，见
[模型验证记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)。
