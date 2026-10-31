# 电化学 PCET：局域弱耦合速率模型

恒电位有效势、显式电子数平衡采样和 HER 表面电流见新增的
[电化学接口](electrochemistry.md)。这些组件复用现有内核，与本页的局域
golden-rule 模型具有不同的系综与速率归一化。

`task = "pcet"` 计算一个固定局域电极–反应物复合物的质子耦合电子转移
（PCET）。第一版使用一个转移质子、两个电子态、弱电子/振电耦合、经典
Gaussian 环境和理想宽带金属电子储库。质子振动态由一维有限差分本征值
问题求得，再与电极 Fermi 分布和环境重组核积分，得到还原/氧化一级速率
常数，单位为 **s⁻¹**。

这是 PyRPMD 中独立的 golden-rule 模型工作流。它不执行环聚合物实时传播，
也不使用气相双分子 RPMD 的远区通量前因子。现有 RPMD、FES 和 dRPMD
工作流保持各自的物理定义。

## 适用范围

| 模型内容 | 当前约定 |
| --- | --- |
| 化学状态 | 一对 oxidized/reduced 局域电子态，一个转移质子 |
| 质子量子效应 | 两个一维势中的振动能级、热布居与波函数重叠，分别求 H/D |
| 电子耦合 | 与电子能量和质子坐标无关的常数 `electronic_coupling` |
| 环境 | 单一、正的经典 Gaussian 重组能 `reorganization_energy` |
| 金属电极 | 常数态密度 `density_of_states`，平衡 Fermi 占据，单个电子储库 |
| 扫描 | 温度和电势的笛卡尔积；各条件的正反向速率及 H/D KIE |
| 归一化 | `local_first_order`，已经形成的同一个局域复合物，s⁻¹ |

该模型未包含显式电极/溶剂构型采样、PBC、电子数或电极电荷采样、双电层、
供受体距离分布、反应物扩散/吸附、酸碱物种分布、自动 pH 依赖及多步反应
网络。它也不描述强电子耦合、溶剂控制区或一般非绝热实时动力学。输入耦合
较小是用户的模型选择；数值收敛报告本身不能证明其物理适用性。

这里的速率不能直接作为气相 cm³/s、通常意义的异相 cm/s 或电流密度。
计算电流还需要明确界面复合物的浓度/覆盖度、面积归一化与反应网络；
需要时还应连接传质模型。

## 能量零点和电势符号

电子态标签固定为 `oxidized`（O）和 `reduced`（R）。`driving_force`
表示在参考电势下，两态势阱最低点对应的环境自由能差：

\[
\Delta G_{\min,\mathrm{ref}}=G_{\min,\mathrm R}-G_{\min,\mathrm O}.
\]

该输入**不包含转移质子的零点能和热振动自由能**。两个质子势曲线分别
去除自身最低值，质子本征能量在各自势阱最低点之上计量，并包含求解得到
的零点能。任何来自别的计算的反应自由能，使用前都需要与这一分解一致；
不能把已经含该质子振动自由能的总反应自由能直接填入该字段。

设电子电荷绝对值为 \(e>0\)，电势 \(U\) 与参考电势使用同一标尺，
则本模型约定：

\[
\Delta G_{\min}(U)=\Delta G_{\min,\mathrm{ref}}+e(U-U_\mathrm{ref}).
\]

因此，电极电势增大（变得更正）使还原的自由能变化增大，即还原在热力学上更不利。
电子能量 \(\epsilon\) 相对于当前电极 Fermi 能级计量，
\(f(\epsilon)=[1+\exp(\beta\epsilon)]^{-1}\)，其中
\(\beta=(k_\mathrm BT)^{-1}\)。不要在输入的 `driving_force` 中再次加入
同一个电势修正。

`potential_reference` 是必需明确的标尺标签；程序记录该标签，不自动做
SHE/RHE 转换或绝对真空能级标定。改变标签而保持数值不变，不代表完成了
物理转换。输入没有 pH 字段，也不会根据电势标签推断质子供体或活度。

## 质子本征态和速率表达式

在共同的一维坐标 \(x\) 上，对同位素质量 \(m\) 分别求解：

\[
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V_a(x)\right]
\chi^a_n(x)=E^a_n\chi^a_n(x),\qquad a\in\{\mathrm O,\mathrm R\}.
\]

网格使用 Dirichlet 边界，波函数在两端为零。势可以是位移谐振子或表格
曲线。两态使用同一个坐标、网格和内积测度，计算
\(S_{\mu\nu}=\langle\chi^{\mathrm O}_\mu|\chi^{\mathrm R}_\nu\rangle\)。
同位素替换保留同一对势曲线，改变质量后重新求全部谱、布居和重叠；
不使用预设 KIE 或单一质量比例修正。

同位素质量沿用 `System` 的中性原子质量约定：H、D、T 分别为
1.00782503223、2.01410177812、3.01604928199 amu。输入可选择这三种标签；
自动 KIE 输出对应同时包含 H 和 D 的扫描。同位素替换只改变这一转移核，
未加入 H₂O/D₂O 溶剂替换或环境参数变化导致的同位素效应。

本实现使用电子非绝热、常耦合近似
\(V_{\mu\nu}=V_\mathrm{el}S_{\mu\nu}\)。定义

\[
g_\lambda(y)=\frac{1}{\sqrt{4\pi\lambda k_\mathrm BT}}
\exp\left[-\frac{(y+\lambda)^2}{4\lambda k_\mathrm BT}\right],\qquad
A_{\mu\nu}=\Delta G_{\min}(U)+E^{\mathrm R}_\nu-E^{\mathrm O}_\mu.
\]

宽带金属态密度为 \(\rho\)，两方向的模型速率为

\[
k_\mathrm{red}=\frac{2\pi}{\hbar}\rho|V_\mathrm{el}|^2
\sum_{\mu\nu}p^{\mathrm O}_\mu|S_{\mu\nu}|^2
\int_{-\infty}^{\infty}d\epsilon\,
f(\epsilon)g_\lambda(A_{\mu\nu}-\epsilon),
\]

\[
k_\mathrm{ox}=\frac{2\pi}{\hbar}\rho|V_\mathrm{el}|^2
\sum_{\mu\nu}p^{\mathrm R}_\nu|S_{\mu\nu}|^2
\int_{-\infty}^{\infty}d\epsilon\,
[1-f(\epsilon)]g_\lambda(-A_{\mu\nu}+\epsilon).
\]

这里 \(p^a_n=\exp(-\beta E^a_n)/Z_a\)，
\(Z_a=\sum_n\exp(-\beta E^a_n)\)。`density_of_states` 的单位是能量的
倒数；它应包含模型中实际计入的电子态，程序不另乘隐含的自旋简并因子。
速率与 \(\rho|V_\mathrm{el}|^2\) 成正比；减小耦合时应恢复 golden-rule
的平方标度。

同一模型应满足详细平衡：

\[
\frac{k_\mathrm{red}}{k_\mathrm{ox}}
=e^{-\beta\Delta G_{\min}(U)}\frac{Z_\mathrm R}{Z_\mathrm O}.
\]

因此，单独比较 \(e^{-\beta\Delta G_{\min}}\) 会漏掉质子配分函数的贡献。
如果两势的曲率不同，这个比值也可以随同位素变化。还原与氧化的 KIE
分别定义为同一条件下 \(k^\mathrm H/k^\mathrm D\)。

为保留极小的逆向速率，程序在 log 空间对有利方向的 Fermi/Gaussian 核
积分，再用配对核的详细平衡恒等式构造另一方向。输出的详细平衡残差
因此是内部一致性诊断，不是独立的物理模型验证。

## 输入与运行

完整可运行输入为
[`examples/pcet/harmonic_scan.inp`](../examples/pcet/harmonic_scan.inp)。
这个教学模型使用位移和曲率都不同的谐振子，没有真实分子或电极材料归属。

```bash
pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/pcet-harmonic
pyrpmd analyze runs/pcet-harmonic
pyrpmd resume runs/pcet-harmonic
```

顶层 `temperature` 可以是单个温度或温度列表。PCET 不需要 `[system]`、
`[potential]`、`beads`、`[dynamics]` 或 `[sampling]`；它使用下面的独立
模型输入，不调用分子 PES 插件。

| 字段 | 约定与示例 |
| --- | --- |
| `pcet.method` | `"golden_rule"` |
| `pcet.normalization` | `"local_first_order"` |
| `pcet.isotopes` | 例如 `["H", "D"]` |
| `pcet.driving_force` | 最低点差 \(\Delta G_{\min,\mathrm{ref}}\)，例如 `"-0.1 eV"` |
| `pcet.reorganization_energy` | 正的环境重组能，例如 `"0.7 eV"` |
| `pcet.electronic_coupling` | 常数电子耦合，例如 `"0.0001 eV"` |
| `pcet.electrode.model` | `"wide_band"` |
| `pcet.electrode.density_of_states` | 例如 `"1 eV^-1"` |
| `pcet.electrode.potential_reference` | 电势标尺标签，例如 `"SHE"` |
| `pcet.electrode.reference_potential_V` | `driving_force` 的参考电势，数值按 V |
| `pcet.electrode.potentials_V` | 扫描的电势列表，数值按 V |
| `pcet.proton.grid_min` / `grid_max` | 共同质子坐标域，例如 `"-1.5 angstrom"` / `"1.5 angstrom"` |
| `pcet.proton.grid_points` / `states` | 网格点数和保留本征态数 |
| `pcet.proton.oxidized` / `reduced` | 各自的一维质子势，见下文 |
| `pcet.numerics.quadrature_rtol` / `quadrature_atol` | 自适应电极能量积分的相对/绝对容差，默认 `1e-9` / `1e-11`；作用于 log-scaled 积分，不是最终速率的 s⁻¹ 容差 |
| `pcet.numerics.convergence_check` | 是否运行内置数值对照 |
| `pcet.numerics.relative_tolerance` | 数值对照的相对容差 |
| `pcet.numerics.absolute_tolerance_s` | 精确零速率的绝对比较容差，按 s⁻¹；不覆盖非零物理速率的相对误差 |

谐振子势写为：

```toml
[pcet.proton.oxidized]
kind = "harmonic"
center = "-0.25 angstrom"
force_constant = "8 eV/angstrom^2"
```

对应 \(V(x)=\tfrac12 k(x-x_0)^2\)。另一电子态使用 `reduced` 子表。
两个势的坐标应使用同一个物理定义，不允许对两列独立重新定位原点后
仍把它们当作同一质子坐标。

表格势使用 `kind="tabulated"`、`file`、`coordinate_unit` 和
`energy_unit`。文件是至少五行的两列数值（坐标、能量），坐标严格递增；
`.csv` 使用逗号，其余后缀使用空白分隔，可用 `#` 写注释。两个表格都必须
覆盖整个质子网格；程序线性插值，不外推。最低点不得位于网格边界。
例如：

```toml
[pcet.proton.oxidized]
kind = "tabulated"
file = "oxidized.csv"
coordinate_unit = "angstrom"
energy_unit = "eV"
```

文件路径按输入文件所在目录解析。能量列用于质子势形状，
两个电子态之间的最低点自由能差仍只由 `driving_force` 提供；表格中任意
加上的整体能量常数不能替代该输入。

## 结果与数值诊断

运行结果包含每个温度、电势和同位素的 `reduction_rate_s` 和
`oxidation_rate_s`，以及 H/D 的分方向 KIE。对应的
`log_reduction_rate_s` 和 `log_oxidation_rate_s` 保留以 s⁻¹ 为单位的速率
数值的自然对数。
详细输出保留质子谱、热布居、振动态对贡献和数值诊断；顶层运行
记录继续保存输入身份和结果摘要。所有速率均按局域一级归一化解释。

| 运行目录文件 | 内容 |
| --- | --- |
| `input/resolved.json`、`input/model.json` | 规范配置、质子网格/势曲线及能量原点；PCET 不生成虚构的分子 `system.json` |
| `rates.csv` | 温度、电势、标尺、同位素、驱动力和两方向速率/对数速率，以及各行数值收敛状态 |
| `proton_spectra.json` | 各同位素、各电子态的能级、质量、能量原点、边界诊断 |
| `proton_wavefunctions.npz` | 网格、势、能级及离散归一化波函数；每列满足 `sum(abs(psi)**2)=1` |
| `pcet_details.json` | 每个扫描点的振动态对贡献、布居、速率与数值检查详情 |
| `summary.json`、`results/summary.json` | 工作流摘要；后者另带软件和运行信息。`kinetic_isotope_effects` 保存 H/D 两方向 KIE 和其对数 |
| `checkpoints/pcet-*.json` | 已完成扫描点及身份/完整性检查信息 |

`resume` 复用身份与校验匹配的已完成扫描点，计算剩余扫描点。改变物理
参数、势曲线或数值设置时应使用新运行目录；这不是延长动力学轨迹。
`analyze` 读取已存摘要，PCET 不支持 FES 的 `--recompute` 重加权选项。
所有扫描由串行模型工作流执行，不使用轨迹并行或传播后端配置。

`convergence_check=true` 比较同一坐标域内的加密网格（由 M 点到 2M−1 点）
和增加保留态数（最多加倍且不超过网格内部态数），并报告边界诊断。
非零物理速率使用对数速率计算相对变化；即使普通浮点速率下溢为零，仍必须
通过相对误差检查。`absolute_tolerance_s` 仅适用于精确零速率（对数为空），
因此缩小电子耦合不会掩盖网格或态截断误差，也不会使 H/D 比值出现虚假的收敛。
电极积分采用 log-scaled 自适应求积以保留很小速率的信息。
`numerical_convergence` 汇总已执行的数值对照，
`physical_applicability_unverified=true` 单独保留物理适用性尚未验证的状态。
计算成功和数值对照通过具有不同含义。数值报告只涉及实际执行的对照，
不代替下列模型与离散参数的研究：

1. 扩大质子坐标域，确认边界未截断相关波函数；仅增加点数不能消除有限盒子误差。
2. 固定物理坐标域加密网格，检查有限差分能级、重叠和最终速率。
3. 增加保留态数，同时检查初态热布居和产物可达振动态；初态激发态很少并不说明只需一个末态。
4. 收紧电极自适应积分容差，检查各温度、电势与同位素的正反向速率。
5. 检查详细平衡和耦合平方标度，再评估弱耦合、Gaussian 环境、宽带与固定局域构型假设。

H 与 D 要分别检查。很小的非零速率保留对数值，并以对数差计算稳定的
相对误差；精确零耦合下的 H/D 比值没有定义。输入参数本身的不确定度和
模型误差不属于确定性求积容差。

独立解析极限、工作流测试之外，固定几何教学模型还与作者公开的 pyPCET
实现进行了交叉验证。输入映射、参考提交、运行脚本和完整数值记录见
[PCET 模型验证记录](validation/pcet-model-2026-10-30/README.md)。

## 理论来源与后续扩展边界

- Charulatha Venkataraman, Alexander V. Soudackov, Sharon Hammes-Schiffer,
  “Theoretical Formulation of Nonadiabatic Electrochemical Proton-Coupled
  Electron Transfer at Metal–Solution Interfaces,” *J. Phys. Chem. C* **112**,
  12386–12397 (2008), [doi:10.1021/jp802171y](https://doi.org/10.1021/jp802171y).
  该文建立更广的界面非绝热 PCET 速率理论；本工作流只实现上述固定局域
  构型、常电子耦合、宽带和经典 Gaussian 环境子模型，不声称覆盖论文
  中的全部供受体振动或界面距离平均。
- Zachary K. Goldsmith, Yan Choi Lam, Alexander V. Soudackov,
  Sharon Hammes-Schiffer, “Proton Discharge on a Gold Electrode from
  Triethylammonium in Acetonitrile: Theoretical Modeling of Potential-Dependent
  Kinetic Isotope Effects,” *J. Am. Chem. Soc.* **141**, 1084–1090 (2019),
  [doi:10.1021/jacs.8b11826](https://doi.org/10.1021/jacs.8b11826).
  该研究说明电势相关 KIE 与振电通道的物理意义；本仓库教学参数未拟合或
  复现该实验体系。
- Menglin Sun, Bin Jin, Xiaolong Yang, Shenzhen Xu,
  “Probing nuclear quantum effects in electrocatalysis via a machine-learning
  enhanced grand canonical constant potential approach,” *Nature Communications*
  **16**, 3600 (2025),
  [doi:10.1038/s41467-025-58871-7](https://doi.org/10.1038/s41467-025-58871-7).
  该文用 GC-HMC 与 PIMC 进行恒电势界面采样和量子自由能计算，是后续
  显式界面模块的参考。它不是本实现的算法，也不提供将 Monte Carlo 步数
  解释为真实 PCET 反应时间的依据。
