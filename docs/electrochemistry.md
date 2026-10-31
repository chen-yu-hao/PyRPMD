# 电极 PCET：电子数相关势能与恒电位模型

电极模块把电子库和电子数相关势能接入 PyRPMD 已有的 PES、周期体系和
自由能工作流。它提供两种有明确区别的模型：逐构型绝热恒电位有效势，
以及核环聚合物共享电子数的巨正则 HMC/MC 平衡采样。两者的统计分布不同，
不能把其中一种的平衡构型直接当作另一种已经热化的初态。

现有 [`task="pcet"`](pcet.md) 继续负责局域弱耦合 golden-rule 速率。
该工作流已包括质子本征态、H/D 同位素效应、正反向电极 Fermi 积分和
详细平衡；本模块复用它作为非绝热模型入口，不重复实现一套速率理论。

## 模型选择

| 目标 | 使用的模型 | 结果含义 |
| --- | --- | --- |
| 在单一绝热面上研究电位对核运动或 PMF 的影响 | 逐构型最小化后的恒电位 PES | 所选绝热模型上的力和自由能 |
| 采样电子数涨落与核构型 | 共享电子数的巨正则 HMC/MC | 指定共享电子数模型的联合平衡样本 |
| 弱电子耦合的局域 PCET 正反向速率和 KIE | 已有 `task="pcet"` | 指定局域复合物的一级速率，s⁻¹ |
| 周期界面的二维自由能 | 已有 `task="fes"` 与新的绝热 PES | 声明 CV 测度下的自由能密度 |
| 绝热表面模型中的单个线性出口速率 | 已有 `task="surface_rate"` 与同一个 PES | 盆地归一化 QTST 与 RPMD 回穿修正 |

这些接口不包含新的多电子态实时动力学、电子摩擦或电子库的真实响应时间。
解析充电模型是可核验的接口示例，没有赋予 Pt(111)、Volmer 或 Heyrovsky
反应的定量参数。用于实际电极时，应提供具有一致能量零点、电子数定义和
周期边界处理的势能模型。

## 电位、电子数与单位

令电子数变量为 \(N\)，正增量表示增加电子，电子电荷绝对值为 \(e>0\)。
电子库的化学势满足

\[
\mu_e(U)=\mu_{e,\mathrm{ref}}-e(U-U_\mathrm{ref}),\qquad
\Omega(\mathbf R,N;U)=E(\mathbf R,N)-\mu_e(U)N.
\]

电位变得更正时，电子化学势降低。在 eV/V 单位下，电位差与每电子能量差
数值相同、符号相反；传入原子单位 PES 前仍须进行 eV 到 hartree 的转换。
电位标尺与能量参考必须由模型明确给出，程序不从 `SHE`、`RHE` 标签猜测
绝对真空标定，也不会自动加入 pH 或双电层修正。

\(N\) 可以是相对于固定参考电子数的连续过剩电子数。此时模型与电子库必须
使用同一个参考；舍去的 \(-\mu N_0\) 是固定电位下的构型无关常数，不改变
核力和归一化概率。不能在一个轨迹中切换电子数参考，或把已包含
\(-\mu N\) 的巨势再次作为普通能量减去电子库项。

核势能契约沿用 PyRPMD：坐标为 bohr，能量为 hartree，力为
\(-\partial E/\partial\mathbf R\)，单位 hartree/bohr。电子数无量纲，
\(\partial E/\partial N\) 的单位为 hartree/电子。势函数返回物理能量和
物理力，不自行乘或除珠子数。

### 电子库与模型接口

```python
from pyrpmd.electrochemistry import ElectronReservoir
from pyrpmd.units import parse_quantity

reservoir = ElectronReservoir(
    potential_V=-0.2,
    reference_potential_V=0.0,
    chemical_potential_at_reference_au=parse_quantity("0 eV", "energy"),
    potential_reference="toy model reference",
)
mu_au = reservoir.chemical_potential_au
```

这里使用教学模型的能量零点，不能把 `0 eV` 当作真实 SHE 的电子化学势。
四个参数都须显式给出。`potential_reference` 是可追溯的标签，不触发能量
转换；`chemical_potential_at_reference_au` 则是与 PES 同一能量零点的数值。

电子数相关模型提供 `model.evaluate(q, electron_numbers)`，其中
`q.shape == (B,N_atoms,3)`、`electron_numbers.shape == (B,)`；第一维为
独立构型批次。返回 `ElectronicPESResult`：

| 字段 | 形状 | 定义 |
| --- | --- | --- |
| `energy` | `(B,)` | \(E(\mathbf R,N)\)，hartree |
| `forces` | `(B,N_atoms,3)` | 固定电子数时的核力，hartree/bohr |
| `denergy_dnumber` | `(B,)` | \(\partial E/\partial N\)，hartree/电子 |
| `status`, `messages`, `virial` | 可选 | 沿用 PES 状态和可选 virial 约定 |

周期模型继续通过 `bind_system(system)` 获取规范原子单位的晶胞和 PBC。
计算器需要保证能量、核力及电子数导数来自同一个模型；已有普通 PES
不会自动得到有物理意义的 `denergy_dnumber`。

## 逐构型绝热恒电位势

对于每一个独立核构型，定义

\[
N^*(\mathbf R;U)=\arg\min_N
\{E(\mathbf R,N)-\mu_e(U)N\},\qquad
\Omega_\mathrm{BO}(\mathbf R;U)
=E(\mathbf R,N^*)-\mu_e(U)N^*.
\]

在内点最优解，\(\partial E/\partial N=\mu_e\)。若解唯一且光滑，包络定理给出

\[
\mathbf F_\Omega
=-\left.\frac{\partial E(\mathbf R,N)}{\partial\mathbf R}\right|_{N=N^*}.
\]

因此，无须把 \(dN^*/d\mathbf R\) 另行加到核力中。势能和导数的不一致、
优化未收敛或超出训练电子数范围，都会破坏这个结论；必须把它们视为模型
求值失败，而不是有效的恒电位结果。

PES 接口的第一维是**独立构型批次**，可以由珠子执行器拆分。每个构型独立
求 \(N^*\)，这样已有的串行、珠子批处理和进程执行方式保持一致。
将该 PES 用于 RPMD 后，核环聚合物势为

\[
\sum_{b=1}^{P}\Omega_\mathrm{BO}(\mathbf q_b;U).
\]

这采用每颗珠子独立电子最小化的绝热有效势近似。它**不等于**对一条环聚合物
只优化一个共享 \(N\) 的模型，也不包含有限温度电子数涨落。
已有传播器继续负责弹簧、温度、质量和固定原子；无需另建一个核传播内核。

### 绝热求解器与已有输入格式

`AdiabaticElectrodePotential(model, reservoir, *, electron_bounds, ...)`
封装上述模型，`evaluate(q)` 返回标准 `PESResult`，可以直接交给已有核传播器。
`solve(q)` 返回 `AdiabaticElectrodeResult`，包括 `pes`、`electron_numbers`、
`chemical_potentials_au` 和 `iterations`。求解器假定指定范围内电子数方向
严格凸、导数单调；`electron_bounds` 必须包围每个构型的内点驻点。
没有包围根时明确失败，不把边界值静默当作收敛最小值。驻点两侧还检查
电子数导数是否严格递增，以拒绝局部极大值及数值上无法分辨的斜率。
这些端点、局部单调性和残差检查不能证明任意外部模型在整个区间都凸；
该物理性质由势能模型保证。

| 数值参数 | 默认值 | 含义 |
| --- | --- | --- |
| `electron_bounds` | 必须提供 | 允许且已验证的电子数区间 |
| `number_tolerance` | `1e-10` | 电子数括区宽度阈值 |
| `chemical_potential_tolerance_au` | `1e-9` | 收敛所需的化学势残差容差，hartree |
| `max_iterations` | `100` | 最大求解次数 |

输入文件使用现有 Python PES factory 入口，无须新增 `task`：

```toml
[potential]
provider = "python"
factory = "pyrpmd.electrochemistry:build_adiabatic_electrode"

[potential.params]
electron_bounds = [-2.0, 2.0]

[potential.params.model]
factory = "your_model:make_model"

[potential.params.model.params]
# 实际模型工厂所需的参数。

[potential.params.reservoir]
potential_V = -0.2
reference_potential_V = 0.0
chemical_potential_at_reference_au = 0.0
potential_reference = "toy model reference"
```

这里的模型工厂名需替换为实际可导入模块；完整教学输入见
[`examples/electrochemistry/fes2d.inp`](../examples/electrochemistry/fes2d.inp)。
外层 Python PES 也必须使用 bohr/hartree 和 `derivative="force"`，不能再给
这个原子单位包装器施加第二次单位或力符号转换。
保持 factory 可导入使现有 worker 能各自构建计算器，避免把一个可变
电子结构上下文共享给并发轨迹。

## 共享电子数的巨正则 HMC/MC 采样

另一个明确的统计模型给整条环聚合物一个共享电子数：

\[
\bar E(\mathbf q,N)=\frac1P\sum_{b=1}^{P}E(\mathbf q_b,N),\qquad
\pi(N\mid\mathbf q,U,T)\propto
\exp[-\beta\{\bar E(\mathbf q,N)-\mu_e(U)N\}],
\quad\beta=(k_BT)^{-1}.
\]

电子数改变而核坐标固定时，弹簧和核动能抵消。使用对称电子数试探，接受概率为

\[
A=\min\left[1,
\exp\{-\beta[\bar E(\mathbf q,N')-\bar E(\mathbf q,N)-\mu_e(N'-N)]\}
\right].
\]

必须使用平均珠子能量。把相同珠子复制为更多珠子不应改变条件电子数分布。
如果采用在边界处截断后重新归一化的试探分布，还须加入逆向/正向试探概率比；
不能只截断区间后继续使用对称提议的接受式。

`GrandCanonicalSampler` 将上述电子数 MC 与固定电子数的核 HMC 组合。
每次核提议重新抽取单珠动量，方差为 \(mP/\beta\)，复用已有环聚合物
积分器传播，然后根据完整核 Hamiltonian 的变化以
\(\min[1,\exp(-\beta\Delta H_P/P)]\) 接受或拒绝。这个 Metropolis 校正是
平衡采样的一部分；仅穿插未校正的有限步长 MD 不会自动具有同样的统计保证。
电子数变更后的下一次核提议使用新的电子数重新求力。

单次固定电子数核 HMC 提议复用已有势能缓存，避免相邻时间步重复求相同
端点的能量和力。缓存作用域在任何电子数更新前结束，不将旧电子数下的力
带入新电子数的提议。

支持与时间和历史无关的偏置。传入的 `bias.energy_force` 必须遵循已有
`UmbrellaBias` 的总珠子约定，即在 \(H_P\) 中加入 \(P W\)；采样分布中的
物理偏置因此是 \(W\)。依赖历史的 metadynamics 或仍在自适应改变的偏置
不能用于这里的固定目标 HMC。

MC 试探次数与真实时间没有对应关系。不能把电子数接受事件数解释为电子
转移速率，或在普通 RPMD 回穿轨迹中加入 MC 跳跃后直接报告真实 PCET 速率。

共享电子数还存在两个不同的消元方案：

\[
\min_N[\bar E(\mathbf q,N)-\mu_eN]
\quad\text{与}\quad
-\beta^{-1}\log\int dN\,
e^{-\beta[\bar E(\mathbf q,N)-\mu_eN]}.
\]

前者仅保留最优电子数，后者保留电子数涨落熵；两者都不同于前一节的逐珠子
绝热有效势。本模块不把这些方案静默互换。

### 采样 API 与输出

`GrandCanonicalState(q, electron_number, sweeps=0)` 保存原子单位核坐标
`q[P,N_atoms,3]`、共享电子数和 sweep 计数。它不保存真实时间或持续传播的
动量，因为每次核 HMC 提议会重新抽样动量。

```python
from pyrpmd.electrochemistry import GrandCanonicalSampler, GrandCanonicalState

# model、masses、q、beta、dt 由同一物理模型提供，均使用原子单位。
state = GrandCanonicalState(q, electron_number=0.0)
sampler = GrandCanonicalSampler(
    model, masses, beta, q.shape[0], dt,
    reservoir=reservoir,
    electron_bounds=(-2.0, 2.0),
    proposal_width=0.5,
    seed=17,
)
record = sampler.step(state, nuclear_steps=5, electron_moves=1)
```

模型若需要 cell/PBC，应在构建采样器前调用自己的 `bind_system(system)`。
构造参数还支持 `native=True`、`integrator="cayley-bcb"`、`fixed_atoms` 和
`fixed_positions`；固定原子沿用核积分器的固定位置与零动量契约。
`native` 指复用编译后的自由环传播微内核，外层 HMC 和电子数 MC 仍由 Python
组织，不表示新增了完整原生恒电位传播器。

`proposal_width` 是电子数对称均匀增量的半宽；超出 `electron_bounds` 的
提议直接拒绝，不 clip、不重抽。`step` 原地更新 state，返回以下诊断：

| 字段 | 含义 |
| --- | --- |
| `sweep` | 本次 sweep 编号，不是物理时间 |
| `nuclear_accepted`, `nuclear_log_acceptance` | 核 HMC 接受情况及对数接受信息 |
| `electron_attempted`, `electron_accepted`, `electron_out_of_bounds` | 本次电子数提议统计 |
| `electron_number` | 更新后的共享电子数 |
| `physical_grand_energy_au` | \(\bar E-\mu N\)，不含弹簧、动能或 umbrella 偏置 |

`physical_grand_energy_au` 是构型能量诊断，不是热力学自由能或完整环聚合物作用量。

`sampler.statistics` 返回累计计数与接受比例的副本。设 `nuclear_steps=0`
可只做固定核条件电子数采样；设 `electron_moves=0` 可只做固定电子数核 HMC；
两者不能同时为零。普通拒绝计入统计；势能求值等异常会回滚 state、RNG 和
计数，避免得到一半更新的采样状态。

这是可组合的 Python 采样 API，尚未增加一个自动管理所有文件的独立
巨正则 CLI 工作流。只保存 `GrandCanonicalState` 不是完整可复现 checkpoint：
还需保存 `sampler.rng.bit_generator.state`，并保留一致的模型、偏置及数值设置。
可执行例子见
[`sample_grand_canonical.py`](../examples/electrochemistry/sample_grand_canonical.py)。

## 复用界面自由能与表面速率

固定晶胞、部分 PBC、固定底层、周期键长和二维 umbrella/MBAR 采样沿用
[表面工作流](surface-catalysis.md)。电子数相关势必须把完整 cell/PBC
传给实际计算器；添加电子库项本身不会生成周期静电、镜像电荷或双电层。

质子转移可使用 `bond_difference` 或显式 `periodic_bond_difference` 构造
自由能坐标，第二维可以是明确实现的溶剂极化或配位坐标。供体身份变化的
Grotthuss 过程通常需要连续
配位型 CV；固定原子编号的键差不能自动追踪交换后的新供体。

`surface_rate` 接受内置 `cartesian`、`surface_height`、`surface_lateral`
以及具有恒定端点质量度量的 `distance`。`periodic_distance` 还须满足
分割面远离最小镜像切换边界的光滑性限制。非线性键差或自定义质子坐标
可以用于 FES，但不自动满足这些速率公式的质量度量条件。
二维 PMF 本身也不提供非绝热电子跃迁概率。

在支持的线性出口上，PMF、受约束母轨迹和回穿子轨迹都必须使用同一个
绝热恒电位 PES。得到的是该近似模型下的一级速率；不得引入气相双分子
远区球面面积前因子。电流转换另需活性位点面密度、覆盖度、反应计量和
反向反应，不能只把自由能势垒乘一个通用常数当作电流。

## HER 覆盖度与电流后处理

`HERRates` 和 `steady_state_her` 将给定的 Volmer、Heyrovsky、Tafel 六个
正反向速率系数组合成单一吸附氢位点的理想平均场网络：

\[
\mathrm{H^++e^-+*}\rightleftharpoons\mathrm{H^*},\qquad
\mathrm{H^*+H^++e^-}\rightleftharpoons\mathrm{H_2+*},\qquad
2\mathrm{H^*}\rightleftharpoons\mathrm{H_2+2*}.
\]

令覆盖度为 \(\theta\)、空位分数为 \(v=1-\theta\)，外部无量纲活度为
\(a_{\mathrm H}\)、\(a_{\mathrm H_2}\)。每个位点每秒的净事件通量为

\[
r_V=k_{V+}a_{\mathrm H}v-k_{V-}\theta,
\]

\[
r_H=k_{H+}a_{\mathrm H}\theta-k_{H-}a_{\mathrm H_2}v,
\qquad
r_T=k_{T+}\theta^2-k_{T-}a_{\mathrm H_2}v^2.
\]

`HERRates` 的六个字段为 `volmer_forward_s`、`volmer_reverse_s`、
`heyrovsky_forward_s`、`heyrovsky_reverse_s`、`tafel_forward_s`、
`tafel_reverse_s`；默认均为零。它们是已经匹配上述活度和位点约定的有效
系数，单位 s⁻¹。Tafel 系数须已包含邻接位点数、配对计数等几何因子；
程序不再乘额外的 `1/2` 或配位数。电位依赖由所给系数决定，后处理不加入
隐藏的 Butler–Volmer、pH 或 SHE/RHE 修正。

\[
\dot\theta=r_V-r_H-2r_T,\qquad
\mathrm{TOF}_{\mathrm H_2}=r_H+r_T.
\]

若 \(\Gamma\) 是 **mol sites/m²**，阳极方向为正的电流密度为

\[
j=-F\Gamma(r_V+r_H)\quad[\mathrm{A/m^2}].
\]

它计算交换电子的净通量。仅在 \(\dot\theta=0\) 时，才可简化为
\(j=-2F\Gamma\,\mathrm{TOF}_{\mathrm H_2}\)。瞬态吸附氢的积累也消耗电子，
因此不能对任意覆盖度直接使用两电子乘 H₂ 通量的式子。

```python
from pyrpmd.electrochemistry import HERRates, steady_state_her

rates = HERRates(volmer_forward_s=3.0, heyrovsky_forward_s=7.0)
result = steady_state_her(
    rates,
    proton_activity=1.0,
    hydrogen_activity=1.0,
    site_density_mol_m2=2.5e-5,
)
print(result.coverage)                 # 0.3
print(result.hydrogen_tof_s)           # 2.1 H2 / site / s
print(result.current_density_A_m2)    # negative: cathodic HER
```

`her_fluxes(coverage, rates, *, proton_activity, hydrogen_activity,
site_density_mol_m2)` 在已知覆盖度处返回瞬态通量；
`steady_state_her(rates, *, ...)` 求物理区间 \([0,1]\) 内的稳态。
两者返回 `HERFluxes`，字段包括 `coverage`、`vacancy`、`volmer_s`、
`heyrovsky_s`、`tafel_s`、`coverage_rate_s`、`hydrogen_tof_s` 和
`current_density_A_m2`。负的净 H₂ 通量表示消耗 H₂。若所有活度加权反应
系数均为零，任何覆盖度都保持不变，稳态求解明确报非唯一解；此时可以在
已知覆盖度调用 `her_fluxes`。

已有 PCET 或表面速率输出必须先映射到相同的局域复合物、位点和活度定义，
才能作为这里的系数。模型假定每个位点最多一个吸附氢、固定外部储库，
未包含侧向相互作用、传质或其他吸附物竞争。正反向系数的热力学一致性
由输入物理模型负责。

## 与已有非绝热 PCET 的关系

已有 `task="pcet"` 使用两套局域电子态的质子势、弱电子耦合和经典 Gaussian
环境。其 `driving_force` 是参考电位下的**势阱最低点自由能差**；转移质子的
零点能及热布居由谱计算加入。新势能接口不会把恒电位 PMF 自动转换为这些
参数；若外部自由能已经包含质子振动贡献，直接填入会重复计数。

固定几何教学模型与官方 pyPCET 的独立比较及能量零点映射见
[验证记录](validation/pcet-model-2026-10-30/README.md)。该记录验证已有局域
golden-rule 实现，不能替代新模型的电子数分布、有效力和周期接口验证。

## 开源参考与实现归属

本模块通过独立编写的接口和公开统计力学公式复用方法，优先复用 PyRPMD
已有 PES 结果验证、核传播、周期 CV、PMF 和表面速率代码。没有将下列项目的
实现文件或模型权重复制到 PyRPMD，也不增加对它们的运行时依赖。

| 参考及固定提交 | 实际参考内容 | 源码授权情况 |
| --- | --- | --- |
| [GC-Constrained-PIHMC](https://github.com/sxu39/GC-Constrained-PIHMC/tree/94337e7e87eeb5f3776bd364145f9650333b156b) · `94337e7e87eeb5f3776bd364145f9650333b156b` | 共享电子数、平均珠子能量、电子库项及边界提议的 Hastings 修正 | 所检查版本没有根 LICENSE；只参考方法，不复制代码 |
| [TP-RPMD](https://github.com/sxu39/TP-RPMD/tree/ae3d6f0dd0b66401094548f394c99e17e1f2a7a1) · `ae3d6f0dd0b66401094548f394c99e17e1f2a7a1` | `E(R,N)`、`dE/dN` 接口与电子库动力学所需的独立定义 | 修改的 LAMMPS 源文件保留 GPL 声明；不引入源码 |
| [pyPCET](https://github.com/shsgroup/pyPCET/tree/031a679eab50e210aa7f76698ef5f302de31a460) · `031a679eab50e210aa7f76698ef5f302de31a460` | 已有局域 golden-rule PCET 的外部验证及零点能映射 | GPLv3；验证脚本导入独立检出版本 |

可直接检查 GC 项目的
[`source/elec_num.cpp`](https://github.com/sxu39/GC-Constrained-PIHMC/blob/94337e7e87eeb5f3776bd364145f9650333b156b/source/elec_num.cpp)
和 [`source/ml_pot.cpp`](https://github.com/sxu39/GC-Constrained-PIHMC/blob/94337e7e87eeb5f3776bd364145f9650333b156b/source/ml_pot.cpp)，
以及 TP 项目的
[`fix_pimd_langevin.cpp`](https://github.com/sxu39/TP-RPMD/blob/ae3d6f0dd0b66401094548f394c99e17e1f2a7a1/lammps-stable_2Aug2023_update2/src/fix_pimd_langevin.cpp)。
后者将共享电子数作为带虚拟质量的扩展自由度传播，并耦合热浴。这里的
绝热最小化和巨正则 HMC/MC 都不是该 TP-RPMD 动力学的复现；本模块不声称复现
其电极反应速率或发表体系。
