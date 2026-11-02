# 非绝热动力学：布居系综、分子 PES 与 NRPMD 碰撞

## 原始论文与复现范围

[方法—论文目录](references.md) 给出完整书目信息和源码位置，
[复现计划](reproduction.md) 列出原文体系、应计算的观测量、现有入口和待补工作。
一次计算应分别引用动力学方法、初态预备和物理模型；通用 LVC 或离散 spin-boson
容器仍需注明所用参数与浴离散化的来源。

| 实现组成 | 原始来源 | 引用对应的范围 |
| --- | --- | --- |
| Focused NRPMD 与 MInt 子步 | [Richardson–Thoss 2013](https://doi.org/10.1063/1.4816124)；[Chowdhury–Huo 2019](https://doi.org/10.1063/1.5096276)；[Church 等 2018](https://doi.org/10.1063/1.5005557) | 2019 论文对应 focused 初态；Church 论文对应积分子步，包未因此实现完整 MQC-IVR。 |
| 带符号热 NRPMD | [Richardson–Thoss 2013](https://doi.org/10.1063/1.4816124) | SEO 热采样与相关函数；专用论文 benchmark 使用原文的最低绝热面分裂，库默认采用 trace 分裂。 |
| 两态 SM-NRPMD | [Bossion 等 2021](https://doi.org/10.1063/5.0051456) | 已实现 focused 自旋传播；原文平衡相关函数还需要其电子热迹和带相位权重的估计量。 |
| MASH 与 CA-RP-MASH | [Mannouch–Richardson 2023](https://doi.org/10.1063/5.0139734)；[Cao 等，ChemRxiv v1，2025](https://doi.org/10.26434/chemrxiv-2025-xps2g) | `MASHDynamics(P=1)` 对应前者；`RPMASHDynamics`/`rp-mash` 对应后者的 CA 变体。 |
| FSSH 与质心 RPSH | [Tully 1990](https://doi.org/10.1063/1.459170)；[Shushkov 等 2012](https://doi.org/10.1063/1.4766449)；[Shakib–Huo 2017](https://doi.org/10.1021/acs.jpclett.7b01343) | 复现时匹配目标论文的初态、动量重标规则及振幅/活动态观测量。 |
| RPSH-BA 与 RPSH-DM | [Y. Shakya 等 2023](https://doi.org/10.1103/PhysRevA.107.062810) | BA 共用一个电子向量；DM 在共同质心基下平均逐珠电子波函数密度；两者均共用一个活动面。 |
| EDC 与 DBOC | [Granucci–Persico 2007](https://doi.org/10.1063/1.2715585)；[Limbu 等 2025](https://doi.org/10.1063/5.0301249) | 是不同修正，各有显式的环聚合适配约定；DBOC 论文的 donor population 需活动态与振幅相干组合的混合估计量。 |
| 复 hopping 与几何力 | [Miao 等 2019](https://doi.org/10.1063/1.5088770) | 局域非简并 NAC/动量方向和 Berry 力处方的环聚合适配。 |
| MV-RPMD 与 KC-RPMD | [Ananth 2013](https://doi.org/10.1063/1.4821590)；[Menzeleev 等 2014](https://doi.org/10.1063/1.4863919) | 各有独立模型 API 和热权重；KC 还需交叉校准与辅助质量。 |
| Isomorphic 势 | [Tao 等 2018](https://doi.org/10.1063/1.5005544) | 包将两态电子迹匹配与 Ehrenfest 组合；原文 QCLE/iso-surface-hopping 结果需要对应传播器。 |
| 准 diabatic 传输 | [Mandal 等 2018](https://doi.org/10.1021/acs.jctc.7b01178)；[Zhou 等 2019](https://doi.org/10.1021/acs.jpclett.9b02747) | QD-NRPMD 复用局域基底策略；解析接口演示不包含原文的 PLDM/SQC 或乙烯即时电子结构计算。 |
| Tully 模型与 NaH₂ PES | [Tully 1990](https://doi.org/10.1063/1.459170)；[Halvick–Truhlar 1992](https://doi.org/10.1063/1.461986) 及 [1994 勘误](https://doi.org/10.1063/1.467282) | 势能模型来源与传播方法分别引用。 |
| 独立一维量子基准 | [Feit 等 1982](https://doi.org/10.1016/0021-9991(82)90091-2) | 有限周期网格上的 Fourier 谱传播，空间区域、网格和时间步需分别收敛。 |

建议首先用 `examples/nonadiabatic/published_nrpmd_benchmark.py` 定量比较
Richardson–Thoss 图 1(c)，明确选择 `--potential-split lowest_adiabatic`。
其他短时布居和分子例子用于检查工作流；若采用合成参数或不同初始分布，需按复现计划
替换后才能对照原文。CA-RP-MASH 引用保留代码实际依据的 v1；截至 2026-11-02
的核对未核实期刊后继版本。

## 布居与相干性工作流

`task="nonadiabatic"` 支持下表中的显式方法选择，使用统一的
`validate → run → analyze → resume` 流程：

```bash
pyrpmd validate examples/nonadiabatic/population.inp --check-pes
pyrpmd run examples/nonadiabatic/population.inp --workers 2 --output runs/population
pyrpmd analyze runs/population --recompute
pyrpmd resume runs/population --workers 1 --block-steps 64
```

| `method` | 电子传播与适用范围 |
| --- | --- |
| `nrpmd` | focused MMST NRPMD；逐珠 mapping 变量，支持复 Hermitian 矩阵 |
| `sm-nrpmd` | 两态 W-spin SM-NRPMD；逐珠单位自旋，支持复 Hermitian 矩阵 |
| `rp-mash` | 两态**质心近似 CA-RP-MASH**；实矩阵，`integrator="adaptive-dop853"`、`native=false` |
| `rpsh` | 原有实矩阵质心 RPSH；保留 `centroid` / `ring` 动量重标度约定及可选 EDC |
| `rpsh-ca` | 扩展质心 RPSH；非简并 Hermitian 矩阵、全环跃迁能量账本、可选 DBOC / 几何力 |
| `rpsh-ba` | 实矩阵 RPSH-BA；一个共同绝热电子向量，由逐珠平均能量和 `v·d` 驱动 |
| `rpsh-dm` | 逐珠电子波函数在共同质心基下形成密度与跃迁电流；支持非简并 Hermitian 矩阵 |
| `soc-rpsh` | 复 SOC 的显式选择项，使用与 `rpsh-ca` 相同的扩展 CA 驱动器 |

程序内置 Tully I–III、一般线性振电耦合（LVC）、显式有限模式 spin-boson 模型，
以及下文介绍的 NaH₂ 分子 PES。参数采用原子单位，provider 给出全部透热势能矩阵及
正梯度。[扩展示例](https://github.com/chen-yu-hao/PyRPMD/tree/main/examples/nonadiabatic/extensions)
包含 spin、BA/DM、DBOC、SOC 和分子体系输入。原有
`examples/nonadiabatic/rpsh_population.inp` 继续演示带可选 EDC 的原质心 RPSH。

初态可选择指定核相点，或独立抽取有限珠数的谐振参考环聚合物分布。后者只对明确指定的
坐标采样，其余坐标和动量为零；它不是传播势能下的完整核—电子热平衡分布。
NRPMD 和 SM 的 `initial.initial_state` 表示固定透热初态；hopping 方法使用
`initial.active_surface`。BA 使用共同的逐珠绝热标签，CA/DM 在质心基下形成用于布居
决策的电子量；MASH 在选定质心绝热半球上按要求进行重要性采样。
相同的整数通常并不代表相同的物理电子初态。

`results/nonadiabatic.npz` 保存逐轨迹的能量、范数、核质心及方法对应的电子观测量；
`results/populations.csv` 给出布居均值和逐时刻标准误。NRPMD 和 SM 保存固定透热基下
的有符号密度估计，不裁剪负值；RPSH 的幅度密度、绝热振幅布居和活跃面占比分别保存。
**MASH 的 CLI 布居是指定重要性采样下的绝热活跃面占比**，不把传播自旋的外积当成
透热密度估计，也不据此输出相干性。一般 MASH 透热布居或相干算符的关联函数，应使用
全球面采样及带算符配对权重的 Python API。

误差以独立轨迹为样本，珠数和时间帧数不增加独立样本量；只有一条轨迹时不提供标准误。

每条轨迹的状态、随机数和观测历史在同一校验断点中保存。中断后可改变轨迹进程数或
断点块大小继续；改变物理参数需要新计算。失败轨迹不会被静默丢弃后继续发布平均结果。
BA 的连续绝热基框架及 MASH 的采样标签也进入断点。短算例是工作流演示，输出保留
`physical_converged=false`。

四个扩展 RPSH 选择项支持 `dboc=true` 和 `geometric_force`。DBOC 同时进入能量、
核力和跃迁能隙；复 hopping 使用明确规定的最大实 NAC 方向与局域 Abelian Berry 力
近似。它要求非简并态，不处理 Kramers 等简并子空间，也不是相空间 surface hopping。
原有/扩展 CA 和实 BA 可选择 `decoherence="edc"`；DM 使用自己的逐珠密度构造，
要求 `decoherence="none"`。扩展类的 `rescaling="bead"|"centroid"` 都按全环能量
处理跃迁，不能与原 CA 的质心能量模式混为一谈。

完整字段见[模型布居工作流](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-workflow.md)，
公式见[模型库](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-models.md)，
方法定义见[spin 方法指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-spin-methods.md)
和[hopping 扩展指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-hopping-extensions.md)。
原有[系综 API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-ensembles.md)
继续支持 prepared NRPMD / 原质心 RPSH 状态与自定义完整矩阵 provider；其他方法使用
自己的驱动器和已接入的 CLI 工作流，不能把不同 State 直接交给这个双方法 API。

## 已支持的真实分子 PES

`model.name="nah2_leps_5g_1992"` 选择历史的 NaH₂ 5G 双态拟合面。
`NaH2LEPS5G` 接受三维 Na/H/H 几何，返回完整实透热矩阵及所有 Cartesian 梯度，
包括耦合项导数。它描述 Na + H₂ 电子猝灭/反应涉及的两个 doublet 面，具有固定的
来源、原子顺序及版本标识，并已与独立编译的原 Fortran 作实现对照。

这是一个化学组成明确的分子 PES，不是任意分子的实时电子结构接口。它不包含 SOC、
更高电子态或其他元素组合；与原程序相符也不代表重新验证了历史拟合面的全部化学精度。
现有分子示例从指定的相互作用区相点出发，尚不能作为振转态选择的收敛散射概率或热反应
速率。NaH₂ 输入中的相同 `initial_state` / `active_surface` 数字也可能对应不同的
电子初始密度，不能直接把两个短算例的差别解释为方法误差。

来源、单位、解析链式梯度与可选固定版本 ChemPotPy 适配器见
[分子 PES 指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/molecular-nonadiabatic.md)，
可运行输入及原 Fortran 对照见
[分子基准](https://github.com/chen-yu-hao/PyRPMD/tree/main/benchmarks/molecular_nonadiabatic)。

## 电子后端与 NRPMD 批处理

布居工作流的 **NRPMD 和 SM-NRPMD** 可选择
`dynamics.electronic_backend="numpy"|"native"|"torch"`。
native 路径把 Hermitian 对角化与 C++/OpenMP 收缩结合，计算固定核坐标下的精确
电子传播及积分核动量冲量。另一个 `native` 参数仍单独控制核自由环积分器。

Torch 路径可在可用时选择 `electronic_device="cpu"` 或 `"cuda[:index]"`。
GPU 范围限于 mapping/impulse 内核；当前驱动器的分子 PES 与核位置传播仍在 CPU，
使用 GPU 时有主机/设备数据传输。缺少依赖或设备会明确报错，不会静默回退。
当前验证没有 CUDA 硬件实测，也不声称 GPU 加速倍数。RPSH 和 MASH 没有接入该 GPU 内核。

`BatchedNRPMDDynamics` 另提供 Python 层的轨迹/珠子联合 batch API，适用于声明各帧
独立的 provider；它并不把所有 CLI 方法自动变成批处理 GPU 轨迹。
具体后端、已测 CPU 对照和限制见
[性能指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-performance.md)。

仍未实现的具体范围包括 SM 电子热迹及相位重加权、SU(N) SM-NRPMD、BA-RP-MASH、
多态/复 SOC MASH、简并子空间的非 Abelian hopping，以及 GPU PES、核传播或
hopping/event 内核。现有 focused / harmonic-reference 初始化不是这些缺失热采样的替代。

## 实验性非绝热碰撞：NRPMD

NRPMD 是 `task="drpmd"` 的实验性模式。它在每个核珠子上加入电子 mapping 变量，在完整的 diabatic 势能矩阵上同时
推进电子与核，并把电子布居与化学产物分类器结合，给出电子态分辨的贡献。

碰撞从一个指定的入射 diabatic 电子态出发，该态在远区解耦。初始条件、碰撞参数和分类沿用普通 dRPMD，见[气相工作流](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/zh/manual/workflows.md)。

## 运行示例

在仓库根目录执行：

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)使用 4 个珠子、两个电子态、
两条母链和 8 条固定碰撞能的轨迹。[矩阵势](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)
是带局域耦合的人工双态模型；通道按 H + HD → H₂ + D 标记。

`collision.sampling="fixed_energy"` 时结果为截面，单位 `bohr^2`。改为 `thermal_flux` 并删除 `energy`，即得到从所选入射态出发的热碰撞分布。

## 配置与矩阵 PES

在普通 dRPMD 输入中加入：

```toml
[potential]
provider = "diabatic_python"
factory = "matrix_pes.py:make"
derivative = "gradient"

[nonadiabatic]
method = "nrpmd"
nstates = 2
initial_state = 0
state_labels = ["entrance", "excited"]
initialization = "focused"
asymptotic_coupling_tolerance = "1e-8 hartree"
```

`initial_state` 从 0 开始，标签对应固定 diabatic 基中的态。factory 返回带 `evaluate(q)` 的对象或可调用对象，
返回 `pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`：

| 数据 | 形状与单位 |
| --- | --- |
| 核坐标 q | `[B,N,3]`，bohr |
| 能量矩阵 | `[B,K,K]`，hartree |
| 矩阵的梯度 | `[B,K,K,N,3]`，hartree/bohr |

B 是批大小，K 是电子态数。两个数组须有限，并在电子指标上满足 Hermitian 条件；可使用实数或复数，每个对角元和耦合都有梯度。矩阵接口使用
**正梯度**（标量势返回的是力）。provider 声明坐标或能量单位时，适配器负责转换。

与态无关的能量放在每个对角元中；该矩阵就是完整的势能。`validate --check-pes` 检查矩阵并对每个矩阵元的梯度做有限差分。

## Mapping 变量与有符号估计量

核变量为 `q,p[P,N,3]`，电子变量为 `mapping_q,mapping_p[P,K]`。母链在入射态的对角势上采样。每次发射为每个珠子和
每个电子态抽取独立相位；focused 初始化使入射态布居为 1、其余为 0。碰撞过程不加恒温器。

记电子变量为 x、y，一条轨迹的布居估计量为

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

各态布居之和在舍入误差内为 1，单个值可以落在 $[0,1]$ 之外。对轨迹求平均得到态分辨的贡献。

化学通道 α 与电子态 s 的联合贡献为 `w * I_alpha * n_s`，以发射轨迹总数归一化；对态求和即得化学贡献。所有通道共用
母链 bootstrap 抽样，从而得到它们的联合协方差。

## 停止条件、输出与续跑

发射时，所有珠子的非对角耦合须低于 `asymptotic_coupling_tolerance`。产物或返回的反应物在几何上分离且电子解耦时
才算已决；否则轨迹运行到时间预算。耦合阈值、分离距离和传播时间需要分别收敛。

| 输出 | 内容 |
| --- | --- |
| `results/summary.json` 中的 `nonadiabatic` | `final_estimates_available`、态标签、布居和误差、通道结果 |
| `results/electronic_channels.csv` | 有符号观察贡献及热速率或截面 |
| `nonadiabatic.channels` | `signed_observed_contributions`、`rate_cm3_molecule_s` 或 `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | 按 `covariance_columns` 排列的联合误差信息 |

若有轨迹失败或未决，最终的电子态速率或截面留空，仍报告观察到的贡献。摘要中带有 `experimental=true` 和
`physically_validated=false`。

检查点保存核与 mapping 数组、时钟、分类持续状态和 RNG 状态，单步失败时电子与核一起回滚。`resume` 在原时间预算内
完成中断的工作；更长的最大碰撞时间需要新的运行。

## 支持的配置

这里的 **`task="drpmd"` 指定初态碰撞流程**仍在 CPU 上用 NumPy 运行，可串行或使用
本机轨迹进程；核自由环可使用编译内核。使用 `loop_backend="auto"` 或 `"python"`。
这个碰撞入口没有暴露上文布居任务的电子后端选择，也没有自动采用 NRPMD batch API；
它不支持 MPI、珠子 worker 拆分、GPU、融合的原生电子轨迹循环、周期或固定原子体系。
矩阵传播支持复 Hermitian SOC，包括电子旋转和核动量冲量中的复相位。

## 其他已有方法和表面电子摩擦

| 方法 | 入口 | 当前范围 |
| --- | --- | --- |
| 热 NRPMD | `ThermalNRPMDSampler`、`ThermalNRPMDDynamics` | 实矩阵、SEO 投影的有符号热采样、有限时间通量/布居速率曲线 |
| 经典核 MASH API | `MASHDynamics` | `P=1` 对应原始两态方法；`P>1` 是实验性质心扩展，与 `RPMASHDynamics` 不同 |
| 准 diabatic / 绝热 API | `QDNRPMDDynamics`、`AdiabaticRPSHDynamics` | 由 provider 提供能量、梯度、导数耦合与电子态重叠 |
| 电子摩擦 EF-RPMD | `diffusion` 的 `[electronic_friction]`；`ElectronicFrictionDynamics` | Markov 质心摩擦张量、涨落–耗散噪声、周期体系及固定原子 |
| MV-RPMD | `MVRPMDDynamics` | SEO 映射迹 Hamiltonian 及其专用传播器 |
| KC-RPMD | `KCRPMDDynamics` | 两个实态、动力学约束、用户提供的校准参数 |
| Isomorphic RPMD | `IsomorphicPotential`、`IsomorphicEhrenfestDynamics` | 两态电子迹匹配与 Ehrenfest 传播组合 |

各方法的初始分布、传播 Hamiltonian 和速率定义不同。模型 API 不会隐式替代现有 `rate` 工作流。
热 NRPMD 保留符号抵消及平台诊断，其热采样仍要求实矩阵；质心电子摩擦不包含完整非
Markov 路径积分 GLE 的隧穿抑制。原有 `RPSHDynamics` 的实矩阵限制不应推广到上文
已经实现、具有自身适用范围的扩展 CA/DM 复 hopping。
计算测试说明的是实现一致性，不是某个真实催化体系的定量验证。

详见 [热 NRPMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/thermal-nrpmd.md)、
[电子摩擦](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md)、
[SOC/RPSH](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-soc-rpsh.md) 和
[KC/MV/isomorphic](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-methods.md) 专题。

用于真实体系时，提供含态间耦合的 diabatic 基、完整矩阵及其梯度，并收敛远区态、能量与 mapping 范数、珠子数、
时间步、母链相关性、碰撞参数和产物判定。方法与来源见 [NRPMD 专题](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md)
和[示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md)。
