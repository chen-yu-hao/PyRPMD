# 原始论文与实现功能对照

本页按**已实现的方法/模型/接口**归并功能，而不是为每个辅助函数硬配论文。核查对象是
[`28cde09`](https://github.com/chen-yu-hao/PyRPMD/commit/28cde092b66707878d99bc45eadf6bdf5950b317)，核查日期为 **2026-11-03**。
下列 **125 个条目、199 条文献记录**将源码、原始方法及实现范围联系起来；
[复现清单](reproduction.md)另列原文体系、应算物理量、现有入口、缺失条件和优先级。
上次核查（`cb60679`，2026-11-02）之后合入的功能，以及范围因这些功能而改变的条目，均按本提交新写或修订；
其余条目的范围说明沿用上次核查结论，仅更新源码链接。

## 如何引用

论文/报告应同时引用使用的**原始算法、具体势能面或电子结构模型、依赖软件**，并记录 PyRPMD 的版本与提交。
例如 H + H₂/BKMP2 的速率研究需同时引用 Bennett–Chandler、RPMD 速率、双分子归一化、BKMP2 势及实际使用的软件。
仅引用 RPMDrate、Caracal 或一篇综述不能代替原始算法出处。

PyRPMD 本身按软件引用，元数据见
[CITATION.cff](https://github.com/chen-yu-hao/PyRPMD/blob/main/CITATION.cff)（GitHub 的 **Cite this repository**
按钮可据此生成 APA 与 BibTeX）。同时写明 `pyrpmd --version` 给出的版本（每次运行的 `manifest.json`
也会记录）；未发布的检出还需注明 `git rev-parse HEAD` 给出的提交哈希。
PyRPMD 目前没有期刊论文，不要用 RPMDrate、Caracal 或方法论文代替软件引用。

本目录有意保留不同出处的身份：原始方法、后续修正、模型/数据、上游软件、公式或背景文献。
**引用不等于完整实现或已完成复现**。组合流程分别引用各组成方法；外部接口只表明能调用该软件；
通用解析控制模型、输入/检查点/并行/报告等工程功能没有唯一科学原始论文时，直接说明这一点。
原始论文的具体版本、勘误及实现差别以各条目的范围说明为准。

### 按工作流选择条目

从下列条目收集一次计算需要引用的文献。常见起点：

| 计算 | 应引用的条目 |
| --- | --- |
| 双分子 RPMD 速率（`rate`） | [rpm-rate](#rpm-rate)、[integrators](#integrators)、[samp-umbrella](#samp-umbrella)、[samp-ui](#samp-ui) 或 [samp-mbar](#samp-mbar)，以及势能面 |
| 直接碰撞（`drpmd`） | [direct-rpmd](#direct-rpmd)、[integrators](#integrators)；选态初始条件另加 [state-rpmd](#state-rpmd) 或 [qct](#qct) |
| 自由能面（`fes`） | 所用方法：[samp-umbrella](#samp-umbrella)、[samp-standard-meta](#samp-standard-meta)、[samp-wt-meta](#samp-wt-meta)、[samp-opes](#samp-opes)、[samp-ves](#samp-ves)、[samp-abf](#samp-abf) 或 [samp-steered](#samp-steered) |
| 平衡 PIMD（`pimd`） | [pimd](#pimd)、所用热浴（[pile](#pile)、[gle](#gle)、[piglet](#piglet)、[nhc](#nhc)）；NPT 另加压浴（[bzp](#bzp)、[mtk](#mtk)、[scr](#scr)） |
| 副本交换与回火 | [samp-reus](#samp-reus) 或 [samp-tempering](#samp-tempering) |
| 路径采样（`path_sampling`） | [samp-tis](#samp-tis)、[samp-retis](#samp-retis)、[samp-infretis](#samp-infretis)、[samp-pptis](#samp-pptis) 或 [samp-ffs](#samp-ffs) |
| 非绝热布居（`nonadiabatic`） | 所选方法条目，如 [na-nrpmd-focused](#na-nrpmd-focused)、[na-carp-mash](#na-carp-mash)、[na-ba-rp-mash](#na-ba-rp-mash)、[na-ms-mash](#na-ms-mash)、[na-iesh](#na-iesh) 或 [na-rpsh-ca](#na-rpsh-ca)，以及所用模型 |
| 电化学 PCET（`pcet`） | [pcet](#pcet)，以及电极与环境模型 |
| 外部力引擎与机器学习势 | 接口条目（[ase](#ase)、[openmm](#openmm)、[lammps](#lammps)、[ipi-socket](#ipi-socket)、[ml-adapters](#ml-adapters)）以及模型论文和所用 checkpoint |

完整结构化目录和 BibTeX 位于仓库的
[method-references.json](../../method-references.json) 与
[references.bib](../../references.bib)。两种语言的本页及 BibTeX 均由
`python tools/reference_catalog.py --write` 从 JSON 生成。

## 平衡统计、动力学、反应速率与光谱

### pimd

**原始可区分粒子 PIMD**

出处：[cw81](#cw81), [pr84](#pr84), [pile10](#pile10)

闭合环聚合物平衡采样；采样时间不是精确量子实时间。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/engine.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/engine.py), [src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py), [src/pyrpmd/workflows/native_pimd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/native_pimd.py)

[该项复现任务](reproduction.md#task-pimd).

### rpmd

**RPMD 实时间关联函数**

出处：[rpm04](#rpm04), [kubo57](#kubo57)

用物理质量环聚合物动力学近似 Kubo 关联；检验 P=1 与谐振子线性可观测量极限。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py), [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py)

[该项复现任务](reproduction.md#task-rpmd).

### rpm-rate

**热平衡 RPMD 双分子反应速率**

出处：[bennett1977](#bennett1977), [chandler1978](#chandler1978), [rate05](#rate05), [rate05b](#rate05b), [bimol09](#bimol09), [rpmrate13](#rpmrate13), [bkmp21996](#bkmp21996)

质心自由能/QTST 因子乘以动力学回穿系数，并处理入口通道归一化。

源码：[src/pyrpmd/workflows/rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate.py), [src/pyrpmd/workflows/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_pmf.py), [src/pyrpmd/workflows/rate_recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_recrossing.py), [src/pyrpmd/analysis/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_pmf.py)

[该项复现任务](reproduction.md#task-rpm-rate).

### direct-rpmd

**直接轨迹热平衡 dRPMD**

出处：[drpmd18](#drpmd18)

热采样入射环聚合物，直接推进碰撞轨迹并统计截面、速率和反应通道。

源码：[src/pyrpmd/workflows/drpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd.py), [src/pyrpmd/workflows/drpmd_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd_sampling.py), [src/pyrpmd/workflows/drpmd_classification.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/drpmd_classification.py), [src/pyrpmd/analysis/scattering.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/scattering.py)

[该项复现任务](reproduction.md#task-direct-rpmd).

### state-rpmd

**谐振近似初态选择 NE-RPMD**

出处：[nerp16](#nerp16), [state20](#state20), [state21](#state21), [state22](#state22)

冷谐振环加质心振动激发，可做 j=0 投影；辅助 β 与碰撞能、热温度不同。

源码：[src/pyrpmd/initial_states.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/initial_states.py), [src/pyrpmd/workflows/state_selected.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/state_selected.py)

[该项复现任务](reproduction.md#task-state-rpmd).

### qct

**谐振 QCT 与全势能面作用量匹配 QCT**

出处：[qct65](#qct65), [state20](#state20), [state22](#state22)

主工作流是 P=1 谐振初态；H₃ 基准另有 J(E)=v+1/2 和按运动时间均匀抽相位的全势能面双原子初始化。

源码：[src/pyrpmd/initial_states.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/initial_states.py), [src/pyrpmd/workflows/state_selected.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/state_selected.py), [benchmarks/h3/diatomic_action.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/diatomic_action.py), [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/state_preparation_study.py)

[该项复现任务](reproduction.md#task-qct).

### state-switch

**实验性平滑切换初态诊断**

出处：[state20](#state20), [state22](#state22)

仓库自定义的制备哈密顿量到真实 PES 的平滑切换；NE-RPMD 文献仅作背景。

源码：[benchmarks/h3/diatomic_switch.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/diatomic_switch.py), [benchmarks/h3/state_preparation_study.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/h3/state_preparation_study.py)

[该项复现任务](reproduction.md#task-state-switch).

### integrators

**速度 Verlet、正常模 BAB 与 Cayley BCB**

出处：[vv82](#vv82), [pile10](#pile10), [cayley19](#cayley19)

物理质量的确定性环聚合物推进；分别使用直接 Verlet、解析自由环和 Cayley 自由环。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py)

[该项复现任务](reproduction.md#task-integrators).

### stochastic-integrators

**BAOAB 与 Cayley BCOCB**

出处：[baoab13](#baoab13), [bcocb20](#bcocb20), [cayley19](#cayley19)

对称 Langevin 劈分；BCOCB 的 C 半步是整步 Cayley 映射的平方根。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-stochastic-integrators).

### classical-baths

**Andersen、Cartesian Langevin 与随机速度缩放温控**

出处：[and80](#and80), [langevin07](#langevin07), [svr07](#svr07)

分别实现全局 Maxwell 重抽样、精确 OU 热浴及包含符号和零动能分支的正则速度缩放。

源码：[src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermostats.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py), [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py)

[该项复现任务](reproduction.md#task-classical-baths).

### pile

**PILE-L 与 PILE-G 正常模温控**

出处：[pile10](#pile10), [svr07](#svr07)

PILE-L 按环频率设局域摩擦；PILE-G 对内模用 PILE、质心用全局随机速度缩放。 内模摩擦按环频率设定，质心耦合单独设置，不能从零频模推断质心无热浴。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-pile).

### nhc

**Nosé–Hoover 链温控**

出处：[nhc92](#nhc92), [yoshida1990](#yoshida1990)

全局链加三阶段 Suzuki–Yoshida 组合；Python/native 共用质量和自由度约定。

源码：[src/pyrpmd/thermostats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermostats.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-nhc).

### gle

**Cartesian/正常模 GLE 热浴**

出处：[gle09](#gle09), [pile10](#pile10)

使用显式稳定漂移矩阵和协方差矩阵的扩展变量 OU 更新，可组合质心 SVR。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-gle).

### piglet

**外部标定矩阵驱动的 PIGLET**

出处：[piglet12](#piglet12)

NVT 正常模 GLE，要求矩阵分别匹配逆温度 β 和珠数 P 并记录来源；没有内置标定表。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-piglet).

### ffl

**Fast-forward Langevin 的 rescale 变体**

出处：[ffl18](#ffl18)

native 路线保留动量方向并更新其模长。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-ffl).

### constraints

**逐珠 SHAKE/RATTLE 键约束**

出处：[shake77](#shake77), [rattle83](#rattle83)

Verlet 旧梯度 SHAKE 位移与动量脉冲，随后做 RATTLE 速度切空间投影。

源码：[src/pyrpmd/bond_constraints.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bond_constraints.py), [src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py)

[该项复现任务](reproduction.md#task-constraints).

### rpc-mts

**环聚合物收缩 RPC 与可逆多时间步**

出处：[rpc08](#rpc08), [respa92](#respa92), [mts16](#mts16), [refrpc16](#refrpc16)

低频珠模式收缩慢力，配合两层 r-RESPA；采用参考势校正方案时，慢势为目标势减参考势。

源码：[src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/contraction.py), [src/pyrpmd/multitime.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/multitime.py), [src/pyrpmd/propagation.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/propagation.py)

[该项复现任务](reproduction.md#task-rpc-mts).

### pacmd

**部分绝热质心分子动力学 PA-CMD**

出处：[cmd93](#cmd93), [cmd94](#cmd94), [pacmd06](#pacmd06)

独立 PACMDDynamics 用内模虚质量、物理质心质量和自由环精确 BAOAB 推进。

源码：[src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cmd.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[该项复现任务](reproduction.md#task-pacmd).

### fcmd

**力匹配快速质心动力学 f-CMD**

出处：[cmd94](#cmd94), [fcmd05](#fcmd05)

导出质心/珠平均物理力，拟合局域二次保守质心势，并记录训练温度；IBI 只是更新模块。

源码：[src/pyrpmd/cmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cmd.py)

[该项复现任务](reproduction.md#task-fcmd).

### qcmd

**弯曲三原子分子的实验性 f-QCMD**

出处：[qcmd2019](#qcmd2019), [fqcmd2021](#fqcmd2021), [cmd94](#cmd94)

对非周期、固定拓扑的弯曲三原子分子，用珠平均键长/键角定义准质心；以迭代 Boltzmann 反演拟合可分离修正势；光谱工作流以 method="f-qcmd"、beads=1 运行经典动力学。约束 PIMD API 在固定平均内坐标下做平衡采样。实时间绝热 QCMD 与 h-CMD 未实现，会明确报错。

源码：[src/pyrpmd/qcmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/qcmd.py), [src/pyrpmd/_qcmd_geometry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_qcmd_geometry.py), [src/pyrpmd/_qcmd_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_qcmd_sampling.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[该项复现任务](reproduction.md#task-qcmd).

### trpmd

**TRPMD 与相对红外频谱**

出处：[trpmd14](#trpmd14), [rpm04](#rpm04), [kubo57](#kubo57), [water09](#water09)

内模摩擦、质心不温控；按 βω² 的 Kubo 偶极关联算相对 IR，不再叠加谐振量子修正。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py), [src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/workflows/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/spectroscopy.py)

[该项复现任务](reproduction.md#task-trpmd).

### trpmd-rates

**TRPMD 速率对照模式**

出处：[trpmd14](#trpmd14), [trprate15](#trprate15)

可用内模温控轨迹检查速率对摩擦的敏感性。

源码：[src/pyrpmd/dynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/dynamics.py), [src/pyrpmd/workflows/rate_recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_recrossing.py)

[该项复现任务](reproduction.md#task-trpmd-rates).

### raman

**各向同性/各向异性相对 Raman 光谱**

出处：[kubo57](#kubo57), [raman21](#raman21)

极化率张量关联分解为标量与无迹张量，明确 VV/VH 权重、Kubo→Stokes 及激光频率因子。

源码：[src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/properties.py)

[该项复现任务](reproduction.md#task-raman).

### sfg

**基于时间关联的相对 SFG 响应**

出处：[sfg02](#sfg02), [kubo57](#kubo57)

由偶极—极化率关联构造因果 R(t)=−βdC/dt，保留带符号 Imχ 和复振幅相干平均。

源码：[src/pyrpmd/analysis/spectroscopy.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/spectroscopy.py), [src/pyrpmd/properties.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/properties.py)

[该项复现任务](reproduction.md#task-sfg).

### ti-pcn

**Takahashi–Imada 迹作用量与 pCN 采样**

出处：[ti84](#ti84), [pcn13](#pcn13), [cw81](#cw81)

原始/TI 闭合路径迹作用量，以内模 pCN 和质心 Metropolis 采样；MC 步不是实时间。

源码：[src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py)

[该项复现任务](reproduction.md#task-ti-pcn).

### thermo-estimators

**量子动能、总能与热容估计器**

出处：[yam05](#yam05), [ti84](#ti84), [cw81](#cw81)

对指定迹作用量取 β 导数，并可使用缩放坐标有限差分 virial 估计及分块误差。

源码：[src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[该项复现任务](reproduction.md#task-thermo-estimators).

### isotopes

**同位素自由能与分馏估计**

出处：[iso13](#iso13), [iso14](#iso14)

热力学/缩放坐标 FEP、两环境分馏比及沿质量的动能热力学积分，并检查权重重叠。

源码：[src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[该项复现任务](reproduction.md#task-isotopes).

### momentum

**原始开路径单分量动量分布**

出处：[cw81](#cw81), [ceperley95](#ceperley95), [mom10](#mom10)

直接采样开链端点位移并 Fourier 得到一个 Cartesian 分量的 n(p)，带 KDE/分块 jackknife 诊断。

源码：[src/pyrpmd/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/path_integral.py), [src/pyrpmd/quantum_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/quantum_statistics.py)

[该项复现任务](reproduction.md#task-momentum).

### suzuki-chin

**有限差分 Suzuki–Chin 路径积分**

出处：[suz95](#suz95), [chin97](#chin97), [sc16](#sc16)

偶数珠交替权重与沿 F/m 的对称有限差分力梯度；Python 可选 α，CLI 当前限 α=0。

源码：[src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py), [src/pyrpmd/suzuki_chin.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/suzuki_chin.py)

[该项复现任务](reproduction.md#task-suzuki-chin).

### bosons

**二次复杂度玻色交换 PIMD**

出处：[bose19](#bose19), [bose23](#bose23)

同质量交换组的 Feldman–Hirshberg O(N²+PN) 递推，使用代表循环并测量对称量。

源码：[src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py)

[该项复现任务](reproduction.md#task-bosons).

### pigs

**正试探波函数的原始 PIGS**

出处：[pigs00](#pigs00)

奇数珠开链、端点半势权重和正 Gaussian/自定义正试探波函数；中点测量投影基态。

源码：[src/pyrpmd/pimd_methods.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pimd_methods.py), [src/pyrpmd/workflows/path_integral.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_integral.py)

[该项复现任务](reproduction.md#task-pigs).

### rpi

**环聚合物瞬子热速率**

出处：[rpi09](#rpi09), [rpi16](#rpi16)

优化驻定虚时间路径，计算半经典涨落行列式/前因子、谐振反应物归一化及负/零模；气相速率可声明外部刚体模。

源码：[src/pyrpmd/instanton.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/instanton.py), [src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[该项复现任务](reproduction.md#task-rpi).

### splitting

**对称束缚双阱的瞬子隧穿劈裂**

出处：[split11](#split11)

单 kink、孤立对称束缚或明确约束双态的半经典劈裂。

源码：[src/pyrpmd/tunnelling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/tunnelling.py), [src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[该项复现任务](reproduction.md#task-splitting).

### low-t-checks

**谐振 TST、Wigner 与 Eckart 对照**

出处：[eyr35](#eyr35), [wig32](#wig32), [eck30](#eck30)

经典/量子谐振 TST、Wigner 低阶修正和非对称 Eckart 透射热积分，并明确零点能/势垒约定。

源码：[src/pyrpmd/low_temperature.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/low_temperature.py)

[该项复现任务](reproduction.md#task-low-t-checks).

## 自由能、增强采样与反应路径

### samp-umbrella

**伞形采样与 1D–3D 自由能面**

出处：[samp-us1974](#samp-us1974), [samp-us1977](#samp-us1977)

实现静态谐和伞势、CV Jacobian 与环聚合物能量缩放。Torrie–Valleau 原论文提供重要性采样原理，不是本包动力学引擎的逐项实现。

源码：[src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/fes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/fes.py)

[该项复现任务](reproduction.md#task-samp-umbrella).

### samp-mbar

**MBAR 自由能、直方图 FES 与重叠诊断**

出处：[samp-mbar2008](#samp-mbar2008)

直接以 SciPy 求解 MBAR，再用目标权重生成直方图 FES；不是另一个 WHAM 求解器。当前误差路径是 bootstrap，不能由此声称实现原文全部渐近协方差公式。

源码：[src/pyrpmd/analysis/mbar.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/mbar.py), [src/pyrpmd/analysis/fes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes.py), [src/pyrpmd/analysis/umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/umbrella.py)

[该项复现任务](reproduction.md#task-samp-mbar).

### samp-ui

**高斯窗口伞形积分**

出处：[samp-ui2005](#samp-ui2005), [samp-uierr2006](#samp-uierr2006)

实现一维高斯窗口近似，并按 RPMDrate 数学形式移植；不包括后续高阶或多维伞形积分。

源码：[src/pyrpmd/analysis/umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/umbrella.py), [src/pyrpmd/analysis/rate_pmf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_pmf.py)

[该项复现任务](reproduction.md#task-samp-ui).

### samp-adaptive

**自适应伞窗布置与覆盖规划**

出处：[samp-mbar2008](#samp-mbar2008), [samp-us1977](#samp-us1977), [samp-ogre2023](#samp-ogre2023)

本包以归一化 MBAR 重叠图、窗口中点和覆盖缺口补点，保持刚度；pilot 后重新生成 production。OGRe 仅提供对照模型，其刚度细化和 confinement/consistency 判据未完整实现。

源码：[src/pyrpmd/workflows/adaptive_umbrella.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/adaptive_umbrella.py), [src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes_statistics.py)

[该项复现任务](reproduction.md#task-samp-adaptive).

### samp-standard-meta

**标准元动力学**

出处：[samp-meta2002](#samp-meta2002)

高斯历史偏置与有限时间 −V 自由能估计。Laio–Parrinello 提供方法起源，本包直接在 MD 工作流沉积偏置。

源码：[src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/metadynamics.py), [src/pyrpmd/analysis/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/metadynamics.py)

[该项复现任务](reproduction.md#task-samp-standard-meta).

### samp-wt-meta

**良调元动力学**

出处：[samp-wt2008](#samp-wt2008)

按偏置指数衰减 hill 高度，以 −γV/(γ−1) 估计自由能；bias_fes 不实现带 c(t) 的时间依赖轨迹重加权。

源码：[src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/workflows/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/metadynamics.py), [src/pyrpmd/analysis/metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/metadynamics.py)

[该项复现任务](reproduction.md#task-samp-wt-meta).

### samp-walkers

**共享偏置的多行者元动力学**

出处：[samp-walkers2006](#samp-walkers2006), [samp-wt2008](#samp-wt2008)

以 walkers_mode=serial、mpi-sync（每个 MPI rank 一个行者、按 rank 顺序同步批次、单一原子性系综 checkpoint）或 async-files（每个行者追加日志、最终合并）共享高斯（MetaD）或核（OPES）历史。共享同一偏置的行者在统计上相关；同步与重启协议是项目工程，原文 WHAM superposition 后处理未一并实现。

源码：[src/pyrpmd/shared_metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/shared_metadynamics.py), [src/pyrpmd/workflows/shared_metadynamics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/shared_metadynamics.py), [src/pyrpmd/walker_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/walker_exchange.py), [src/pyrpmd/workflows/_distributed_meta_mpi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_meta_mpi.py), [src/pyrpmd/workflows/_distributed_meta_async.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_meta_async.py), [src/pyrpmd/workflows/_distributed_opes_mpi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_opes_mpi.py), [src/pyrpmd/workflows/_distributed_opes_async.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/_distributed_opes_async.py)

[该项复现任务](reproduction.md#task-samp-walkers).

### samp-grid-meta

**网格元动力学与自由能投影**

出处：[samp-meta2002](#samp-meta2002), [samp-wt2008](#samp-wt2008)

网格插值、导数和带 bin 体积的 Boltzmann 边际化，是既有偏置与 FES 的数值表示，没有单独原创方法论文；缺失支持的掩膜投影不是 hill filtering。

源码：[src/pyrpmd/bias.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/bias.py), [src/pyrpmd/analysis/projection.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/projection.py)

[该项复现任务](reproduction.md#task-samp-grid-meta).

### samp-opes

**OPES_METAD 与冻结偏置重加权**

出处：[samp-opes2020](#samp-opes2020)

默认固定宽度，用未压缩且归一化的周期包裹高斯 KDE 与中心采样 Z 估计；冻结偏置后的直方图用 exp(βV)，不扣时间依赖 c(t)。

源码：[src/pyrpmd/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/opes.py), [src/pyrpmd/workflows/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/opes.py)

[该项复现任务](reproduction.md#task-samp-opes).

### samp-opes-explore

**OPES-explore 与可选自适应宽度**

出处：[samp-explore2022](#samp-explore2022), [samp-opes2020](#samp-opes2020)

explore 使用等权目标分布 KDE 与 (γ−1)/β 系数。可选 Welford/Scott-ESS 宽度更新是本包明确规则，区别于 PLUMED diffusion-window 规则。

源码：[src/pyrpmd/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/opes.py), [src/pyrpmd/workflows/opes.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/opes.py)

[该项复现任务](reproduction.md#task-samp-opes-explore).

### samp-ves

**质心 CV 上的变分增强采样（VES）**

出处：[samp-ves2014](#samp-ves2014), [samp-wtves2015](#samp-wtves2015)

fes.method="ves" 在一或两个非周期质心 CV 上优化张量积 Legendre 线性偏置，采用带完整协方差 Hessian 的平均随机梯度下降；均匀或 well-tempered 目标用 Gauss–Legendre 积分；冻结偏置生产与重加权输出通用 FESResult。不是 PLUMED VES 接口，不与其他基组、周期 CV、压浴或副本交换组合。

源码：[src/pyrpmd/ves.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/ves.py), [src/pyrpmd/workflows/ves.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/ves.py)

[该项复现任务](reproduction.md#task-samp-ves).

### samp-abf

**质心 ABF 与扩展系统 ABF（CZAR）**

出处：[samp-abf2001](#samp-abf2001), [samp-abf2008](#samp-abf2008), [samp-abfproj2015](#samp-abfproj2015), [samp-eabf2017](#samp-eabf2017)

fes.method="abf"/"eabf" 适用于一或两个非周期、Jacobian 行满秩的质心 CV。局部平均力由全环 Jacobian 与带线性 ramp 的质量加权对偶场给出；一维沿边积分、二维最小二乘 Helmholtz 投影得到 FES。eABF 对谐振耦合的虚拟变量施加偏置，并用 CZAR 重构 FES。要求全部原子可动并使用正则热浴；拒绝固定原子、周期胞/CV、压浴、副本交换和速率通量测度。

源码：[src/pyrpmd/abf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/abf.py), [src/pyrpmd/workflows/abf.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/abf.py)

[该项复现任务](reproduction.md#task-samp-abf).

### samp-steered

**质心拉伸动力学（steered MD）与非平衡功估计**

出处：[samp-jarzynski1997](#samp-jarzynski1997), [samp-crooks1999](#samp-crooks1999), [samp-bar1976](#samp-bar1976), [samp-shadow2013](#samp-shadow2013)

fes.method="steered" 沿正向和/或反向移动质心谐振约束，用 Jarzynski 或 Crooks/Bennett 接受比方程估计受约束端点间的自由能差。功按 H_P/P 计，总功 = 协议功 + 有限步长 shadow work，不含热浴热量。不生成 FESResult 网格，未实现 Hummer–Szabo 无偏 PMF 重构。

源码：[src/pyrpmd/steered.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/steered.py), [src/pyrpmd/workflows/steered.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/steered.py)

[该项复现任务](reproduction.md#task-samp-steered).

### samp-reus

**同温伞形采样副本交换**

出处：[samp-reus2000](#samp-reus2000)

固定一维/二维伞窗间交换完整环构型，使用 β/P 测度和 P 倍伞势；该工作流不进行温度交换。

源码：[src/pyrpmd/workflows/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/replica_exchange.py)

[该项复现任务](reproduction.md#task-samp-reus).

### samp-remd

**温度与哈密顿量副本交换 API**

出处：[samp-remd1999](#samp-remd1999), [samp-reus2000](#samp-reus2000)

串行 Python 引擎，对目标温度重新计算环弹簧作用并缩放动量；带 checkpoint 与 MPI 的可续跑 [tempering] 输入工作流见单独条目 samp-tempering。

源码：[src/pyrpmd/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/replica_exchange.py)

[该项复现任务](reproduction.md#task-samp-remd).

### samp-tempering

**可续跑的并行回火与哈密顿量副本交换工作流**

出处：[samp-pt1996](#samp-pt1996), [samp-remd1999](#samp-remd1999), [samp-reus2000](#samp-reus2000), [samp-mbar2008](#samp-mbar2008)

[tempering] 用于 task="pimd" 与部分 task="fes" 输入：显式或几何温度梯以及哈密顿量梯。交换计算所有状态的约化势（含各目标温度的环弹簧作用），动量按 sqrt(beta_i/beta_j) 缩放。按状态保存轨迹，报告接受率、往返、状态空间扩散与 MBAR 重叠；PIMD 估计量对每个目标状态重新计算。串行、进程与 MPI 执行顺序一致；与同温 [replica_exchange] REUS 相互独立。

源码：[src/pyrpmd/workflows/tempering.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/tempering.py), [src/pyrpmd/replica_exchange.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/replica_exchange.py)

[该项复现任务](reproduction.md#task-samp-tempering).

### samp-biased-clock

**有偏时间加速与删失速率推断**

出处：[samp-imeta2013](#samp-imeta2013), [samp-flooding2022](#samp-flooding2022)

后处理 API 按积分区间计算 exp[β(V−V_TS)] 时钟，支持删失曝光的指数 MLE 和重新拟合的 KS bootstrap；不是原生 infrequent-MetaD/OPES-flooding 生产工作流。

源码：[src/pyrpmd/biased_kinetics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/biased_kinetics.py)

[该项复现任务](reproduction.md#task-samp-biased-clock).

### samp-tis

**双向正则 shooting 的过渡界面采样**

出处：[samp-tis2003](#samp-tis2003), [samp-tis2005](#samp-tis2005), [samp-tps1998](#samp-tps1998), [samp-shoot2025](#samp-shoot2025)

Python API 使用确定性实质量环动力学，均匀选择 shooting frame 并重抽全部正则动量；路径有有限长度上限，绝对速率还需独立匹配的有效通量。

源码：[src/pyrpmd/transition_interface.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/transition_interface.py)

[该项复现任务](reproduction.md#task-samp-tis).

### samp-retis

**可选 minus 系综的交换式 TIS**

出处：[samp-retis2007](#samp-retis2007), [samp-tis2003](#samp-tis2003)

交换相同动力学下的界面路径系综并保留拒绝/重复路径。minus=true 增加 [0-] 及 [0-]<->[0+] 交换，并由停留时间在内部估计有效通量；否则需要匹配的外部通量。提供持久 checkpoint 与 task="path_sampling"；异步无限交换见单独条目 samp-infretis。

源码：[src/pyrpmd/retis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/retis.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[该项复现任务](reproduction.md#task-samp-retis).

### samp-pptis

**部分路径 TIS（PPTIS）与 REPPTIS**

出处：[samp-pptis2004](#samp-pptis2004), [samp-repptis2023](#samp-repptis2023), [samp-tis2003](#samp-tis2003)

以三界面局部路径系综和记忆约化递推计算跨越概率；REPPTIS 在相邻路径系综间增加副本交换。提供 Python API 与 task="path_sampling"，带持久 checkpoint。记忆丢失假设须用界面间距与完整 TIS 对照检验。

源码：[src/pyrpmd/pptis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pptis.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[该项复现任务](reproduction.md#task-samp-pptis).

### samp-ffs

**直接前向通量采样（FFS）**

出处：[samp-ffs2005](#samp-ffs2005), [samp-ffs2009](#samp-ffs2009)

对显式提供的平稳、时间齐次 Markov 步进回调 step(state, rng) 逐界面分裂：由 A 区历史得到通量、由试验得到条件概率，带持久 checkpoint，也可经 task="path_sampling" 运行。不需要逆向动力学；采样器不替用户选择热浴或物理过程。

源码：[src/pyrpmd/ffs.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/ffs.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[该项复现任务](reproduction.md#task-samp-ffs).

### samp-infretis

**异步无限交换 RETIS（∞RETIS）**

出处：[samp-infretis2022](#samp-infretis2022), [samp-infretis2024](#samp-infretis2024), [samp-retis2007](#samp-retis2007)

exchange="infinite" 以精确的积和式比值边缘概率把空闲路径分配给空闲正界面系综，嵌套界面下用二次复杂度递推计算。异步进程执行器配合系综锁与按票据顺序提交，保证精确续跑；minus 系综单独成块。保留完整 Maxwell 双向 shooting；未实现 2024 年子轨迹移动、MPI 与 bead/GPU 工作池。

源码：[src/pyrpmd/infinite_retis.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/infinite_retis.py), [src/pyrpmd/parallel/path_executor.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/parallel/path_executor.py), [src/pyrpmd/workflows/path_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/path_sampling.py)

[该项复现任务](reproduction.md#task-samp-infretis).

### samp-committor

**固定完整环构型的 RPMD committor**

出处：[samp-comm2000](#samp-comm2000), [samp-cp1934](#samp-cp1934)

固定所有 bead 坐标，独立重抽正则动量并传播无偏确定性 first-hit 轨迹。P=1 是经典构型 committor；P>1 是近似 RPMD committor，未决轨迹用精确二项区间构造保守包络。

源码：[src/pyrpmd/committor.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/committor.py)

[该项复现任务](reproduction.md#task-samp-committor).

### samp-pathcv

**对齐 RMSD 与参考路径 CV**

出处：[samp-kabsch1976](#samp-kabsch1976), [samp-path2007](#samp-path2007)

加权 proper Kabsch 对齐、显式尺度的平滑 s/z 路径变量。二面角、配位数和周期仿射 CV 是数学构件；TorchScript 仅提供用户模型值/导数，不包含学习 CV 的训练方法。

源码：[src/pyrpmd/advanced_cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_cv.py)

[该项复现任务](reproduction.md#task-samp-pathcv).

### samp-mlcolvar

**mlcolvar 学习的质心 CV（Deep-LDA / Deep-TICA，TorchScript 部署）**

出处：[samp-mlcolvar2023](#samp-mlcolvar2023), [samp-deeplda2020](#samp-deeplda2020), [samp-deeptica2021](#samp-deeptica2021)

训练在外部 mlcolvar 库中完成；PyRPMD 用 TorchScriptCV 求值导出的 CPU float64 TorchScript 图（预处理、网络与投影），包括笛卡尔导数和环珠缩放。依赖可选；部署检查不能证明 CV 质量。

源码：[src/pyrpmd/advanced_cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_cv.py)

[该项复现任务](reproduction.md#task-samp-mlcolvar).

### samp-string

**欧氏平均力与约束自适应 string**

出处：[samp-string2002](#samp-string2002), [samp-stringcv2006](#samp-stringcv2006)

固定端点的梯度松弛、重参数化和插点；需要调用者提供一致的自由能梯度。约束扩展仍用缩放欧氏几何，没有完整一般非线性 CV 的 MFEP 度量和条件项。

源码：[src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py), [src/pyrpmd/adaptive_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_nrpmd.py)

[该项复现任务](reproduction.md#task-samp-string).

### samp-ft-string

**质心 CV 空间中的有限温度弦方法**

出处：[samp-fts2005](#samp-fts2005), [samp-stringcv2006](#samp-stringcv2006)

在质心 CV 上做约束采样，使用平均 CV 质量度规、块诊断、固定端点的路径更新与精确的迭代边界重放，并复用已有弧长重参数化。像数和端点固定；限非周期、导数满秩的质心 CV。有限约束刚度下得到的是平滑化自由能路径。

源码：[src/pyrpmd/finite_temperature_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/finite_temperature_string.py), [src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py)

[该项复现任务](reproduction.md#task-samp-ft-string).

### samp-prfo

**分区有理函数鞍点优化**

出处：[samp-prfo1985](#samp-prfo1985)

每次生成有限差分 Hessian，分别求上坡/下坡增广本征问题并限制步长；两侧下降仅作连通性诊断，不是经过认证的量子化学 IRC。

源码：[src/pyrpmd/workflows/saddle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/saddle.py), [src/pyrpmd/workflows/prepare.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/prepare.py)

[该项复现任务](reproduction.md#task-samp-prfo).

### samp-cineb

**CI-NEB、改进切向与 FIRE 松弛**

出处：[samp-nebtangent2000](#samp-nebtangent2000), [samp-cineb2000](#samp-cineb2000), [samp-fire2006](#samp-fire2006)

实现能量加权切向、攀爬像和 FIRE；当前端到端 rate preparation 面向非周期分子笛卡尔体系，不能直接承担原论文表面 slab 计算。

源码：[src/pyrpmd/workflows/neb.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/neb.py)

[该项复现任务](reproduction.md#task-samp-cineb).

### samp-connectivity

**共价半径连通图与反应通道推断**

出处：[samp-radii2008](#samp-radii2008)

采用 Cordero 的 H–Cm 半径表，加本包 bond_scale=1.25 阈值和用户覆盖；这是近似图分类，不是电子键级模型。

源码：[src/pyrpmd/covalent.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/covalent.py), [src/pyrpmd/workflows/prepare.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/prepare.py)

[该项复现任务](reproduction.md#task-samp-connectivity).

### samp-metric

**一般非线性质心坐标的约束质量度量**

出处：[samp-bluemoon1989](#samp-bluemoon1989), [samp-blue1998](#samp-blue1998)

将受约束父样本转换为条件系综权重，并检验坐标重参数化不变性；属于 RPMD 速率理论组合，不等于实现全部 blue-moon 热力学积分变体。

源码：[src/pyrpmd/workflows/rate_constraints.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/rate_constraints.py), [src/pyrpmd/cv.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cv.py), [src/pyrpmd/cv_periodic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cv_periodic.py)

[该项复现任务](reproduction.md#task-samp-metric).

### samp-rate-postprocess

**二维通量与带符号多通道后处理**

出处：[samp-blue1998](#samp-blue1998), [bennett1977](#bennett1977), [chandler1978](#chandler1978), [rate05](#rate05), [bimol09](#bimol09)

基于 flux-over-population / Bennett–Chandler RPMD 的项目扩展；需要同时引用核心速率论文。FES 概率密度测度与速率通量测度具有不同因子，没有独立离线管线原创论文。

源码：[src/pyrpmd/analysis/rate_postprocess.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_postprocess.py), [src/pyrpmd/analysis/rate_io.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/rate_io.py), [src/pyrpmd/analysis/bond_flux.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/bond_flux.py), [src/pyrpmd/analysis/bond_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/bond_rate.py), [src/pyrpmd/analysis/entrance.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/entrance.py)

[该项复现任务](reproduction.md#task-samp-rate-postprocess).

### samp-statistics

**相关诊断与同步循环块 bootstrap**

出处：[samp-cblock1992](#samp-cblock1992), [samp-geyer1992](#samp-geyer1992)

循环块重采样依照 Politis–Romano；正相邻对截断仅是相关诊断。CV/平方/周期谐波探针、ceil(5g) 和 N/2 上限是项目规则，交换窗口与出口通道保持共同重采样时钟。

源码：[src/pyrpmd/analysis/fes_statistics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/fes_statistics.py), [src/pyrpmd/analysis/recrossing.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/recrossing.py)

[该项复现任务](reproduction.md#task-samp-statistics).

### samp-plumed

**PLUMED 质心偏置接口**

出处：[samp-plumed2009](#samp-plumed2009), [samp-plumed2014](#samp-plumed2014)

连接外部 PLUMED kernel，不能视为内部实现全部 PLUMED action；实际使用时同时引用软件及 plumed.dat 选择的每个 CV/偏置算法。

源码：[src/pyrpmd/plumed.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/plumed.py), [src/pyrpmd/workflows/biased_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/biased_sampling.py)

[该项复现任务](reproduction.md#task-samp-plumed).

## 非绝热方法与矩阵模型

### na-nrpmd-focused

**Focused 非平衡 MMST-NRPMD**

出处：[na-mm1979](#na-mm1979), [na-stock1997](#na-stock1997), [na-nrpmd2013](#na-nrpmd2013), [na-focused2019](#na-focused2019), [na-mint2018](#na-mint2018)

每个核 bead 配一组 MMST 变量，支持多电子态实/复 Hermitian 矩阵。采用 focused 初始电子态和有符号、未裁剪的布居估计量；不包括 PLDM、SQC，也不等同于 SEO 热平衡采样。

源码：[src/pyrpmd/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic.py), [src/pyrpmd/nonadiabatic_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_ensemble.py), [src/pyrpmd/workflows/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic.py)

[该项复现任务](reproduction.md#task-na-nrpmd-focused).

### na-mint

**MInt 电子映射积分子步**

出处：[na-mint2018](#na-mint2018)

固定核位置时精确传播电子映射旋转及其核动量冲量，再与自由环聚合子步对称组合。引用的是 Church 等人论文中的积分组成部分，包未实现该文完整 MQC-IVR。

源码：[src/pyrpmd/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic.py), [src/pyrpmd/spin_mapping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/spin_mapping.py)

[该项复现任务](reproduction.md#task-na-mint).

### na-thermal-nrpmd

**带符号 SEO 热平衡 NRPMD 与相关/速率估计量**

出处：[na-nrpmd2013](#na-nrpmd2013), [na-stock1997](#na-stock1997), [na-mint2018](#na-mint2018)

仅实对称矩阵：对 SEO 初始权重的绝对值采样，相关函数保留初始 sign。默认 scalar/traceless 分裂不同于 2013 论文的最低绝热面分裂；专用 published benchmark 明确实现后者。热分布不是普通 exp(-βH/P)。

源码：[src/pyrpmd/thermal_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermal_nrpmd.py), [src/pyrpmd/workflows/nonadiabatic_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_rate.py)

[该项复现任务](reproduction.md#task-na-thermal-nrpmd).

### na-adaptive-thermal

**自适应约束路径与带符号 NRPMD 组合工作流**

出处：[na-nrpmd2013](#na-nrpmd2013), [samp-stringcv2006](#samp-stringcv2006), [samp-mbar2008](#samp-mbar2008)

组合 SEO 热采样、约束路径适应、自由能重加权及冻结路径后的独立生产采样。未核实存在一篇同时提出 PyRPMD 完整组合的原始论文；应分别引用 NRPMD、有限温度 string 和 MBAR。

源码：[src/pyrpmd/adaptive_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_nrpmd.py), [src/pyrpmd/adaptive_string.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adaptive_string.py), [src/pyrpmd/thermal_nrpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/thermal_nrpmd.py)

[该项复现任务](reproduction.md#task-na-adaptive-thermal).

### na-sm-nrpmd

**Focused 两态自旋映射 NRPMD**

出处：[na-sm2021](#na-sm2021)

每个 bead 一个 W-spin，半径 √3/2，focused u_z=±1/√3；支持两态实/复 Hermitian 矩阵。当前未实现通用 SU(N)，也未实现原文 Γ_W 平衡 trace/phase 采样。

源码：[src/pyrpmd/spin_mapping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/spin_mapping.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[该项复现任务](reproduction.md#task-na-sm-nrpmd).

### na-mash

**经典核 MASH 与实验性质心扩展**

出处：[na-mash2023](#na-mash2023), [na-mashrates2024](#na-mashrates2024)

MASHDynamics 通过自旋半球确定活动态。P=1 对应原始两态 MASH；P>1 是实验性质心扩展，与独立的 CA-RP-MASH 类不同。跳跃定位在步末，因此需要时间步收敛。

源码：[src/pyrpmd/mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mash.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[该项复现任务](reproduction.md#task-na-mash).

### na-carp-mash

**CA-RP-MASH（命令行 rp-mash）**

出处：[na-carp2025](#na-carp2025), [na-mash2023](#na-mash2023), [na-mashrates2024](#na-mashrates2024)

整条环共用一个自旋和活动面，核力在各 bead 求值、电子动力学在质心求值；限两态实对称、非零能隙。DOP853 事件求根定位过赤道跳跃并按全环能量重标动量；BA-RP-MASH 与多态 MASH 见单独条目（na-ba-rp-mash、na-ms-mash），不含复 SOC。

源码：[src/pyrpmd/rp_mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_mash.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[该项复现任务](reproduction.md#task-na-carp-mash).

### na-ba-rp-mash

**BA-RP-MASH（命令行 rp-mash-ba）**

出处：[na-carp2025](#na-carp2025), [na-mash2023](#na-mash2023)

两态珠近似：整条环共用一个自旋并采用共同绝热表象，各珠分别计算矩阵、梯度、能隙与导数耦合，电子演化取珠平均。状态、工作流名称和科学身份均与 CA-RP-MASH（rp-mash）分开；不是多态或复 SOC MASH。

源码：[src/pyrpmd/rp_mash_ba.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_mash_ba.py), [src/pyrpmd/nonadiabatic_registry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_registry.py)

[该项复现任务](reproduction.md#task-na-ba-rp-mash).

### na-ms-mash

**原始多态 MASH（命令行 multistate-mash）**

出处：[na-msmash2023](#na-msmash2023), [na-mash2023](#na-mash2023)

一个归一化复映射矢量，活动绝热面取映射布居最大的态，使用 Runeson–Manolopoulos 带符号观测量估计；限实哈密顿量、非简并能级、单珠。实现的是 2023 年原始表述，不是 unSMASH 或后续初始化变体；即使两态也不同于两态 MASH 估计量。

源码：[src/pyrpmd/mash.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mash.py), [src/pyrpmd/workflows/nonadiabatic_multistate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_multistate.py)

[该项复现任务](reproduction.md#task-na-ms-mash).

### na-rpsh-ca

**FSSH 与质心近似 RPSH**

出处：[na-tully1990](#na-tully1990), [na-rpsh2012](#na-rpsh2012), [na-rpsh2017](#na-rpsh2017), [na-rpshdm2023](#na-rpshdm2023)

实固定 diabatic 的 RPSHDynamics、扩展 CA 类和直接 adiabatic 接口均共用一个环活动面；P=1 是经典核 FSSH 极限。动量重标、受阻跃迁及初态约定应与目标论文一致。

源码：[src/pyrpmd/surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/surface_hopping.py), [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[该项复现任务](reproduction.md#task-na-rpsh-ca).

### na-rpsh-ba

**Bead 平均表面跳跃 RPSH-BA**

出处：[na-rpshdm2023](#na-rpshdm2023), [na-rpsh2012](#na-rpsh2012), [na-tully1990](#na-tully1990)

只有一组电子振幅，由各 bead 的能量和速度耦合项平均驱动；整条环共用活动态，并非每个 bead 单独选电子面。

源码：[src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[该项复现任务](reproduction.md#task-na-rpsh-ba).

### na-rpsh-dm

**密度矩阵表面跳跃 RPSH-DM**

出处：[na-rpshdm2023](#na-rpshdm2023)

各 bead 独立传播电子振幅，密度变换至共同质心基底后平均，因此可出现纯度降低；环仍共用活动面。此变体不叠加 EDC。

源码：[src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[该项复现任务](reproduction.md#task-na-rpsh-dm).

### na-edc

**能量型去相干修正 EDC**

出处：[na-edc2007](#na-edc2007)

按能隙和所选动能定义阻尼非活动态振幅，并保留活动振幅相位地重新归一化；RPSH 的质心/全环能量是显式适配约定。支持 CA/BA，不作为 DM 的额外修正。

源码：[src/pyrpmd/surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/surface_hopping.py), [src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[该项复现任务](reproduction.md#task-na-edc).

### na-soc-rpsh

**复 Hermitian / SOC 表面跳跃与局域几何力**

出处：[na-soc2019](#na-soc2019), [na-tully1990](#na-tully1990), [na-rpsh2012](#na-rpsh2012)

按 Miao 等人 method 2 从复 NAC 选择实动量重标方向，并可用 Cayley 旋转加入反对称 Berry 力。限局域非简并修正，不等同完整几何相位干涉或简并态非 Abelian 算法。

源码：[src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py), [src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[该项复现任务](reproduction.md#task-na-soc-rpsh).

### na-dboc-rpsh

**投影对角 Born–Oppenheimer 修正 RPSH**

出处：[na-dboc2025](#na-dboc2025)

计算所含电子态的 Σ|d_ij|²/(2m) 并差分其梯度，同时修正核力和跃迁能隙。截断态、近简并和差分步长限制应注明；效果随动力学区间变化。

源码：[src/pyrpmd/advanced_surface_hopping.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/advanced_surface_hopping.py)

[该项复现任务](reproduction.md#task-na-dboc-rpsh).

### na-iesh

**经典 IESH 与质心 RP-IESH（命令行 iesh、rp-iesh）**

出处：[na-iesh2009](#na-iesh2009), [na-iesh2023](#na-iesh2023), [na-rpsh2012](#na-rpsh2012), [na-rpsh2017](#na-rpsh2017)

离散金属能带中独立电子的单个无自旋 Slater 行列式；核在裸能量加占据轨道能之和上运动，按最少跳跃做单激发跳跃。RP-IESH 整条环共用一个行列式并在质心传播电子。限实哈密顿量、非简并能级；输出单粒子密度布居；不是 GPU 或原生全轨迹求解器。

源码：[src/pyrpmd/iesh.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/iesh.py), [src/pyrpmd/rp_iesh.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/rp_iesh.py), [src/pyrpmd/workflows/nonadiabatic_multistate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/nonadiabatic_multistate.py)

[该项复现任务](reproduction.md#task-na-iesh).

### na-qd-interface

**准 diabatic NRPMD 与即时绝热电子结构接口**

出处：[na-qd2018](#na-qd2018), [na-qd2019](#na-qd2019), [na-nrpmd2013](#na-nrpmd2013), [na-mint2018](#na-mint2018), [na-tully1990](#na-tully1990)

provider 提供能量、梯度、NAC 和态重叠；用局域基底重叠传输映射变量及重建非对角梯度。QDNRPMD 是 QD 策略与 NRPMD 的组合，并未实现原文的 PLDM/SQC。缺少真实重叠时的 endpoint-NAC 传输属于额外近似。

源码：[src/pyrpmd/adiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/adiabatic.py)

[该项复现任务](reproduction.md#task-na-qd-interface).

### na-excited-state

**在线激发态计算接口与基于重叠的耦合**

出处：[na-hst1994](#na-hst1994), [na-npi2014](#na-npi2014), [na-npi2016](#na-npi2016), [pyscf2018](#pyscf2018)

ExcitedStateProvider 向已有绝热动力学（如 AdiabaticRPSHDynamics）提供各态总能量、正梯度、导数耦合、相邻帧重叠、态追踪和可续跑的电子上下文。由重叠得到耦合时采用 Hammes-Schiffer–Tully 有限差分，或在 Löwdin 正交化后采用多态广义 NPI。PySCF TDA/TDDFT 已测试；ORCA、Q-Chem、OpenMolcas 文件适配器仅经解析与模拟子进程测试。该接口不选择动力学方法。

源码：[src/pyrpmd/potentials/excited_state](https://github.com/chen-yu-hao/PyRPMD/tree/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/excited_state)

[该项复现任务](reproduction.md#task-na-excited-state).

### na-mv-rpmd

**映射变量环聚合动力学 MV-RPMD**

出处：[na-mv2013](#na-mv2013)

实固定 diabatic 多态 signed projected-Wigner trace 采样，保留符号。包使用精确有限 bead 电子矩阵指数，原文使用 primitive 短时近似；非可分电子流用 implicit midpoint，遇 trace 节点/符号跨越报错。

源码：[src/pyrpmd/mv_rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/mv_rpmd.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[该项复现任务](reproduction.md#task-na-mv-rpmd).

### na-kc-rpmd

**动力学约束环聚合方法 KC-RPMD**

出处：[na-kc2014](#na-kc2014)

实现原始 primitive 两态正耦合构造，用户给定 a/b、交叉校准 η 与辅助质量。限 0<a≤1、b>0、P≥2、βK/P<1；不自动包含后续 KC 变体。

源码：[src/pyrpmd/kc_rpmd.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/kc_rpmd.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[该项复现任务](reproduction.md#task-na-kc-rpmd).

### na-isomorphic

**两态同构势与 Ehrenfest 动力学**

出处：[na-iso2018](#na-iso2018)

实现实两态、整环 trace 匹配的 isomorphic Hamiltonian，再用单一归一化电子波函数做 Ehrenfest。不是原文 QCLE 或 iso-surface-hopping，也不含通用多态构造或隐含热采样。

源码：[src/pyrpmd/isomorphic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/isomorphic.py), [src/pyrpmd/_matrix_path.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/_matrix_path.py)

[该项复现任务](reproduction.md#task-na-isomorphic).

### na-tully-models

**Tully I/II/III 解析矩阵势**

出处：[na-tully1990](#na-tully1990)

实现单避免交叉、双避免交叉和扩展耦合反射三种势及解析导数。核质量属于 system/采样配置，不会由势类自动指定。

源码：[src/pyrpmd/potentials/nonadiabatic_models.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nonadiabatic_models.py)

[该项复现任务](reproduction.md#task-na-tully-models).

### na-generic-matrix-models

**通用 LVC / 离散 spin-boson 与谐振环初态**

出处：[na-focused2019](#na-focused2019), [na-sm2021](#na-sm2021)

LVC 和离散 spin-boson 是参数化容器，须为每个应用另引参数来源。harmonic ring 采样是显式声明的有限 P 非耦合谐振初态，不自动给出耦合系统热平衡密度，也未内置特定 pyrazine/FMO 参数或通用浴离散规则。

源码：[src/pyrpmd/potentials/nonadiabatic_models.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nonadiabatic_models.py), [src/pyrpmd/nonadiabatic_sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/nonadiabatic_sampling.py)

[该项复现任务](reproduction.md#task-na-generic-matrix-models).

### na-nah2-pes

**NaH₂ LEPS-5G 两态 diabatic 势**

出处：[na-nah21992](#na-nah21992), [na-nah2err1994](#na-nah2err1994)

历史中性 NaH₂ 的两个最低双重态 ²A′ 耦合势及矩阵导数；应同时引用物理 PES 原文与勘误，软件适配器引用单列。

源码：[src/pyrpmd/potentials/nah2_leps.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/nah2_leps.py)

[该项复现任务](reproduction.md#task-na-nah2-pes).

### na-quantum-reference

**有限区域矩阵分裂算符量子基准**

出处：[na-feit1982](#na-feit1982)

一维周期 Fourier 网格加逐点 Hermitian 矩阵指数和二阶对称分裂，支持纯态及谐振热混合态。没有吸收边界和通用多维传播；需分别收敛空间区域、网格、时间步与热截断。

源码：[src/pyrpmd/analysis/quantum_reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/quantum_reference.py)

[该项复现任务](reproduction.md#task-na-quantum-reference).

## 材料、电化学、势能与接口

### bzp

**各向同性 BZP 恒压 PIMD**

出处：[bzp2009](#bzp2009), [ipi2014](#ipi2014)

只缩放质心，使用物理位力及 PIMD 系综测度修正；要求三维周期且全部原子可动。

源码：[src/pyrpmd/barostat.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/barostat.py), [src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-bzp).

### mtk

**柔性晶胞 MTTK/MTK PIMD**

出处：[mtk1994](#mtk1994), [nptpimd1999](#nptpimd1999)

上三角应变率、下三角行晶格矢量；静水压及可选晶胞分量掩码。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-mtk).

### rgb

**RGB 外应力耦合**

出处：[rgb2011](#rgb2011)

显式对称外应力及参考晶胞应变能模型。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-rgb).

### scr

**随机晶胞重缩放 SCR**

出处：[scr2020](#scr2020)

仅各向同性 P=1；平方根体积随机提议，无 Metropolis 校正。

源码：[src/pyrpmd/native_ensemble.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/native_ensemble.py)

[该项复现任务](reproduction.md#task-scr).

### diffusion

**RPMD 扩散、MSD/VACF 与张量分析**

出处：[pd2024](#pd2024), [ni2012](#ni2012)

展开质心的 Einstein/Green–Kubo 估计及副本/分块误差；材料应用论文与基础 RPMD 文献同时引用。

源码：[src/pyrpmd/analysis/diffusion.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/diffusion.py), [src/pyrpmd/workflows/materials.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/materials.py)

[该项复现任务](reproduction.md#task-diffusion).

### site-surface

**盆地归一化位点/表面速率**

出处：[sds2025](#sds2025), [pd2024](#pd2024), [bazro2008](#bazro2008)

单分界面与反应物盆地归一化；全可动位点非线性坐标和固定基底表面坐标的支持范围不同。

源码：[src/pyrpmd/workflows/site_rate.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/site_rate.py), [src/pyrpmd/workflows/surface.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/surface.py), [src/pyrpmd/workflows/surface_source.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/surface_source.py)

[该项复现任务](reproduction.md#task-site-surface).

### nialh-model

**NiAlH EAM 试算势**

出处：[nialh1995](#nialh1995)

使用外部固定版本势表进行集成试算。

源码：[benchmarks/materials/README.md](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/benchmarks/materials/README.md)

[该项复现任务](reproduction.md#task-nialh-model).

### pcet

**局域黄金规则电化学 PCET**

出处：[pcet2008](#pcet2008), [pypcet2025](#pypcet2025), [gold2019](#gold2019)

单质子网格本征态、固定供受体构型、常电子耦合、宽带金属及经典 Gaussian 环境。

源码：[src/pyrpmd/pcet.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pcet.py), [src/pyrpmd/workflows/pcet.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/pcet.py)

[该项复现任务](reproduction.md#task-pcet).

### gc

**共享电子数的巨正则 HMC/MC**

出处：[hmc1987](#hmc1987), [hastings1970](#hastings1970), [gc2025](#gc2025)

项目组合：核坐标 HMC 与有界连续共享电子数 Metropolis 更新；MC 步数不代表物理时间。

源码：[src/pyrpmd/electrochemistry/sampling.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/sampling.py)

[该项复现任务](reproduction.md#task-gc).

### adiabatic-electrode

**绝热恒电位标量与矩阵包装**

出处：[gc2025](#gc2025)

项目近似：逐构型最小化 E(R,N)-mu*N；矩阵版本给所有透热态加入相同标量充电能。

源码：[src/pyrpmd/electrochemistry/potentials.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/potentials.py), [src/pyrpmd/electrochemistry/nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/nonadiabatic.py), [src/pyrpmd/electrochemistry/diabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/diabatic.py)

[该项复现任务](reproduction.md#task-adiabatic-electrode).

### her

**Volmer–Heyrovsky–Tafel 平均场后处理**

出处：[gc2025](#gc2025)

使用外部给定正逆基元系数及理想独立位点的项目质量作用定律平衡。

源码：[src/pyrpmd/electrochemistry/kinetics.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electrochemistry/kinetics.py)

[该项复现任务](reproduction.md#task-her).

### efrpmd

**Markov 质心电子摩擦 RPMD**

出处：[efrpmd2024](#efrpmd2024)

珠平均半正定摩擦作用于质心；各珠共享随机冲量，内部模不受该浴直接热化。

源码：[src/pyrpmd/electronic_friction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electronic_friction.py)

[该项复现任务](reproduction.md#task-efrpmd).

### pyscf

**PySCF 参考数据与即时受力接口**

出处：[pyscf2018](#pyscf2018)

将 RHF/UHF/RKS/UKS 能量、梯度及所支持 Hessian 委托给外部 PySCF。

源码：[src/pyrpmd/pesgen/reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/reference.py), [src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[该项复现任务](reproduction.md#task-pyscf).

### populations

**Hirshfeld 电荷与 Mayer 键级**

出处：[hirshfeld1977](#hirshfeld1977), [mayer1983](#mayer1983)

中性球对称 HF 原子参考的 stockholder 分区；非限制 Mayer 包含自旋密度。

源码：[src/pyrpmd/pesgen/reference.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/reference.py)

[该项复现任务](reproduction.md#task-populations).

### qmdff

**QMDFF 势生成**

出处：[qmdff2014](#qmdff2014), [caracal2023](#caracal2023)

以平衡构型 QM 数据调用固定源码版本 Caracal QMDFF 生成器。

源码：[src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py), [src/pyrpmd/pesgen/formats.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/formats.py)

[该项复现任务](reproduction.md#task-qmdff).

### deevb

**dE-EVB / EVB-QMDFF 耦合拟合**

出处：[deevb2015](#deevb2015), [caracal2023](#caracal2023)

以给定路径能量及上游 sp/1g 耦合拟合连接端点 QMDFF。

源码：[src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py)

[该项复现任务](reproduction.md#task-deevb).

### treq

**TREQ 势生成**

出处：[treq2019](#treq2019), [caracal2023](#caracal2023)

使用上游反应路径/二阶展开构造；公开桥接明确使用四阶差分力。

源码：[src/pyrpmd/pesgen/workflow.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/workflow.py), [src/pyrpmd/pesgen/build.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/build.py)

[该项复现任务](reproduction.md#task-treq).

### jaxpip

**线性 PIP / PIP-NN 拟合与 JaxPIP 导出**

出处：[pip2010](#pip2010), [pipnn2013](#pipnn2013), [jaxpip2026](#jaxpip2026)

显式训练/验证/测试集、能量/力损失、JaxPIP 描述符及原生 ONNX 导出；模型导出与势能面验证是不同步骤。

源码：[src/pyrpmd/pesgen/jaxpip_fit.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_fit.py), [src/pyrpmd/pesgen/jaxpip_export.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_export.py), [src/pyrpmd/pesgen/jaxpip_data.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/pesgen/jaxpip_data.py)

[该项复现任务](reproduction.md#task-jaxpip).

### committee

**委员会不确定度与样本采集**

出处：[committee2020](#committee2020)

固定提供器均值、总体标准差分歧、离线采集，以及 ActiveLearningLoop：用固定委员会采样、在标注预算内选择超阈值候选、标注、重训练并安装下一代委员会。分歧不是经过校准的误差界；采样时不施加分歧偏置。

源码：[src/pyrpmd/potentials/lifecycle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lifecycle.py), [src/pyrpmd/active_learning.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/active_learning.py)

[该项复现任务](reproduction.md#task-committee).

### delta

**加性 Δ 学习与 RPC/MTS**

出处：[delta2015](#delta2015)

将给定校正模型加到固定参考势，并提供快/慢力分解；RPC/MTS 另有方法来源。

源码：[src/pyrpmd/potentials/lifecycle.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lifecycle.py), [src/pyrpmd/potentials/contraction.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/contraction.py)

[该项复现任务](reproduction.md#task-delta).

### ase

**ASE 计算器接口**

出处：[ase2017](#ase2017)

能量、力、周期晶胞和应力转换；使用与力一致的电子自由能。

源码：[src/pyrpmd/potentials/optional.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/optional.py)

[该项复现任务](reproduction.md#task-ase).

### openmm

**OpenMM 力接口**

出处：[openmm2017](#openmm2017)

外部受力计算、虚拟位点与周期位力；核动力学由 PyRPMD 执行。

源码：[src/pyrpmd/potentials/openmm.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/openmm.py)

[该项复现任务](reproduction.md#task-openmm).

### lammps

**LAMMPS 库接口**

出处：[lammps1995](#lammps1995)

每个上下文调用外部库受力/晶胞；不是重写 LAMMPS 算法。

源码：[src/pyrpmd/potentials/lammps.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/lammps.py)

[该项复现任务](reproduction.md#task-lammps).

### ipi-socket

**i-PI socket 接口**

出处：[ipi2014](#ipi2014)

具有显式晶胞和单位约定的 socket 受力协议。

源码：[src/pyrpmd/potentials/socket.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/socket.py)

[该项复现任务](reproduction.md#task-ipi-socket).

### torch

**Torch/GPU 势及张量传播**

出处：[torch2019](#torch2019)

自动微分能量/力及已有物理积分器的张量版本。

源码：[src/pyrpmd/potentials/optional.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/optional.py), [src/pyrpmd/torch_backend.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/torch_backend.py)

[该项复现任务](reproduction.md#task-torch).

### ml-adapters

**MACE / NequIP / DeePMD / fairchem 接口**

出处：[mace2022](#mace2022), [nequip2022](#nequip2022), [deepmd2018](#deepmd2018), [uma2025](#uma2025)

调用外部推理引擎；模型论文及准确权重需另引，不能称为仓库重实现。

源码：[src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[该项复现任务](reproduction.md#task-ml-adapters).

### tblite

**tblite 的 GFN1/GFN2/IPEA1-xTB**

出处：[gfn12017](#gfn12017), [gfn22019](#gfn22019), [ipea2017](#ipea2017)

将指定电子哈密顿量委托给 tblite；明确电荷、多重度及电子温度。

源码：[src/pyrpmd/potentials/adapters.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/adapters.py)

[该项复现任务](reproduction.md#task-tblite).

### h5md-export

**H5MD、extxyz 与 ASE 轨迹导出**

出处：[h5md2014](#h5md2014), [ase2017](#ase2017)

H5MD 格式来源与 ASE 互操作格式；不是新动力学方法。

源码：[src/pyrpmd/trajectory_export.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory_export.py)

[该项复现任务](reproduction.md#task-h5md-export).

### engineering

**输入、续跑、并行、报告与运行组织**

出处：—

项目工程：TOML/Python 规范、哈希、续跑、C ABI/f2py、原子性传播块、process/MPI/OpenMP/GPU 调度、报告及速率矩阵装配。

源码：[src/pyrpmd/input.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/input.py), [src/pyrpmd/checkpoint.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/checkpoint.py), [src/pyrpmd/parallel](https://github.com/chen-yu-hao/PyRPMD/tree/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/parallel), [src/pyrpmd/trajectory.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory.py), [src/pyrpmd/analysis/reporting.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/analysis/reporting.py), [src/pyrpmd/reactions.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/reactions.py), [src/pyrpmd/workflows/network.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/workflows/network.py), [src/pyrpmd/cli.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/cli.py), [src/pyrpmd/runner.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/runner.py), [src/pyrpmd/input_schema.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/input_schema.py), [src/pyrpmd/batched.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/batched.py), [src/pyrpmd/batched_nonadiabatic.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/batched_nonadiabatic.py), [src/pyrpmd/trajectory_batch.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/trajectory_batch.py), [src/pyrpmd/electronic_backends.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/electronic_backends.py), [src/pyrpmd/geometry.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/geometry.py)

[该项复现任务](reproduction.md#task-engineering).

### analytic-models

**解析自由粒子、谐振子、Morse、四次双阱与周期余弦模型**

出处：[morse1929](#morse1929)

MorsePairPotential 使用 Morse 函数形式；自由粒子、谐振子、可分离四次双阱和周期余弦是明确给定的解析控制模型，不是拟合的分子或金属势。

源码：[src/pyrpmd/potentials/builtin.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/builtin.py)

[该项复现任务](reproduction.md#task-analytic-models).

### rate-studies

**温度/同位素扫描、Arrhenius 拟合与数值收敛研究**

出处：[bimol09](#bimol09), [iso13](#iso13)

对已有速率与同位素方法进行研究组织和回归拟合；这个包内扫描/报告组合不硬配一篇独立原始论文。

源码：[src/pyrpmd/studies.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/studies.py)

[该项复现任务](reproduction.md#task-rate-studies).

### bkmp2-model

**BKMP2 H₃ 论文基准与外部势能面溯源**

出处：[bkmp21996](#bkmp21996), [bimol09](#bimol09), [rpmrate13](#rpmrate13)

基准使用外部 BKMP2 H₃ 势，不是 PyRPMD 新拟合的势；除速率理论和软件接口外还要引用势能面原论文。

源码：[src/pyrpmd/potentials/native.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/native.py), [src/pyrpmd/potentials/cabi.py](https://github.com/chen-yu-hao/PyRPMD/blob/28cde092b66707878d99bc45eadf6bdf5950b317/src/pyrpmd/potentials/cabi.py)

[该项复现任务](reproduction.md#task-bkmp2-model).

## 完整参考文献

下列“用途”描述保留英文以精确区分出处身份；论文题名使用出版方原题名。

### cw81

D. Chandler; P. G. Wolynes. **Exploiting the isomorphism between quantum theory and classical statistical mechanics of polyatomic fluids**. *J. Chem. Phys. 74, 4078–4095* (1981). DOI: [10.1063/1.441588](https://doi.org/10.1063/1.441588).

用途 / Role: Ring-polymer statistical-mechanical foundation.

### pr84

M. Parrinello; A. Rahman. **Study of an F center in molten KCl**. *J. Chem. Phys. 80, 860–867* (1984). DOI: [10.1063/1.446740](https://doi.org/10.1063/1.446740).

用途 / Role: Early path-integral molecular-dynamics application.

### rpm04

I. R. Craig; D. E. Manolopoulos. **Quantum statistics and classical mechanics: Real time correlation functions from ring polymer molecular dynamics**. *J. Chem. Phys. 121, 3368–3373* (2004). DOI: [10.1063/1.1777575](https://doi.org/10.1063/1.1777575).

用途 / Role: Original method.

### rate05

I. R. Craig; D. E. Manolopoulos. **Chemical reaction rates from ring polymer molecular dynamics**. *J. Chem. Phys. 122, 084106* (2005). DOI: [10.1063/1.1850093](https://doi.org/10.1063/1.1850093).

用途 / Role: Original method.

### rate05b

I. R. Craig; D. E. Manolopoulos. **A refined ring polymer molecular dynamics theory of chemical reaction rates**. *J. Chem. Phys. 123, 034102* (2005). DOI: [10.1063/1.1954769](https://doi.org/10.1063/1.1954769).

用途 / Role: 2005 RPMD rate-theory refinement.

### bimol09

R. Collepardo-Guevara; Y. V. Suleimanov; D. E. Manolopoulos. **Bimolecular reaction rates from ring polymer molecular dynamics**. *J. Chem. Phys. 130, 174713* (2009). DOI: [10.1063/1.3127145](https://doi.org/10.1063/1.3127145).

用途 / Role: Bimolecular extension; consult its errata and corrected conventions.

### rpmrate13

Y. V. Suleimanov; J. W. Allen; W. H. Green. **RPMDrate: Bimolecular chemical reaction rates from ring polymer molecular dynamics**. *Comput. Phys. Commun. 184, 833–840* (2013). DOI: [10.1016/j.cpc.2012.10.017](https://doi.org/10.1016/j.cpc.2012.10.017).

用途 / Role: Implementation/workflow source; also cite the underlying rate-method papers.

### drpmd18

Y. V. Suleimanov; A. Aguado; S. Gómez-Carrasco; O. Roncero. **A Ring Polymer Molecular Dynamics Approach to Study the Transition between Statistical and Direct Mechanisms in the H₂ + H₃⁺ → H₃⁺ + H₂ Reaction**. *J. Phys. Chem. Lett. 9, 2133–2137* (2018). DOI: [10.1021/acs.jpclett.8b00783](https://doi.org/10.1021/acs.jpclett.8b00783).

用途 / Role: Original method.

### qct65

M. Karplus; R. N. Porter; R. D. Sharma. **Exchange Reactions with Activation Energy. I. Simple Barrier Potential for (H, H₂)**. *J. Chem. Phys. 43, 3259–3287* (1965). DOI: [10.1063/1.1697301](https://doi.org/10.1063/1.1697301).

用途 / Role: Foundational quasiclassical-trajectory application.

### nerp16

R. Welsch; K. Song; Q. Shi; S. C. Althorpe; T. F. Miller III. **Non-equilibrium dynamics from RPMD and CMD**. *J. Chem. Phys. 145, 204118* (2016). DOI: [10.1063/1.4967958](https://doi.org/10.1063/1.4967958).

用途 / Role: Original method.

### state20

A. Marjollet; R. Welsch. **Nuclear quantum effects in state-selective scattering from ring polymer molecular dynamics**. *J. Chem. Phys. 152, 194113* (2020). DOI: [10.1063/5.0004179](https://doi.org/10.1063/5.0004179).

用途 / Role: Original method.

### state21

A. Marjollet; R. Welsch. **State-selective cross sections from ring polymer molecular dynamics**. *Int. J. Quantum Chem. 121, e26447* (2021). DOI: [10.1002/qua.26447](https://doi.org/10.1002/qua.26447).

用途 / Role: Detailed state-selected methodology and tests; first published online 2020.

### state22

A. Marjollet; L. Inhester; R. Welsch. **Initial state-selected scattering for the reactions H + CH₄/CHD₃ and F + CHD₃ employing ring polymer molecular dynamics**. *J. Chem. Phys. 156, 044101* (2022). DOI: [10.1063/5.0076216](https://doi.org/10.1063/5.0076216).

用途 / Role: Polyatomic extension; harmonic initialization equations used by the implementation.

### cayley19

R. Korol; N. Bou-Rabee; T. F. Miller III. **Cayley modification for strongly stable path-integral and ring-polymer molecular dynamics**. *J. Chem. Phys. 151, 124103* (2019). DOI: [10.1063/1.5120282](https://doi.org/10.1063/1.5120282).

用途 / Role: Original method.

### bcocb20

R. Korol; J. L. Rosa-Raíces; N. Bou-Rabee; T. F. Miller III. **Dimension-free path-integral molecular dynamics without preconditioning**. *J. Chem. Phys. 152, 104102* (2020). DOI: [10.1063/1.5134810](https://doi.org/10.1063/1.5134810).

用途 / Role: Original method.

### vv82

W. C. Swope; H. C. Andersen; P. H. Berens; K. R. Wilson. **A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: Application to small water clusters**. *J. Chem. Phys. 76, 637–649* (1982). DOI: [10.1063/1.442716](https://doi.org/10.1063/1.442716).

用途 / Role: Velocity-Verlet implementation reference.

### and80

H. C. Andersen. **Molecular dynamics simulations at constant pressure and/or temperature**. *J. Chem. Phys. 72, 2384–2393* (1980). DOI: [10.1063/1.439486](https://doi.org/10.1063/1.439486).

用途 / Role: Collision thermostat; PyRPMD uses a global discrete collision-clock variant.

### nhc92

G. J. Martyna; M. L. Klein; M. Tuckerman. **Nosé–Hoover chains: The canonical ensemble via continuous dynamics**. *J. Chem. Phys. 97, 2635–2643* (1992). DOI: [10.1063/1.463940](https://doi.org/10.1063/1.463940).

用途 / Role: Original method.

### pile10

M. Ceriotti; M. Parrinello; T. E. Markland; D. E. Manolopoulos. **Efficient stochastic thermostatting of path integral molecular dynamics**. *J. Chem. Phys. 133, 124104* (2010). DOI: [10.1063/1.3489925](https://doi.org/10.1063/1.3489925).

用途 / Role: Original method.

### svr07

G. Bussi; D. Donadio; M. Parrinello. **Canonical sampling through velocity rescaling**. *J. Chem. Phys. 126, 014101* (2007). DOI: [10.1063/1.2408420](https://doi.org/10.1063/1.2408420).

用途 / Role: Original method.

### gle09

M. Ceriotti; G. Bussi; M. Parrinello. **Langevin Equation with Colored Noise for Constant-Temperature Molecular Dynamics Simulations**. *Phys. Rev. Lett. 102, 020601* (2009). DOI: [10.1103/PhysRevLett.102.020601](https://doi.org/10.1103/PhysRevLett.102.020601).

用途 / Role: Original method.

### piglet12

M. Ceriotti; D. E. Manolopoulos. **Efficient First-Principles Calculation of the Quantum Kinetic Energy and Momentum Distribution of Nuclei**. *Phys. Rev. Lett. 109, 100604* (2012). DOI: [10.1103/PhysRevLett.109.100604](https://doi.org/10.1103/PhysRevLett.109.100604).

用途 / Role: PIGLET thermostat; paper's transient anisotropic Gaussian momentum estimator is not implemented here.

### ffl18

M. Hijazi; D. M. Wilkins; M. Ceriotti. **Fast-forward Langevin dynamics with momentum flips**. *J. Chem. Phys. 148, 184109* (2018). DOI: [10.1063/1.5029833](https://doi.org/10.1063/1.5029833).

用途 / Role: Current native implementation uses the rescale variant.

### langevin07

G. Bussi; M. Parrinello. **Accurate sampling using Langevin dynamics**. *Phys. Rev. E 75, 056707* (2007). DOI: [10.1103/PhysRevE.75.056707](https://doi.org/10.1103/PhysRevE.75.056707).

用途 / Role: Langevin splitting/OU integration reference.

### baoab13

B. Leimkuhler; C. Matthews. **Robust and efficient configurational molecular sampling via Langevin dynamics**. *J. Chem. Phys. 138, 174102* (2013). DOI: [10.1063/1.4802990](https://doi.org/10.1063/1.4802990).

用途 / Role: BAOAB construction; ring-polymer extensions also require PILE/BCOCB references.

### shake77

J.-P. Ryckaert; G. Ciccotti; H. J. C. Berendsen. **Numerical integration of the Cartesian equations of motion of a system with constraints: molecular dynamics of n-alkanes**. *J. Comput. Phys. 23, 327–341* (1977). DOI: [10.1016/0021-9991(77)90098-5](https://doi.org/10.1016/0021-9991(77)90098-5).

用途 / Role: Original method.

### rattle83

H. C. Andersen. **Rattle: A 'velocity' version of the shake algorithm for molecular dynamics calculations**. *J. Comput. Phys. 52, 24–34* (1983). DOI: [10.1016/0021-9991(83)90014-1](https://doi.org/10.1016/0021-9991(83)90014-1).

用途 / Role: Original method.

### rpc08

T. E. Markland; D. E. Manolopoulos. **An efficient ring polymer contraction scheme for imaginary time path integral simulations**. *J. Chem. Phys. 129, 024105* (2008). DOI: [10.1063/1.2953308](https://doi.org/10.1063/1.2953308).

用途 / Role: Original method.

### respa92

M. Tuckerman; B. J. Berne; G. J. Martyna. **Reversible multiple time scale molecular dynamics**. *J. Chem. Phys. 97, 1990–2001* (1992). DOI: [10.1063/1.463137](https://doi.org/10.1063/1.463137).

用途 / Role: Original method.

### mts16

V. Kapil; J. VandeVondele; M. Ceriotti. **Accurate molecular dynamics and nuclear quantum effects at low cost by multiple steps in real and imaginary time: Using density functional theory to accelerate wavefunction methods**. *J. Chem. Phys. 144, 054111* (2016). DOI: [10.1063/1.4941091](https://doi.org/10.1063/1.4941091).

用途 / Role: Original method.

### refrpc16

O. Marsalek; T. E. Markland. **Ab initio molecular dynamics with nuclear quantum effects at classical cost: Ring polymer contraction for density functional theory**. *J. Chem. Phys. 144, 054112* (2016). DOI: [10.1063/1.4941093](https://doi.org/10.1063/1.4941093).

用途 / Role: Reference-potential correction application.

### cmd93

J. Cao; G. A. Voth. **A new perspective on quantum time correlation functions**. *J. Chem. Phys. 99, 10070–10073* (1993). DOI: [10.1063/1.465512](https://doi.org/10.1063/1.465512).

用途 / Role: Early centroid-dynamics paper; quartic model in existing benchmark.

### cmd94

J. Cao; G. A. Voth. **The formulation of quantum statistical mechanics based on the Feynman path centroid density. II. Dynamical properties**. *J. Chem. Phys. 100, 5106–5117* (1994). DOI: [10.1063/1.467176](https://doi.org/10.1063/1.467176).

用途 / Role: Original method.

### pacmd06

T. D. Hone; P. J. Rossky; G. A. Voth. **A comparative study of imaginary time path integral based methods for quantum dynamics**. *J. Chem. Phys. 124, 154103* (2006). DOI: [10.1063/1.2186636](https://doi.org/10.1063/1.2186636).

用途 / Role: Original method.

### fcmd05

T. D. Hone; S. Izvekov; G. A. Voth. **Fast centroid molecular dynamics: A force-matching approach for the predetermination of the effective centroid forces**. *J. Chem. Phys. 122, 054105* (2005). DOI: [10.1063/1.1836731](https://doi.org/10.1063/1.1836731).

用途 / Role: Original method.

### qcmd2019

G. Trenins; M. J. Willatt; S. C. Althorpe. **Path-integral dynamics of water using curvilinear centroids**. *J. Chem. Phys. 151, 054109* (2019). DOI: [10.1063/1.5100587](https://doi.org/10.1063/1.5100587).

用途 / Role: Original quasi-centroid molecular dynamics (QCMD).

### fqcmd2021

T. Fletcher; A. Zhu; J. E. Lawrence; D. E. Manolopoulos. **Fast quasi-centroid molecular dynamics**. *J. Chem. Phys. 155, 231101* (2021). DOI: [10.1063/5.0076704](https://doi.org/10.1063/5.0076704).

用途 / Role: Original fast QCMD with a separable correction potential.

### trpmd14

M. Rossi; M. Ceriotti; D. E. Manolopoulos. **How to remove the spurious resonances from ring polymer molecular dynamics**. *J. Chem. Phys. 140, 234116* (2014). DOI: [10.1063/1.4883861](https://doi.org/10.1063/1.4883861).

用途 / Role: Original method.

### trprate15

T. J. H. Hele; Y. V. Suleimanov. **Should thermostatted ring polymer molecular dynamics be used to calculate thermal reaction rates?**. *J. Chem. Phys. 143, 074107* (2015). DOI: [10.1063/1.4928599](https://doi.org/10.1063/1.4928599).

用途 / Role: Assessment of TRPMD rate limitations, not a new accuracy guarantee.

### water09

S. Habershon; T. E. Markland; D. E. Manolopoulos. **Competing quantum effects in the dynamics of a flexible water model**. *J. Chem. Phys. 131, 024501* (2009). DOI: [10.1063/1.3167790](https://doi.org/10.1063/1.3167790).

用途 / Role: q-TIP4P/F potential and liquid-water dynamical benchmark.

### kubo57

R. Kubo. **Statistical-Mechanical Theory of Irreversible Processes. I. General Theory and Simple Applications to Magnetic and Conduction Problems**. *J. Phys. Soc. Jpn. 12, 570–586* (1957). DOI: [10.1143/JPSJ.12.570](https://doi.org/10.1143/JPSJ.12.570).

用途 / Role: Original linear-response/correlation foundation.

### raman21

K. O. H. M. Dundas; M. T. P. Beerepoot; M. Ringholm; S. Reine; R. Bast; N. H. List; J. Kongsted; K. Ruud; J. M. H. Olsen. **Harmonic Infrared and Raman Spectra in Molecular Environments Using the Polarizable Embedding Model**. *J. Chem. Theory Comput. 17, 3599–3617* (2021). DOI: [10.1021/acs.jctc.0c01323](https://doi.org/10.1021/acs.jctc.0c01323).

用途 / Role: Formula/convention source for Raman invariants; not the historical origin of Raman theory, and polarizable embedding is not implemented.

### sfg02

A. Morita; J. T. Hynes. **A Theoretical Analysis of the Sum Frequency Generation Spectrum of the Water Surface. II. Time-Dependent Approach**. *J. Phys. Chem. B 106, 673–685* (2002). DOI: [10.1021/jp0133438](https://doi.org/10.1021/jp0133438).

用途 / Role: Original dipole–polarizability time-correlation SFG response framework; online 2001.

### ti84

M. Takahashi; M. Imada. **Monte Carlo Calculation of Quantum Systems. II. Higher Order Correction**. *J. Phys. Soc. Jpn. 53, 3765–3769* (1984). DOI: [10.1143/JPSJ.53.3765](https://doi.org/10.1143/JPSJ.53.3765).

用途 / Role: Original method.

### yam05

T. M. Yamamoto. **Path-integral virial estimator based on the scaling of fluctuation coordinates: Application to quantum clusters with fourth-order propagators**. *J. Chem. Phys. 123, 104101* (2005). DOI: [10.1063/1.2013257](https://doi.org/10.1063/1.2013257).

用途 / Role: Original method.

### iso13

M. Ceriotti; T. E. Markland. **Efficient methods and practical guidelines for simulating isotope effects**. *J. Chem. Phys. 138, 014112* (2013). DOI: [10.1063/1.4772676](https://doi.org/10.1063/1.4772676).

用途 / Role: Original method.

### iso14

B. Cheng; M. Ceriotti. **Direct path integral estimators for isotope fractionation ratios**. *J. Chem. Phys. 141, 244112* (2014). DOI: [10.1063/1.4904293](https://doi.org/10.1063/1.4904293).

用途 / Role: Original method.

### mom10

L. Lin; J. A. Morrone; R. Car; M. Parrinello. **Displaced Path Integral Formulation for the Momentum Distribution of Quantum Particles**. *Phys. Rev. Lett. 105, 110602* (2010). DOI: [10.1103/PhysRevLett.105.110602](https://doi.org/10.1103/PhysRevLett.105.110602).

用途 / Role: Related original momentum-distribution method; displaced-path free-energy estimator is not implemented.

### ceperley95

D. M. Ceperley. **Path integrals in the theory of condensed helium**. *Rev. Mod. Phys. 67, 279–355* (1995). DOI: [10.1103/RevModPhys.67.279](https://doi.org/10.1103/RevModPhys.67.279).

用途 / Role: Review/background for off-diagonal density matrices, explicitly not an original-method claim.

### pcn13

S. L. Cotter; G. O. Roberts; A. M. Stuart; D. White. **MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster**. *Statistical Science 28, 424–446* (2013). DOI: [10.1214/13-STS421](https://doi.org/10.1214/13-STS421).

用途 / Role: pCN Gaussian-preserving Metropolis proposals.

### sc16

V. Kapil; J. Behler; M. Ceriotti. **High order path integrals made easy**. *J. Chem. Phys. 145, 234103* (2016). DOI: [10.1063/1.4971438](https://doi.org/10.1063/1.4971438).

用途 / Role: Finite-difference Suzuki–Chin force implementation.

### suz95

M. Suzuki. **Hybrid exponential product formulas for unbounded operators with possible applications to Monte Carlo simulations**. *Phys. Lett. A 201, 425–428* (1995). DOI: [10.1016/0375-9601(95)00266-6](https://doi.org/10.1016/0375-9601(95)00266-6).

用途 / Role: Higher-order factorization foundation.

### chin97

S. A. Chin. **Symplectic integrators from composite operator factorizations**. *Phys. Lett. A 226, 344–348* (1997). DOI: [10.1016/S0375-9601(97)00003-0](https://doi.org/10.1016/S0375-9601(97)00003-0).

用途 / Role: Force-gradient factorization foundation.

### bose19

B. Hirshberg; V. Rizzi; M. Parrinello. **Path integral molecular dynamics for bosons**. *Proc. Natl. Acad. Sci. USA 116, 21445–21449* (2019). DOI: [10.1073/pnas.1913365116](https://doi.org/10.1073/pnas.1913365116).

用途 / Role: Original method.

### bose23

Y. M. Y. Feldman; B. Hirshberg. **Quadratic scaling bosonic path integral molecular dynamics**. *J. Chem. Phys. 159, 154107* (2023). DOI: [10.1063/5.0173749](https://doi.org/10.1063/5.0173749).

用途 / Role: Quadratic recurrence implemented by PyRPMD.

### pigs00

A. Sarsa; K. E. Schmidt; W. R. Magro. **Path integral ground state method**. *J. Chem. Phys. 113, 1366–1371* (2000). DOI: [10.1063/1.481926](https://doi.org/10.1063/1.481926).

用途 / Role: Original method.

### rpi09

J. O. Richardson; S. C. Althorpe. **Ring-polymer molecular dynamics rate-theory in the deep-tunneling regime: Connection with semiclassical instanton theory**. *J. Chem. Phys. 131, 214106* (2009). DOI: [10.1063/1.3267318](https://doi.org/10.1063/1.3267318).

用途 / Role: Original method.

### rpi16

J. O. Richardson. **Derivation of instanton rate theory from first principles**. *J. Chem. Phys. 144, 114106* (2016). DOI: [10.1063/1.4943866](https://doi.org/10.1063/1.4943866).

用途 / Role: Later first-principles derivation of instanton rate theory.

### split11

J. O. Richardson; S. C. Althorpe. **Ring-polymer instanton method for calculating tunneling splittings**. *J. Chem. Phys. 134, 054109* (2011). DOI: [10.1063/1.3530589](https://doi.org/10.1063/1.3530589).

用途 / Role: Original method.

### eck30

C. Eckart. **The Penetration of a Potential Barrier by Electrons**. *Phys. Rev. 35, 1303–1309* (1930). DOI: [10.1103/PhysRev.35.1303](https://doi.org/10.1103/PhysRev.35.1303).

用途 / Role: Original method.

### wig32

E. Wigner. **Über das Überschreiten von Potentialschwellen bei chemischen Reaktionen**. *Z. Phys. Chem. B 19, 203–216* (1932). DOI: [10.1515/zpch-1932-1920](https://doi.org/10.1515/zpch-1932-1920).

用途 / Role: Original method.

### eyr35

H. Eyring. **The Activated Complex in Chemical Reactions**. *J. Chem. Phys. 3, 107–115* (1935). DOI: [10.1063/1.1749604](https://doi.org/10.1063/1.1749604).

用途 / Role: Original method.

### yoshida1990

H. Yoshida. **Construction of higher order symplectic integrators**. *Physics Letters A 150, 262–268* (1990). DOI: [10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601(90)90092-3).

用途 / Role: Original higher-order symmetric composition coefficients.

### samp-us1974

G. M. Torrie; J. P. Valleau. **Monte Carlo free energy estimates using non-Boltzmann sampling: Application to the sub-critical Lennard-Jones fluid**. *Chemical Physics Letters 28, 578–581* (1974). DOI: [10.1016/0009-2614(74)80109-0](https://doi.org/10.1016/0009-2614(74)80109-0).

用途 / Role: Original method.

### samp-us1977

G. M. Torrie; J. P. Valleau. **Nonphysical sampling distributions in Monte Carlo free-energy estimation: Umbrella sampling**. *Journal of Computational Physics 23, 187–199* (1977). DOI: [10.1016/0021-9991(77)90121-8](https://doi.org/10.1016/0021-9991(77)90121-8).

用途 / Role: Original method.

### samp-mbar2008

M. R. Shirts; J. D. Chodera. **Statistically optimal analysis of samples from multiple equilibrium states**. *Journal of Chemical Physics 129, 124105* (2008). DOI: [10.1063/1.2978177](https://doi.org/10.1063/1.2978177).

用途 / Role: Original method.

### samp-ui2005

J. Kästner; W. Thiel. **Bridging the gap between thermodynamic integration and umbrella sampling provides a novel analysis method: “Umbrella integration”**. *Journal of Chemical Physics 123, 144104* (2005). DOI: [10.1063/1.2052648](https://doi.org/10.1063/1.2052648).

用途 / Role: Original method.

### samp-uierr2006

J. Kästner; W. Thiel. **Analysis of the statistical error in umbrella sampling simulations by umbrella integration**. *Journal of Chemical Physics 124, 234106* (2006). DOI: [10.1063/1.2206775](https://doi.org/10.1063/1.2206775).

用途 / Role: Statistical analysis / window design background.

### samp-ogre2023

S. Borgmans; S. M. J. Rogge; L. Vanduyfhuys; V. Van Speybroeck. **OGRe: Optimal Grid Refinement Protocol for Accurate Free Energy Surfaces and Its Application in Proton Hopping in Zeolites and 2D COF Stacking**. *Journal of Chemical Theory and Computation 19, 9032–9048* (2023). DOI: [10.1021/acs.jctc.3c01028](https://doi.org/10.1021/acs.jctc.3c01028).

用途 / Role: Benchmark-model source and algorithm comparison; not PyRPMD planner identity.

### samp-meta2002

A. Laio; M. Parrinello. **Escaping free-energy minima**. *PNAS 99, 12562–12566* (2002). DOI: [10.1073/pnas.202427399](https://doi.org/10.1073/pnas.202427399).

用途 / Role: Original method.

### samp-wt2008

A. Barducci; G. Bussi; M. Parrinello. **Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method**. *Physical Review Letters 100, 020603* (2008). DOI: [10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603).

用途 / Role: Original method.

### samp-walkers2006

P. Raiteri; A. Laio; F. L. Gervasio; C. Micheletti; M. Parrinello. **Efficient Reconstruction of Complex Free Energy Landscapes by Multiple Walkers Metadynamics**. *Journal of Physical Chemistry B 110, 3533–3539* (2006). DOI: [10.1021/jp054359r](https://doi.org/10.1021/jp054359r).

用途 / Role: Original method.

### samp-opes2020

M. Invernizzi; M. Parrinello. **Rethinking Metadynamics: From Bias Potentials to Probability Distributions**. *Journal of Physical Chemistry Letters 11, 2731–2736* (2020). DOI: [10.1021/acs.jpclett.0c00497](https://doi.org/10.1021/acs.jpclett.0c00497).

用途 / Role: Original method.

### samp-explore2022

M. Invernizzi; M. Parrinello. **Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling**. *Journal of Chemical Theory and Computation 18, 3988–3996* (2022). DOI: [10.1021/acs.jctc.2c00152](https://doi.org/10.1021/acs.jctc.2c00152).

用途 / Role: Original method.

### samp-ves2014

O. Valsson; M. Parrinello. **Variational Approach to Enhanced Sampling and Free Energy Calculations**. *Phys. Rev. Lett. 113, 090601* (2014). DOI: [10.1103/PhysRevLett.113.090601](https://doi.org/10.1103/PhysRevLett.113.090601).

用途 / Role: Original variationally enhanced sampling functional.

### samp-wtves2015

O. Valsson; M. Parrinello. **Well-Tempered Variational Approach to Enhanced Sampling**. *J. Chem. Theory Comput. 11, 1996–2002* (2015). DOI: [10.1021/acs.jctc.5b00076](https://doi.org/10.1021/acs.jctc.5b00076).

用途 / Role: Well-tempered target distribution for VES.

### samp-abf2001

E. Darve; A. Pohorille. **Calculating free energies using average force**. *J. Chem. Phys. 115, 9169–9183* (2001). DOI: [10.1063/1.1410978](https://doi.org/10.1063/1.1410978).

用途 / Role: Original adaptive biasing force (average force) method.

### samp-abf2008

E. Darve; D. Rodríguez-Gómez; A. Pohorille. **Adaptive biasing force method for scalar and vector free energy calculations**. *J. Chem. Phys. 128, 144120* (2008). DOI: [10.1063/1.2829861](https://doi.org/10.1063/1.2829861).

用途 / Role: Multidimensional ABF mean-force formula with first derivatives.

### samp-abfproj2015

H. Alrachid; T. Lelièvre. **Long-time convergence of an adaptive biasing force method: Variance reduction by Helmholtz projection**. *SMAI J. Comput. Math. 1, 55–82* (2015). DOI: [10.5802/smai-jcm.4](https://doi.org/10.5802/smai-jcm.4).

用途 / Role: Helmholtz projection of the estimated mean force used for 2D integration.

### samp-eabf2017

A. Lesage; T. Lelièvre; G. Stoltz; J. Hénin. **Smoothed Biasing Forces Yield Unbiased Free Energies with the Extended-System Adaptive Biasing Force Method**. *J. Phys. Chem. B 121, 3676–3685* (2017). DOI: [10.1021/acs.jpcb.6b10055](https://doi.org/10.1021/acs.jpcb.6b10055).

用途 / Role: eABF and the CZAR free-energy estimator.

### samp-jarzynski1997

C. Jarzynski. **Nonequilibrium Equality for Free Energy Differences**. *Phys. Rev. Lett. 78, 2690–2693* (1997). DOI: [10.1103/PhysRevLett.78.2690](https://doi.org/10.1103/PhysRevLett.78.2690).

用途 / Role: Original nonequilibrium work equality.

### samp-crooks1999

G. E. Crooks. **Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences**. *Phys. Rev. E 60, 2721* (1999). DOI: [10.1103/PhysRevE.60.2721](https://doi.org/10.1103/PhysRevE.60.2721).

用途 / Role: Original work fluctuation theorem behind the bidirectional estimator.

### samp-bar1976

C. H. Bennett. **Efficient estimation of free energy differences from Monte Carlo data**. *J. Comput. Phys. 22, 245–268* (1976). DOI: [10.1016/0021-9991(76)90078-4](https://doi.org/10.1016/0021-9991(76)90078-4).

用途 / Role: Original acceptance-ratio (BAR) estimator used for bidirectional work.

### samp-shadow2013

D. A. Sivak; J. D. Chodera; G. E. Crooks. **Using Nonequilibrium Fluctuation Theorems to Understand and Correct Errors in Equilibrium and Nonequilibrium Simulations of Discrete Langevin Dynamics**. *Phys. Rev. X 3, 011007* (2013). DOI: [10.1103/PhysRevX.3.011007](https://doi.org/10.1103/PhysRevX.3.011007).

用途 / Role: Shadow-work interpretation of finite-timestep Langevin error.

### samp-remd1999

Y. Sugita; Y. Okamoto. **Replica-exchange molecular dynamics method for protein folding**. *Chemical Physics Letters 314, 141–151* (1999). DOI: [10.1016/S0009-2614(99)01123-9](https://doi.org/10.1016/S0009-2614(99)01123-9).

用途 / Role: Original method.

### samp-reus2000

Y. Sugita; A. Kitao; Y. Okamoto. **Multidimensional replica-exchange method for free-energy calculations**. *Journal of Chemical Physics 113, 6042–6051* (2000). DOI: [10.1063/1.1308516](https://doi.org/10.1063/1.1308516).

用途 / Role: Original method.

### samp-pt1996

K. Hukushima; K. Nemoto. **Exchange Monte Carlo Method and Application to Spin Glass Simulations**. *J. Phys. Soc. Jpn. 65, 1604–1608* (1996). DOI: [10.1143/JPSJ.65.1604](https://doi.org/10.1143/JPSJ.65.1604).

用途 / Role: Original exchange Monte Carlo (parallel tempering) method.

### samp-imeta2013

P. Tiwary; M. Parrinello. **From Metadynamics to Dynamics**. *Physical Review Letters 111, 230602* (2013). DOI: [10.1103/PhysRevLett.111.230602](https://doi.org/10.1103/PhysRevLett.111.230602).

用途 / Role: Original method.

### samp-flooding2022

D. Ray; N. Ansari; V. Rizzi; M. Invernizzi; M. Parrinello. **Rare Event Kinetics from Adaptive Bias Enhanced Sampling**. *Journal of Chemical Theory and Computation 18, 6500–6509* (2022). DOI: [10.1021/acs.jctc.2c00806](https://doi.org/10.1021/acs.jctc.2c00806).

用途 / Role: Acceleration-clock applicability; native OPES-flooding exclusion not implemented.

### samp-tis2003

T. S. van Erp; D. Moroni; P. G. Bolhuis. **A novel path sampling method for the calculation of rate constants**. *Journal of Chemical Physics 118, 7762–7774* (2003). DOI: [10.1063/1.1562614](https://doi.org/10.1063/1.1562614).

用途 / Role: Original method.

### samp-tis2005

T. S. van Erp; P. G. Bolhuis. **Elaborating transition interface sampling methods**. *Journal of Computational Physics 205, 157–181* (2005). DOI: [10.1016/j.jcp.2004.11.003](https://doi.org/10.1016/j.jcp.2004.11.003).

用途 / Role: Detailed path/flux measure and extensions.

### samp-retis2007

T. S. van Erp. **Reaction rate calculation by parallel path swapping**. *Physical Review Letters 98, 268301* (2007). DOI: [10.1103/PhysRevLett.98.268301](https://doi.org/10.1103/PhysRevLett.98.268301).

用途 / Role: Original method.

### samp-pptis2004

D. Moroni; P. G. Bolhuis; T. S. van Erp. **Rate constants for diffusive processes by partial path sampling**. *J. Chem. Phys. 120, 4055–4065* (2004). DOI: [10.1063/1.1644537](https://doi.org/10.1063/1.1644537).

用途 / Role: Original partial-path TIS memory-reduction method.

### samp-repptis2023

W. Vervust; D. T. Zhang; T. S. van Erp; A. Ghysels. **Path sampling with memory reduction and replica exchange to reach long permeation timescales**. *Biophysical Journal 122, 2960–2972* (2023). DOI: [10.1016/j.bpj.2023.02.021](https://doi.org/10.1016/j.bpj.2023.02.021).

用途 / Role: Original REPPTIS formulation.

### samp-ffs2005

R. J. Allen; P. B. Warren; P. R. ten Wolde. **Sampling Rare Switching Events in Biochemical Networks**. *Phys. Rev. Lett. 94, 018104* (2005). DOI: [10.1103/PhysRevLett.94.018104](https://doi.org/10.1103/PhysRevLett.94.018104).

用途 / Role: Original forward flux sampling method.

### samp-ffs2009

R. J. Allen; C. Valeriani; P. R. ten Wolde. **Forward flux sampling for rare event simulations**. *J. Phys.: Condens. Matter 21, 463102* (2009). DOI: [10.1088/0953-8984/21/46/463102](https://doi.org/10.1088/0953-8984/21/46/463102).

用途 / Role: FFS review; stochastic-splitting construction followed by the implementation.

### samp-infretis2022

S. Roet; D. T. Zhang; T. S. van Erp. **Exchanging Replicas with Unequal Cost, Infinitely and Permanently**. *J. Phys. Chem. A 126, 8878–8886* (2022). DOI: [10.1021/acs.jpca.2c06004](https://doi.org/10.1021/acs.jpca.2c06004).

用途 / Role: Original asynchronous infinite-swap replica exchange (∞RETIS) algorithm.

### samp-infretis2024

D. T. Zhang; L. Baldauf; S. Roet; A. Lervik; T. S. van Erp. **Highly parallelizable path sampling with minimal rejections using asynchronous replica exchange and infinite swaps**. *Proc. Natl. Acad. Sci. USA 121, e2318731121* (2024). DOI: [10.1073/pnas.2318731121](https://doi.org/10.1073/pnas.2318731121).

用途 / Role: Later ∞RETIS developments; its subtrajectory moves are not implemented.

### samp-tps1998

P. G. Bolhuis; C. Dellago; D. Chandler. **Sampling ensembles of deterministic transition pathways**. *Faraday Discussions 110, 421–436* (1998). DOI: [10.1039/A801266K](https://doi.org/10.1039/A801266K).

用途 / Role: Deterministic trajectory sampling antecedent.

### samp-shoot2025

S. Falkner; A. Coretti; B. Peters; P. G. Bolhuis; C. Dellago. **Revisiting shooting point Monte Carlo methods for transition path sampling**. *Journal of Chemical Physics 163, 034105* (2025). DOI: [10.1063/5.0261744](https://doi.org/10.1063/5.0261744).

用途 / Role: Acceptance-rule analysis for shooting indices.

### samp-comm2000

P. G. Bolhuis; C. Dellago; D. Chandler. **Reaction coordinates of biomolecular isomerization**. *PNAS 97, 5877–5882* (2000). DOI: [10.1073/pnas.100127697](https://doi.org/10.1073/pnas.100127697).

用途 / Role: Primary molecular committor application; not a new exact-quantum committor algorithm.

### samp-cp1934

C. J. Clopper; E. S. Pearson. **The use of confidence or fiducial limits illustrated in the case of the binomial**. *Biometrika 26, 404–413* (1934). DOI: [10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404).

用途 / Role: Exact binomial interval; PyRPMD composes two bounds into a censoring envelope.

### samp-path2007

D. Branduardi; F. L. Gervasio; M. Parrinello. **From A to B in free energy space**. *Journal of Chemical Physics 126, 054103* (2007). DOI: [10.1063/1.2432340](https://doi.org/10.1063/1.2432340).

用途 / Role: Original method.

### samp-mlcolvar2023

L. Bonati; E. Trizio; A. Rizzi; M. Parrinello. **A unified framework for machine learning collective variables for enhanced sampling simulations: mlcolvar**. *J. Chem. Phys. 159, 014801* (2023). DOI: [10.1063/5.0156343](https://doi.org/10.1063/5.0156343).

用途 / Role: software; external training library for learned CVs.

### samp-deeplda2020

L. Bonati; V. Rizzi; M. Parrinello. **Data-Driven Collective Variables for Enhanced Sampling**. *J. Phys. Chem. Lett. 11, 2998–3004* (2020). DOI: [10.1021/acs.jpclett.0c00535](https://doi.org/10.1021/acs.jpclett.0c00535).

用途 / Role: Original Deep-LDA collective variables.

### samp-deeptica2021

L. Bonati; G. Piccini; M. Parrinello. **Deep learning the slow modes for rare events sampling**. *Proc. Natl. Acad. Sci. USA 118, e2113533118* (2021). DOI: [10.1073/pnas.2113533118](https://doi.org/10.1073/pnas.2113533118).

用途 / Role: Original Deep-TICA collective variables.

### samp-kabsch1976

W. Kabsch. **A solution for the best rotation to relate two sets of vectors**. *Acta Crystallographica A 32, 922–923* (1976). DOI: [10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873).

用途 / Role: Weighted optimal rotation underpinning RMSD; not a molecular dynamics method.

### samp-string2002

W. E; W. Ren; E. Vanden-Eijnden. **String method for the study of rare events**. *Physical Review B 66, 052301* (2002). DOI: [10.1103/PhysRevB.66.052301](https://doi.org/10.1103/PhysRevB.66.052301).

用途 / Role: Original method.

### samp-stringcv2006

L. Maragliano; A. Fischer; E. Vanden-Eijnden; G. Ciccotti. **String method in collective variables: Minimum free energy paths and isocommittor surfaces**. *Journal of Chemical Physics 125, 024106* (2006). DOI: [10.1063/1.2212942](https://doi.org/10.1063/1.2212942).

用途 / Role: CV-string theoretical background; general nonlinear-CV metric terms not fully implemented.

### samp-fts2005

W. E; W. Ren; E. Vanden-Eijnden. **Finite Temperature String Method for the Study of Rare Events**. *J. Phys. Chem. B 109, 6688–6693* (2005). DOI: [10.1021/jp0455430](https://doi.org/10.1021/jp0455430).

用途 / Role: Original finite-temperature string method.

### samp-prfo1985

A. Banerjee; N. Adams; J. Simons; R. Shepard. **Search for stationary points on surfaces**. *Journal of Physical Chemistry 89, 52–57* (1985). DOI: [10.1021/j100247a015](https://doi.org/10.1021/j100247a015).

用途 / Role: Original method.

### samp-nebtangent2000

G. Henkelman; H. Jónsson. **Improved tangent estimate in the nudged elastic band method for finding minimum energy paths and saddle points**. *Journal of Chemical Physics 113, 9978–9985* (2000). DOI: [10.1063/1.1323224](https://doi.org/10.1063/1.1323224).

用途 / Role: Original method.

### samp-cineb2000

G. Henkelman; B. P. Uberuaga; H. Jónsson. **A climbing image nudged elastic band method for finding saddle points and minimum energy paths**. *Journal of Chemical Physics 113, 9901–9904* (2000). DOI: [10.1063/1.1329672](https://doi.org/10.1063/1.1329672).

用途 / Role: Original method.

### samp-fire2006

E. Bitzek; P. Koskinen; F. Gähler; M. Moseler; P. Gumbsch. **Structural Relaxation Made Simple**. *Physical Review Letters 97, 170201* (2006). DOI: [10.1103/PhysRevLett.97.170201](https://doi.org/10.1103/PhysRevLett.97.170201).

用途 / Role: Original method.

### samp-radii2008

B. Cordero; V. Gómez; A. E. Platero-Prats; M. Revés; J. Echeverría; E. Cremades; F. Barragán; S. Alvarez. **Covalent radii revisited**. *Dalton Transactions, 2832–2838* (2008). DOI: [10.1039/B801115J](https://doi.org/10.1039/B801115J).

用途 / Role: Published parameter data; connectivity threshold is a PyRPMD heuristic.

### samp-bluemoon1989

E. A. Carter; G. Ciccotti; J. T. Hynes; R. Kapral. **Constrained reaction coordinate dynamics for the simulation of rare events**. *Chemical Physics Letters 156, 472–477* (1989). DOI: [10.1016/S0009-2614(89)87314-2](https://doi.org/10.1016/S0009-2614(89)87314-2).

用途 / Role: Constrained-to-conditional ensemble reweighting antecedent.

### samp-blue1998

M. Sprik; G. Ciccotti. **Free energy from constrained molecular dynamics**. *Journal of Chemical Physics 109, 7737–7744* (1998). DOI: [10.1063/1.477419](https://doi.org/10.1063/1.477419).

用途 / Role: Mass-metric / constrained-ensemble relation.

### samp-plumed2009

M. Bonomi; D. Branduardi; G. Bussi; C. Camilloni; D. Provasi; P. Raiteri; D. Donadio; F. Marinelli; F. Pietrucci; R. A. Broglia; M. Parrinello. **PLUMED: A portable plugin for free-energy calculations with molecular dynamics**. *Computer Physics Communications 180, 1961–1972* (2009). DOI: [10.1016/j.cpc.2009.05.011](https://doi.org/10.1016/j.cpc.2009.05.011).

用途 / Role: External software interface; cite selected PLUMED algorithms separately.

### samp-plumed2014

G. A. Tribello; M. Bonomi; D. Branduardi; C. Camilloni; G. Bussi. **PLUMED 2: New feathers for an old bird**. *Computer Physics Communications 185, 604–613* (2014). DOI: [10.1016/j.cpc.2013.09.018](https://doi.org/10.1016/j.cpc.2013.09.018).

用途 / Role: External software interface.

### samp-cblock1992

D. N. Politis; J. P. Romano. **A Circular Block-Resampling Procedure for Stationary Data**. *In Exploring the Limits of Bootstrap, R. LePage and L. Billard (eds.), Wiley, pp. 263–270* (1992). [Source / author version](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data).

用途 / Role: Original circular-block resampling; linked 1991 technical-report precursor, no DOI asserted.

### samp-geyer1992

C. J. Geyer. **Practical Markov Chain Monte Carlo**. *Statistical Science 7, 473–483* (1992). DOI: [10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137).

用途 / Role: Positive-sequence variance-estimation background; local pairing and observables are implementation choices.

### samp-wham1992

S. Kumar; J. M. Rosenberg; D. Bouzida; R. H. Swendsen; P. A. Kollman. **The weighted histogram analysis method for free-energy calculations on biomolecules. I. The method**. *Journal of Computational Chemistry 13, 1011–1021* (1992). DOI: [10.1002/jcc.540130812](https://doi.org/10.1002/jcc.540130812).

用途 / Role: Historical/comparison background, not an independent WHAM solver in this code.

### na-mm1979

H.-D. Meyer and W. H. Miller. **A classical analog for electronic degrees of freedom in nonadiabatic collision processes**. *The Journal of Chemical Physics 70, 3214–3223* (1979). DOI: [10.1063/1.437910](https://doi.org/10.1063/1.437910).

用途 / Role: Original classical electronic mapping antecedent; the complete classical S-matrix calculation is not implemented merely by using MMST variables.

### na-stock1997

G. Stock and M. Thoss. **Semiclassical Description of Nonadiabatic Quantum Dynamics**. *Physical Review Letters 78, 578–581* (1997). DOI: [10.1103/PhysRevLett.78.578](https://doi.org/10.1103/PhysRevLett.78.578). [Source / author version](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.578).

用途 / Role: Quantum mapping / SEO antecedent, not evidence that the package implements the full semiclassical IVR method.

### na-nrpmd2013

J. O. Richardson and M. Thoss. **Communication: Nonadiabatic ring-polymer molecular dynamics**. *The Journal of Chemical Physics 139, 031102* (2013). DOI: [10.1063/1.4816124](https://doi.org/10.1063/1.4816124).

用途 / Role: Original NRPMD and its SEO-projected thermal construction; direct origin for the published Fig. 1(c) benchmark.

### na-nrpmd2017

J. O. Richardson, P. Meyer, M.-O. Pleinert, and M. Thoss. **An analysis of nonadiabatic ring-polymer molecular dynamics and its application to vibronic spectra**. *Chemical Physics 482, 124–134* (2017). DOI: [10.1016/j.chemphys.2016.09.036](https://doi.org/10.1016/j.chemphys.2016.09.036). [Source / author version](https://www.sciencedirect.com/science/article/abs/pii/S0301010416305717).

用途 / Role: Theoretical analysis and generalized independent mapping/nuclear bead construction; background only for features beyond one mapping vector per nuclear bead.

### na-focused2019

S. N. Chowdhury and P. Huo. **State dependent ring polymer molecular dynamics for investigating excited nonadiabatic dynamics**. *The Journal of Chemical Physics 150, 244102* (2019). DOI: [10.1063/1.5096276](https://doi.org/10.1063/1.5096276). [Source / author version](https://huogroup.github.io/pdf/jcp8.pdf).

用途 / Role: Focused nonequilibrium electronic preparation and population estimator used by the public NRPMD path, especially Eqs. 20–23.

### na-mint2018

M. S. Church, T. J. H. Hele, G. S. Ezra, and N. Ananth. **Nonadiabatic semiclassical dynamics in the mixed quantum-classical initial value representation**. *The Journal of Chemical Physics 148, 102326* (2018). DOI: [10.1063/1.5005557](https://doi.org/10.1063/1.5005557). [Source / author version](https://arxiv.org/abs/1709.07474).

用途 / Role: MInt exact mapping/nuclear-momentum subflow component; the paper's full MQC-IVR algorithm is not implemented by this substep.

### na-sm2021

D. Bossion, S. N. Chowdhury, and P. Huo. **Non-adiabatic ring polymer molecular dynamics with spin mapping variables**. *The Journal of Chemical Physics 154, 184106* (2021). DOI: [10.1063/5.0051456](https://doi.org/10.1063/5.0051456). [Source / author version](https://huogroup.github.io/pdf/jcp11.pdf).

用途 / Role: Original SM-NRPMD; the implemented focused two-state equations do not include the paper's full Gamma_W equilibrium initialization.

### na-mash2023

J. R. Mannouch and J. O. Richardson. **A mapping approach to surface hopping**. *The Journal of Chemical Physics 158, 104111* (2023). DOI: [10.1063/5.0139734](https://doi.org/10.1063/5.0139734). [Source / author version](https://arxiv.org/abs/2212.11773).

用途 / Role: Original two-state MASH with deterministic spin-hemisphere surface selection.

### na-mashrates2024

J. E. Lawrence, J. R. Mannouch, and J. O. Richardson. **Recovering Marcus Theory Rates and Beyond without the Need for Decoherence Corrections: The Mapping Approach to Surface Hopping**. *The Journal of Physical Chemistry Letters 15, 707–716* (2024). DOI: [10.1021/acs.jpclett.3c03197](https://doi.org/10.1021/acs.jpclett.3c03197). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC10823533/).

用途 / Role: Population sampling/ET-rate follow-up used in interpreting MASH and CA-RP-MASH observables.

### na-carp2025

Z. Cao, V. A. Suarez, and J. S. Kretchmer. **A ring polymer molecular dynamics extension of the mapping approach to surface hopping: Incorporation of nuclear quantum effects**. *ChemRxiv, version 1 (7 December 2025), preprint* (2025). DOI: [10.26434/chemrxiv-2025-xps2g](https://doi.org/10.26434/chemrxiv-2025-xps2g). [Source / author version](https://client.prod.orp.cambridge.org/engage/chemrxiv/article-details/692ba382a10c9f5ca169b8de).

用途 / Role: Implemented CA-RP-MASH algorithm and operator-pair weights; record the implemented preprint version explicitly.

### na-msmash2023

J. E. Runeson; D. E. Manolopoulos. **A multi-state mapping approach to surface hopping**. *J. Chem. Phys. 159, 094115* (2023). DOI: [10.1063/5.0158147](https://doi.org/10.1063/5.0158147).

用途 / Role: Original multistate MASH and its signed observable estimator.

### na-tully1990

J. C. Tully. **Molecular dynamics with electronic transitions**. *The Journal of Chemical Physics 93, 1061–1071* (1990). DOI: [10.1063/1.459170](https://doi.org/10.1063/1.459170).

用途 / Role: Original fewest-switches algorithm and Tully's three two-state scattering models.

### na-rpsh2012

P. Shushkov, R. Li, and J. C. Tully. **Ring polymer molecular dynamics with surface hopping**. *The Journal of Chemical Physics 137, 22A549* (2012). DOI: [10.1063/1.4766449](https://doi.org/10.1063/1.4766449).

用途 / Role: Original RPSH proposal, with thermal scattering-rate tests.

### na-rpsh2017

F. A. Shakib and P. Huo. **Ring Polymer Surface Hopping: Incorporating Nuclear Quantum Effects into Nonadiabatic Molecular Dynamics Simulations**. *The Journal of Physical Chemistry Letters 8, 3073–3080* (2017). DOI: [10.1021/acs.jpclett.7b01343](https://doi.org/10.1021/acs.jpclett.7b01343). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01343).

用途 / Role: Nonequilibrium RPSH benchmark and centroid/full-ring rescaling convention discussion; Tully models and low-momentum scattering.

### na-iesh2009

N. Shenvi; S. Roy; J. C. Tully. **Nonadiabatic dynamics at metal surfaces: Independent-electron surface hopping**. *J. Chem. Phys. 130, 174107* (2009). DOI: [10.1063/1.3125436](https://doi.org/10.1063/1.3125436).

用途 / Role: Original independent-electron surface hopping.

### na-iesh2023

J. Gardner; D. Corken; S. M. Janke; S. Habershon; R. J. Maurer. **Efficient implementation and performance analysis of the independent electron surface hopping method for dynamics at metal surfaces**. *J. Chem. Phys. 158, 064101* (2023). DOI: [10.1063/5.0137137](https://doi.org/10.1063/5.0137137).

用途 / Role: IESH implementation analysis and discretized metal model followed by the implementation.

### na-rpshdm2023

Y. Shakya, R. Welsch, L. Inhester, and R. Santra. **Capturing electronic decoherence in quantum-classical dynamics using the ring-polymer-surface-hopping–density-matrix approach**. *Physical Review A 107, 062810* (2023). DOI: [10.1103/PhysRevA.107.062810](https://doi.org/10.1103/PhysRevA.107.062810). [Source / author version](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062810).

用途 / Role: Explicit CA, bead-average (BA), and bead-electronic density-matrix (DM) formulations; first author is Yashoj Shakya, initial Y., not P.

### na-dboc2025

D. K. Limbu, S. Bhusal, D. M. Castañeda-Bagatella, and F. A. Shakib. **Diagonal Born–Oppenheimer corrections in condensed-phase ring polymer surface hopping**. *The Journal of Chemical Physics 163, 234102* (2025). DOI: [10.1063/5.0301249](https://doi.org/10.1063/5.0301249).

用途 / Role: DBOC-RPSH extension and condensed-phase spin-boson benchmark; not a guarantee of accuracy improvement in every regime.

### na-soc2019

G. Miao, N. Bellonzi, and J. E. Subotnik. **An extension of the fewest switches surface hopping algorithm to complex Hamiltonians and photophysics in magnetic fields: Berry curvature and ‘magnetic’ forces**. *The Journal of Chemical Physics 150, 124101* (2019). DOI: [10.1063/1.5088770](https://doi.org/10.1063/1.5088770). [Source / author version](https://arxiv.org/pdf/1902.05115).

用途 / Role: Complex derivative-coupling rescaling direction (method 2) and local geometric force; ring-polymer use is an adaptation.

### na-edc2007

G. Granucci and M. Persico. **Critical appraisal of the fewest switches algorithm for surface hopping**. *The Journal of Chemical Physics 126, 134114* (2007). DOI: [10.1063/1.2715585](https://doi.org/10.1063/1.2715585).

用途 / Role: Energy-based decoherence correction, especially the amplitude damping prescription in Eq. 17; centroid/ring energies are implementation conventions.

### na-qd2018

A. Mandal, S. S. R. K. C. Yamijala, and P. Huo. **Quasi-Diabatic Representation for Nonadiabatic Dynamics Propagation**. *Journal of Chemical Theory and Computation 14, 1828–1840* (2018). DOI: [10.1021/acs.jctc.7b01178](https://doi.org/10.1021/acs.jctc.7b01178). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.7b01178).

用途 / Role: QD local-basis/overlap strategy; its PLDM calculation is not implemented merely by reusing QD transport.

### na-qd2019

W. Zhou, A. Mandal, and P. Huo. **Quasi-Diabatic Scheme for Nonadiabatic On-the-Fly Simulations**. *The Journal of Physical Chemistry Letters 10, 7062–7070* (2019). DOI: [10.1021/acs.jpclett.9b02747](https://doi.org/10.1021/acs.jpclett.9b02747). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02747).

用途 / Role: On-the-fly QD demonstration with ethylene; package QD-NRPMD is a composition with a different dynamical approximation.

### na-hst1994

S. Hammes-Schiffer; J. C. Tully. **Proton transfer in solution: Molecular dynamics with quantum transitions**. *J. Chem. Phys. 101, 4657–4667* (1994). DOI: [10.1063/1.467455](https://doi.org/10.1063/1.467455).

用途 / Role: Finite-difference time-derivative coupling from adjacent-frame overlaps.

### na-npi2014

G. A. Meek; B. G. Levine. **Evaluation of the Time-Derivative Coupling for Accurate Electronic State Transition Probabilities from Numerical Simulations**. *J. Phys. Chem. Lett. 5, 2351–2356* (2014). DOI: [10.1021/jz5009449](https://doi.org/10.1021/jz5009449).

用途 / Role: Original norm-preserving interpolation of overlap-based couplings.

### na-npi2016

A. Jain; E. Alguire; J. E. Subotnik. **An Efficient, Augmented Surface Hopping Algorithm That Includes Decoherence for Use in Large-Scale Simulations**. *J. Chem. Theory Comput. 12, 5256–5268* (2016). DOI: [10.1021/acs.jctc.6b00673](https://doi.org/10.1021/acs.jctc.6b00673).

用途 / Role: Multistate generalized NPI time-derivative matrix used by the overlap-coupling path; its A-FSSH decoherence is not implemented here.

### na-mv2013

N. Ananth. **Mapping variable ring polymer molecular dynamics: A path-integral based method for nonadiabatic processes**. *The Journal of Chemical Physics 139, 124102* (2013). DOI: [10.1063/1.4821590](https://doi.org/10.1063/1.4821590). [Source / author version](https://arxiv.org/abs/1309.4476).

用途 / Role: Original MV-RPMD signed mapping partition function and model dynamics.

### na-kc2014

A. R. Menzeleev, F. Bell, and T. F. Miller III. **Kinetically constrained ring-polymer molecular dynamics for non-adiabatic chemical reactions**. *The Journal of Chemical Physics 140, 064103* (2014). DOI: [10.1063/1.4863919](https://doi.org/10.1063/1.4863919). [Source / author version](https://arxiv.org/abs/1401.3033).

用途 / Role: Original primitive two-state KC-RPMD, crossing calibration and auxiliary-coordinate dynamics.

### na-iso2018

X. Tao, P. Shushkov, and T. F. Miller III. **Path-integral isomorphic Hamiltonian for including nuclear quantum effects in non-adiabatic dynamics**. *The Journal of Chemical Physics 148, 102327* (2018). DOI: [10.1063/1.5005544](https://doi.org/10.1063/1.5005544). [Source / author version](https://arxiv.org/abs/1709.06722).

用途 / Role: Trace-matched isomorphic Hamiltonian; the package implements an explicit two-state Ehrenfest consumer, not all QCLE/SH dynamics in the paper.

### na-nah21992

P. Halvick and D. G. Truhlar. **A new diabatic representation of the coupled potential energy surfaces for Na(3p 2P)+H2 → Na(3s 2S)+H2 or NaH+H**. *The Journal of Chemical Physics 96, 2895–2909* (1992). DOI: [10.1063/1.461986](https://doi.org/10.1063/1.461986).

用途 / Role: Original coupled NaH2 diabatic potential; physical origin is independent of the selected propagation method.

### na-nah2err1994

P. Halvick and D. G. Truhlar. **Erratum: A new diabatic representation of the coupled potential energy surfaces for Na(3p 2P)+H2 → Na(3s 2S)+H2 or NaH+H**. *The Journal of Chemical Physics 100, 4718* (1994). DOI: [10.1063/1.467282](https://doi.org/10.1063/1.467282).

用途 / Role: Required correction accompanying the original NaH2 potential citation.

### na-feit1982

M. D. Feit, J. A. Fleck Jr., and A. Steiger. **Solution of the Schrödinger equation by a spectral method**. *Journal of Computational Physics 47, 412–433* (1982). DOI: [10.1016/0021-9991(82)90091-2](https://doi.org/10.1016/0021-9991(82)90091-2). [Source / author version](https://www.sciencedirect.com/science/article/pii/0021999182900912).

用途 / Role: Fourier spectral/split-operator quantum propagation antecedent; the exactness claim requires finite-box/grid/time convergence.

### bzp2009

G. Bussi; T. Zykova-Timan; M. Parrinello. **Isothermal-isobaric molecular dynamics using stochastic velocity rescaling**. *J. Chem. Phys. 130, 074101* (2009). DOI: [10.1063/1.3073889](https://doi.org/10.1063/1.3073889). [Source / author version](https://arxiv.org/abs/0901.0779).

用途 / Role: method.

### mtk1994

G. J. Martyna; D. J. Tobias; M. L. Klein. **Constant pressure molecular dynamics algorithms**. *J. Chem. Phys. 101, 4177–4189* (1994). DOI: [10.1063/1.467468](https://doi.org/10.1063/1.467468).

用途 / Role: method.

### nptpimd1999

G. J. Martyna; A. Hughes; M. E. Tuckerman. **Molecular dynamics algorithms for path integrals at constant pressure**. *J. Chem. Phys. 110, 3275–3290* (1999). DOI: [10.1063/1.478193](https://doi.org/10.1063/1.478193).

用途 / Role: method.

### rgb2011

P. Raiteri; J. D. Gale; G. Bussi. **Reactive force field simulation of proton diffusion in BaZrO3 using an empirical valence bond approach**. *J. Phys.: Condens. Matter 23, 334213* (2011). DOI: [10.1088/0953-8984/23/33/334213](https://doi.org/10.1088/0953-8984/23/33/334213). [Source / author version](https://arxiv.org/abs/1104.0773).

用途 / Role: method.

### scr2020

M. Bernetti; G. Bussi. **Pressure control using stochastic cell rescaling**. *J. Chem. Phys. 153, 114107* (2020). DOI: [10.1063/5.0020514](https://doi.org/10.1063/5.0020514). [Source / author version](https://arxiv.org/abs/2006.09250).

用途 / Role: method.

### ipi2014

M. Ceriotti; J. More; D. E. Manolopoulos. **i-PI: A Python interface for ab initio path integral molecular dynamics simulations**. *Comput. Phys. Commun. 185, 1019–1026* (2014). DOI: [10.1016/j.cpc.2013.10.027](https://doi.org/10.1016/j.cpc.2013.10.027). [Source / author version](https://arxiv.org/abs/1402.1045).

用途 / Role: software / PIMD formulation.

### pd2024

Y. Kataoka; J. Haruyama; O. Sugino; M. Shiga. **Predictive evaluation of hydrogen diffusion coefficient on Pd(111) surface by path integral simulations using neural network potential**. *Phys. Rev. Research 6, 043224* (2024). DOI: [10.1103/PhysRevResearch.6.043224](https://doi.org/10.1103/PhysRevResearch.6.043224). [Source / author version](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.043224).

用途 / Role: application / estimator convention.

### ni2012

Y. V. Suleimanov. **Surface Diffusion of Hydrogen on Ni(100) from Ring Polymer Molecular Dynamics**. *J. Phys. Chem. C 116, 11141–11153* (2012). DOI: [10.1021/jp302453z](https://doi.org/10.1021/jp302453z). [Source / author version](https://pubs.acs.org/doi/10.1021/jp302453z).

用途 / Role: application.

### bazro2008

Q. Zhang; G. Wahnström; M. E. Björketun; S. Gao; E. Wang. **Path Integral Treatment of Proton Transport Processes in BaZrO3**. *Phys. Rev. Lett. 101, 215902* (2008). DOI: [10.1103/PhysRevLett.101.215902](https://doi.org/10.1103/PhysRevLett.101.215902).

用途 / Role: application; centroid-TST, not a full RPMD recrossing reproduction.

### sds2025

C. Li; L. Zhang; B. Jiang; H. Guo. **Unified ring polymer molecular dynamics rate calculations for reactions with separable and non-separable reactants**. *J. Chem. Phys. 163, 024105* (2025). DOI: [10.1063/5.0273941](https://doi.org/10.1063/5.0273941). [Source / author version](https://arxiv.org/abs/2503.22964).

用途 / Role: method.

### nialh1995

J. E. Angelo; N. R. Moody; M. I. Baskes. **Trapping of hydrogen to lattice defects in nickel**. *Modelling Simul. Mater. Sci. Eng. 3, 289–307* (1995). DOI: [10.1088/0965-0393/3/3/001](https://doi.org/10.1088/0965-0393/3/3/001).

用途 / Role: potential model.

### pcet2008

C. Venkataraman; A. V. Soudackov; S. Hammes-Schiffer. **Theoretical Formulation of Nonadiabatic Electrochemical Proton-Coupled Electron Transfer at Metal–Solution Interfaces**. *J. Phys. Chem. C 112, 12386–12397* (2008). DOI: [10.1021/jp802171y](https://doi.org/10.1021/jp802171y). [Source / author version](https://pubs.acs.org/doi/10.1021/jp802171y).

用途 / Role: method.

### gold2019

Z. K. Goldsmith; Y. C. Lam; A. V. Soudackov; S. Hammes-Schiffer. **Proton Discharge on a Gold Electrode from Triethylammonium in Acetonitrile: Theoretical Modeling of Potential-Dependent Kinetic Isotope Effects**. *J. Am. Chem. Soc. 141, 1084–1090* (2019). DOI: [10.1021/jacs.8b11826](https://doi.org/10.1021/jacs.8b11826).

用途 / Role: application; broader model than local workflow.

### pypcet2025

P. Hutchison; K. Cui; J. Zhong; S. Hammes-Schiffer. **Tutorial on computing nonadiabatic proton-coupled electron transfer rate constants**. *J. Chem. Phys. 163, 091501* (2025). DOI: [10.1063/5.0284337](https://doi.org/10.1063/5.0284337). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12646331/).

用途 / Role: software / comparison; not original 2008 electrode theory.

### gc2025

M. Sun; B. Jin; X. Yang; S. Xu. **Probing nuclear quantum effects in electrocatalysis via a machine-learning enhanced grand canonical constant potential approach**. *Nature Communications 16, 3600* (2025). DOI: [10.1038/s41467-025-58871-7](https://doi.org/10.1038/s41467-025-58871-7). [Source / author version](https://pmc.ncbi.nlm.nih.gov/articles/PMC12000419/).

用途 / Role: related grand-canonical algorithm / application.

### efrpmd2024

R.-H. Bi; W. Dou. **Electronic friction near metal surface: Incorporating nuclear quantum effect with ring polymer molecular dynamics**. *J. Chem. Phys. 160, 074110* (2024). DOI: [10.1063/5.0187646](https://doi.org/10.1063/5.0187646). [Source / author version](https://arxiv.org/abs/2311.08779).

用途 / Role: method.

### hmc1987

S. Duane; A. D. Kennedy; B. J. Pendleton; D. Roweth. **Hybrid Monte Carlo**. *Physics Letters B 195, 216–222* (1987). DOI: [10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693(87)91197-X). [Source / author version](https://www.sciencedirect.com/science/article/pii/037026938791197X).

用途 / Role: method.

### hastings1970

W. K. Hastings. **Monte Carlo sampling methods using Markov chains and their applications**. *Biometrika 57, 97–109* (1970). DOI: [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97).

用途 / Role: method.

### pyscf2018

Q. Sun et al.. **PySCF: the Python-based simulations of chemistry framework**. *WIREs Comput. Mol. Sci. 8, e1340* (2018). DOI: [10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340). [Source / author version](https://arxiv.org/abs/1701.08223).

用途 / Role: software.

### hirshfeld1977

F. L. Hirshfeld. **Bonded-atom fragments for describing molecular charge densities**. *Theoretica Chimica Acta 44, 129–138* (1977). DOI: [10.1007/BF00549096](https://doi.org/10.1007/BF00549096).

用途 / Role: method.

### mayer1983

I. Mayer. **Charge, bond order and valence in the AB initio SCF theory**. *Chemical Physics Letters 97, 270–274* (1983). DOI: [10.1016/0009-2614(83)80005-0](https://doi.org/10.1016/0009-2614(83)80005-0).

用途 / Role: method.

### qmdff2014

S. Grimme. **A General Quantum Mechanically Derived Force Field (QMDFF) for Molecules and Condensed Phase Simulations**. *J. Chem. Theory Comput. 10, 4497–4514* (2014). DOI: [10.1021/ct500573f](https://doi.org/10.1021/ct500573f). [Source / author version](https://pubs.acs.org/doi/10.1021/ct500573f).

用途 / Role: method.

### deevb2015

B. Hartke; S. Grimme. **Reactive force fields made simple**. *Phys. Chem. Chem. Phys. 17, 16715–16718* (2015). DOI: [10.1039/C5CP02580J](https://doi.org/10.1039/C5CP02580J). [Source / author version](https://pubs.rsc.org/en/content/articlehtml/2015/cp/c5cp02580j).

用途 / Role: method.

### treq2019

J. Steffen. **A new class of reaction path based potential energy surfaces enabling accurate black box chemical rate constant calculations**. *J. Chem. Phys. 150, 154105* (2019). DOI: [10.1063/1.5092589](https://doi.org/10.1063/1.5092589).

用途 / Role: method.

### caracal2023

J. Steffen. **Caracal: A Versatile Ring Polymer Molecular Dynamics Simulation Package**. *J. Chem. Theory Comput. 19, 5334–5355* (2023). DOI: [10.1021/acs.jctc.3c00568](https://doi.org/10.1021/acs.jctc.3c00568). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00568).

用途 / Role: software / upstream PES implementation.

### pipnn2013

B. Jiang; H. Guo. **Permutation invariant polynomial neural network approach to fitting potential energy surfaces**. *J. Chem. Phys. 139, 054112* (2013). DOI: [10.1063/1.4817187](https://doi.org/10.1063/1.4817187).

用途 / Role: method.

### jaxpip2026

J. Li; K. Song; H. Guo; J. Li. **Efficient Tensorized Evaluation of Permutation Invariant Polynomials for Representing Potential Energy Surfaces**. *J. Chem. Theory Comput. 22, 9940–9953* (2026). DOI: [10.1021/acs.jctc.6c01659](https://doi.org/10.1021/acs.jctc.6c01659). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.6c01659).

用途 / Role: method / software.

### committee2020

C. Schran; K. Brezina; O. Marsalek. **Committee neural network potentials control generalization errors and enable active learning**. *J. Chem. Phys. 153, 104105* (2020). DOI: [10.1063/5.0016004](https://doi.org/10.1063/5.0016004).

用途 / Role: method.

### delta2015

R. Ramakrishnan; P. O. Dral; M. Rupp; O. A. von Lilienfeld. **Big Data Meets Quantum Chemistry Approximations: The Δ-Machine Learning Approach**. *J. Chem. Theory Comput. 11, 2087–2096* (2015). DOI: [10.1021/acs.jctc.5b00099](https://doi.org/10.1021/acs.jctc.5b00099). [Source / author version](https://pubs.acs.org/doi/10.1021/acs.jctc.5b00099).

用途 / Role: method.

### ase2017

A. H. Larsen et al.. **The atomic simulation environment—a Python library for working with atoms**. *J. Phys.: Condens. Matter 29, 273002* (2017). DOI: [10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e).

用途 / Role: software.

### openmm2017

P. Eastman et al.. **OpenMM 7: Rapid development of high performance algorithms for molecular dynamics**. *PLoS Computational Biology 13, e1005659* (2017). DOI: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659). [Source / author version](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005659).

用途 / Role: software.

### lammps1995

S. Plimpton. **Fast Parallel Algorithms for Short-Range Molecular Dynamics**. *J. Comput. Phys. 117, 1–19* (1995). DOI: [10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039). [Source / author version](https://www.sciencedirect.com/science/article/pii/S002199918571039X).

用途 / Role: software / parallel algorithm.

### torch2019

A. Paszke et al.. **PyTorch: An Imperative Style, High-Performance Deep Learning Library**. *NeurIPS 32* (2019). DOI: [10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703). [Source / author version](https://arxiv.org/abs/1912.01703).

用途 / Role: software; DOI is the author preprint.

### mace2022

I. Batatia; D. P. Kovács; G. N. C. Simm; C. Ortner; G. Csányi. **MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields**. *NeurIPS 35* (2022). DOI: [10.48550/arXiv.2206.07697](https://doi.org/10.48550/arXiv.2206.07697). [Source / author version](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html).

用途 / Role: external model; DOI is the author preprint.

### nequip2022

S. Batzner et al.. **E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials**. *Nature Communications 13, 2453* (2022). DOI: [10.1038/s41467-022-29939-5](https://doi.org/10.1038/s41467-022-29939-5). [Source / author version](https://www.nature.com/articles/s41467-022-29939-5).

用途 / Role: external model.

### deepmd2018

L. Zhang; J. Han; H. Wang; R. Car; W. E. **Deep Potential Molecular Dynamics: A Scalable Model with the Accuracy of Quantum Mechanics**. *Phys. Rev. Lett. 120, 143001* (2018). DOI: [10.1103/PhysRevLett.120.143001](https://doi.org/10.1103/PhysRevLett.120.143001). [Source / author version](https://arxiv.org/abs/1707.09571).

用途 / Role: external model.

### uma2025

B. M. Wood et al.. **UMA: A Family of Universal Models for Atoms**. *arXiv:2506.23971 (v1 2025; v2 2026)* (2025). [Source / author version](https://arxiv.org/abs/2506.23971).

用途 / Role: external model preprint; cite the selected checkpoint and manuscript version.

### gfn12017

S. Grimme; C. Bannwarth; P. Shushkov. **A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parametrized for All spd-Block Elements (Z = 1–86)**. *J. Chem. Theory Comput. 13, 1989–2009* (2017). DOI: [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118).

用途 / Role: external electronic method.

### gfn22019

C. Bannwarth; S. Ehlert; S. Grimme. **GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions**. *J. Chem. Theory Comput. 15, 1652–1671* (2019). DOI: [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176).

用途 / Role: external electronic method.

### ipea2017

V. Ásgeirsson; C. A. Bauer; S. Grimme. **Quantum chemical calculation of electron ionization mass spectra for general organic and inorganic molecules**. *Chemical Science 8, 4879–4895* (2017). DOI: [10.1039/C7SC00601B](https://doi.org/10.1039/C7SC00601B). [Source / author version](https://pubs.rsc.org/en/content/articlehtml/2017/sc/c7sc00601b).

用途 / Role: IPEA-xTB origin; PyRPMD does not implement QCEIMS.

### h5md2014

P. de Buyl; P. H. Colberg; F. Höfling. **H5MD: A structured, efficient, and portable file format for molecular data**. *Comput. Phys. Commun. 185, 1546–1553* (2014). DOI: [10.1016/j.cpc.2014.01.018](https://doi.org/10.1016/j.cpc.2014.01.018).

用途 / Role: file format.

### bennett1977

C. H. Bennett. **Molecular Dynamics and Transition State Theory: The Simulation of Infrequent Events**. *Algorithms for Chemical Computations, ACS Symposium Series 46, 63–97* (1977). DOI: [10.1021/bk-1977-0046.ch004](https://doi.org/10.1021/bk-1977-0046.ch004).

用途 / Role: Original classical reactive-flux / transition-state factorization; book chapter.

### chandler1978

D. Chandler. **Statistical mechanics of isomerization dynamics in liquids and the transition state approximation**. *J. Chem. Phys. 68, 2959–2970* (1978). DOI: [10.1063/1.436049](https://doi.org/10.1063/1.436049).

用途 / Role: Original classical reactive-flux and plateau theory.

### bkmp21996

A. I. Boothroyd; W. J. Keogh; P. G. Martin; M. R. Peterson. **A refined H3 potential energy surface**. *J. Chem. Phys. 104, 7139–7152* (1996). DOI: [10.1063/1.471430](https://doi.org/10.1063/1.471430). [Source / author version](https://www.cita.utoronto.ca/~boothroy/bkmp2.html).

用途 / Role: Original BKMP2 model and author source.

### morse1929

P. M. Morse. **Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels**. *Phys. Rev. 34, 57–64* (1929). DOI: [10.1103/PhysRev.34.57](https://doi.org/10.1103/PhysRev.34.57).

用途 / Role: Original Morse potential and analytic vibrational levels.

### pip2010

Z. Xie; J. M. Bowman. **Permutationally Invariant Polynomial Basis for Molecular Energy Surface Fitting via Monomial Symmetrization**. *J. Chem. Theory Comput. 6, 26–34 (online 2009)* (2010). DOI: [10.1021/ct9004917](https://doi.org/10.1021/ct9004917).

用途 / Role: Original monomial-symmetrization PIP basis method.
