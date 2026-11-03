# 能力与验证边界

[文档索引](../index.md)是选择计算任务的入口。本页集中保存原来放在仓库 README 中的详细能力、
限制与验证说明；方程、允许的设置、示例和检查点约定仍以对应专题指南为准。
软件能够运行、pilot 检查通过和物理预测收敛是不同阶段。将估计值用于研究结论前，
请按[科学约定](science.md)完成相应检查。

## 选择适当的工作流

| 计算 | 工作流提供的内容 | 后续指南 |
| --- | --- | --- |
| 气相热速率 | 过渡态/端点准备、反应通道检查、伞形/冻结偏置/外部 PMF、受约束母链和成对回穿子轨迹 | [反应工作流](workflows.md)、[PMF 选择](../rate-pmf-selection.md)、[过渡态搜索](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-state-search.md) |
| 速率精度 | 分离 pilot 与冻结后的独立生产、预设检查、统计/数值验收和强制 PES 帧预算 | [精度控制器与 CLI](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate_precision.md) |
| 散射与反应网络 | dRPMD 碰撞、近似振动态 NE-RPMD 和 P=1 QCT、未决轨迹延长、按入口/通道/TS 汇总的速率矩阵 | [反应工作流](workflows.md)、[态选择散射](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/state-selected.md) |
| 自由能与反应路径 | 1–3 维伞形重建、重叠/有效样本诊断、自适应窗口、metadynamics、OPES、REUS、ABF/eABF、驱动功和受限 CV string | [增强采样](../enhanced-sampling.md)、[扩展 CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-sampling.md)、[有限温度 string](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/finite-temperature-string.md) |
| 稀有事件 | 固定环构型的 committor shots；含 minus 系综与内部通量的 TIS/RETIS、REPPTIS 和直接 FFS | [Committor](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/committor.md)、[路径采样](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md) |
| 材料与表面 | 平衡 PIMD、支持的 NPT 系综、扩散、固定层、位点/表面逃逸速率、估计量和轨迹输出 | [材料与表面](periodic.md)、[凝聚相核量子效应](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md) |
| 光谱 | PA-CMD、二次 f-CMD、实验性弯曲三原子 f-QCMD、TRPMD，以及由用户性质模型得到的相对 IR/Raman/SFG | [性质光谱](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-spectroscopy.md)、[f-QCMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/qcmd.md)、[TRPMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/trpmd-spectroscopy.md) |
| 低温量子效应 | 分子 instanton 优化、速率/劈裂收敛扫描、Hessian 复用、高斯涨落预因子和正常区 golden-rule instanton | [Instanton](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md)、[低温方法](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/low-temperature.md)、[量子估计量](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-statistics.md) |
| 电子动力学与电化学 | NRPMD/自旋映射/表面跳跃、经典 IESH/多态 MASH、SA-CASSCF 数据、spin–boson HEOM、局域 PCET 和有效电极模型 | [非绝热方法](nonadiabatic.md)、[IESH/MASH](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-iesh-multistate.md)、[电子数据/HEOM](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-nonadiabatic.md)、[PCET](pcet.md)、[电极模型](electrochemistry.md) |
| 模型构建 | PySCF 参考数据、PES 拟合/导出、committee 不确定性、delta 修正和可续算的采样—标注—训练闭环 | [PES 生成](pes-generation.md)、[ML 生命周期](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-ml-lifecycle.md) |

普通输入工作流使用 `validate`、`run`、`analyze` 和 `resume`。
`precision`、`instanton`、`nonadiabatic-rate` 和仅有 Python API 的方法各有独立约定。
存在某个 Python API，不代表存在对应 TOML 任务、MPI 实现或所有恒温器/偏置组合。

## 引擎、并行与续算

现有输入保持原引擎及科学身份。默认材料工作流、显式选择的原生 PIMD 引擎与 Torch 工作流，
仍有各自支持的组合和检查点读取器；不能把一种引擎的检查点直接解释为另一种引擎的状态。

[原生系综 API](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/native-ensembles.md)持有常驻 C++
物理状态、恒温/恒压辅助量、力缓存和 RNG 状态。[内置解析势的原生绑定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/builtin-native-potentials.md)
使这些模型可进入融合 C++ 循环。即便积分算子是原生实现，Python PES 桥接仍然是 Python 力评估。
[统一传播栈](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/propagation-stack.md)说明整块原子提交、
成功前缀提交及兼容的后端切换。

[Torch 工作流](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/torch-materials-workflows.md)
把 float64 `[replicas, beads, atoms, 3]` 状态保留在选定 CPU/CUDA 设备，整批原子提交。
它支持全部原子可动、固定晶胞的 PIMD 和扩散；拒绝 NPT、固定原子、约束、交换/高阶作用量、摩擦、
偏置以及 MPI/worker 池组合。请求的设备不可用或模型不兼容时会明确失败；不会据此降低精度或切换到主机。
精确续算的约定限于同一运行环境。

[执行指南](execution.md)解释进程池、珠子 worker、静态/动态 MPI 组和 OpenMP 控制。
[批量 ML 势](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-ml-potentials.md)与
[力引擎适配器](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potential-adapters.md)
分别规定模型、晶胞、线程和设备要求。OpenMM copy-bank 路径评估多个副本，并不以 OpenMM 轨迹替换 PyRPMD 动力学。
LAMMPS 常驻实例也不保证提速。

检查点保存完整工作流状态和 RNG 数据，并进行原子发布与完整性检查。科学指纹拒绝变化后的模型/源码文件和物理设置；
只有特定读取器支持时才允许调整执行参数。见[执行与续算](execution.md)、
[可移植发布](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/portable-persistence.md)及
[Windows 身份规则](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md#windows-runs-with-missing-historical-source-hashes)。
历史 Windows 运行若缺少盘符路径引用源码的哈希，会在迁移前拒绝续算，必须在新目录重跑。

PLUMED 的 MD 快照不能单独恢复不透明的 action 历史。自动续跑要求在首次运行前选定经验证的
`[plumed] restart="files"` 方式，见 [PLUMED 续跑](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md)。

## 需要检查的科学限制

**速率与精度。** 等价通道、源盆地归一化、回穿平台、能量守恒和不确定性都应与实际物理问题匹配。
连通的投影 FES 不是速率或分支比。精度执行器目前要求固定的局域伞形设计、显式 Python 推进和串行执行；
不支持的采样/并行设置会被拒绝。达到统计目标不等于验收通过，未要求的数值对照也不会被推定为通过。

**量子采样。** Suzuki–Chin、交换与 PIGS 采样平衡或投影分布，不能当作物理实时轨迹。
费米结果通过对正的玻色采样做符号重加权得到，需要足够的平均符号。周期交换选择收敛的高斯像求和，
或显式的最小像近似。其各向同性 NPT 使用全珠子 BZP 推进器，要求物理 PES virial 与晶胞更新；
不能改用质心 BZP 组合。当前没有超流估计量，见[路径积分作用量](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-path-integral.md)。

**GLE/PIGLET。** [参数导入与生成](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md#generating-and-importing-glepiglet-parameters)
记录频率范围、拟合目标、残差和来源。谐振拟合通过不能证明非谐观测量准确。
归档的 P=4 PIGLET 参数未通过非谐压力检查；这个负结果保留在
[凝聚相验证](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md)中。

**光谱。** 实验性 f-QCMD 使用固定拓扑弯曲三原子分子、可分离 IBI 修正与 P=1 生产。
固定准质心 PIMD 采样器是平衡工具。实时绝热 QCMD 与 h-CMD 尚未实现，所要求的 f-QCMD 收敛光谱证据也仍待补充。
用户给出的偶极/极化率模型决定相对强度；不自动给出绝对光学强度或 Fresnel 修正。

**Instanton。** 优化作用量本身不是多维速率；必须提供反应物配分函数以及对称性/刚体模式约定。
致密预因子最多处理 4096 个活动珠子坐标。Golden-rule 支持常数弱耦合、两个束缚/受约束 diabatic 势面和正常区。
数值驻定与有限预因子仍保留 `physical_converged=false`；珠子数、虚时间长度、PES 与半经典近似精度应单独检查。

**电子方法与模型有效性。** IESH 和多态 MASH 当前要求 P=1、实 Hamiltonian 和非简并绝热能级。
PySCF 桥接实现 SA-CASSCF，HEOM 参考实现其指南规定的 spin–boson 热浴与层级；两者都不代表任意电子结构或开放体系。
PCET 假设局域单质子复合物、宽带金属和经典 Gaussian 环境。Committee 分歧不是校准后的模型误差，训练闭环成功也不证明 PES 有效。

## 归档证据与性能

下面保留原来从 README 访问的证据。这些是有日期和模型范围的研究记录，文档更新没有重新测量这些数值。
[文献验证清单](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/literature-validation.md)和
[复现清单](reproduction.md)列出剩余要求。

| 记录 | 证据及实际限制 |
| --- | --- |
| [H + H₂/BKMP2 PMF 细化](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/h3-pmf-refinement-2026-10-29/README.md) | 1500 K、P=16 下四组独立随机种子使用 848,000 个样本和 1,024 条新回穿轨迹；UI/MBAR 势垒差的配对区间包含零，速率联合相对标准误为 20–23%。这是该设置的有限采样证据。 |
| [态选择 BKMP2 研究](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/STATE_PREPARATION_STUDY.md) | 区分质心筛选、全 PES 作用量 QCT 和实验性平滑准备，并保留未决轨迹和低能区差异。 |
| [PCET 谐振对照](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md) | 指定谐振模型上 12 个正反向速率与固定版本 pyPCET 相差不超过 0.062%。 |
| [PMF 数值检查](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pmf-accuracy-2026-10-30/README.md) | 独立解析积分和 PyMBAR 检查不同珠子数、积分器、周期 CV 与相关误差。 |
| [H + BrCl 模型](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/brhcl/README.md)与[自适应伞形审查](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/umbrella-reliability-2026-10-31/README.md) | 保留隐藏坐标混合、重叠/不确定性修复，以及 pilot 阈值与精度的区别。 |
| [Ni–H EAM pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)与[水 pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/water/README.md) | 固定来源的材料模型和短模型检查；没有声称完整收敛地复现所有物理量。 |
| [凝聚相核量子效应](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md)、[instanton](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md)及[非绝热对照](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-iesh-multistate.md) | 独立参考、有限珠子/步长限制与报告的不确定性分别对应各自模型。 |

CPU 耗时、模型大小、硬件与线程数分别记录在
[原生势基准](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/builtin-native-potentials.md)、
[批量 ML 测量](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-ml-potentials.md)和
[大体系力引擎报告](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/t19-performance.md)中。
已有 CUDA 测量驱动，但**这些报告没有给出 GPU 耗时或 GPU 提速的实测证据**。
便宜解析内核的提速不能直接推广到昂贵势模型，CPU 吞吐量也不能作为 CUDA 性能结论。

[周期材料审查](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/periodic-materials-audit.md)、
[实现审查](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/implementation-review-2026-11-03.md)及
[开发路线](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md)保存更长的工程评估与未完成项。

## 输入、扩展与来源

最小速率/散射输入见[输入与命令行](input.md)，完整可运行任务见[示例](examples.md)。
Python 输入作为受信任的可执行代码运行；可复用 factory 应放在可导入模块中。
PES 接收原子单位的 `float64[B,N,3]` 坐标，返回 `[B]` 能量和 `[B,N,3]` 力，
其中 `forces = -dV/dq`，不进行珠子缩放。自定义 CV 应给出完整环 Jacobian，
详见[势能接口](potentials.md)和 [Python API](api.md)。

[输出与分析](results.md)解释 manifest、样本、诊断、轨迹格式与重新分析；
[开发指南](development.md)解释测试、构建和贡献规则。
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md)
保留源码版本、上游归属与第三方声明。研究应引用实际采用的方法、模型、参数、上游软件及 PyRPMD commit；
可用[方法目录](references.md)与[复现清单](reproduction.md)确定引用来源。
