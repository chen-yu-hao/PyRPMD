# 周期材料与表面催化：实现审查和改进清单

审查日期：2026-10-31。代码基线为
[`c4fa743f552c1d71db35fab791a2e63409bc1c3e`](https://github.com/chen-yu-hao/PyRPMD/tree/c4fa743f552c1d71db35fab791a2e63409bc1c3e)。
本记录以该基线的实际实现、测试和中英文指南为依据，区分已实现能力、可复现缺陷和仍需研究验证的扩展。
它不把数值单元测试、短真实势能 pilot 或“程序运行完成”视为材料物理结果已经收敛。

## 结论与审查范围

PyRPMD 已具备可用于开展周期材料研究的基础：倾斜晶胞和部分 PBC、连续展开的环坐标、
冻结基底、周期 PES、PIMD、RPMD 扩散、自由能采样及盆地归一化局部速率。
本次优先修复会影响表面几何、能量一致性和速率单位的明确问题，并同步已经落后于代码的指南。
真实催化机理、有限尺寸和采样收敛仍应针对具体材料建立证据。

| 审查层 | 代码与验证入口 | 已有能力及主要边界 |
| --- | --- | --- |
| 输入与系统 | `input.py`、`system.py`、`units.py`；`test_materials_input_cv.py`、`test_periodic_materials.py` | 原子单位、行晶格矢量、严格 PBC/固定原子检查；气相 `rate`/`drpmd` 拒绝周期体系 |
| 周期几何与 CV | `geometry.py`、`cv.py`、`cv_periodic.py`；`test_periodic_geometry_cache.py`、`test_surface_geometry.py` | 一般倾斜晶胞的精确 MIC、梯度、固定晶格缓存；普通分子 CV 不自动切换成周期 CV |
| 传播与统计系综 | `dynamics.py`、`barostat.py`、`native_ensemble.py`、`native/`；`test_surface_dynamics.py`、`test_barostat.py`、`test_native_ensembles.py` | 固定层每珠保持参考位置和零动量；默认 BZP 与独立原生柔性晶胞路径已有，slab/固定层 NPT 不支持 |
| 势能与执行 | `potentials/`、`parallel/execution.py`；`test_cabi_v2.py`、`test_torch_periodic.py`、`test_potential_adapters.py`、`test_ipi_clients.py` | 周期 C ABI v2、ASE/ML、Python/Torch、tblite、i-PI；提供 cell-aware 求值不自动等于支持 NPT 的 `set_cell` |
| 工作流与恢复 | `workflows/materials.py`、`native_pimd.py`、`fes.py`、`metadynamics.py`、`site_rate.py`、`surface.py`；stream/native/workflow 测试 | 流式轨迹、原子提交、完整性检查；native 与 legacy 检查点不可互换 |
| 物理量与速率 | `analysis/diffusion.py`、`surface.py`、`projection.py`、`recrossing.py`；diffusion/surface/statistics 测试 | 自扩散张量、非扩散窗口筛选、盆地归一化和相关块误差；投影源 FES 的完整速率误差尚未闭合 |
| 相关扩展 | `electronic_friction.py`、`electrochemistry/`、`potentials/contraction.py`、`multitime.py`、`transition_interface.py` | 已有 EF-RPMD、局部电化学、RPC/MTS、TIS 接口，各有明确适用范围；并非全部接通固定层表面速率 |
| 指南与实例 | README EN/ZH、`docs/manual/`、`docs/zh/manual/`、材料/表面专题、`examples/materials/`、`examples/surface/`、`benchmarks/materials/` | 示例与数学约定丰富，但周期势能、原生引擎和非线性速率能力存在文档漂移 |

本次是仓库实现和使用路径的审查及有针对性的回归验证；未重新完成所有可选电子结构、GPU、多节点
和长期生产基准，也未为所有非绝热/量子统计扩展重新推导物理方法。

## 本次已落实的改进

### 1. 周期最小镜像不应依赖等价晶格基的写法

基线对 `cell=[[1000,1,0],[1,0,0],[0,0,15]]`、`pbc=[true,true,false]` 的最近像搜索
可能因候选数上限报“晶胞过度倾斜”，而交换两个周期基矢就能成功。两者描述同一个二维周期晶格。
问题来自未约化晶格基下的球搜索范围，不能据此把合法等价表示视为物理错误。

`geometry.py` 现在先在**周期子空间**做整数 size/Lovasz 约化，再使用既有精确 QR 球搜索。
体系原始 cell/PBC、坐标和物理晶格保持不变；没有逐珠子折回，也没有改成分数坐标简单取整。
新测试覆盖八种 PBC、等价整系数换基、倾斜晶胞穷举、周期 CV 梯度及独立 ASE `find_mic` 对照。
ASE 的一般 MIC 实现同样采用晶格约化思想；本次使用独立的预条件实现，没有复制 ASE 源码。[S1]

### 2. 金属电子展宽计算使用与力一致的能量

基线 ASE adapter 默认请求普通 `energy`。部分电子结构 calculator 会用该属性返回零温外推能，
其核力和应力却是有限电子温度自由能的导数；把这两者组合会影响能量守恒诊断及自由能权重。
ASE 明确区分这两个能量接口。[S2]

现在优先请求 `get_potential_energy(force_consistent=True)`。只在 calculator 未声明
`free_energy` 且抛出 `PropertyNotImplementedError` 时回退普通能量；已声明的自由能求值失败仍报错。
这一电子自由能不是核 PMF，电子温度也不自动等于核恒温器温度。
旧的、采用外推能的电子展宽计算应在新输出目录重跑，不能拼接旧能量记录或旧权重。
经典 calculator 的既有能量约定保持适用。

### 3. Python 周期模型失败绑定不得污染包装器状态

Python adapter 重新绑定体系时，现在先验证候选 cell/PBC 并调用外部模型的绑定接口，全部成功后
再提交包装器的 cell/PBC 与调用模式。无效晶胞、无效边界或 callback 异常不会令包装器留下部分新状态。
这只保证 PyRPMD 包装器的提交边界，不承诺撤销任意外部 Python 对象自身已经产生的副作用。

### 4. 表面面积由实际周期平面决定

基线二维速率归一化假设面内矢量总是 `cell[0]`、`cell[1]`，因此合法的 `[true,false,true]`
和 `[false,true,true]` slab 被错误拒绝，三维全周期晶胞也可能在未明确晶面的情况下获得面积。

现在只在恰有两个周期晶格矢量时由它们的叉积推导面积；其他情形要求显式 `area`。
输出记录面积来源及用于推导的周期矢量索引。转换仍为 `k2=k_cell*A/n`，单位 `cm²/s`，
没有引入覆盖度相互作用或新的速率统计模型。对原先含糊的三维全周期输入，需显式给出研究表面面积。

### 5. 把 ASE 能量约定纳入运行身份

只修改能量调用而允许无条件续跑，会使一个输出目录混合两种 PES 能量约定。
新的 ASE 能量契约因此进入运行身份校验：旧 ASE-backed 检查点的续跑和重新分析会被拒绝，
包括无法在离线时证明两种能量一致的经典 calculator 任务；已存摘要仍可读取。
新约定的运行正常支持恢复。这个保守兼容性门限保护 PMF 权重、能量记录及检查点的一致性，
不表示旧的每个经典轨迹都存在物理错误。
静态指纹中的契约标记为 `potential_energy_convention="ase-force-consistent-v1"`，
应用于明确的 `ase`、`mace`、`nequip`、`fairchem` provider；它不是新增用户输入参数。
任意 Python factory 内部隐藏的 ASE calculator 无法通过静态 provider 名称发现，仍须记录其源码与能量约定。

### 6. 原生 PIMD 的公开准备与验证入口

基线原生 `resolve_materials` 返回原生引擎的选项结构，而 `runner.prepare` 仍直接读取 legacy
专有的 `barostat`/`storage` 字段。因此有效的原生 PIMD 输入会在 `prepare` 或
`validate --check-pes` 路径出现 `KeyError`，指南列出的验证命令无法使用。

修复为独立处理原生和 legacy 的准备结果，并对原生引擎进行实际的轻量 preflight，
避免将原生 barostat 参数错误地送入 legacy BZP。回归应覆盖公开 native NVT/NPT 示例及其失败输入。
普通绝热 PES 的 `validate --check-pes` 仍是接口、求值与有限值检查，不代替有限差分力、
晶格平移和有限应变的科学验证。

### 7. 使使用指南与已有实现一致

中英文周期指南与材料/表面说明现在明确：

- C ABI v2、显式周期 Torch、命名 ML 等接口已经存在；仅 C ABI v1/f2py 等保持分子边界。
- 默认 Python 材料工作流与显式原生 PIMD 是不同引擎；原生 MTTK/RGB 并非缺失功能，
  但也不意味着固定层 slab NPT 已经成立。
- 非线性 `site_rate` 已有约束母系综度量修正，`surface_rate` 已支持连续距离子集；
  不能再把所有非线性速率描述成未实现。
- EF-RPMD、恒电位有效 PES 和局部 HER 后处理已有相应模块，其物理限制不应被“完全未实现”的旧文字掩盖。
- 新增从结构、PES 契约、热化到速率/扩散及有限尺寸收敛的实际使用顺序。

## 开源实现对照

| 对照来源 | 可借鉴的具体实践 | PyRPMD 的现状与改进方向 |
| --- | --- | --- |
| ASE geometry / Atoms / socketio [S1–S3] | 一般 MIC 先约化晶格；明确 force-consistent energy；显式交换晶胞与物理 virial | 本次改进 MIC 与能量契约；继续扩大真实 calculator 的力/应变验证，不能由 adapter 名称推断模型适用性 |
| i-PI [S4] | 原子传播与力计算解耦，恒压和柔性晶胞，RPC 与 MTS 的显式 force splitting | 上述核心能力在 PyRPMD 中各有实现；下一步应接通已存在接口并验证材料，不重复增加同名算法 |
| LAMMPS PIMD / i-PI client [S5] | 区分热平衡 PIMD 与动力学、记录 image flags、清楚定义坐标与正常模数据、定义变胞 client 责任 | 保持 PyRPMD unwrapped q/centroid 约定；增加真实 force engine 互操作与相同系综对照，而非逐轨迹随机数一致性要求 |
| PLUMED 2.10 [S6] | 区分 nearest-image distance、连续分量和分子/COM 镜像重建 | PyRPMD 有周期 pair CV，但组 COM 的初始一致镜像仍需用户保证；可提供显式拓扑辅助工具，不改变传播坐标 |
| GPAW [S8] | 明确实空间与平面波 slab 的静电边界及偶极层修正 | PyRPMD 传递 cell/PBC，电子结构边界和长程修正仍由 PES 配置；全周期 slab 须明确表面法向和归一化面积 |
| OpenPathSampling [S7] | TIS 交叉概率须与独立有效通量相乘，完整路径和独立链的统计支持不可省略 | PyRPMD TIS 已有该分工；表面多通道和长期动力学可在既有 API 上验证，不能用一次 MetaD 越垒时间直接换速率 |

这里的开源项目是算法和接口设计对照，不代表彼此在任意 PES、温度、珠子数或约束下数值等价。

## TODO：优先级与验收条件

`P0` 为直接影响数值/单位正确性的修复；`P1` 为研究可靠性和周期能力衔接；`P2` 为成本、易用性及扩展。
已勾选项为本 PR 的改动，未勾选项保留为后续工作，不能计作已经交付的科学结果。

- [x] **P0 — 等价晶格基下的 MIC 可靠性。** 周期子空间约化；须通过独立最近像和梯度对照，保留一般倾斜/部分 PBC 行为。
- [x] **P0 — ASE 能量/力一致性。** 对声明的自由能严格求值；覆盖电子展宽、经典 fallback 和失败传播。
- [x] **P0 — ASE 能量约定的运行身份。** 阻止旧检查点续跑/重算混用能量；新约定的恢复和已有摘要只读访问应保持有效。
- [x] **P0 — Python 绑定的提交边界。** 无效输入或 callback 失败后包装器保留原晶胞、PBC 和调用模式。
- [x] **P0 — 表面面积与单位。** 覆盖三种二维周期组合、晶格行置换、显式面积及全周期歧义拒绝，保留一阶速率默认行为。
- [x] **P0 — 原生 PIMD 公开 preflight。** `prepare` 和 `validate --check-pes` 按实际引擎解析和验证，覆盖 native NVT/NPT，不读取不存在的 legacy 字段。
- [x] **P1 — 同步中英文指南。** 准确区分 legacy/native、已有功能和未验证扩展，并说明兼容性变化与研究准备流程。
- [ ] **P1 — 真实周期材料的物理验证矩阵。** 在固定来源 PES 上开展 H/D、珠子数、步长、热化、独立种子、超胞、层数、真空和覆盖度扫描。
  电子结构参考另需收敛 k 点、截断、电子展宽及 SCF 力精度，明确带电／非对称 slab 的静电边界和偶极处理；PBC 标志不替代这些设置。[S8]
  验收：独立误差与系统误差分别报告；扩散具有稳定长时窗口，速率具有连通 PMF 和回穿平台；保留失败/未收敛样本。
- [ ] **P1 — 源 FES → 表面速率的联合误差。** 目前 `surface.free_energy` 保留 `projected_pmf_pilot`，正式 `rate_s_inv` 为空。
  验收：对源采样相关块、条件积分域及母链/子对联合重采样，传播支持失败与协方差，并用已知模型验证覆盖率。
- [ ] **P1 — 固定层下的一般非线性速率。** 复用已有全可动 `site_rate` 的度量框架，不能只把冻结原子的速度置零。
  验收：可动子空间质量度量、曲面约束测度、通量权重和联合误差一致，并通过坐标重参数化及解析条件分布测试。
- [ ] **P1 — 周期 provider 的变胞契约补齐。** Torch/C ABI v2/tblite/DeePMD 的 cell-aware 求值不自动提供 legacy NPT 所需的动态 `set_cell`；当前 guard 应保留。
  验收：逐一实现明确的更新接口，以有限应变检查 virial，并验证失败回滚、checkpoint 恢复和 provider 内部邻居表更新。
- [ ] **P1 — 显式分子与珠子镜像准备。** 参照 PLUMED 的 ordered topology 重建思想，增加可选择的组镜像初始化及 ring winding 诊断。
  验收：跨边界分子的 COM/键长一致，输入重排与晶格平移不改变物理量；持续动力学 q 仍保持展开，不能逐珠子 wrap。
- [ ] **P2 — 接通已有 RPC/MTS 的材料工作流。** 当前 Python API 已有 force split 和收缩，但未提供完整 workflow、压力/virial、偏置/约束与专用 checkpoint 集成。
  验收：先定义 `Vfast + Vslow`，分别收敛收缩珠数和外时间步；报告目标 PES 调用数及端到端时间，并与 full-bead 物理量比较。[S4]
- [ ] **P2 — 更高效的真实周期 PES 执行。** C ABI v2 批量接口与 `native_periodic` 整轨迹 ABI 目前不同；ASE 模型逐帧求值。
  验收：对明确模型提供 cell-aware 原生批量/设备常驻路径，保留物理 virial、状态错误和内存所有权契约；报告不同 N、P 和 PES 成本下的性能。
- [ ] **P2 — 任意表面方向的输运及结构分析。** 目前 `axes` 是笛卡尔分量；长轨迹离线分析仍需要选定原子或标量历史。
  验收：显式正交分析基下扩散张量协变、面内 Einstein 因子正确；增加按需的结构统计及有界内存相关分析，避免把 self-D 当作集体电导率。
- [ ] **P2 — 周期距离分割面的等价晶格判定。** `surface.py` 当前用原始周期基最小奇异值的一半作为安全半径，
  可避免镜像切换，但可能因等价整数剪切而过度拒绝合法分割面；本次 MIC 改进没有改变这一保守速率门限。
  验收：采用经验证的最短晶格矢量或约化基保守界，使等价晶格表示具有一致可用范围，同时保留镜像接缝保护。
- [ ] **P2 — 表面网络、覆盖度和入射过程。** 已有局部 HER 后处理不能替代一般吸附/脱附/反应 kMC；气体入射与 sticking 也需要新初态和归一化。
  验收：清楚定义标准态、位点与通道计数、详细平衡和覆盖度相互作用；网络不重复计入 FES 已包含的等价路径。

不建议在本修复中直接增加“只给 H 多珠、金属一珠”的混合模型。现有全可动原子共用 P，
把不同原子的珠数随意改成 1 会改变哈密顿量和统计测度。优先验证已有 RPC/MTS 的受控近似，
再为按原子区域的收缩或经典基底建立独立推导与验证。

## 验证记录

本次在源码冻结后重新执行全套测试及四组真实 Ni–H EAM pilot。完整命令、环境、源码哈希和原始输出见
[本次验证证据](validation/periodic-materials-2026-10-31/README.md)。计算在提交前的修改工作树中进行，
归档明确记录基线提交与各文件 SHA256，不把基线提交当作含修复的版本。

| 本 PR 检查 | 结果 |
| --- | --- |
| 新增回归 | 80 项：MIC/CV 14、ASE 能量及应变 7、能量身份/恢复 28、Python 失败绑定 3、表面面积 13、原生预检 15 |
| 最终全套 pytest | **3,885 passed、114 skipped、264 subtests passed**，零失败；耗时 471.94 s |
| 真实 Ni–H EAM | 四组 H/D × P=1/4 的接口及有限步长检查通过；计算期间所记录的 16 个源码文件未变化 |
| 中英文 Sphinx | HTML、singlehtml 共四次 `-W --keep-going` 严格构建通过，无警告 |

全套测试启用了已编译原生扩展、ASE 以及固定来源 EAM；可选电子结构、PyTorch、PLUMED、
外部 PES 与 MPI 等未满足环境条件的检查按自身条件跳过。具体原因保留在原始日志中，未执行 GPU 或多节点验证。

本次 [`EAM 原始报告`](validation/periodic-materials-2026-10-31/eam-validation.json) 使用
27 Ni + H/D、600 K、0.1 fs 步长、64 步平衡及 160 步无恒温生产，每组保存 81 帧：

| 同位素 | 珠子数 | 最大能量漂移／珠子（hartree） | 平均环回转半径平方（bohr²） | 扩散诊断 |
| --- | ---: | ---: | ---: | --- |
| H | 1 | 1.6704e-7 | 0 | `ballistic_or_drifting` |
| D | 1 | 1.1438e-7 | 0 | `ballistic_or_drifting` |
| H | 4 | 1.8282e-7 | 0.039323 | `ballistic_or_drifting` |
| D | 4 | 1.6073e-7 | 0.019857 | `ballistic_or_drifting` |

所选有限差分力的最大绝对误差为 `4.4348e-10 eV/angstrom`；这验证的是接口与有限步长实现。
四组仍没有可接受的扩散系数，`physical_converged=false`、`equilibration_converged=false`。
这些短轨迹的环尺寸也不能解释为平衡同位素效应。势文件固定来源与 SHA256 见
[`Ni–H EAM pilot` 说明](../benchmarks/materials/README.md)，本次没有重新分发势文件。

其他已有材料证据的边界：
[`CO/O/Pt(111)`](../examples/surface/pt111/README.md) 具备真实 EANN 和完整活动原子传播，
但其短 pilot 不覆盖完整 FES，也未复现论文速率，本次没有重跑其生产计算。
完整长算、PES 物理准确性和材料结论仍属于上面的 P1 验证工作。

## 参考实现与一手来源

来源查阅于 2026-10-31。可变文档链接的取值日期与源码固定版本分开列出；没有复制上游实现代码。

- **[S1] ASE 周期几何。** [一般 MIC 源码](https://docs.ase-lib.org/_modules/ase/geometry/geometry.html)、
  [geometry API](https://docs.ase-lib.org/ase/geometry.html)。用于比较晶格约化、部分 PBC 和几何导数约定。
- **[S2] ASE 能量契约。** [Atoms.get_potential_energy](https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_potential_energy)、
  [Atoms 源码](https://docs.ase-lib.org/_modules/ase/atoms.html)。`force_consistent` 区分力一致的电子自由能和外推能。
- **[S3] ASE i-PI 协议。** [socketio 源码](https://docs.ase-lib.org/_modules/ase/calculators/socketio.html)。
  用于 cell/inverse/force/virial 协议约定；协议本身没有逐方向 PBC 标志，外部 driver 必须实现匹配的边界条件。
- **[S4] i-PI。** [功能与理论出处](https://docs.ipi-code.org/features.html)、
  [RPC/MTS 作者教程](https://atomistic-cookbook.org/examples/pi-mts-rpc/mts-rpc.html)、
  [固定源码 `93162d280d15fd954da6dc3580a5ed3503e207ac`](https://github.com/i-pi/i-pi/tree/93162d280d15fd954da6dc3580a5ed3503e207ac)。
- **[S5] LAMMPS。** [`fix pimd/langevin`](https://docs.lammps.org/fix_pimd.html)、
  [`fix ipi`](https://docs.lammps.org/fix_ipi.html)，文档标记 release `30Sep2026`；
  [查阅时 develop 源码 `7dbed49448a17dbc9d4bc3eee5512e9ea2ad3a04`](https://github.com/lammps/lammps/tree/7dbed49448a17dbc9d4bc3eee5512e9ea2ad3a04)。
- **[S6] PLUMED 2.10。** [DISTANCE](https://www.plumed.org/doc-v2.10/user-doc/html/_d_i_s_t_a_n_c_e.html)、
  [COM](https://www.plumed.org/doc-v2.10/user-doc/html/_c_o_m.html)、
  [WHOLEMOLECULES](https://www.plumed.org/doc-v2.10/user-doc/html/_w_h_o_l_e_m_o_l_e_c_u_l_e_s.html)。
- **[S7] OpenPathSampling 作者论文。** [Python framework, part 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6374749/)，
  对照 TIS 的完整路径系综、条件交叉概率及有效通量归一化。
- **[S8] GPAW。** [Dipole-layer corrections](https://gpaw.readthedocs.io/tutorialsexercises/electrostatics/dipole_correction/dipole.html)，
  对照非对称 slab 的实空间／平面波边界、真空电势和偶极层修正；本 PR 没有运行 GPAW 计算。
