---
orphan: true
---

# 开发路线与待办清单

更新日期：2026-11-02。本页跟踪按任务卡 T01–T17 分派给独立 agent 的改进工作，以及集成时发现的后续事项。各任务的验证记录在对应的 `docs/validation/`、`benchmarks/` 和功能指南中；本页只汇总状态，不重复科学结论。

集成分支 `claude/clever-cray-renag1` 以 `main@8cc402f` 为基准，合并了下列已完成的任务分支以及 PR #78、#79、#80。集成后在 Linux（Python 3.13，4 核，未安装 ASE、Torch、OpenMM、MDAnalysis、matplotlib）上运行全量测试，结果为 **9990 passed、361 skipped、0 failed**（14 分钟）；`-m "not campaign"` 共收集 8443 个测试；中英文 Sphinx `-W` 构建均通过。合并前 main 的基线为 7992 passed、1 failed。

## 第一波任务状态

| 任务 | 状态 | 来源分支 | 交付内容 | 遗留问题 |
| --- | --- | --- | --- | --- |
| T01 工程卫生与跨平台 CI | 已合并 | `t01/engineering-hygiene-cross-platform-ci` | 新增 `campaign` marker（`pytest -m "not campaign"` 为快速库测试）；新增 macOS/Windows 库测试 job；可选依赖测试加守卫；只读数据清单脚本 `tools/data_manifest.py` 与 [数据归档计划](data-archive-plan.md) | Windows 上发现 4 类运行时缺陷，已在 `tests/conftest.py` 中按测试名精确跳过，修复见 T18 |
| T02 约束父链度规 O(N) 化 | 已合并 | `perf/t02-metric-hvp` | Python 与两份原生实现都用解析 Hessian–向量积替代稠密 CV Hessian；新增 `centroid_value_gradient_hvp` API；run identity 不变 | — |
| T03 内置解析势原生绑定 | 已合并（PR #80） | `codex/t03-builtin-native-bindings` | Free、Harmonic、MorsePair、DoubleWell 和固定 cell 的 PeriodicCosine 使用 C ABI 内核（`native/builtin_pes.hpp`）；快速入门和底层示例全部走原生步；run fingerprint 不变；旧 checkpoint 可续算。P=32、N=3 时提速 34.6× | 非绝热矩阵势（diabatic、Tully/LVC/spin-boson、NaH2 等）需要矩阵 ABI，尚未绑定 |
| T04 ML 势批量评估 | 已合并 | `feat/t04-batched-ml-potentials` | 新增 `mace-batched`、`nequip-batched`、`fairchem-batched`、`deepmd-batched`、`metatomic` provider；`evaluate_tensor` 设备接口；附 CPU 吞吐基准 | 只在 CPU 上测量，CUDA 未验证，见 T19 |
| T05 OpenMM/LAMMPS 适配器 | 已合并（部分达标） | `feat/t05-force-engine-batching` | OpenMM 持久并发 Context：8 workers 时 CPU 上提速 3.1×（192 原子）和 5.1×（2187 原子）；LAMMPS 珠子亲和的持久实例与私有 MPI 子通信器（`pes_ranks`） | LAMMPS 持久模式在 256 原子时只快 1.26×，2048 原子时为 0.92×（反而变慢）；OpenMM GPU 驻留和 `RPMDIntegrator` 路线未做，见 T19 |
| T06 Torch 后端接入工作流 | **未交付** | 无分支 | — | 需要重新分派 |
| T07 f-QCMD | 已合并（部分达标） | `feat/t07-fqcmd` | 实验性 `method="f-qcmd"`，面向固定拓扑的弯曲三原子分子：IBI 训练、固定准质心的 PIMD 采样器、单珠生产与续算（[指南](qcmd.md)） | 没有记录相对 CMD 的 OH 伸缩谱移动结果；h-CMD 和实时 QCMD 未做，见 T20。重复分支 `t07-fqcmd-curvilinear` 未合并 |
| T08 费米子与周期交换 | 已合并 | `feat/t08-quantum-exchange` | `method="fermionic"`：符号重加权，平均符号过小时拒绝给出结论；`periodic_mode="minimum_image"`；联合块误差 | 交换 + NPT 显式拒绝；没有 winding 求和 |
| T09 生产级 instanton | 已合并 | `codex/t09-production-instanton` | Bofill 更新与 Hessian provider；新命令 `pyrpmd instanton`。H + H₂（BKMP2）在 200、250、300 K 下的速率都落在文献两位有效数字的舍入区间内，最大偏差 2.1%（200 K） | 致密预因子最多支持 4096 个活动珠子坐标 |
| T10 路径采样 | 已合并 | `t10/path-sampling-20261102` | RETIS [0⁻] minus 系综与自身通量、REPPTIS、直接 FFS；在解析双阱上与精确速率一致 | 未做 ∞RETIS 调度和 TOML 工作流 |
| T11 增强采样 | 已合并 | `feat/t11-enhanced-sampling-20261102` | `[fes] method="abf"`、`"eabf"`、`"steered"`；有限温度 string；PLUMED `restart="files"` 续算；mlcolvar → TorchScript CV 路线 | 未做 VES；steered 只给出自由能差 |
| T12 IESH 与多态 MASH | 已合并 | `feat/t12-iesh-multistate-mash` | `method="iesh"`（Newns–Anderson 模型）和 `method="multistate-mash"`，均为 P=1（[指南](nonadiabatic-iesh-multistate.md)） | 没有环聚合物版本 |
| T13 PySCF 多态桥与 HEOM | 已合并 | `feat/t13-pyscf-heom` | `PySCFMultistatePotential`（SA-CASSCF）及 LiH RPSH 示例；`SpinBosonHEOM` 复现了已发表的布居曲线 | 未做 TDDFT NAC |
| T14 主动学习闭环 | 已合并 | `feat/t14-active-learning-loop` | `ActiveLearningLoop`：可续算的采样 → 标注 → 训练轮次；CPU 上的 MACE 参考训练器 | — |
| T15 GLE/PIGLET 参数 | 已合并 | `feat/t15-gle-parameter-tools` | gle4md 文件导入；optimal GLE 与 PIGLET 矩阵生成并进入 run identity；PIGLET 谐振子能量误差约 1.5×10⁻⁴ | — |

## 其他已合并的 PR

| PR | 内容 | 说明 |
| --- | --- | --- |
| #78 | QCT 初始化新增 `j0_energy_policy="preserve_actions"` | 默认行为与旧 checkpoint 不变；新策略是实验性的约束相位系综 |
| #79 | 精度驱动的热 RPMD 速率控制器（`pyrpmd.precision*`） | 原为 draft，全部为新增文件；CLI 接入和执行层对接未做，见 T22 |
| #80 | T03 内置解析势原生绑定 | 原为 draft；详见上表 T03 行 |

## 未合并分支

| 分支 | 原因 |
| --- | --- |
| `t07-fqcmd-curvilinear` | 另一个 agent 独立完成的 f-QCMD 实现，模块、文档和测试与已合并的 `feat/t07-fqcmd` 同名，两者大面积冲突，只能保留一个。该分支附带气相水单体光谱验证记录，正好是已合并版本缺少的证据。保留哪一个由维护者决定；也可以把它的验证移植到已合并的实现上（见 T20） |
| `staging/reus-reviewed-20261031`、`feat/ecosystem-trust-20261101`、`codex/stage-mixed-sampling-20261102` | 暂存或数据传输分支，其内容已通过其他提交进入 main；建议维护者确认后删除 |

## 第二波与新增待办

| 编号 | 事项 | 前置条件 | 验收要点 |
| --- | --- | --- | --- |
| T06 | 设备常驻 Torch 后端接入 `pimd`/`diffusion`（重新分派；可直接使用 T04 的 `evaluate_tensor`） | 无 | 精确量子统计对照；完整的 run → interrupt → resume 测试 |
| T16a | 合并两份原生 CV/偏置实现，以 T02 的 HVP 接口为准 | 已具备（T02、T03 均已合并） | 原生与 Python 一致性测试全部通过 |
| T16b | legacy 与 native PIMD 引擎收敛，提供 checkpoint 迁移工具 | T06 合并后 | 旧 checkpoint 可续算，现有输入身份不变 |
| T17 | 精简 README、汇总参考文献与发布说明、中英同步 | T03、T06、T16 合并后 | Sphinx `-W` 通过，链接检查通过 |
| T18 | 修复 Windows 运行时缺陷：原生轨迹发布时对只读描述符 fsync；盘符路径解析；归档映射键中 `\` 与 `/` 不一致；并发 `os.replace` | 无 | 逐项删除 `tests/conftest.py` 中对应的 Windows 跳过登记，并附回归测试 |
| T19 | GPU 与大体系性能验证：T04 的 CUDA 实测；OpenMM CUDA/`RPMDIntegrator` 驻留路线；分析 LAMMPS 持久实例在 2048 原子时没有收益的原因 | 需要 GPU 环境 | 在注明硬件的前提下给出正确性与每步耗时 |
| T20 | 记录 f-QCMD 相对 CMD 的光谱结果；实现 h-CMD | 先由维护者在两个 f-QCMD 实现中选定一个 | 生成并归档 JSON 报告（可参考 `t07-fqcmd-curvilinear` 中的验证）；h-CMD 分区与身份测试 |
| T21 | 消除间歇失败的测试：`tests/test_h3_p256_parent_campaign.py::test_worker_failure_stops_admissions_and_drains_all_owned_processes`（基线上约 1/30 失败）和 `tests/test_h3_quantum_isolated_continuation.py::test_independent_worker_kernel_caps_survive_blocked_owner_alarm[CPU]` | 无 | 修复调度时序假设，重复运行 100 次不失败 |
| T22 | 精度控制器（PR #79）接入 CLI 和执行层 | 无 | 中断与重放不重复计数；用真实速率作业做端到端测试 |
| 数据 | 按[数据归档计划](data-archive-plan.md)迁移验证数据 | 需维护者决定 | 不改写历史的增量迁移；本次合并又新增了若干验证日志 |

## 可选研究方向

∞RETIS 并行调度；RETIS/REPPTIS/FFS 的 TOML 工作流；VES；环聚合物 IESH 与多态 RP-MASH；交换与 NPT 的严格结合及 winding；TDDFT NAC 桥；腔与外场驱动动力学。
