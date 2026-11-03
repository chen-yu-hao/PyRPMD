---
orphan: true
---

# 开发路线与待办清单

更新日期：2026-11-03（第三轮集成）。本页跟踪按任务卡分派给独立 agent 的改进工作、集成中做出的决定，以及后续待办。各任务的验证记录在对应的 `docs/validation/`、`benchmarks/` 和功能指南中；本页只汇总状态，不重复科学结论。

## 集成记录

| 轮次 | PR | 内容 | 验证 |
| --- | --- | --- | --- |
| 第一轮 | #81（合并为 `f877051`） | T01–T05、T07–T15，PR #78、#79、#80，以及 CI 发现问题后的修复：精度方差抵消、PLUMED 重启格式、double-well FMA、Windows 可移植性 | 60 个 CI 检查全部通过（Linux 3.11–3.14、macOS、Windows、Torch、真实 PLUMED） |
| 第二轮 | #86（合并为 `8f32b37`） | PR #82、#83（T18）、#84（T06），以及 T04、T08、T09 分支合并后追加的提交；全量测试的 CI 时限调整为 `fused` 40 分钟、macOS/Windows 库测试 45 分钟 | CI 全部通过；Linux 本地 10311 passed、360 skipped |
| 第四轮 | 本轮集成 | PR #93（T16b，部分）、#96（T23）、#99（T24）、#97（T25a）、#94（T25b）、#106（T37）、#104（T38）、#103（T40）；未合并的分支与原因见下文“第四轮审查结论” | 全量测试进行中（结果在合并前补记） |
| 第三轮 | #91（合并为 `ceee59c`） | PR #85（T18b）、#87（T21）、#88（T22）、#89（T19，部分）、#90（T16a） | Linux 本地（加载真实 PLUMED 2.9.4）：10491 passed、405 skipped、0 failed，新增的跳过都来自需要 OpenMM、LAMMPS 或 CUDA 的 T19 测试；在真实 OpenMM 8.6.1 和 LAMMPS（2025-07-22）上，T19 与现有引擎测试 102 passed（7 个仅限 CUDA 的跳过）；中英文 Sphinx `-W` 通过；CI 结果见 PR |

## 任务状态

| 任务 | 状态 | 交付内容 | 遗留问题 |
| --- | --- | --- | --- |
| T01 工程卫生与跨平台 CI | 已合并 | `campaign` marker（`pytest -m "not campaign"` 为快速库测试）；macOS/Windows 库测试 job；可选依赖守卫；只读数据清单脚本 `tools/data_manifest.py` 与[数据归档计划](data-archive-plan.md) | Windows 跳过名单由 T18b 清理 |
| T02 约束父链度规 O(N) 化 | 已合并 | 用解析 Hessian–向量积替代稠密 CV Hessian（Python 与两份原生实现）；新增 `centroid_value_gradient_hvp`；run identity 不变 | — |
| T03 内置解析势原生绑定 | 已合并（PR #80） | Free、Harmonic、MorsePair、DoubleWell 和固定 cell 的 PeriodicCosine 使用 C ABI 内核；P=32、N=3 时提速 34.6×；double-well 能量不依赖浮点精确抵消 | 非绝热矩阵势需要矩阵 ABI |
| T04 ML 势批量评估 | 已合并（含后续提交） | `mace-batched`、`nequip-batched`、`fairchem-batched`、`deepmd-batched`、`metatomic`；`evaluate_tensor`；后续提交加入 DeePMD 精度修复、NequIP AOTI 校验、减少 MACE 图拷贝、CUDA stream 排序 | 只在 CPU 上测量过，见 T19 |
| T05 OpenMM/LAMMPS 适配器 | 已合并（部分达标） | OpenMM 持久并发 Context，CPU 上提速 3–5×；LAMMPS 珠子亲和的持久实例与私有 MPI 子通信器 | LAMMPS 大体系无收益；OpenMM GPU 驻留路线未做，见 T19 |
| T06 Torch 后端接入工作流 | 已合并（PR #84） | `sampling.engine="torch"` 用于 `pimd`/`diffusion`：所有副本放在一个 `[B,P,N,3]` 批次中，原子 checkpoint 包含 RNG 状态，可精确续算（[指南](torch-materials-workflows.md)） | CUDA 未实测，见 T19；不支持 NPT、约束、交换、摩擦、MPI 分组 |
| T07 f-QCMD | 已合并（部分达标） | 实验性 `method="f-qcmd"`，面向弯曲三原子分子（[指南](qcmd.md)）。**已决定保留此实现** | 光谱证据与 h-CMD 见 T20 |
| T08 费米子与周期交换 | 已合并（含后续提交） | `method="fermionic"` 符号重加权；`periodic_mode="full_winding"`（高斯像求和）或 `minimum_image`；周期交换使用全珠子各向同性 BZP NPT | 无超流估计量；centroid BZP 仍不能与周期交换同时使用 |
| T09 生产级 instanton | 已合并（含后续提交） | Bofill 优化与 Hessian provider，`pyrpmd instanton`；H + H₂ 速率在文献两位有效数字以内；后续提交加入正常区 golden-rule instanton，并与独立参考对照 | 致密预因子最多支持 4096 个活动珠子坐标 |
| T10 路径采样 | 已合并 | RETIS [0⁻] minus 系综与自身通量、REPPTIS、直接 FFS | ∞RETIS 与 TOML 工作流见 T23 |
| T11 增强采样 | 已合并 | ABF、eABF、steered、有限温度 string、PLUMED `restart="files"`、mlcolvar 路线 | VES 见 T24 |
| T12 IESH 与多态 MASH | 已合并 | `method="iesh"`、`method="multistate-mash"`，均为 P=1 | 环聚合物版本见 T25 |
| T13 PySCF 多态桥与 HEOM | 已合并 | `PySCFMultistatePotential`（SA-CASSCF）；`SpinBosonHEOM` | TDDFT NAC 见 T25 |
| T14 主动学习闭环 | 已合并 | `ActiveLearningLoop`；CPU 上的 MACE 参考训练器 | — |
| T15 GLE/PIGLET 参数 | 已合并 | gle4md 导入；optimal GLE 与 PIGLET 生成 | — |
| T18 Windows 运行时缺陷 | 已合并（PR #83） | 原生发布在只读描述符上的 fsync 与内存映射 unlink、归档路径分隔符、并发 JSON 替换。盘符路径哈希由第一轮的 `f9e85b4` 修复 | 这三类的 Windows 跳过登记仍在，见 T18c |
| T18b Windows 旧运行身份 | 已合并（PR #85） | 已保存的平台为 Windows、且以盘符路径引用的插件或模型未被哈希的旧运行，在迁移或分派之前就拒绝续算，并提示在新目录重跑；删除盘符路径类跳过登记（37→22 个名称），恢复 16 个 Windows 用例，新增 44 个回归测试；Windows CI 实测通过 | #83 三类跳过登记的清理见 T18c |
| T16a 合并原生 CV 实现 | 已合并（PR #90） | 两份原生 CV、偏置和度规实现合并为一份共享实现；开启 FMA 的构建与默认构建都通过；N=3 时 12 组基准的变慢都不超过 2% | 可选的 nanobind/pybind11 试点被推迟 |
| T19 GPU 与大体系性能 | 部分合并（PR #89） | CUDA 测量驱动脚本（5 个 ML provider 与 PIMD/扩散工作流）、OpenMM RPMD copy-bank 评估路线、LAMMPS 校验向量化；CPU 实测：N=4096 时 OpenMM Contexts-4 提速 1.9–2.5× | **没有任何 GPU 测量**（环境中无 NVIDIA 设备）；部分 LAMMPS 配置仍偏慢 |
| T21 消除间歇失败测试 | 已合并（PR #87） | 根因在测试中的时序假设：子进程等长睡眠不能保证先后顺序；1 秒 CPU 时间与 3 秒墙钟时间相互竞争。修正后在负载下各运行 100 次，0 失败；未改动生产代码 | — |
| T22 精度控制器 CLI | 已合并（PR #88） | 有预算、可续算的 CLI 执行：中断和重放不会重复计数，覆盖崩溃窗口，67 个测试；跨平台加锁 | 示例只是小规模工程算例，不代表物理收敛 |
| T16b PIMD 引擎收敛 | 部分合并（PR #93） | 统一的 `EngineRoute` 分派与预检；原生 checkpoint 改用共享原子容器（v1 载荷不变，旧文件可读）；`python -m pyrpmd.pimd_checkpoint inspect` 与 legacy 内存/流式存储迁移；7 个由 `ceee59c` 生成的 checkpoint 实际续算逐字节一致 | 三套数值内核仍各自独立；没有 `pyrpmd migrate` 跨引擎迁移，也没有三引擎 NVE 等价测试 |
| T23 路径采样工作流 | 已合并（PR #96） | `task="path_sampling"`：RETIS、REPPTIS、FFS 与异步 ∞RETIS，原子 checkpoint、精确续算、报告；四次独立运行与解析四次势速率在 2 SE 内一致 | 无 aimless shooting/似然最大化；不支持 MPI 和续算时改变 worker 数；FFS 噪声大 |
| T24 VES | 已合并（PR #99） | `[fes] method="ves"`：一维、二维质心 CV 的 Legendre 展开，uniform 与 well-tempered 目标，冻结后重加权，精确续算；与数值积分和 MBAR 对照 | 仅非周期 CV、无约束全可动体系；P=1 的模型验证不代表分子体系收敛 |
| T25a 环聚合物 IESH | 已合并（PR #97） | `method="rp-iesh"`：Gardner 2023 质心研究近似，逐珠受力、共享行列式，记录跳跃造成的完整环能量变化；独立 Fock 空间参考对照 | `method_status="research"`；仅实对称、非简并、独立电子模型 |
| T25b BA-RP-MASH | 已合并（PR #94） | `method="rp-mash-ba"`：双态 bead 近似 RP-MASH，事件驱动跳跃，三路 RNG 完整续算；有限能隙 Fourier 参考对照 | 重复实现 PR #95 未合并（见“维护者的决定”） |
| T37 副本交换工作流 | 已合并（PR #106） | `[tempering]`：PIMD/FES 的 PT 与 HREMD，复用 G×B 执行器，原子系综 checkpoint，按热力学态重组的轨迹与 MBAR | 固定晶胞；不支持压浴、约束、Torch/native 引擎、OPES 等组合 |
| T38 分布式 multiple walkers | 已合并（PR #104） | MetaD/OPES 的 `walkers_mode="mpi-sync"\|"async-files"`；同步模式逐位等价，异步模式带校验和日志与偏移续算；共享固定带宽 OPES | 实现采用“逐字节私有副本 + 导入时打补丁”的兼容层（含 `sys.modules` 替换），需要后续去重；异步模式依赖共享文件系统语义 |
| T40 在线激发态接口 | 已合并（PR #103） | `potentials/excited_state/`：态追踪与相位对齐、HST/NPI 耦合、PySCF TDA/TDDFT 后端，ORCA/OpenMolcas/Q-Chem 文件适配器 | 三个外部程序均未实机运行；仅闭壳层 RHF/RKS；无 SOC、QM/MM |

其他已合并的 PR：
- #78：QCT `j0_energy_policy="preserve_actions"`；
- #79：精度驱动速率控制器，CLI 接入见 T22；
- #82：隧穿劈裂相位投影降到二次复杂度。

## 维护者的决定

- **f-QCMD：** 保留 `feat/t07-fqcmd`（已在 main 中）。`t07-fqcmd-curvilinear` 只作 T20 的参考，T20 完成后删除。
- **暂存分支：** `staging/reus-reviewed-20261031`（`d3243dd`）、`feat/ecosystem-trust-20261101`（`3d096d5`）、`codex/stage-mixed-sampling-20261102`（`06bab4e`）的内容已在 main 中，可以删除。

- **BA-RP-MASH：** PR #94 与 #95 实现了同一方法 `method="rp-mash-ba"`。第四轮合并 #94（对已有 CA 代码改动更小，且 Linux 全量 CI 已通过）；#95 作为重复实现关闭，其中 P=1、4、8 精确求积参考和串行/进程等价测试可以移植到 #94 的模块。

## 第四轮审查结论（2026-11-03）

以下分支没有合并，需要退回修改或重新提交：

| 分支 / PR | 结论 | 原因 |
| --- | --- | --- |
| `agent/t26-aux-operators`（#102） | 退回 | `nonadiabatic.py` 删除了 `_positive_integer` 等被 `adiabatic.py` 等模块导入的名称，32 个测试模块在收集阶段报错；提交者未运行任何测试；兼容层采用复制旧模块再在导入时替换 `_save`/`_load` 的方式 |
| `agent/t27-native-threading`（#105） | 退回 | `native/ring_fft.h` 中 `forward_two_real` 使用未声明的 `j`，无法编译；源文件被按字节切成 `*_part1..4.hpp` 并删去原有注释；中大体系轨迹与 `ceee59c` 不再逐位一致（未版本化）；N=3、P=1 单线程变慢约 46%（Cayley）至 110%（normal-mode）；N=1000、P=32 的 4 线程加速 1.43×，未达 2.5× |
| `agent/t36-plumed-bridge`（无 PR） | 退回 | 测试向 `sys.modules` 注入的模拟 `plumed` 模块未清理，导致已有的真实内核测试报错而非跳过；包含导出源码的临时工作流；兼容层同为复制加补丁 |
| `agent/t30-kinetics-export`、`agent/t32-scalable-instanton`、`agent/t35-quantum-free-energy`、`agent/t39-photo-initial-lvc-gpt56` | 拒绝 | 不含源码，只有编码载荷和带 `contents: write` 权限、在 CI 中解码并提交代码的工作流（T32 对载荷执行 `exec`；T39 改写了已有工作流 `paper-validation-source-export.yml` 并以维护者身份提交）。T30 载荷与其自带校验和不符；T39 载荷被截断；T32 载荷是只使用 `pathlib` 的文件改写脚本，可以由人工审查后作为普通提交重新提交 |
| `agent/t31-vtst-sct`（无 PR） | 未完成 | 只有 MEP/CVT/ZCT/SCT 数值核心；测试引用的 `workflows/vtst.py` 与 CLI 不存在 |
| #95 | 关闭 | 与 #94 重复 |
| #92（T20）、#98（MACE Hessian）、#100（T17）、#101（T35） | 草稿 | #92 待上传验证归档；#98 未运行测试且修改 T32 独占的 `instanton.py`；#100 需在本轮合并后变基；#101 只有占位内容 |

`agent/t16b-pimd-engine-convergence`、`agent/t28-ab-initio-socket`、`agent/t29-product-states`、`agent/t34-structure-transport`、`agent/t37-source-export`、`agent/t39-photo-initial-lvc`、`agent/t39-photo-initial-lvc-clean` 没有任何提交。

## 待办

| 编号 | 事项 | 前置条件 | 验收要点 |
| --- | --- | --- | --- |
| T18c | 在 Windows CI 上实测后，删除 #83 三类问题（原生发布、归档、JSON 替换）对应的跳过登记，并更新 `docs/development.md` | 无 | Windows 库测试通过，跳过名单再变短 |
| T19（续） | 在真实 GPU 上运行 `benchmarks/efficiency/t19_cuda.py` 等测量脚本：5 个 ML provider、Torch 工作流、OpenMM CUDA | GPU 环境 | 正确性与耗时表，注明硬件 |
| T20 | f-QCMD 光谱证据（可参考 `t07-fqcmd-curvilinear`）与 h-CMD——**尚未交付，需要重新分派** | 无 | 可复现的归档报告；h-CMD 身份与续算测试 |
| T16b（续） | 跨引擎迁移工具 `pyrpmd migrate` 与三引擎确定性 NVE 等价测试（分派与 checkpoint 存储已由 #93 合并） | 无 | 迁移目录记录来源与语义；三引擎 1000 步 `q,p` 一致 |
| T17 | README、参考文献、发布说明总整合，覆盖第四轮合并的 T16b、T23、T24、T25a、T25b、T37、T38、T40（草稿 PR #100 需变基） | 第四轮合并之后 | README 约 12 KB 以内，链接检查通过 |
| 去重 | 去掉 T38 的“逐字节私有副本 + 导入时补丁”兼容层，把改动直接写入原模块 | 无 | 旧 checkpoint 与运行身份不变；不再使用 `sys.modules` 替换 |
| 数据 | 按[数据归档计划](data-archive-plan.md)迁移验证数据 | 需维护者决定 | 不改写历史的增量迁移 |

## 第四轮任务卡

2026-11-03 的对标评审（与 i-PI、NQCDynamics.jl、RPMDrate/Caracal、SHARC、LAMMPS/CP2K/Tinker-HP、
PLUMED/infretis、TorchSim 的比较）拆成了可以独立分派的任务卡，放在 [`docs/tasks/`](tasks/README.md)：
共 22 张卡（T16b、T23 扩充版、T26–T41、T42a/b、T43–T46），附通用执行规则、文件所有权、
波次与依赖关系，以及按 agent 数量划分的车道。T25 中的 TDDFT NAC 由 T40 取代；
T24（VES）优先通过 T36 的 PLUMED 路线满足；T19（续）的 GPU 实测并入 T43。

## 可选研究方向

- T23：∞RETIS 与路径采样 TOML 工作流；
- T24：VES；
- T25：环聚合物 IESH、多态 RP-MASH、TDDFT NAC（每项单独分派）；
- 腔与外场驱动动力学。
