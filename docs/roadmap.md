---
orphan: true
---

# 开发路线与待办清单

更新日期：2026-11-03。本页跟踪按任务卡分派给独立 agent 的改进工作、集成中做出的决定，以及后续待办。各任务的验证记录在对应的 `docs/validation/`、`benchmarks/` 和功能指南中；本页只汇总状态，不重复科学结论。

## 集成记录

| 轮次 | PR | 内容 | 验证 |
| --- | --- | --- | --- |
| 第一轮 | #81（合并为 `f877051`） | T01–T05、T07–T15，PR #78、#79、#80，以及 CI 发现问题后的修复：精度方差抵消、PLUMED 重启格式、double-well FMA、Windows 可移植性 | 60 个 CI 检查全部通过（Linux 3.11–3.14、macOS、Windows、Torch、真实 PLUMED） |
| 第二轮 | 本轮集成 | PR #82、#83（T18）、#84（T06），以及 T04、T08、T09 分支合并后追加的提交 | Linux 本地（加载真实 PLUMED 2.9.4）：10311 passed、360 skipped、0 failed；`-m "not campaign"` 收集 8758 个测试；中英文 Sphinx `-W` 通过；CI 结果见 PR |

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
| T18 Windows 运行时缺陷 | 已合并（PR #83，部分） | 原生发布在只读描述符上的 fsync 与内存映射 unlink、归档路径分隔符、并发 JSON 替换。盘符路径哈希由第一轮的 `f9e85b4` 修复 | 跳过名单清理与 Windows 旧运行身份见 T18b |

其他已合并的 PR：
- #78：QCT `j0_energy_policy="preserve_actions"`；
- #79：精度驱动速率控制器，CLI 接入见 T22；
- #82：隧穿劈裂相位投影降到二次复杂度。

## 维护者的决定

- **f-QCMD：** 保留 `feat/t07-fqcmd`（已在 main 中）。`t07-fqcmd-curvilinear` 只作 T20 的参考，T20 完成后删除。
- **暂存分支：** `staging/reus-reviewed-20261031`（`d3243dd`）、`feat/ecosystem-trust-20261101`（`3d096d5`）、`codex/stage-mixed-sampling-20261102`（`06bab4e`）的内容已在 main 中，可以删除。

## 待办（第三波任务卡）

| 编号 | 事项 | 前置条件 | 验收要点 |
| --- | --- | --- | --- |
| T18b | Windows 收尾：按类别删除跳过名单中已修复的条目；为 `f9e85b4` 之前、以盘符路径引用插件或模型的 Windows 运行设计明确的身份处理（清晰报错或带版本号的迁移，不得静默接受） | 无 | Windows 库测试通过，跳过名单变短，Linux 身份不变 |
| T19 | GPU 与大体系性能验证：T04 provider 与 T06 工作流的 CUDA 实测；OpenMM 设备驻留；LAMMPS 大体系剖析 | GPU 环境 | 正确性与耗时表，注明硬件 |
| T16a | 合并两份原生 CV/偏置实现 | 无 | 一致性测试通过；开启 FMA 的构建同样通过 |
| T20 | f-QCMD 光谱证据（可参考 curvilinear 分支）与 h-CMD | 无 | 可复现的归档报告；h-CMD 身份与续算测试 |
| T21 | 消除两个间歇失败的 campaign 测试 | 无 | 负载下重复 100 次 0 失败，并给出根因 |
| T22 | 精度控制器接入 CLI 与执行层 | 无 | 端到端中断与续算测试；三个平台的 CI 通过 |
| T16b | legacy、native 与 torch 三套 PIMD 引擎收敛 | 无（T06 已合并） | 旧 checkpoint 可续算；迁移工具有往返测试 |
| T17 | README、参考文献、发布说明总整合 | 最后做 | README 约 12 KB 以内，链接检查通过 |
| 数据 | 按[数据归档计划](data-archive-plan.md)迁移验证数据 | 需维护者决定 | 不改写历史的增量迁移 |

## 可选研究方向

- T23：∞RETIS 与路径采样 TOML 工作流；
- T24：VES；
- T25：环聚合物 IESH、多态 RP-MASH、TDDFT NAC（每项单独分派）；
- 腔与外场驱动动力学。
