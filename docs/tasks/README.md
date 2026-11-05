---
orphan: true
---

# 独立 agent 任务卡（第四轮：对标评审后续）

更新日期：2026-11-03。基线：`main@ceee59c`（第三轮集成 #91）。

本目录把 2026-11-03 对标评审（PyRPMD 与 i-PI、NQCDynamics.jl、RPMDrate/Caracal、
SHARC/Newton-X、LAMMPS/CP2K/Tinker-HP、PLUMED/infretis、TorchSim 的比较）得出的功能改进，
拆成可以交给彼此独立的 agent 执行的任务卡。每张卡都写明前置条件、独占文件、共享注册点、
交付物和验收标准。只要遵守下面的通用规则，同一波次的卡可以同时执行，合并时只会在少数
登记好的注册点上产生追加式冲突。

**用法：** 交给 agent 时，同时提供本页（通用规则）和对应的卡片文件。agent 必须先完整阅读
两者再动手。

> "原生路径替换"专项任务卡（NP0–NP8，让整条轨迹尽量留在 C++）见
> [native-path/README.md](native-path/README.md)，同样遵守本页的 G1–G15。

## 通用执行规则

所有卡片都遵守以下规则；卡片中的具体要求优先于本节。

- **G1 基线与分支。** 从最新的 `main`（不早于 `ceee59c`）建分支，分支名用卡片建议的名字；
  一张卡对应一个分支、一个 PR。不要合并其他尚未合并的 PR；提交 PR 前 rebase 到最新 `main`。
- **G2 文件所有权。** 只修改卡片“独占文件”中列出的文件、新建的文件，以及“共享注册点”。
  如果必须修改其他卡独占的文件，停下来，在 PR 的“需要协调”一节说明原因和最小改动，
  不要自行扩大修改范围。
- **G3 保留文件（只由集成者修改）。** `README.md`、`README.zh-CN.md`、`docs/index.md`、
  `docs/zh/**`、`docs/manual/**`、`docs/roadmap.md`、`docs/references.bib`、
  `docs/method-references.json`、`docs/releases/**`、`src/pyrpmd/__init__.py` 的公开导出、
  所有版本号。需要新增的参考文献、toctree 条目、README 段落写进 PR 的“需集成者处理”一节。
  如果复制或改写了第三方代码，必须追加 `THIRD_PARTY_NOTICES.md`（这是唯一的例外）。
- **G4 共享注册点（只允许追加式的最小改动）。** `src/pyrpmd/input.py`、`input_tasks.py`、
  `input_schema.py`、`runner.py`、`cli.py`、`workflows/__init__.py`、`potentials/__init__.py`、
  `pyproject.toml`（只限 `optional-dependencies`）、`.github/workflows/*.yml`（只限新增 job 或步骤）、
  `docs/conf.py`（只限向 `include_patterns` 追加新指南）。
  只加入新分支、新条目或新子命令，不重排、不重构已有代码；PR 中逐一列出这些改动。
- **G5 科学与持久化契约。**
  - 不改变已有输入的物理身份（run identity / fingerprint）和已有 checkpoint 格式，除非卡片明确要求；
    确需改变时必须版本化，保留旧格式读取器并加测试。
  - 新的随机过程把全部状态（含 RNG、噪声缓冲、计数器）写进原子 checkpoint；
    CPU 确定性路径上“中断后续算”必须与“不中断”逐位一致。
  - 新输入在产生任何副作用（创建输出目录、启动外部程序、计算 PES、消耗 RNG）之前完成严格校验。
  - 内部一律使用原子单位，输入输出处显式标注单位。
  - 结果默认 `physical_converged=false`；失败、未判定的样本要保留并单独统计，不能丢弃。
- **G6 测试。** 新增单元测试，并至少加一个端到端（CLI 或公开 API）测试。
  `python -m pytest -q -m "not campaign"` 必须全部通过；耗时长的验证标记为 `campaign`；
  可选依赖缺失时跳过而不是失败；不写依赖时序的测试（见 T21 的教训）。
- **G7 原生代码。** 改动 `native/` 时，开启和关闭 OpenMP 都要能编译；`PYRPMD_NO_NATIVE=1` 的
  NumPy 路径必须继续可用；保留并通过原生与 NumPy 的一致性测试；开启 FMA 的构建也要通过。
- **G8 文档。** 新的英文指南放在 `docs/<topic>.md`，文件开头写 `orphan: true` front matter
  （由集成者加入 toctree），并把文件名追加到 `docs/conf.py` 的 `include_patterns`。
  手册只构建该白名单中的文件，不加入就不会被 `-W` 检查。中文版可选。
  `python -m sphinx -W --keep-going -b html docs docs/_build/html` 必须通过。
  指向尚不存在的文件时用反引号写路径，不要写成 Markdown 链接。
  本目录的任务卡和 `docs/validation/` 下的记录不进入手册构建，在 GitHub 上阅读。
- **G9 验证证据。** 记录放在 `docs/validation/<topic>-<YYYY-MM-DD>/README.md`，可附小型 JSON，
  每张卡合计不超过 1 MB。原始轨迹和大型二进制文件不要提交进 git，只记录 SHA256、
  复现命令、软件版本和硬件。
- **G10 诚实边界。** 没有证据时，不得声称 GPU 加速、硬件实测、物理收敛或复现了文献。
  没有实机运行过的外部程序配方标注为“未实测”。
- **G11 PR 描述。** 至少包含以下各节：
  - 改动摘要；
  - 文件清单（标出共享注册点的改动）；
  - 验证命令与原始 pytest 汇总行；
  - 未完成事项与限制；
  - 需要协调；
  - 需集成者处理（README、toctree、参考文献、发布说明）；
  - AI 协助披露（见 `docs/publication-readiness.md`）。
- **G12 不发布。** 不打 tag、不发布 PyPI/conda 包、不推送 `main`、不 force push 他人的分支。

以下三条来自第四轮审查（2026-11-03）中发现的实际问题，违反任何一条的分支都不会被合并：

- **G13 只提交可审查的源码。** 不提交编码载荷（base64、base85、压缩包）；不新增或修改带
  `contents: write` 权限的工作流；不在 CI 中生成、解码或推送代码；不以他人身份提交。
  环境无法直接推送时，停止工作并报告，不要绕过。`tools/check_workflow_policy.py`（由
  “Workflow policy” 工作流和 `tests/test_workflow_policy.py` 运行）会自动拒绝 `.github/` 中的
  非元数据文件、写权限请求，以及推送、解码或动态执行代码的工作流步骤。
- **G14 先跑测试再提交。** 本地无法克隆仓库、安装包或运行测试时，不要只凭 `py_compile`
  或 CI 排队就提交 PR；应停止并报告环境问题。推送的代码必须与实际测试过的代码完全一致：
  不要事后“重建”或“物化”源码。
- **G15 保持代码结构。** 不按文件大小把源文件切成 `*_partN` 片段；不删除与改动无关的注释；
  不用“复制旧模块 + 导入时替换函数或 `sys.modules`”的方式规避编辑大文件，应直接修改原模块；
  性能改动必须保持已有运行的逐位结果，或者作为带版本号的数值契约改动单独说明。

规模估计：**S** 表示一个模块加测试；**M** 表示新模块、工作流接入和小型验证；
**L** 表示跨模块重构，或新的工作流加外部验证。

## 卡片总览

### 第 1 波：可立即开始，彼此可以并行

| 卡片 | 标题 | 规模 | 前置 | 主要独占文件 |
| --- | --- | --- | --- | --- |
| [T16b](T16b-pimd-engine-convergence.md) | 三套 PIMD 引擎收敛（既有待办，补全卡片） | L | 无 | `workflows/materials.py`、`native_pimd.py`、`torch_materials.py`、`native_ensemble.py`、`torch_backend.py`、`native/thermal.*` |
| [T26](T26-core-auxiliary-operators.md) | 统一传播核心 I：辅助自由度算符协议与 NRPMD 试点 | L | 无 | `engine.py`、`dynamics.py`（钩子）、`nonadiabatic.py` |
| [T27](T27-native-threading.md) | 原生内核多线程扩展与原生周期 LJ | M | 无 | `native/cayley.cpp`、`native/propagation.h`、`native/fused.hpp`、`native/ring_fft.h`、`native/builtin_pes.hpp` |
| [T28](T28-ab-initio-socket.md) | 从头算 PIMD/RPMD：i-PI 服务端配方与稳健性 | M | 无 | `potentials/socket.py`、`examples/ab_initio/` |
| [T29](T29-product-states-dcs.md) | dRPMD/QCT 产物态分辨与微分截面 | M | 无 | `analysis/scattering.py`、`workflows/drpmd_analysis.py` |
| [T30](T30-kinetics-export.md) | 速率拟合与动力学模型导出 | S | 无 | `studies.py`（拟合部分） |
| [T31](T31-vtst-sct.md) | 变分过渡态理论与 SCT 隧穿对照 | L | 无 | 新 `vtst.py`、`workflows/vtst.py` |
| [T32](T32-scalable-instanton.md) | 可扩展 instanton 预因子 | M | 无 | `instanton.py`、`low_temperature.py`、`tunnelling.py` |
| [T33](T33-qtb.md) | 量子热浴 QTB/adQTB | M | 无（与 T26 共用 `engine.py` 注册点） | 新 `qtb.py` |
| [T34](T34-structure-transport.md) | 凝聚相结构与输运分析 | M | 无 | 新 `analysis/structure.py`、`analysis/transport.py` |
| [T35](T35-quantum-free-energy.md) | 绝对量子自由能热力学积分 | L | 无 | 新 `free_energy_ti.py`、`workflows/free_energy.py` |
| [T36](T36-plumed-bridge.md) | PLUMED 桥升级：多 walker、珠子 CV、MPI | M | 无 | `plumed.py`、`workflows/plumed.py` |
| [T37](T37-replica-exchange-workflow.md) | 并行温度与哈密顿副本交换工作流 | M | 无 | `replica_exchange.py`、新 `workflows/tempering.py` |
| [T38](T38-async-walkers.md) | 多进程 multiple-walker MetaD/OPES | M | 无 | `shared_metadynamics.py`、`opes.py`、对应工作流 |
| [T23](T23-path-sampling-workflow.md) | 路径采样工作流（扩充原 T23） | L | 无 | `retis.py`、`pptis.py`、`ffs.py`、`transition_interface.py`、`committor.py` |
| [T39](T39-photo-initial-lvc.md) | 光动力学初始条件与 LVC 模型构建 | M | 无 | 新 `photo_initial.py`、`potentials/lvc_builder.py` |
| [T40](T40-excited-state-interface.md) | 在线激发态电子结构接口 | L | 无 | 新 `potentials/excited_state/` |
| [T41](T41-community-plugins.md) | 社区文件、可执行教程与插件入口 | M | 无 | `.github/ISSUE_TEMPLATE/`、`docs/tutorials/`、新 `plugins.py` |

### 第 2 波：前置卡合并后开始

| 卡片 | 标题 | 规模 | 前置 |
| --- | --- | --- | --- |
| [T42a](T42a-port-mapping-methods.md) | 统一核心 II-a：映射类非绝热方法移植与 SU(N) 推广 | L | T26 |
| [T42b](T42b-port-hopping-methods.md) | 统一核心 II-b：跃迁与摩擦类方法移植 | L | T26 |
| [T43](T43-torch-parity-gpu.md) | Torch 引擎功能对齐与 GPU 实测 | L | T16b；GPU 实测需要 NVIDIA 硬件 |
| [T44](T44-constraints-slab-npt.md) | 约束积分器扩展与板（slab）NPT | M | T16b、T26 |
| [T45](T45-driven-cavity.md) | 外场驱动与腔 RPMD | M | T26 |
| [T46](T46-rpc-mts-workflows.md) | RPC/MTS 接入材料与从头算工作流 | M | T16b、T28 |

### 第 3 波

T17（README、参考文献与发布说明的总整合，见 `docs/roadmap.md`）放在最后，
由集成者统一处理各 PR 的“需集成者处理”一节。

## 依赖与冲突

- **`engine.py`：** 由 T26 独占结构性改动。T33 只追加 `qtb`/`adqtb` 的参数表条目和热浴工厂分支；
  T16b 不修改 `engine.py`，确有需要时写进“需要协调”。T44 和 T45 在 T26 合并后开始。
- **`native/`：** T27 负责传播、自由环和内置 PES 内核；T16b 负责 `native/thermal.hpp`、
  `native/thermal.cpp`（原生系综引擎）。T42 不改原生代码。
- **非绝热：** T26 只移植 NRPMD（`nonadiabatic.py`）；其余方法分给 T42a/T42b。
  T40 不修改 `adiabatic.py` 中的动力学类（第 254 行之后），提供者契约只允许向后兼容的追加。
- **低温速率：** T32 独占 `instanton.py`、`low_temperature.py`、`tunnelling.py`；T31 只能调用
  其中的公开函数，不能修改。
- **共享注册点：** T23、T29、T30、T31、T35、T37、T39、T41 都会在 `cli.py`、`runner.py` 或
  `input*.py` 追加条目。这类冲突全是追加式的，由集成者按合并顺序解决。

```text
T16b ──┬─> T43 (GPU)
       ├─> T44 <── T26 ──┬─> T42a
       └─> T46 <── T28   ├─> T42b
                         └─> T45
其余第 1 波卡片彼此独立；全部合并后由 T17 统一整合。
```

## 建议分工

每条“车道”交给一个 agent 按顺序执行。不同车道之间在第 1 波没有独占文件冲突
（只有共享注册点上的追加式冲突）。遇到第 2 波卡片时，如果前置卡还没有合并，
就先做本车道中的下一张，或者等待。

**4 个 agent：**

| 车道 | 方向 | 卡片顺序 |
| --- | --- | --- |
| A | 核心 | T26 → T42a → T42b → T45 |
| B | 引擎 | T16b → T28 → T46 → T44（T44 还需要 T26） |
| C | 速率与散射 | T30 → T29 → T31 → T32 → T23 |
| D | 凝聚相与采样 | T27 → T33 → T34 → T35 → T36 → T38 → T37 |

剩余的 T39、T40、T41、T43，由先空闲的车道领取。

**6 个 agent：**

| 车道 | 方向 | 卡片顺序 |
| --- | --- | --- |
| A | 核心 | T26 → T42a → T45 |
| B | 引擎 | T16b → T44 → T46（T46 等待 F 车道的 T28） |
| C | 速率与散射 | T30 → T29 → T31 → T32 |
| D | 凝聚相 | T27 → T33 → T34 → T35 |
| E | 增强与路径采样 | T36 → T38 → T37 → T23 |
| F | 电子结构与生态 | T28 → T39 → T40 → T41 |

剩余的 T42b（等待 T26）和 T43（等待 T16b，并需要 GPU）由先空闲的车道领取。

**8 个 agent：** 在 6 车道的基础上，把 T42b 交给 G（等待 T26），把 T43 交给拥有 NVIDIA GPU 的 H
（等待 T16b）。

优先级：T26、T16b、T27、T28 决定核心竞争力，应最先开始；T29–T32 让 PyRPMD 成为最完整的
RPMD 速率与散射程序；其余卡片补齐与 i-PI、SHARC、PLUMED 生态的差距。

## 与既有待办的关系

- **T16b：** 原路线图中的待办，本目录补全为可分派的卡片。
- **T17：** 保持最后执行，并吸收本轮各 PR 的“需集成者处理”。
- **T18c：** 不变，仍然需要 Windows CI。
- **T19（续）：** 合入 T43 的 GPU 实测部分；没有硬件时只能完成 CPU 正确性部分。
- **T20：** 不变（f-QCMD 光谱证据与 h-CMD）。
- **T23：** 本目录给出扩充后的卡片，取代原来的一行描述。
- **T24（VES）：** 优先通过 T36 的 PLUMED VES 路线满足；原生 VES 仍是可选研究方向。
- **T25：** TDDFT NAC 由 T40 取代；环聚合物 IESH 和多态 RP-MASH 仍是研究方向，
  在 T42a/T42b 的统一核心上实现会更容易。
- **数据迁移：** 仍然需要维护者决定，本轮不处理。

## 未拆卡事项

- 旗舰算例的生产级收敛（H + H₂、H + CH₄、q-TIP4P/F 扩散）：需要大量计算资源，
  按 `docs/literature-validation.md` 的计划继续，不适合一次性交给 agent。
- PyPI 与 conda-forge 正式发布：需要维护者凭据与审核，见 `docs/distribution.md`。
- 瞬态吸收、TRPES 等非绝热光谱可观测量：在 T39、T40 合并后再拆卡。
- QM/MM 嵌入：在 T40 合并后，作为单独的电子结构卡处理。
