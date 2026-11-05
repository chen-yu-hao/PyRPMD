---
orphan: true
---

# T52 RPMDrate 输入导入器

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1（第五轮） |
| 前置 | 无 |
| 建议分支 | `agent/t52-rpmdrate-import` |

## 背景

RPMDrate 的用户是 PyRPMD 最直接的潜在用户：PyRPMD 的速率算法本身改编自 RPMDrate，
并且已经能加载 RPMDrate 的 f2py PES 模块（`provider="rpmdrate"`）。
但 RPMDrate 的**输入文件**目前无法使用，用户必须手工改写成 PyRPMD 的 schema-2 TOML。

RPMDrate 的输入是一段 Python 脚本，由 `rpmdrate/input.py` 的 `loadInputFile(path, T, Nbeads, processes)`
在注入了下列名称的命名空间中执行（以 `https://github.com/chen-yu-hao/RPMDrate` 为准，
实现时请固定到具体提交）：

| RPMDrate 名称 | 参数 |
| --- | --- |
| `reactants` | `atoms`、`reactant1Atoms`、`reactant2Atoms`、`Rinf` |
| `transitionState` | `geometry`、`formingBonds`、`breakingBonds` |
| `equivalentTransitionState` | `formingBonds`、`breakingBonds` |
| `thermostat` | `type`（`'Andersen'`、`'GLE'`）及其参数（如 `A=('gle_A.txt','s^-1')`） |
| `generateUmbrellaConfigurations` | `dt`、`evolutionTime`、`xi_list`、`kforce` |
| `Window` | `xi`、`kforce`、`trajectories`、`equilibrationTime`、`evolutionTime` |
| `conductUmbrellaSampling` | `dt`、`windows`、`saveTrajectories` |
| `computePotentialOfMeanForce` | `windows`、`xi_min`、`xi_max`、`bins` |
| `computeRecrossingFactor` | `dt`、`equilibrationTime`、`childTrajectories`、`childSamplingTime`、`childrenPerSampling`、`childEvolutionTime`、`xi_current`、保存选项 |
| `computeRateCoefficient` | 无 |
| `numpy`、`T`、`Nbeads` | 注入的模块与命令行参数 |

输入脚本通常还会导入 PES，例如 `from h3_bkmp2 import get_potential`。
温度和珠子数来自命令行（`rpmdrate.py input.py T Nbeads -p N`），不在输入文件中。

PyRPMD 的 schema-2 速率输入（`examples/rate/research.inp`、`docs/rate-input-v2.md`）
已有对应的概念：`[reaction]` 的 `reactant_fragments`、`forming_bonds`、`breaking_bonds`、
`rinf`、`equivalent_paths`；`[umbrella]` 的 `windows`/`regions` 与 `force_constant`；
`[recrossing.parent]` 与子轨迹设置。`docs/rate-input-v2.md` 第 61–91 行专门说明了
RPMDrate 的 `kforce = 0.1 * T` 约定与 PyRPMD 的 `force_constant` 之间的换算，这是最容易出错的地方。

## 目标

提供 `pyrpmd import-rpmdrate`，把一个 RPMDrate 速率输入转换为等价的 PyRPMD schema-2 输入，
同时生成逐项转换报告。无法等价转换的设置必须显式报告，不能静默替换。

## 范围

**要做：**

1. **命令。**
   `pyrpmd import-rpmdrate INPUT.py --temperature T --beads P [--pes-module PATH] --output DIR [--allow-unmapped]`
   输出 `DIR/rate.inp`、`DIR/ts.xyz`，以及 `DIR/import_report.json`（逐个 RPMDrate 调用、其参数、
   对应的 PyRPMD 字段、单位换算和无法转换的项）。输出目录必须为空或不存在。
2. **执行方式。** 与 RPMDrate 相同，把输入当作可信的 Python 代码执行（与 PyRPMD 的 Python 输入同一信任模型，
   在文档中写明），但注入的是**只记录参数的桩函数**，不导入 `rpmdrate` 包，也不调用 PES。
   对输入中的 PES 导入：记录模块名与符号，映射为 `[potential] provider="rpmdrate"`；
   如果需要在转换时避免执行真实的 PES 模块，使用只在转换期间生效的导入钩子，转换结束后恢复 `sys.modules`
   （使用上下文管理器并加测试，避免第四轮 T36 的残留问题）。
3. **映射规则。** 至少覆盖：
   - 原子索引从 1 起改为从 0 起（片段、成键、断键、等价过渡态）；
   - 几何与 `Rinf` 的单位（`"bohr"`、`"angstrom"` 等）；
   - `equivalentTransitionState` → `equivalent_paths`，并检查 `symmetry_factor` 的约束（`reactions.py:320`）；
   - 窗口列表 → 显式 `[umbrella] windows`，逐窗口保留 `kforce`、轨迹数、平衡与采样时间；
   - `kforce` 按 `docs/rate-input-v2.md` 的约定换算，并在报告中给出每个窗口最终的力常数（hartree 原子单位）；
   - `generateUmbrellaConfigurations` → 窗口构型生成阶段的设置；
   - `computePotentialOfMeanForce` → `[umbrella.analysis]`；
   - `computeRecrossingFactor` → 父链与子轨迹设置，`xi_current` → 分界面；
   - `thermostat('Andersen')` → Andersen 采样；`'GLE'` 如果 PyRPMD 的速率伞形采样阶段有等价设置则转换，
     否则列为无法转换项。
4. **无法转换的项。** 默认报错退出，并列出全部无法转换项；`--allow-unmapped` 时写出输入，
   在报告中逐项标明采用的 PyRPMD 默认值，并在 `rate.inp` 顶部写注释。
   dRPMD 等非速率输入（RPMDrate 分支的 `--drpmd` 模式）明确拒绝。
5. **文档与示例。** 英文指南 `docs/rpmdrate-import.md`（`orphan: true`）：映射表、单位约定、
   信任模型和限制；`examples/rpmdrate_import/` 放一个最小的 RPMDrate 风格输入（自行编写，不复制上游示例全文；
   如复制，追加 `THIRD_PARTY_NOTICES.md`）。

**不做：**

- 不实现 RPMDrate 输出文件或 checkpoint 的导入。
- 不修改速率工作流本身；发现 PyRPMD 缺少某个 RPMDrate 能力时，记录在验证记录中，另行立卡。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/rpmdrate_import.py`、`tests/test_rpmdrate_import.py`、
  `docs/rpmdrate-import.md`、`examples/rpmdrate_import/`。
- **共享注册点：** `cli.py`（追加 `import-rpmdrate` 子命令）、`docs/conf.py`（`include_patterns`）、
  T47 合并后的注册表（登记命令）。

## 交付物

- 导入命令、转换报告格式及其测试。
- 验证记录 `docs/validation/rpmdrate-import-<日期>/README.md`：四个上游示例的转换结果与逐项对照。

## 验收标准

- 在固定提交的 RPMDrate 仓库中，`examples/` 下的 H+H2、H+CH4、H+C2H6、OH+CH4 四个输入都能转换；
  每个转换结果都能通过 `pyrpmd validate`（PES 不可用时用 `--pes` 覆盖为测试势，只检查结构与单位）。
- H+H2：转换结果的反应身份字段（片段、成键与断键、`equivalent_paths`、过渡态几何、`rinf`）
  经 `JobSpec.resolve()` 后，与手写的 BKMP2 速率配置（`benchmarks/h3/rpmd.py` 中
  `configuration(stage="rate")` 与 `benchmarks/h3/convergence_inputs/base.inp`）逐字段一致；
  采样预算本来就不同，不参与比较。每个窗口的力常数（hartree）与独立按 RPMDrate Fortran 偏置公式
  计算的值一致，推导写进验证记录。
- 原子索引、单位、等价过渡态都有单元测试，包括错误输入（越界索引、未知单位、重复键）的报错。
- 转换过程中不调用 PES，不导入 `rpmdrate` 包；转换结束后 `sys.modules` 与转换前一致（测试断言）。
- 存在无法转换项且未给 `--allow-unmapped` 时，命令以非零状态退出，不写任何输出文件。
- 端到端测试走真实的 CLI 与输入解析（G19）。
- 满足通用规则 G5–G8。

## 验证命令

```bash
python -m pytest -q tests/test_rpmdrate_import.py
python -m pytest -q -m "not campaign"
pyrpmd import-rpmdrate examples/rpmdrate_import/input.py --temperature 1000 --beads 4 --output runs/t52-import
pyrpmd validate runs/t52-import/rate.inp
```
