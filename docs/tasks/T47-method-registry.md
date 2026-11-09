---
orphan: true
---

# T47 方法与工作流注册表：能力矩阵与成熟度标签

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1（第五轮） |
| 前置 | 无 |
| 后续依赖本卡 | G18 生效；T16b（续）的引擎能力行；T26 的组合白名单迁入 |
| 建议分支 | `agent/t47-method-registry` |

## 背景

PyRPMD 有 14 个以上的 task、几十个 method 选择器、十余种热浴和三套 PIMD 引擎，
但“哪些名称存在、哪些组合允许、成熟到什么程度”没有统一的来源：

- `pyrpmd info` 的能力列表是 `src/pyrpmd/cli.py` 第 523–570 行中手写的常量。
  它已经过时：没有 `path_sampling` task，没有 VES、ABF、eABF、steered，也没有 `[tempering]`；
  `temperature_hamiltonian_replica_exchange` 仍被列为“仅 Python API”，而 T37 已经提供了工作流。
- 组合校验分散在各个输入模块中，写成 `if task not in (...)`，例如 `input.py:220`、
  `input_fes.py:59`、`input_stages.py:220`、`input_system.py:42`、`input_tasks.py:108–115`。
- PIMD 引擎能力有单独的 `EngineRoute`（`workflows/pimd.py:50`）；T16b 原卡还要求另做一份引擎能力矩阵。
- 方法成熟度只在个别地方出现（`workflows/nonadiabatic_multistate.py:79` 的 `method_status="research"`），
  用户从输出中无法区分经过独立验证的方法和研究性实现。

i-PI、LAMMPS 等程序的文档与输入校验来自同一份声明。PyRPMD 方法数量多，更需要这一点。

## 目标

建立一份机器可读的注册表，作为方法名称、能力、组合限制和成熟度的唯一来源。
`pyrpmd info`、文档中的能力表和成熟度标签都从注册表生成；测试保证注册表与实际的校验和分派一致。
本卡**不改变任何输入的接受或拒绝行为**，只负责描述与交叉校验。

## 范围

**要做：**

1. **注册表模块。** 新建 `src/pyrpmd/registry.py`，用不可变数据类声明：
   - task：名称、入口、支持的命令（`run`/`resume`/`analyze`/`extend`）、是否支持 MPI 和进程池；
   - method：所属 task 与选择键（例如 `[fes] method`、`[nonadiabatic] method`、`[path_integral] method`、
     spectroscopy method、`[path_sampling] method`）、原始文献键（对应 `docs/method-references.json`）、
     成熟度、已知限制的简短说明；
   - 热浴、压浴、积分器、引擎（legacy/native/torch）及其能力（固定原子、键约束、NPT、交换、
     流式轨迹、设备）；
   - 允许的组合：哪些 task/method 可以与哪些热浴、压浴、引擎、偏置、约束同时使用。
2. **成熟度定义。** 三个等级：
   - `stable`：有公开工作流、精确续算测试，并且 `docs/validation/` 中有独立参考对照；
   - `experimental`：有公开入口和测试，但缺少独立参考对照或已知有范围限制；
   - `research`：研究性近似或只在模型体系上检验过。
   agent 按上述规则给出初始分级表，逐项写明依据；最终分级在 PR 的“需要维护者决定”一节中由维护者确认。
3. **`pyrpmd info` 由注册表生成。** 保留现有 JSON 的全部键和含义（向后兼容），值改为由注册表计算；
   新增 `registry` 键，包含完整的 task/method/成熟度列表。
4. **成熟度写入运行目录。** 在 `runner.py` 写 `manifest.json` 的统一出口（第 101、123 行附近）
   追加 `method_maturity` 字段（task、method、等级、注册表版本）。这是附加的记录字段，
   不进入 run identity，不影响续算判定。
5. **一致性测试。** 新建 `tests/test_registry.py`：
   - 输入校验接受的每个 task 和 method 都在注册表中，注册表中的每一项也都能被输入校验接受；
   - 对注册表声明为“禁止”的每个组合，构造最小输入并断言校验阶段报错且不创建输出目录；
     对声明为“允许”的组合，断言校验通过（不运行动力学）；
   - `pyrpmd info` 输出与注册表一致；
   - 文档能力表是最新的（见交付物）。
6. **吸收 T16b 的引擎能力矩阵要求。** 引擎能力行由本卡建立，并与 `EngineRoute` 交叉校验。

**不做：**

- 不把分散的校验逻辑改写为由注册表驱动（另行立卡，等本卡稳定后再做）；
  本卡只描述并用测试锁定现状。发现描述与实际行为不一致时，在验证记录中列出，不修改行为。
- 不修改任何 run identity、checkpoint 格式或输入字段。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/registry.py`、`tests/test_registry.py`、
  `docs/method-registry.md`（`orphan: true`）、`cli.py` 中 `info` 子命令的分支。
- **共享注册点：** `runner.py`（只追加 `method_maturity` 写入）、`docs/conf.py`（`include_patterns`）。
- **只读：** 各 `input*.py`、`workflows/pimd.py`（只读取，不修改）。
- **协调：** T26 的组合白名单先以独立数据结构合并；本卡合并后，由后合并的一方负责把它迁入注册表。

## 交付物

- 注册表模块、初始分级表及每项依据。
- `python -m pyrpmd.registry --markdown` 生成的能力与成熟度表，写入 `docs/method-registry.md`。
- 验证记录 `docs/validation/method-registry-<日期>/README.md`，列出注册表描述与实际校验不一致之处。

## 验收标准

- `pyrpmd info` 列出 `path_sampling`、VES、ABF、eABF、steered、tempering 等第四轮已合并的入口；
  原有 JSON 键全部保留。
- `tests/test_registry.py` 覆盖所有 task、所有 method 选择器、所有热浴模式和三套引擎；
  在任意输入模块中新增一个未登记的 method 名称会使测试失败（在 PR 中演示一次）。
- `docs/method-registry.md` 与 `python -m pyrpmd.registry --markdown` 的输出逐字一致，由测试保证。
- 每个新运行的 `manifest.json` 含有 `method_maturity`；已有运行照常续算，run identity 不变。
- 满足通用规则 G5–G8、G19。

## 验证命令

```bash
python -m pytest -q tests/test_registry.py
python -m pytest -q -m "not campaign"
pyrpmd info | python -c "import json,sys; d=json.load(sys.stdin); print(sorted(d['registry']['tasks']))"
python -m pyrpmd.registry --markdown | diff - docs/method-registry.md
```
