---
orphan: true
---

# T26 统一传播核心 I：辅助自由度算符协议与 NRPMD 试点

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1（最高优先级） |
| 前置 | 无 |
| 后续依赖本卡 | T42a、T42b、T44、T45 |
| 建议分支 | `agent/t26-aux-operators` |

## 背景

`Dynamics` 和 `src/pyrpmd/engine.py` 中的 `PropagationPlan`/`_Composition` 负责算符顺序
（B/2–A–B/2、热浴、压浴、力分裂、约束投影），并有原生执行器。但十个非绝热动力学类各自实现
自己的 `step` 循环，都不接入这套传播栈：

`NRPMDDynamics`（`nonadiabatic.py`）、`SMNRPMDDynamics`（`spin_mapping.py`）、
`RPMASHDynamics`（`rp_mash.py`）、`RPSHDynamics`（`surface_hopping.py`）、
`AdvancedRPSHDynamics` 及其子类（`advanced_surface_hopping.py`）、`MASHDynamics`/`MultistateMASHDynamics`
（`mash.py`）、`IESHDynamics`（`iesh.py`）、`KCRPMDDynamics`、`MVRPMDDynamics`、
`IsomorphicEhrenfestDynamics`。

结果是：这些方法都不能使用恒温器、压浴、周期晶胞、MTS/RPC、约束或偏置，
每种组合都要单独实现。NQCDynamics.jl 和 i-PI 的做法是：所有方法共享同一套系综、输出
和算符框架。

## 目标

在传播栈中引入“辅助自由度”协议。电子映射变量、自旋矢量、活动面、摩擦记忆等作为附加状态，
携带自己的算符（电子传播、与核坐标耦合的力贡献、能量、checkpoint 与身份）；
传播栈负责把它们与已有的核算符按指定顺序组合。以 NRPMD 作为试点，完整迁移到新协议上。

## 范围

**要做：**

1. **协议设计。** 在 `engine.py` 中定义辅助自由度接口：
   - 状态数组；
   - `propagate(dt, q)`，即固定核坐标下的电子演化；
   - `nuclear_force(q)`，即依赖电子态的核力；
   - `energy`；
   - `state_dict`/`restore`；
   - `identity`；
   - 声明自身允许插入的分裂位置（例如 MInt 的对称分裂）。
   协议必须支持“一个环一份”和“每个珠子一份”两种布局，并支持任意电子态数。
2. **组合白名单。** 每种辅助自由度声明与恒温器、压浴、周期边界、力分裂、约束、偏置的
   允许组合，并给出物理理由。未列入白名单的组合在输入校验阶段拒绝。
   不能因为代码能跑通，就默认这种组合在物理上成立。
3. **NRPMD 试点。** 把 `NRPMDDynamics` 改写为该协议的一个实现，保留原类名和公开 API 作为适配层。
   至少开放以下组合，并给出测试：
   - 周期晶胞中的 NRPMD；
   - NRPMD 与 `SplitPotential`/RPC；
   - 核坐标准备阶段的 Andersen/PILE 采样（电子变量冻结或按已发表的方法处理，需注明来源）。
4. **工作流接入。** `task="nonadiabatic"` 下 `method="nrpmd"` 的路径改为调用统一核心，
   checkpoint 格式版本化，旧格式仍可读取并续算。

**不做：**

- 不移植其他非绝热方法（见 T42a、T42b）。
- 不为非绝热方法实现 NPT（等协议稳定后另行立卡）。
- 不修改原生内核；Python 实现优先，原生执行器保持对纯核动力学的现有加速。

## 文件所有权

- **独占：** `src/pyrpmd/engine.py`（结构性改动）、`dynamics.py`（仅限组合钩子）、
  `nonadiabatic.py`、`batched_nonadiabatic.py`、新建 `src/pyrpmd/auxiliary.py`、
  `workflows/nonadiabatic.py` 中的 NRPMD 分支、相关测试。
- **共享注册点：** `nonadiabatic_input.py`、`input_tasks.py`（组合白名单校验）。
- **注意：** T33 会在 `normalize_thermostat` 的参数表中追加 `qtb`/`adqtb` 条目；
  合并时保留这些条目，不要重排。

## 交付物

- 协议实现与英文设计文档 `docs/auxiliary-operators.md`（`orphan: true`）。文档写清算符顺序、
  时间可逆性和辛性分析、checkpoint 契约、组合白名单表。
- 迁移后的 NRPMD 及其适配层。
- 验证记录 `docs/validation/aux-operators-<日期>/README.md`。

## 验收标准

- NVE 下，新核心上的 NRPMD 与旧 `NRPMDDynamics` 在相同初值下 10⁴ 步的
  核坐标、映射变量和总能量一致：相对误差 ≤ 1e-12，或者给出逐位一致的证据。
- 已有的 NRPMD 测试、`examples/nonadiabatic/` 示例和已发表的 2013 NRPMD 两态模型验证
  （`docs/validation/published-nrpmd-2013-2026-11-01/`）结果不变。
- 新开放的组合各有至少一个数值测试：周期晶胞下的力与有限差分对照、RPC 下的能量守恒、
  准备阶段采样的分布检验。
- 未列入白名单的组合（例如 NRPMD + NPT）在输入校验阶段报错，并且不创建输出目录。
- 旧格式的 NRPMD checkpoint 可以读取并续算；新格式带版本号。
- 满足通用规则 G5–G8。

## 验证命令

```bash
python -m pytest -q -m "not campaign"
python -m pytest -q tests -k "nrpmd or nonadiabatic or propagation_stack or engine"
python -m pyrpmd run examples/nonadiabatic/exchange.inp --output runs/t26-exchange
```
