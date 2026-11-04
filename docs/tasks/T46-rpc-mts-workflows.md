---
orphan: true
---

# T46 RPC/MTS 接入材料与从头算工作流

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | T16b、T28 合并 |
| 建议分支 | `agent/t46-rpc-mts-workflows` |

## 背景

Python API 已经支持环聚合物收缩（RPC）与多时间步长（MTS），见 `docs/rpc-mts.md`、
`src/pyrpmd/multitime.py`、`potentials/contraction.py`，以及
`DeltaLearningPotential.split(P, Pc)`。

但周期材料审计（`docs/periodic-materials-audit.md` 的 P2 项）指出：完整的工作流接入、
压力与维里、偏置与约束的组合，以及专门的 checkpoint 都还没有完成。

对于从头算 PIMD，这是 i-PI 用户最常用的降本手段：廉价参考势（DFTB、xTB 或 ML）加上少数
收缩珠子上的从头算修正。

## 目标

让 `pimd`、`diffusion` 和速率工作流可以通过 TOML 声明 V = V_fast + V_slow 的分裂、收缩珠子数和
外层/内层时间步长。提供正确的维里与压力估计量、完整的 checkpoint，以及调用次数与成本报告。
与 T28 的 socket 提供者组合，形成可以使用的从头算 RPC 路线。

## 范围

**要做：**

1. **输入。** 新增 `[force_splitting]` 表：快势、慢势（各自是完整的 `[potential]` 定义，
   可以是 socket 提供者）、`contracted_beads`、`inner_steps`；严格校验，例如 P 能否被收缩珠子数整除等约束。
2. **维里与压力。** 收缩后的慢势维里按 RPC 约定重新展开；NPT 下的压力估计量与完整珠子计算对照。
3. **组合。** 与偏置（质心偏置作用在哪一层要写明）、约束（需要 T44）、固定原子的组合规则，
   写进白名单。
4. **checkpoint。** 内外层时钟、两个提供者的身份与状态，在中断后续算时逐位一致
   （外部程序的 SCF 差异按 T28 的约定处理）。
5. **成本报告。** 慢势和快势的调用次数、墙钟时间、每皮秒成本；与不分裂的计算并列比较，
   计算达到相同误差所需的成本。

**不做：**

- 不实现区域收缩，即为不同原子设置不同珠子数。原审计明确指出这会改变哈密顿量，
  需要单独推导和验证。

## 文件所有权

- **独占：** `src/pyrpmd/multitime.py`、`potentials/contraction.py`、相关测试、`docs/rpc-mts.md`、
  `examples/rpc_mts/`。
- **共享注册点：** `input.py`、`input_tasks.py`、`input_schema.py`（`[force_splitting]`）、
  `workflows/materials.py` 与 `workflows/rate*.py`（只追加分裂分支；冲突时以 T16b 为准）、`runner.py`。

## 交付物

- 工作流接入、英文指南更新和示例：一个解析模型示例，以及一个经 T28 socket 的 DFTB+/xTB + 参考势示例。
- 验证记录 `docs/validation/rpc-mts-workflows-<日期>/README.md`。

## 验收标准

- **谐振与非谐解析模型：** RPC 的估计量随收缩珠子数收敛到完整珠子的结果；
  MTS 的能量守恒随内层步数的变化符合预期。
- **NPT：** RPC 下的平均压力与完整珠子计算在统计误差内一致；维里与有限应变差分对照。
- **从头算示例：** 报告慢势调用次数减少的倍数，以及观测量（动能、RDF 或扩散）与完整计算的差异；
  不声称物理收敛。
- 中断后续算与不中断一致。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "contraction or multitime or rpc or mts"
```
