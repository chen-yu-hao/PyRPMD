---
orphan: true
---

# T44 约束积分器扩展与板（slab）NPT

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | T16b、T26 合并 |
| 建议分支 | `agent/t44-constraints-slab-npt` |

## 背景

- **键约束：** 只支持“逐珠子 SHAKE/RATTLE、velocity-Verlet、固定晶胞”这一种组合
  （`pyrpmd info` 中的 `bond_constraints`，见 `docs/bond-constraints.md`），
  不能与 Cayley-BCB、normal-mode-BAB 或 NPT 组合。
- **板 NPT：** 周期材料审计（`docs/periodic-materials-audit.md`）指出，带固定层的板模型 NPT
  不受支持；原生引擎的 MTTK 虽然有 `cell_mask`，但不能用于固定层的板模型。

这两项限制使刚性水和表面、界面体系难以做恒压的量子模拟，而 i-PI 和 LAMMPS 都支持其中部分组合。

## 目标

把逐珠子约束扩展到 Cayley-BCB 和 normal-mode-BAB 积分器，以及 NPT；
实现带固定层的板模型的面内 NPT，并给出正确的约束维里与压力估计量。

## 范围

**要做：**

1. **约束与自由环传播。** 在 B–A–B 分裂中，于 A（自由环）步之后做逐珠子的 RATTLE 投影；
   分析并记录这种分裂的时间可逆性和误差阶，或者采用经过验证的已发表方案（实现前核实出处）。
2. **约束与 NPT。** 在压力估计量中计入约束力的维里；BZP 和 MTTK 与约束组合时的晶胞更新顺序
   和投影顺序要写清楚，并经过测试。
3. **板 NPT。** 只对周期面内的晶胞分量施加压力（或面内应力）；固定层原子随面内晶胞仿射移动，
   或保持分数坐标不变（两种约定都要写明，默认选一种并给出理由）；法向晶胞和真空层不变；
   面积与面内应力的估计量正确。
4. **组合白名单。** 更新 T26 的白名单表，以及 `pyrpmd info` 中的能力报告。

**不做：**

- 不实现刚体（四元数）积分器。
- 不处理带电板的偶极修正（属于 PES 设置）。

## 文件所有权

- **独占：** `src/pyrpmd/bond_constraints.py`、`barostat.py`、`exchange_barostat.py`
  （只限与约束和板相关的部分）、相关测试、`docs/bond-constraints.md`、`docs/materials-npt.md`。
- **共享注册点：** `engine.py`（组合白名单与投影顺序，追加式）、`input_tasks.py`、`input.py`
  （放开组合校验）、`workflows/materials.py`（只追加分支；如与 T16b 的后续改动冲突，以 T16b 为准）。

## 交付物

- 实现与更新后的英文指南。
- 验证记录 `docs/validation/constraints-slab-npt-<日期>/README.md`。

## 验收标准

- **刚性双原子或刚性水分子：** 在 Cayley-BCB 和 normal-mode-BAB 下，约束保持在设定的容差内；
  NVE 能量漂移随时间步长的变化符合所报告的误差阶。
- **约束 NPT：** 刚性分子流体的平均压力与有限应变能量差分得到的压力一致；
  使用 `examples/materials/rigid_water.inp` 一类的设置，体积分布稳定。
- **板 NPT：** 二维周期的谐振晶格在面内外压下的平衡面积与解析值一致；
  固定层在面内晶胞变化时按所选约定移动，法向晶胞不变。
- 白名单之外的组合会被拒绝；已有约束和 NPT 的测试结果不变。
- 满足通用规则 G5–G9。

## 验证命令

```bash
python -m pytest -q tests -k "constraint or barostat or npt or slab"
```
