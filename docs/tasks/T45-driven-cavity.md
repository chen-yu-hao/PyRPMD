---
orphan: true
---

# T45 外场驱动与腔 RPMD

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2 |
| 前置 | T26 合并（需要依赖时间的算符） |
| 建议分支 | `agent/t45-driven-cavity` |

## 背景

原路线图把“腔与外场驱动动力学”列为可选研究方向。i-PI 3.0 已经提供了腔分子动力学
（参见 Li、Subotnik、Nitzan，PNAS 117, 18324, 2020 的 CavMD 方法）以及外电场驱动动力学。

PyRPMD 已有偶极与极化率性质模型（`src/pyrpmd/properties.py`：`PropertyModel`、
`CallablePropertyModel`、`TorchScriptPropertyModel`），但 PES 和偏置接口
（`bias.energy_force(q)`）都不带时间参数，无法表达依赖时间的哈密顿量。

## 目标

利用 T26 的算符协议，加入依赖时间的外场耦合项 −μ(q)·E(t)；把腔光子模作为附加的动力学自由度
（经典，或者量子化为环聚合物），通过偶极耦合和偶极自能项与分子相互作用。用于非线性振动光谱、
太赫兹泵浦和振动强耦合研究。

## 范围

**要做：**

1. **外场。** 每个珠子上的势能项为 −μ(q_j)·E(t)，力需要偶极的核坐标导数（即 Born 有效电荷）；
   性质模型不提供导数时，拒绝运行，不使用未经声明的有限差分。
   E(t) 支持解析脉冲（高斯包络、余弦载波、啁啾）与用户函数。
   时钟与 checkpoint 一致，续算时 E(t) 的相位不变。
2. **腔模。** 光子坐标与动量作为辅助自由度，采用 CavMD 形式的哈密顿量（含偶极自能项），
   支持多个模式与偏振方向；光子模可以选择经典（P=1）或与核同样的 P 个珠子。
   实现前核实所采用的哈密顿量与单位约定。
3. **可观测量。** 吸收（由输入功率或偶极响应得到）、光子能量与布居、由偶极轨迹得到的
   时间分辨光谱；复用 `analysis/spectroscopy.py` 中的已有工具。
4. **组合白名单。** 允许与 TRPMD、NVE 组合；与恒温器组合时，在文档中说明其物理含义。

**不做：**

- 不处理电子激发态的光–物质耦合（属于非绝热方向）。
- 不实现与晶胞耦合的宏观电场（Berry 相极化），并在文档中写明这一限制。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/driven.py`、新建 `src/pyrpmd/cavity.py`、相关测试、
  新建 `docs/driven-cavity.md`（`orphan: true`）、`examples/cavity/`。
- **只读：** `properties.py`、`analysis/spectroscopy.py`。需要修改时写进“需要协调”。
- **共享注册点：** `engine.py`/`auxiliary.py` 的白名单表（追加式）、`input_tasks.py`、`runner.py`
  （作为 `pimd` 或 `spectroscopy` 的可选 `[field]`、`[cavity]` 表）。

## 交付物

- 外场与腔模实现、英文指南和示例。
- 验证记录 `docs/validation/driven-cavity-<日期>/README.md`。

## 验收标准

- **受驱谐振子：** 带线性偶极的谐振子在共振与非共振驱动下的响应，与解析解一致（P=1 与 P>1）。
- **线性响应一致性：** 弱场下的吸收谱与平衡态偶极自相关（已有 TRPMD 或 RPMD IR 工具）
  给出的谱线在统计误差内一致。
- **腔模：** 单个谐振分子模与单个腔模耦合时，Rabi 劈裂与解析的耦合谐振子频率一致；
  偶极自能项缺失时测试会失败，保证这一项被正确计入。
- 中断后续算与不中断逐位一致，包括 E(t) 相位与光子模状态。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "driven or cavity or spectroscopy"
```
