---
orphan: true
---

# T42b 统一核心 II-b：跃迁与摩擦类方法移植

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 2 |
| 前置 | T26 合并 |
| 可并行 | T42a（两张卡的文件不重叠） |
| 建议分支 | `agent/t42b-hopping-methods` |

## 背景

以下方法带有离散活动面、随机跃迁或摩擦记忆，目前都在各自的循环中运行，
不能使用传播栈的热浴、周期边界或 RPC：

- `RPSHDynamics`（`surface_hopping.py`）；
- `AdvancedRPSHDynamics` 及其子类 `ComplexRPSHDynamics`、`RPSHBADynamics`、`RPSHDMDynamics`
  （`advanced_surface_hopping.py`）；
- `AdiabaticRPSHDynamics`（`adiabatic.py`）；
- `IESHDynamics`（`iesh.py`）；
- `KCRPMDDynamics`（`kc_rpmd.py`）；
- `MVRPMDDynamics`（`mv_rpmd.py`）；
- `ElectronicFrictionDynamics`（`electronic_friction.py`，继承自 `Dynamics`，但摩擦是单独实现的）。

## 目标

把这些方法移植到 T26 的辅助自由度协议上。离散活动面和跃迁事件作为带事件处理的算符；
摩擦作为与核动量耦合的随机算符。使它们获得统一的组合、checkpoint 和输出。

## 范围

**要做：**

1. **事件算符。** 扩展协议（与 T26 的设计一致，追加式），支持：
   - 步内的跃迁事件及其能量重标度；
   - 失败跃迁的处理；
   - 退相干修正（EDC）；
   - 事件对 RNG 的消耗顺序固定，保证续算的确定性。
2. **逐个移植。** 每个方法保留原类名作为适配层；NVE（或固定 RNG 下的随机过程）与旧实现
   逐位一致，或误差 ≤ 1e-12。
3. **EF-RPMD。** 把电子摩擦实现为随机算符，可以与 PILE 等热浴按白名单组合；
   周期晶胞中的 EF-RPMD 与现有的扩散工作流结果一致。
4. **工作流。** 对应的 `method` 改为调用统一核心；旧 checkpoint 可以续算。

**不做：**

- 不实现环聚合物 IESH（仍属研究方向，见原 T25）。
- 不实现相空间面跳跃或非阿贝尔简并态跃迁。

## 文件所有权

- **独占：** `src/pyrpmd/surface_hopping.py`、`advanced_surface_hopping.py`、
  `adiabatic.py` 中的 `AdiabaticRPSHDynamics` 部分、`iesh.py`、`kc_rpmd.py`、`mv_rpmd.py`、
  `electronic_friction.py`、`workflows/nonadiabatic_multistate.py` 中的跃迁类分支、相关测试。
- **共享注册点：** `engine.py`/`auxiliary.py` 的白名单表与事件接口（追加式）、`nonadiabatic_input.py`。

## 交付物

- 事件算符扩展、移植后的方法、更新后的指南（`docs/nonadiabatic-hopping-extensions.md`、
  `docs/electronic-friction.md`、`docs/auxiliary-operators.md`）。
- 验证记录 `docs/validation/hopping-port-<日期>/README.md`。

## 验收标准

- 固定 RNG 下，每个移植方法与旧实现的轨迹、跃迁事件序列和布居逐位一致；
  已有 Tully I–III、spin-boson 和 Newns–Anderson 示例的结果不变。
- 中断后续算与不中断逐位一致，包括跃迁事件的 RNG 消耗。
- EF-RPMD 与 PILE 组合时，平衡温度与涨落–耗散关系的检验通过。
- 白名单之外的组合会被拒绝。
- 满足通用规则 G5–G9。

## 验证命令

```bash
python -m pytest -q tests -k "rpsh or hopping or iesh or kc_rpmd or mv_rpmd or friction"
python -m pytest -q -m "not campaign"
```
