---
orphan: true
---

# T33 量子热浴 QTB/adQTB

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无（与 T26 共用 `engine.py` 的热浴注册点） |
| 建议分支 | `agent/t33-qtb` |

## 背景

PyRPMD 的恒温器清单（`src/pyrpmd/engine.py` 中 `normalize_thermostat` 的参数表：
none、andersen、langevin、pile-l、pile-g、svr、gle、nm-gle、nm-gle-g、piglet、trpmd、ffl、nhc）
已与 i-PI 相当，但没有量子热浴（QTB）。

QTB 用有色噪声，让单珠子经典动力学近似再现量子涨落–耗散关系，代价接近经典 MD；
自适应 QTB（adQTB）通过监测涨落–耗散比来自适应地抑制零点能泄漏。
Tinker-HP 已在大体系中使用 adQTB。

在 PyRPMD 中，QTB 可以作为 PIMD 的廉价对照，也可以用于预热大体系的平衡构型。

## 目标

实现 QTB 与 adQTB 恒温算符，提供完整的 checkpoint 与续算支持。在同一程序中，
用谐振与非谐体系定量比较 QTB、adQTB 与 PIMD 的差异。

## 范围

**要做：**

1. **QTB。** 按 Dammak 等（PRL 103, 190601, 2009）实现：噪声功率谱为
   θ(ω,T) = ħω[1/2 + 1/(e^{βħω}−1)]，可选去掉零点项作对照。
   有色噪声用分段滤波生成，例如 FFT 卷积核或等价的时域滤波，分段长度与频率截断可以配置。
2. **adQTB。** 按 Mangaud 等（JCTC 15, 2863, 2019）实现：逐频率地自适应调整摩擦系数 γ(ω)，
   使涨落–耗散比趋于 1；自适应速率、频率分辨率和平均窗口都可以配置。
3. **限制。** 只允许 P=1（经典核）。P>1 的 PIQTB（Brieuc 等，JCTC 12, 1351, 2016）为可选扩展，
   不作为本卡的验收内容。不允许与 PIMD 恒温器同时使用。
4. **持久化。** 噪声缓冲区、滤波器状态、自适应 γ(ω)、RNG 和计数器全部写入 checkpoint；
   中断后续算与不中断逐位一致。
5. **接入。** 在 `normalize_thermostat` 中追加 `qtb`、`adqtb` 的参数表条目，
   在热浴工厂中追加对应分支；`pimd` 与 `diffusion` 工作流的 `[sampling]` 或传播栈输入可以选择它们。

**不做：**

- 不修改 `engine.py` 的算符结构（归 T26），只追加注册条目与工厂分支。
- 不声称 QTB 精确；非谐体系上与 PIMD 的偏差必须如实报告。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/qtb.py`、相关测试、新建 `docs/qtb.md`（`orphan: true`）。
- **共享注册点：** `engine.py`（`normalize_thermostat` 的参数表条目和热浴工厂分支，追加式）、
  `workflows/sampling.py`（只追加允许的恒温器名称）。

## 交付物

- QTB 与 adQTB 实现、英文指南和示例输入。
- 验证记录 `docs/validation/qtb-<日期>/README.md`。

## 验收标准

- **谐振子：** 在多个 βħω（0.5–10）下，QTB 的 ⟨x²⟩ 与 ⟨p²⟩ 与精确量子值一致，误差在统计误差
  与离散化误差之内，并给出频率截断和分段长度的收敛表。
- **非谐体系：** 在 Morse 振子（或 T27 交付后的 LJ 团簇）上，比较 QTB、adQTB 与收敛 PIMD 的
  动能和位置分布；adQTB 的涨落–耗散比随时间收敛到 1；如实报告零点能泄漏的量级。
- 中断续算与不中断的轨迹逐位一致。
- 选择 P>1 或与其他恒温器组合时，在输入校验阶段报错，并且不创建输出目录。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "qtb or thermostat or propagation_stack"
python -m pytest -q -m "not campaign"
```
