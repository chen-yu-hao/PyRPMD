---
orphan: true
---

# T31 变分过渡态理论与 SCT 隧穿对照

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1 |
| 前置 | 无（只读调用 T32 独占的 `low_temperature.py` 公开函数） |
| 建议分支 | `agent/t31-vtst-sct` |

## 背景

RPMD 速率论文通常把 RPMD 与常规 TST、正则变分 TST（CVT）加小曲率隧穿（SCT）或
μOMT（Polyrate 的标准输出），以及 instanton 和精确量子结果并列比较。

PyRPMD 已经有：

- 数值 Hessian、两侧下降、CI-NEB 和 P-RFO（`src/pyrpmd/workflows/prepare.py`、`saddle.py`、`neb.py`）；
- 简谐 TST，以及 Wigner/Eckart 修正（`low_temperature.py`）；
- instanton。

但还没有质量加权最小能量路径（IRC/MEP）、沿路径的广义简正模分析、CVT，或沿 MEP 的
ZCT/SCT 隧穿。因此用户无法在同一个程序里、用同一个 PES 和同一套约定得到这些对照结果。

## 目标

实现基于 MEP 的变分过渡态理论与多维隧穿修正，输出与 RPMD 速率使用相同单位、
相同对称数和电子配分函数约定的 k(T)，使同一 PES 上的 RPMD、TST、CVT/SCT 和 instanton
可以直接并列比较。

## 范围

**要做：**

1. **MEP。** 在质量标度坐标下从鞍点出发，双向求 MEP（Page–McIver 或等价的稳定积分器，
   步长可配置），并输出沿弧长 s 的能量、梯度和 Hessian；
   终点用已有的通道与连通性判据检验，确认连接的是反应物和产物。
2. **广义简正模。** 沿路径对 Hessian 投影掉梯度方向、平动和转动后，得到实频与虚频；
   默认使用笛卡尔投影，并在文档中说明这一选择对 SCT 的影响。
3. **CVT。** 平动、转动（刚性转子，带对称数）、振动（量子简谐）和电子配分函数，
   给出 ΔG(T, s) 及其极大值；双分子反应使用相对平动配分函数，单分子反应使用相应约定。
4. **隧穿修正。** 实现 ZCT 和 SCT（有效约化质量，以 Liu、Lu、Gonzalez-Lafont、Truhlar、
   Garrett 1993 及 Polyrate 手册为准），并给出 CVT/ZCT 和 CVT/SCT 的透射系数；
   实现前先核实公式来源，并在指南中逐式引用。LCT 和 μOMT 为可选项，不作为本卡的验收内容。
5. **入口。** `pyrpmd vtst INPUT --output DIR` 复用速率输入中的 `[system]`、`[potential]`、
   `[reaction]` 和 `[preparation]`；输出 `results/vtst.json` 与 CSV。

**不做：**

- 不修改 `low_temperature.py`、`instanton.py`、`tunnelling.py`（归 T32），只调用其公开函数。
- 不修改 RPMD 速率工作流的结果格式。
- 不做多结构扭转非谐（MS-T）修正。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/vtst.py`、`src/pyrpmd/workflows/vtst.py`、相关测试；
  可以在新模块中复用 `workflows/prepare.py` 的公开函数，但不修改该文件。
- **共享注册点：** `cli.py`（`vtst` 子命令）。

## 交付物

- 模块、CLI 和英文指南 `docs/vtst.md`（`orphan: true`），写明全部约定：
  坐标标度、投影、对称数、零点能、能量参考。
- 验证记录 `docs/validation/vtst-<日期>/README.md`。

## 验收标准

- **一维 Eckart 势垒：** 在没有路径曲率时 SCT 退化为 ZCT；ZCT 透射系数与独立计算的
  一维半经典（基态 WKB）积分一致，并报告它与解析 Eckart 量子透射之间的已知差异。
- **对称体系检验：** 对 H + H₂ 共线对称路径，正向和反向的 CVT 自由能剖面关于 s=0 对称；
  反应物配分函数与 `low_temperature.py` 的简谐 TST 结果一致。
- **H + H₂/BKMP2：** 给出 200–1000 K 的 TST、CVT、CVT/ZCT、CVT/SCT 速率，并与经过核实的
  已发表 Polyrate 或文献表格对照；没有核实的参考值时，只报告结果，不声称一致。
- MEP 步长减半时，CVT 速率的变化小于报告的数值容差，并给出收敛表。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "vtst or prepare or saddle"
python -m pyrpmd vtst examples/rate/research.inp --output runs/t31-vtst
```
