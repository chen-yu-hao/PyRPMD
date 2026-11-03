---
orphan: true
---

# T29 dRPMD/QCT 产物态分辨与微分截面

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t29-product-states` |

## 背景

dRPMD 与态选择 NE-RPMD/QCT 工作流已经保存每条碰撞轨迹的 `final_q`、`final_p` 和质心轨迹
（`src/pyrpmd/workflows/drpmd_trajectory.py`），`workflows/drpmd_classification.py` 按质心连通性
把结局分为 product、nonreactive、unresolved、failed，`analysis/scattering.py` 给出固定能量下的
等权截面估计。

但目前没有以下分析：

- 产物振转态分配；
- 散射角与微分截面（DCS）；
- 产物平动能分布、opacity function、激发函数 σ(E)；
- 态–态速率常数。

这些都是反应散射研究（VENUS 类 QCT 程序、NQCDynamics 的 QuantisedDiatomic 分析）的标准输出。

## 目标

从已保存的碰撞记录离线计算产物态分辨和角分辨的可观测量，并给出带不确定度的结果，
使 PyRPMD 的 dRPMD/QCT 结果可以直接和交叉分子束实验、量子散射计算对比。

## 范围

**要做：**

1. **产物几何与运动学。** 根据已有分类得到产物碎片；用质心（RPMD）或单珠（P=1 QCT）
   计算碎片的质心位置与速度、相对速度、内能，以及散射角
   θ = ∠(初始相对速度, 末态产物相对速度)。环聚合物情形默认使用质心约定，
   并在输出中记录所用的约定。
2. **双原子产物的振转态分配。** 转动量子数由经典角动量 j(j+1)ħ² = L² 得到；
   振动量子数由含离心项的有效径向势上的半经典 EBK 作用量积分得到；
   提供直方图分箱（HB）和高斯分箱（GB，Bonnet–Rayez 1997）两种方式。
   多原子产物只报告内能分布，不做量子数分配。
3. **可观测量。**
   - 微分截面 dσ/dΩ（θ 分箱，按冲击参数抽样方式正确加权）；
   - opacity function P(b)；
   - 态分辨截面 σ(v′, j′)；
   - 产物平动能分布；
   - 多个碰撞能下的激发函数；
   - 在给定热分布时的态–态速率常数。
4. **不确定度。** 计数类结果给出 Wilson 区间，分箱后的连续量给出自助法误差；
   unresolved 和 failed 轨迹单独计数，不悄悄并入分母。
5. **入口。** `pyrpmd analyze RUN --product-states` 写出 `results/product_states.json` 和 CSV；
   同时提供 Python API。

**不做：**

- 不修改碰撞传播与分类算法，不重新运行已保存的轨迹。
- 不做量子散射计算，不对零点能泄漏做加权修正（只报告诊断量，例如 v′ 落入零点以下的比例）。

## 文件所有权

- **独占：** `src/pyrpmd/analysis/scattering.py`、新建 `analysis/product_states.py`、
  `workflows/drpmd_analysis.py`、相关测试。
- **共享注册点：** `cli.py`（`analyze` 的新选项）。

## 交付物

- 分析模块、CLI 选项和英文指南 `docs/product-states.md`（`orphan: true`）。
- 验证记录 `docs/validation/product-states-<日期>/README.md`。

## 验收标准

- **硬球散射解析检验：** 经典硬球（半径 a）在均匀冲击参数抽样下的 DCS 各向同性，
  且 dσ/dΩ = a²/4；数值结果与解析值在统计误差内一致。
- **Morse 双原子：** EBK 振动能级与 Morse 解析本征值对照，
  j=0 与 j>0 都需覆盖，误差在给定积分精度之内。
- **刚性转子：** 由合成轨迹得到的 j′ 分配与构造值完全一致。
- **H + H₂/BKMP2：** 在已有的 `benchmarks/h3` 设置上运行一个小型 QCT（P=1）试点，
  报告产物 v′、j′ 分布和 DCS；如果找到并核实了文献表格，再做比较；否则只作为流程示例，
  不声称复现文献。
- 已保存的运行重新分析时不改动任何源结果文件；输出带输入记录的哈希。
- 满足通用规则 G5–G9。

## 验证命令

```bash
python -m pytest -q tests -k "scattering or product_state or drpmd"
python -m pyrpmd run examples/drpmd/free_scattering.inp --output runs/t29
python -m pyrpmd analyze runs/t29 --product-states
```
