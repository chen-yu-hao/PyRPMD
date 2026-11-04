---
orphan: true
---

# T34 凝聚相结构与输运分析

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t34-structure-transport` |

## 背景

材料工作流已经能够流式保存质心轨迹，`src/pyrpmd/analysis/diffusion.py` 提供自扩散张量、MSD 和
VACF，并能拒绝非扩散的拟合窗口；`PESResult.virial` 提供每帧物理维里。

但目前缺少凝聚相模拟软件的常用分析：

- 径向分布函数（RDF，含分组分）、结构因子 S(q)；
- 振动态密度（VDOS）；
- 由应力自相关得到的 Green–Kubo 粘度；
- 集体离子电导率。

周期材料审计（`docs/periodic-materials-audit.md`）还列出了两项 P2 待办：在任意正交基下的
扩散张量协方差，以及内存有界的相关函数计算。

## 目标

为 `pimd`/`diffusion` 运行提供结构与输运分析，给出带不确定度的结果；
使用内存有界的流式相关算法，适用于长轨迹。

## 范围

**要做：**

1. **结构。**
   - 分组分 RDF g_ab(r)：支持通用斜晶胞最小像（复用 `geometry.py`），可选质心或全部珠子
     统计（珠子平均对应量子结构）；
   - 由 RDF 或直接求和得到 S(q)；
   - 配位数积分。
2. **振动谱。** 由 RPMD 质心 VACF 或 TRPMD 速度得到 VDOS，窗函数与零填充可以配置。
   文档写明 RPMD 与 TRPMD 谱的已知伪影，不把它们当作精确量子谱。
3. **输运。**
   - 由非对角应力（维里加动能项）的自相关得到 Green–Kubo 剪切粘度；
   - 由用户给定的形式电荷与质心位移得到 Einstein 形式的集体离子电导率；
   - 报告 Nernst–Einstein 估计，并与集体结果对照。
   热导率需要 PES 提供热流，本卡只写文档说明，不实现。
4. **相关函数算法。** 使用 multiple-tau 或分块 FFT 的流式相关器，内存与轨迹长度无关；
   在任意正交基下计算张量分量；用副本间的分块自助法估计不确定度。
5. **入口。** `pyrpmd analyze RUN --structure`、`pyrpmd analyze RUN --transport`，同时提供 Python API。

**不做：**

- 不修改轨迹的写入格式和材料工作流（归 T16b），只读取已保存的数据流。
- 不把自扩散系数当作集体电导率。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/analysis/structure.py`、`analysis/transport.py`、
  `analysis/correlators.py`、相关测试、新建 `docs/structure-transport.md`（`orphan: true`）。
- **只读：** `analysis/diffusion.py`、`trajectory.py`、`trajectory_export.py`。
  需要修改时写进“需要协调”。
- **共享注册点：** `cli.py`（`analyze` 的新选项）。

## 交付物

- 分析模块、CLI 选项、英文指南和示例。
- 验证记录 `docs/validation/structure-transport-<日期>/README.md`。

## 验收标准

- **理想气体：** 无相互作用粒子的 g(r) → 1，在统计误差之内。
- **完美 FCC 晶格：** RDF 峰位与配位数与解析晶格壳层一致。
- **谐振晶体或谐振子集合：** VDOS 峰位与解析频率一致。
- 流式相关器与直接 O(N²) 相关在测试数据上的结果一致（相对误差 ≤ 1e-10），内存不随长度增长。
- **经典 LJ 流体粘度：** 在 P=1 的已知态点上与经过核实的文献值对照。
  可以使用 T27 的原生 LJ，或 ASE/OpenMM 的 LJ；轨迹长度不足时如实报告，不声称收敛。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "structure or transport or correlator or diffusion"
python -m pyrpmd run examples/materials/diffusion.inp --output runs/t34
python -m pyrpmd analyze runs/t34 --structure --transport
```
