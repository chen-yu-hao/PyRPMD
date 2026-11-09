---
orphan: true
---

# L5 材料、表面与输运

| 字段 | 内容 |
| --- | --- |
| CPU | 24–26（3 核） |
| 预算 | 138 核时（计划 110） |
| 工作流 | `diffusion`（MSD/VACF、电子摩擦）、`site_rate`、`surface_rate`、固定层 slab、`export`；可选 q-TIP4P/F 经典输运 |
| 外部模型 | Ni–H EAM（L0 下载）；OpenMM（可选，L5-d） |
| 建议分支 | `agent/c32-l5-materials` |
| 记录 | `docs/validation/c32-l5-materials-<日期>/README.md` |
| 复核 | L5 复核 L4；L4 复核 L5 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L5 ...` |

## 背景

- `surface_rate` 与 `site_rate` 都走 PMF + 约束母链 + 子轨迹的完整速率链；同一模型上用长时间无偏轨迹
  直接数跃迁，是对其归一化（盆的定义、通量测度、方向因子）最直接的检验，目前没有做过。
- q-TIP4P/F 的 OpenMM 经典参考已完成（VACF D=0.19780892 Å²/ps，区间 [0.18561141, 0.21000642]），
  但"PyRPMD 生产输运"从未运行（[文献验证清单](../../literature-validation.md)）。
- Ni–H 只有静态实现证据，没有动力学。

## 子测试

### L5-a 跨路线跃迁速率：`surface_rate` / `site_rate` 与直接计数（约 55 核时）

模型：`examples/surface/diffusion.inp` 使用的固定基底模型（`toy_surface.py`），以及
`examples/materials/site_rate.inp` 的周期模型。在 `benchmarks/campaign_c32/L5/` 写输入，选一个温度使势垒约
4–6 kBT（直接计数可行），P=1 与 P=8 各做一次。

1. **速率链：** `surface_rate`（以及 `site_rate`）各 3 次独立运行，得到定向跃迁速率 k 及其区间。
2. **直接计数：** 同模型、同温度、同 P，用 `diffusion` 任务（或 Python API 的无偏 RPMD 传播）跑 8 条独立的长轨迹，
   按与速率链**相同的盆定义**数首次进入相邻盆的事件，k = 事件数 / 盆内停留总时间；
   预注册事件计数规则（防止同一次跃迁的快速回跳被重复计数：用与 κ 相同的承诺判据）。
3. **P=1 的谐振 TST**（从势能面的二阶导解析计算）作为第三个独立参考，只用于 P=1。
4. 判据：ln k_chain − ln k_count 的 95% 区间包含 0 判为"一致"；区间在 ±0.2 内判为"等价"。
   对 P=8 还报告 k(P=8)/k(P=1) 的区间（量子效应，非判据）。

### L5-b 扩散估计量的内部一致性（约 20 核时）

1. 自由粒子（`examples/materials/diffusion.inp`）：确认分析把弹道区标记出来而不是报告扩散系数。
2. 一个 LJ 流体（经典 P=1，三个尺寸 N=108、256、500）：MSD 与 VACF 两种 D 的区间是否相交；
   D 随 1/L 的趋势（有限尺寸修正只作描述）；P=4 与 P=1 在重原子质量下的差（经典极限，应在区间内为 0）。
3. `pyrpmd export` 把一条质心轨迹分别导出为 extxyz、H5MD、ASE `.traj`，再读回比较坐标与单位。

### L5-c 电子摩擦（约 5 核时）

`examples/nonadiabatic/friction_surface.inp` 放大采样，扫 3 个摩擦强度：扩散系数应随 γ 单调下降；
γ 很大时与 Einstein–Smoluchowski 极限 D = kBT/(mγ) 比较（预注册比较区间）。

### L5-d 可选：q-TIP4P/F 经典 PyRPMD 生产输运（OpenMM 可用时，约 30 核时）

阅读 `benchmarks/water/README.md` 与 `campaign.py`，确认哪个驱动用 **PyRPMD 传播器**（不是 OpenMM 积分器）
调用 q-TIP4P/F 力；P=1、与 OpenMM 参考同样的密度、温度、步长和 8 个种子，按参考的"制备 + 25 × 4 ps NVE"结构，
pilot 后按预算缩放（缩短时写明）。判据：与 OpenMM 参考 VACF D 的差的区间落在 ±0.01 Å²/ps 内
（与已登记的等价边界相同）。量子水不做。

### L5-e Ni–H EAM 动力学试点（约 10 核时）

`benchmarks/materials/ni_h_eam.py --model $C32_DATA/shared/NiAlH_jea.eam.alloy --output ...`（非 `--quick`），
然后在同一模型上用 `site_rate` 做一个 H 跃迁的 P=1 与 P=4 试点。明确标注为工作流验证，
不是 Suleimanov 2012 的复现（原论文协议未确定）。

## 不做

- 不把直接计数与速率链的样本合并；不为了让两者一致而移动分割面或盆定义（盆定义在预注册中固定）。

## 交付物

- `benchmarks/campaign_c32/L5/`：输入、计数与分析脚本、预注册；验证记录与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L5 agent，负责"材料、表面与输运"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L5-materials-transport.md；docs/manual/periodic.md；docs/surface-catalysis.md；
benchmarks/water/README.md。

要求：
- 你的 CPU 是 24–26（3 核），预算 138 核时；所有计算用 $C32_LAUNCH run --lane L5 启动。
- 核心问题是"速率链给出的跃迁速率是否等于长时间无偏轨迹直接数出来的速率"。盆定义、承诺判据和
  比较区间在生产前提交；两条路线的种子独立。
- q-TIP4P/F 只做经典 P=1，且只有 L0 报告 OpenMM 可用时才做；先确认使用的是 PyRPMD 传播器。
- Ni–H 只作工作流试点，记录中不得称为复现。
- T+24 h、T+40 h 更新 $C32_DATA/L5/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l5-materials 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L4。
```
