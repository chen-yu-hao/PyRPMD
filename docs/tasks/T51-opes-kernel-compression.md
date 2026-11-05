---
orphan: true
---

# T51 OPES 核压缩

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2（第五轮） |
| 前置 | T48（去除 MetaD/OPES 兼容层） |
| 建议分支 | `agent/t51-opes-compression` |

## 背景

PyRPMD 的 OPES 不做核压缩：每次沉积都保存一个新核，偏置及其梯度的计算量与已沉积核数成正比。
这一点在代码和文档中都有明确说明：

- `src/pyrpmd/_opes_serial.py:11`：“Kernel compression and shared walkers are absent.”
- `docs/opes.md:14`、`:47`：默认是固定带宽、不压缩；核压缩未实现。
- `workflows/_opes_serial.py:407` 的警告：“Uncompressed OPES: cost grows with kernel count”。

PLUMED 的 `OPES_METAD` 默认开启压缩（`COMPRESSION_THRESHOLD`，默认 1，以带宽为单位）。
压缩后核数大致随已探索的 CV 区域增长，而不随模拟长度增长。没有压缩时，长时间的生产运行越跑越慢，
也无法与 PLUMED 的 OPES 结果做数值层面的对照。

## 目标

实现与 Invernizzi & Parrinello (2020) 和 PLUMED 文档一致的核合并规则，
作为可选功能（默认关闭），使长时间 OPES 运行的单步偏置计算量有界，
并保持关闭压缩时的结果逐位不变。

## 范围

**要做：**

1. **合并规则。** 每次沉积新核时，找出在带宽度量下距离最近的已有核；距离小于
   `compression_threshold` 时与其合并，否则作为新核加入。合并后的高度、中心和带宽满足
   矩守恒（二元合并的权重和、一阶矩和二阶矩不变）。依据原始论文及其补充材料和 PLUMED
   公开文档独立实现，在指南中写出公式；不复制 PLUMED 源码（LGPL）。
   如果参考 PLUMED 的行为细节（例如合并后是否递归检查），在指南中注明来源与版本。
2. **周期 CV。** 合并时正确处理周期坐标的中心平均，给出测试。
3. **归一化。** OPES 的归一化因子 Z 依赖核集合；压缩后按 PLUMED 的处理方式更新 Z，
   并在指南中写明与未压缩估计之间的差异。
4. **共享与分布式 walker。** 同步共享（`mpi-sync`）与异步（`async-files`）模式下，
   合并顺序必须确定：按沉积事件的全局顺序执行，使任意 walker 数和任意续算点上的结果一致。
5. **输入与持久化。** `[opes] compression_threshold`：`0` 表示不压缩（默认）；正数表示启用。
   启用后的核历史以新的 checkpoint schema 版本保存；schema 1、2 继续可读，旧运行照常续算，
   不允许在续算时开启或关闭压缩（属于不同的科学设置）。
6. **冻结偏置重加权。** 压缩历史同样支持 frozen-bias 生产与重加权。

**不做：**

- 不实现 PLUMED 的扩散窗口带宽自适应（与已有的 Welford/Scott 规则是不同的方法）。
- 不改变默认值；默认是否改为启用，由维护者看到验证记录后决定。

## 文件所有权

- **独占：** T48 合并后的 `src/pyrpmd/opes.py` 与 `workflows/opes.py`、
  `workflows/_distributed_opes_*.py`（只限压缩相关改动）、`docs/opes.md`、相关测试。
- **共享注册点：** `input_fes.py`（追加 `compression_threshold` 字段）、
  T47 合并后的注册表（登记新选项及其成熟度）。

## 交付物

- 合并规则实现、指南中的公式与参数说明。
- 验证记录 `docs/validation/opes-compression-<日期>/README.md`。

## 验收标准

- `compression_threshold = 0` 及所有未设置该字段的输入：结果与改动前逐位一致；旧 checkpoint 照常续算。
- 合并规则的单元测试：两核合并的矩守恒（解析对照）；周期 CV 的跨边界合并；阈值边界行为。
- 在二维解析双阱（或已有的 `examples/opes/harmonic_2d.inp` 的加长版本）上：
  - 开启压缩后，核数随模拟长度饱和，并给出核数随时间的曲线；
  - 单步偏置计算耗时随模拟长度基本不变（G16 同机基准）；
  - 重建的 FES 与不压缩运行、与解析或数值积分参考在统计误差内一致（独立种子重复给出误差）。
- 真实 PLUMED 可用时（CI 中已有真实 PLUMED 的 job），对同一条 CV 轨迹分别用 PyRPMD 和 PLUMED
  `OPES_METAD`（相同带宽、阈值、壁垒与 pace）逐步计算偏置，差异在给定容差内，并写明容差的来源；
  不可用时跳过而不是失败。
- `mpi-sync` 与 `async-files` 模式下，不同 walker 数、在任意块边界中断续算，核集合与不中断时逐位一致。
- 满足通用规则 G5–G8、G16、G19。

## 验证命令

```bash
python -m pytest -q tests -k "opes"
python -m pytest -q -m "not campaign"
pyrpmd run examples/opes/harmonic_2d.inp --output runs/t51-uncompressed
```
