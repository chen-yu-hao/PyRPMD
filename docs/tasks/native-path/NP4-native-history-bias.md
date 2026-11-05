---
orphan: true
---

# NP4 C++：元动力学与墙偏置的原生描述符

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无（与第四轮 T27 的 `cv_core.hpp` 重叠见本轮 README） |
| 等价性等级 | E1 |
| 建议分支 | `agent/np4-history-bias` |

## 背景

原生块目前只认识两种偏置：伞形和 RPMDrate flux-metric。偏置描述符有**两条独立入口**，不要混淆：

| 入口 | Python 描述符 | C++ 解析 | 格式 |
| --- | --- | --- | --- |
| `Dynamics.propagate_block` → `advance` | `block_propagation.py:119` 的 `bias_descriptors()`，由 `_native_plan()` 调用 | `native/cv_core.hpp` `Bias<false>`（`tagged` 分支，`propagation.h:25`） | 字符串 `kind`：`"umbrella"`/`"flux"` |
| `Dynamics.propagate` → `run_block` | `propagation.py:196` 的 `_biases()` | `Bias<true>`（`numeric` 分支，`fused.hpp:18`） | 数值 `kind`：0/1 |

其他偏置在两条入口都会抛出 `NotNative`，走 Python。NP7 使用的是 `propagate_block`，所以**第一条入口是本卡必须打通的路径**；
第二条入口可选。

元动力学（`bias.py:188` 的 `MetadynamicsBias`、`bias.py:406` 的 `GridMetadynamicsBias`）只在 `pace`
边界沉积 hill，两次沉积之间偏置是固定的。如果偏置能在 C++ 中求值，就可以每个 `pace` 只返回 Python 一次
（接入归 NP7）。墙偏置（`bias.py:155` 的 `WallBias`）是固定的，常与元动力学组合。

**注意：** `MetadynamicsBias` 的文档写明"deliberately has no frozen native-bias descriptor"。
这是为了防止原生侧持有过期的 hill 历史。本卡必须用"带版本的冻结快照"解决这个问题，而不是忽略它。

## 目标

为 `WallBias`、`MetadynamicsBias`（standard / well-tempered，含周期维的环绕高斯）、
`GridMetadynamicsBias`（`tensor_hermite_v1`，1–2 维）及其 `CompositeBias` 组合提供原生描述符，
供 `propagate_block` 使用。描述符是某一时刻历史的不可变快照，带 `hill_count` 版本号。

## 范围

**要做：**

1. **C++（`native/cv_core.hpp` 的偏置部分）。** 在 `Bias<false>`（`tagged`，供 `advance`）中新增偏置类型；
   `Bias<true>`（`numeric`，供 `run_block`）可选，不做时保持只认识 0/1 并对新类型报错：
   - `wall`：`W = κ·max(lower−s,0)² + κ·max(s−upper,0)²`，哈密顿量中乘 P，与 `bias.py:179-184` 相同约定。
   - `hills`：中心 `[H, D]`、高度 `[H]`、宽度 `[D]`、周期（可空）；周期维按 Python 实现的镜像求和规则
     （读 `MetadynamicsBias._evaluate` 确认镜像数和截断方式，必须一致）。能量加 P·V，力用 CV 的完整环雅可比。
   - `grid`：`tensor_hermite_v1` 插值，网格数组与边界由 Python 传入，越界行为与 Python 一致。
   - 组合：多个偏置项求和（现有 `biases` 序列已支持多项，确认后复用）。
   不改 CV 部分与已有偏置的数学。追加能力常量 `NATIVE_HISTORY_BIAS_API = 1`（`native/cayley.cpp` 共享注册点）。
2. **Python 描述符。** 在 `bias.py` 中为上述类**追加** `native_descriptor()` 方法（不改已有方法），返回包含
   `hill_count`（或网格版本）的不可变快照；把 `MetadynamicsBias` 文档中那句说明改为描述"冻结快照 + 版本"契约。
   在 `block_propagation.py` 的 `bias_descriptors()` 中识别这些类型（`CompositeBias` 递归已存在）；
   子类化或覆写方法的偏置仍走 Python（沿用 `propagation._check_bias_methods`）。
   `propagation._biases()`（`propagate()` 路径）可选支持；不支持时保持抛出 `NotNative`。
3. **过期保护。** 描述符在每次 `propagate_block` 调用时由 `_native_plan()` 新建，不得跨块缓存（加测试：
   沉积后下一块使用新历史）。另在 `block_propagation.py` 新增私有函数 `_check_bias_snapshot(bias, descriptors)`：
   校验 `hill_count`/网格版本与快照一致，不一致时报错。在 `propagate_block` 原生 `advance` 返回之后、提交状态之前
   **插入一行**对它的调用（这是本卡在 `propagate_block` 函数体内唯一允许的改动）。用测试制造过期快照，证明会被拒绝。
4. 在 `docs/native-propagation.md` 追加小节。

**不做：**

- 不接入元动力学工作流（归 NP7）。
- 不做 OPES、ABF、VES、steered（它们每步更新统计量或涉及时间相关协议，留作后续评估；
  在 PR"未完成事项"中写明各自的阻碍）。
- 不改 `native/propagation.h`（归 NP1）。`advance` 通过 `options["biases"]` 把描述符交给 `Bias<false>` 构造函数，
  新类型只需在 `cv_core.hpp` 中解析；如确需改 `propagation.h`，写进"需要协调"。
- 不改 `block_propagation.py` 中除 `bias_descriptors()`、新增 `_check_bias_snapshot()` 和那一行调用之外的任何代码（其余归 NP1）。

## 文件所有权

- **独占：** `native/cv_core.hpp` 的 `struct Bias` 及新增偏置类型；`block_propagation.py` 的 `bias_descriptors()`
  与新增 `_check_bias_snapshot()`；`propagation.py` 的 `_biases` 与 `_check_bias_methods`；
  `bias.py` 中新增的 `native_descriptor` 方法与上述文档句；新测试文件。
- **共享插入点：** `block_propagation.propagate_block` 中对 `_check_bias_snapshot()` 的一行调用（与 NP1 协调，后合并者 rebase）。
- **共享注册点：** `native/cayley.cpp`（能力常量一行）、`docs/native-propagation.md`（追加小节）。

## 交付物

- 原生偏置类型、Python 描述符、过期保护与测试；`native/README.md` 追加一段说明（只追加）。

## 验收标准

- **数值一致（E1）：** 对 1 维与 2 维、含周期与非周期维、H ∈ {0, 1, 100, 5000} 个 hill、
  standard 与 well-tempered、网格内外的随机点，原生与 Python 的偏置能量和环力相对误差不超过 1e-12
  （或给出更严格的实测值），PR 写明最大误差。墙偏置在边界两侧与边界上一致。
- **块一致：** 在固定 hill 历史下，`propagate_block(bias=…)` 的原生与 Python 路径轨迹差异不超过现有
  伞形偏置融合/非融合对比测试的容差。
- **契约：** 子类化或覆写的偏置走 Python；过期快照被拒绝；缺少 `NATIVE_HISTORY_BIAS_API` 时 `auto` 回退、
  `native` 在修改状态前报错。
- 有限差分验证原生力；OpenMP 开/关、`PYRPMD_NO_NATIVE=1`、开启 FMA 的构建都通过。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
OMP_NUM_THREADS=1 python -m pytest -q tests -k "metadynamics or wall or bias or native"
python -m pytest -q -m "not campaign"
```
