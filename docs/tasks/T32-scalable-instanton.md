---
orphan: true
---

# T32 可扩展 instanton 预因子

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t32-scalable-instanton` |

## 背景

T09 已经交付了生产级 instanton：Bofill 优化、Hessian provider、`pyrpmd instanton` 命令，
H + H₂ 速率与文献吻合到两位有效数字。但致密作用量 Hessian 限制在 4096 个活动珠子坐标以内：

- `src/pyrpmd/instanton.py:240` 与 `:495` 的 `if p * d > 4096: raise ValueError(...)`；
- `low_temperature.py:421` 有同样的限制。

在 P=64 时，这只相当于约 21 个活动原子。另一方面，`_from_hessians` 构造的矩阵本身是
块循环三对角结构：对角块是 Δτ·H_j + 2m/Δτ，相邻珠子之间只有质量对角块，并非真正的稠密矩阵。

## 目标

利用 Hessian 的块循环三对角结构，以 O(P d³) 的计算量和 O(P d²) 的内存完成预因子的
行列式与低本征值计算，去掉 4096 坐标上限；同时减少物理 Hessian 的调用次数。

## 范围

**要做：**

1. **稀疏结构上的代数运算。**
   - 用块结构求低本征对（负模、零模、近零模），例如 shift-invert Lanczos 或分块代数；
   - 对去掉零模和负模后的矩阵计算 log|det|，例如分块 LU 或 Cholesky 加循环边界的
     秩修正（Woodbury），或结构化稀疏分解；
   - 零模处理与现有致密实现保持同一约定。
2. **Hessian 成本。** 沿 instanton 路径复用已收敛的物理 Hessian，或用 Bofill/Powell 更新代替
   部分 Hessian 调用；对称或重复的珠子（instanton 前后两半对称）只计算一次。
   每次运行的报告中写明物理 Hessian 的调用次数。
3. **兼容性。** 小体系保留致密路径作为参考实现和对照；输入可选 `algorithm="dense"|"structured"|"auto"`，
   其中 `auto` 在 4096 以内使用致密路径，保证已有结果不变。
4. **低温交叉检验。** 让 `low_temperature.py` 中的高斯涨落预因子共用同一套结构化代数，
   同样去掉上限。

**不做：**

- 不改变 instanton 的物理定义或已有输出字段的含义；新字段采用追加方式。
- 不实现 GPR 加速 instanton（可另行立卡）。

## 文件所有权

- **独占：** `src/pyrpmd/instanton.py`、`low_temperature.py`、`tunnelling.py`、
  `workflows/instanton.py`、`examples/instanton/`、相关测试。
- **注意：** T31 只调用这些文件的公开函数；不要改变现有公开函数的签名。

## 交付物

- 结构化代数实现与英文说明（追加到 `docs/instanton.md`）。
- 验证记录 `docs/validation/instanton-structured-<日期>/README.md`。

## 验收标准

- 在 4096 坐标以内的所有现有测试和 `examples/instanton/` 示例上，结构化路径与致密路径的
  log|det|、本征值和速率相对误差 ≤ 1e-8。
- **解析检验：** 一维 Eckart 势垒耦合 N 个未耦合谐振浴模，浴模的贡献可以解析得到；
  在 P×d ≥ 10⁵ 时结果与解析值一致，并给出耗时和内存随 P、d 的标度。
- 现有 H + H₂ instanton 结果与 T09 的记录一致。
- `algorithm="auto"` 下，已有运行的物理身份和结果不变。
- 满足通用规则 G5–G9。

## 验证命令

```bash
python -m pytest -q tests -k "instanton or low_temperature or tunnelling"
python examples/instanton/production.py
```
