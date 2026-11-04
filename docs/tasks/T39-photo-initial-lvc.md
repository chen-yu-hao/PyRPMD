---
orphan: true
---

# T39 光动力学初始条件与 LVC 模型构建

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t39-photo-initial-lvc` |

## 背景

非绝热工作流目前为约化模型提供精确有限 P 的谐振高斯核初态，带显式的活动坐标与固定旁观坐标
（见 `docs/nonadiabatic-comparison.md`）；`potentials/nonadiabatic_models.py` 中已有
`LinearVibronicCoupling` 模型。

但对于真实分子的光激发动力学，还缺少两项 SHARC 和 Newton-X 中的标准功能：

- 由基态简正模得到的 Wigner 初始条件，以及按激发能窗口和振子强度选择初态；
- 由量子化学数据构建 LVC 模型（参见 Plasser、Gómez、Menger、Mai、González，PCCP 21, 57, 2019），
  以便用很低的成本对环聚合物非绝热方法做基准测试。

## 目标

提供分子光动力学的初始条件生成工具，以及 LVC 模型构建器；
输出可以直接被 `task="nonadiabatic"` 工作流读取的初始系综文件和模型定义。

## 范围

**要做：**

1. **简正模分析。** 复用 `workflows/prepare.py` 的数值 Hessian，或读取用户提供的 Hessian；
   做质量加权，投影掉平动和转动，给出频率与简正模，并检查虚频。
2. **Wigner 采样。**
   - 基态和有限温度（每个模式按 tanh(βħω/2) 加宽）的谐振 Wigner 分布；
   - 低频模式可以按选项排除或按经典处理；
   - 处理同位素与线性分子；
   - 输出质心坐标与动量；对于 P>1，额外提供“基态 PIMD 热采样后垂直激发”的环聚合物初态，
     使用已有采样器即可。
3. **激发选择。** 通过用户回调（或 T40 合并后的激发态提供者）得到垂直激发能与振子强度，
   支持能量窗口筛选、按振子强度的接受–拒绝抽样，并记录每个初态的选择概率。
4. **LVC 构建器。** 由参考几何上的激发能、激发态梯度、基态 Hessian，以及态间耦合
   （由有限位移下的重叠矩阵得到）构建 `LinearVibronicCoupling` 的参数；
   保存为带来源哈希的模型文件，可以在 CLI 中通过模型注册使用。
5. **入口。** `pyrpmd photo-sample INPUT --output DIR` 输出初始系综文件；
   `pyrpmd lvc-build DATA --output model.json`；`nonadiabatic_input.py` 接受
   `initial_ensemble = "file:..."` 一类的引用（追加式改动）。

**不做：**

- 不实现电子结构计算（归 T40）。
- 不修改非绝热动力学类（归 T26、T42a、T42b）。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/photo_initial.py`、新建 `potentials/lvc_builder.py`、
  `potentials/nonadiabatic_models.py`（只限 LVC 序列化与构建所需的追加接口）、相关测试、
  新建 `docs/photo-initial-conditions.md`（`orphan: true`）、`examples/photodynamics/`。
- **共享注册点：** `nonadiabatic_input.py`（初始系综引用）、`cli.py`。

## 交付物

- 采样器、LVC 构建器、CLI、英文指南和示例。
- 验证记录 `docs/validation/photo-initial-<日期>/README.md`。

## 验收标准

- **谐振分子：** Wigner 样本的位置与动量二阶矩与解析值一致（统计误差之内），
  平动和转动分量为零（数值精度之内），能量分布与理论分布一致。
- **有限温度：** 加宽因子正确，高温极限趋于经典 Boltzmann 分布。
- **LVC 往返：** 由已知 LVC 模型生成合成的“量子化学”数据，构建器恢复出的参数误差 ≤ 1e-8（相对）；
  加入数值噪声时，误差随有限位移步长的变化符合预期。
- **小分子示例：** 用 PySCF（SA-CASSCF 或 TDA，例如乙烯或甲醛）完成一次从计算参考数据到
  构建 LVC 再到短程 `task="nonadiabatic"` 运行的完整流程；只验证流程，不声称物理收敛。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "photo or lvc or wigner or nonadiabatic_models"
```
