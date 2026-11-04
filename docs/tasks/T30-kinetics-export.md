---
orphan: true
---

# T30 速率拟合与动力学模型导出

| 字段 | 内容 |
| --- | --- |
| 规模 | S |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t30-kinetics-export` |

## 背景

`pyrpmd study` 已能完成温度、同位素和数值收敛扫描。`src/pyrpmd/studies.py:45` 的
`arrhenius_fit` 只拟合两参数形式 ln k = ln A − Ea/(kB T)，并给出参数协方差。

目前没有修正 Arrhenius 拟合，也没有导出 Cantera YAML 或 Chemkin 格式的功能。
燃烧、大气和天体化学的动力学建模需要这些导出，才能直接使用 RPMD 速率。

注意：RPMD 双分子热速率是不依赖压力的高压极限速率，因此 PLOG 和 Chebyshev
等压力相关形式不在本卡范围内。

## 目标

从已完成的 study 或速率运行，得到带协方差的修正 Arrhenius 拟合，并导出可被 Cantera 和
Chemkin 直接读取的反应条目。导出时附带适用温度范围和不确定度元数据。

## 范围

**要做：**

1. **拟合模型。** 新增修正 Arrhenius 拟合 k = A (T/T₀)ⁿ exp(−Ea/kB T)，T₀ 默认 300 K 并可配置：
   - 加权最小二乘，残差定义在 ln k 上；
   - 给出参数协方差，并给出适用温度区间内 ln k 的置信带；
   - 提供残差诊断；
   - 当温度点少于参数个数，或设计矩阵病态时拒绝拟合。
2. **单位换算。** 区分 cm³ molecule⁻¹ s⁻¹ 与 cm³ mol⁻¹ s⁻¹（乘以 N_A）、单分子 s⁻¹，
   以及活化能单位（J/mol、cal/mol、kJ/mol、K）。用户必须显式给出反应方程式
   （物种名称）和反应级数，程序不从 PES 自动猜测。
3. **导出。**
   - Cantera YAML 的 `reactions` 条目（`rate-constant: {A, b, Ea}`）；
   - Chemkin `REACTIONS` 行，带 `UNITS` 声明；
   - 注释中写明温度范围、拟合残差、不确定度和数据来源（运行目录与身份哈希）。
4. **入口。** 新增 `pyrpmd kinetics RUN_OR_STUDY --equation "H + H2 <=> H2 + H" --format cantera|chemkin`，
   同时提供 Python API。

**不做：**

- 不处理压力相关形式，不做 RRKM 或主方程。
- 不修改速率计算本身，也不修改 `study` 的规划与执行逻辑。

## 文件所有权

- **独占：** `src/pyrpmd/studies.py`（只改拟合函数部分，`arrhenius_fit` 保持兼容）、
  新建 `analysis/kinetics_export.py`、相关测试。
- **共享注册点：** `cli.py`（`kinetics` 子命令）、`pyproject.toml`（可选依赖 `cantera`，仅用于测试）。

## 交付物

- 拟合与导出模块、CLI 和英文指南 `docs/kinetics-export.md`（`orphan: true`）。
- 一个由 `examples/rate/` 或已保存的 study 结果生成的示例导出文件。

## 验收标准

- 用已知参数加噪声生成的合成数据，拟合得到的参数在协方差给出的置信区间内；
  对无噪声数据，参数恢复误差 ≤ 1e-10（相对）。
- **Cantera 往返：** 安装 `cantera` 时，用 `cantera.Solution` 加载导出的 YAML（附带最小物种定义），
  在拟合温度点上计算正向速率常数，与拟合值的相对误差 ≤ 1e-8；未安装时跳过。
- Chemkin 输出的单位声明与数值，和 Cantera 的结果在单位换算后一致（由测试中的解析检查保证）。
- 现有 `arrhenius_fit` 的输出不变，已有 study 测试全部通过。
- 满足通用规则 G5、G6、G8。

## 验证命令

```bash
python -m pytest -q tests -k "study or arrhenius or kinetics"
```
