---
orphan: true
---

# T16b 三套 PIMD 引擎收敛

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1 |
| 前置 | 无（T16a 已合并：共享原生 CV、偏置与度规内核） |
| 后续依赖本卡 | T43、T44、T46 |
| 建议分支 | `agent/t16b-pimd-engine-convergence` |

## 背景

`task="pimd"` 和 `task="diffusion"` 目前有三套执行引擎，功能子集和 checkpoint 格式各不相同：

- **legacy（默认）：** `src/pyrpmd/workflows/materials.py`，基于 `Dynamics` 和传播栈，
  支持固定原子、键约束、BZP NPT、流式轨迹。
- **native：** `src/pyrpmd/workflows/native_pimd.py`、`native_ensemble.py`、`native/thermal.*`，
  支持持久 C++ 恒温器和 MTTK/RGB/SCR 柔性晶胞，以日志式记录为提交点。
- **torch：** `src/pyrpmd/workflows/torch_materials.py`、`torch_backend.py`，
  所有副本放在一个 `[B,P,N,3]` 批次中；不支持 NPT、约束、交换、摩擦和 MPI 分组。

周期材料审计（`docs/periodic-materials-audit.md`）和实现评审（`docs/implementation-review-2026-11-03.md`）
都指出：这几套引擎的相位时钟、RNG 所有权和已提交轨迹前缀的语义互不相同，
旧 checkpoint 不能互相续算。

## 目标

让三套引擎共享同一份可验证的科学契约：同一输入在任何引擎上都有相同的物理身份字段集合、
相同的系综定义和相同的估计量；引擎只是执行方式不同。提供显式的跨引擎迁移工具，
迁移时记录语义和来源，绝不静默地把一个引擎的状态当作另一个引擎的续算。

## 范围

**要做：**

1. 写出引擎能力矩阵（恒温器、压浴、约束、固定原子、交换、流式记录、MPI、设备），
   并从代码自动生成，用测试保证矩阵与实际校验逻辑一致。
2. 统一相位时钟、RNG 所有权、观测节拍和已提交前缀的定义；不同引擎在同一初始相空间点、
   同一确定性设置（NVE、无随机热浴）下的轨迹在浮点容差内一致。
3. 实现迁移工具 `pyrpmd migrate RUN --engine {legacy,native,torch} --output NEW_RUN`：
   从已提交的 checkpoint 生成新运行目录；新目录的身份中记录来源运行、源引擎和迁移语义
   （例如“相空间点相同，随机流重新播种”）。原运行目录保持只读、不被修改。
4. 所有引擎的估计量（原始估计量、质心维里估计量、压力）共享一份实现，或者在测试中逐一对照。

**不做：**

- 不为 torch 引擎补 NPT 和约束（见 T43、T44）。
- 不修改 `engine.py` 的算符结构（归 T26 所有）。
- 不改变已有运行的物理身份；旧运行必须照常续算。

## 文件所有权

- **独占：** `src/pyrpmd/workflows/materials.py`、`workflows/native_pimd.py`、
  `workflows/torch_materials.py`、`native_ensemble.py`、`torch_backend.py`、
  `torch_normal_modes.py`、`native/thermal.hpp`、`native/thermal.cpp`、
  新建 `src/pyrpmd/migrate.py`、相关测试。
- **共享注册点：** `cli.py`（`migrate` 子命令）、`runner.py`（引擎分派）、`input_tasks.py`。

## 交付物

- 能力矩阵生成函数及其英文指南 `docs/pimd-engines.md`（`orphan: true`）。
- `pyrpmd migrate` 命令及往返测试：legacy → native → legacy、legacy → torch → legacy。
- 估计量一致性测试。
- 验证记录 `docs/validation/pimd-engines-<日期>/README.md`。

## 验收标准

- 第三轮集成之前生成的所有 legacy、native、torch checkpoint 都能用新代码原样续算，
  测试中至少各覆盖一个保存的样例。
- 确定性 NVE 下，三套引擎从同一相空间点出发 1000 步的 `q,p` 一致，相对误差 ≤ 1e-12
  （NumPy 与原生之间允许 FMA 带来的舍入差异，需写明实际观察到的上限）。
- 谐振子 PIMD 下，三套引擎的动能、势能和质心维里估计量与解析有限 P 值一致，
  误差在统计误差之内。
- 迁移得到的运行目录身份中含有 `migrated_from` 来源记录；把迁移目录当作原运行续算时会被拒绝。
- 满足通用规则 G5–G8。

## 验证命令

```bash
python -m pytest -q -m "not campaign"
python -m pytest -q tests -k "materials or native_pimd or torch_materials or migrate"
```

## 第五轮调整（2026-11-04）

PR #93 已合并 `EngineRoute` 分派、预检和 checkpoint 存储。本卡续做的部分调整如下，优先于上文的冲突内容：

- **只剩两项：** `pyrpmd migrate` 跨引擎迁移工具，以及三引擎确定性 NVE 等价测试（1000 步 `q,p` 一致）。
- **能力矩阵移交 [T47](T47-method-registry.md)：** 引擎能力行由注册表建立并与 `EngineRoute` 交叉校验。
  本卡不再单独生成能力矩阵。
- **`native/thermal.hpp`、`native/thermal.cpp`、`native_ensemble.py` 移交 [T49](T49-thermal-kernel-performance.md)：**
  本卡在 T49 合并后开始；如需修改这些文件，写进“需要协调”。
- **波次：** 第五轮第 2 波（T49 之后）。T43 仍依赖本卡。
