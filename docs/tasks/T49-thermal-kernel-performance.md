---
orphan: true
---

# T49 热浴原生内核与组合传播的性能

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1（第五轮） |
| 前置 | 无 |
| 后续依赖本卡 | T27（重做，使用 `native/philox.h`）、T16b（续） |
| 建议分支 | `agent/t49-thermal-kernel-performance` |

## 背景

2026-11-04 评审在 4 vCPU 容器上（Python 3.13.18、NumPy 2.5.4、GCC，`OMP_NUM_THREADS=1`）
测量了 `Dynamics.propagate()` 的吞吐量。条件：谐振势，P=32，Cayley-BCB，`loop_backend="auto"`。

| 原子数 | NVE（融合 Cayley 内核） | PILE-L（原生热浴引擎） | i-PI 3.3.1，PILE-L |
| --- | --- | --- | --- |
| 3 | 89,089 步/秒 | 38,709 | 333 |
| 512 | 857 | 212 | 97 |
| 4096 | 95.8 | 19.5 | 18.1 |

打开 PILE-L 后，大体系吞吐量下降约 4–5 倍，N=4096 时与 i-PI 持平。剖析（cProfile，
N=4096，P=32，一次 `propagate(state, 2)`）找到两个原因：

1. **每次调用的回滚快照。** `propagation.py:229` 以及 `block_propagation.py:475`、`:716`
   在每次 `propagate()`/`propagate_block()` 调用开始时执行 `dynamics.operator_state_dict()`，
   用于失败时整块回滚。该快照经 `engine.py:365` 的 `_Composition.snapshot()` 包含
   `native_engine.state()` 的完整 `q`、`p`、力、能量等数组，再由 `engine.py:48` 的 `_json`
   逐元素转成 Python 列表。一次 2 步的调用中，这部分耗时 **0.42 秒**（其中 `ndarray.tolist`
   0.29 秒），而原生计算本身只用 0.16 秒。分块传播、频繁观测或短子轨迹会反复支付这一固定开销。
   `native_ensemble.py:810` 的 `_state_digest` 哈希另占约 0.02 秒。
2. **热浴内核是单线程的。** `native/thermal.hpp` 中没有任何 OpenMP 指令；
   噪声由 `std::mt19937_64` 加 `std::normal_distribution<double>` 串行生成（第 82 行），
   N=4096、P=32 时每步约 39 万个高斯数。T27 原卡明确不修改 `thermal.*`，
   T16b 原卡的目标是引擎收敛而不是性能，因此这一部分没有任何卡负责。

串行梅森旋转器也是第四轮 T27 失败的根源之一：只要噪声按线程划分，结果就依赖线程数，
无法满足“不同线程数下逐位一致”。计数器型 RNG（例如 Philox4x32-10）按
（种子，步，珠子，自由度）直接计算随机数，与线程划分无关。

## 目标

- 回滚快照的开销与体系规模无关的 Python 对象数量脱钩：只做数组拷贝，不做 JSON 转换。
- 热浴原生内核支持 OpenMP，并提供与线程数无关、可逐位复现的计数器型随机流（按 G17 版本化、可选启用）。
- 同一台机器上，带热浴的单线程吞吐量不低于 NVE 融合内核的 50%（目前约 20%）。

## 范围

**要做：**

1. **二进制回滚令牌。** 在 `dynamics.py` 中追加两个私有方法，例如
   `_operator_rollback_token()` 与 `_restore_operator_rollback_token(token)`：
   令牌保存 NumPy 数组的拷贝、RNG 状态对象和原生引擎状态，不经过 `_json`。
   `propagation.py` 与 `block_propagation.py` 的回滚路径改用令牌。
   `operator_state_dict()` 作为 checkpoint 接口保持原样（格式与内容不变）。
   原生引擎如果能以更便宜的方式回滚（例如只记录 RNG 计数器与起点 `q,p`），可以在 `native_ensemble.py`
   中提供，但必须保证失败后的状态与旧实现逐位相同。
2. **剖析。** 分别测量 FFT/正则模变换、PES、kick、热浴噪声生成与施加、观测记录、回滚，
   结果写进验证记录。
3. **OpenMP。** 在 `native/thermal.hpp` 中按自由度块并行化正则模变换、热浴施加、kick 和内置 PES；
   归约量（能量等）使用固定顺序的分块归约，并说明可能的 ulp 级差异。
4. **计数器型 RNG。** 新建 `native/philox.h`（在仓库内独立实现，算法依据 Salmon et al., SC'11；
   不复制带不兼容许可证的代码），提供按（种子，步，珠子，自由度，用途）索引的标准正态数。
   在热浴引擎中作为可选随机流，例如 `[dynamics] rng = "philox-v1"`：
   - 默认仍为现有的 `mt19937_64`，现有运行与 checkpoint 逐位不变；
   - `philox-v1` 写入 run identity 与 checkpoint，`thermal.cpp` 第 200 行附近的 `rng_abi` 同步加版本；
   - 两种随机流都能保存与续算。默认值是否切换由维护者决定（见 [round-5](round-5.md)）。
5. **基准脚本。** 新建 `benchmarks/efficiency/thermal_throughput.py`，复现上面的基线表，
   并扩展到 N ∈ {64, 1000, 4096}、P ∈ {8, 32, 128}、线程数 ∈ {1, 2, 4}，
   热浴覆盖 PILE-L、PILE-G、GLE/PIGLET、NHC，以及支持时的 BZP NPT。

**不做：**

- 不修改融合 Cayley 路径的 `native/cayley.cpp`、`fused.hpp`、`propagation.h`、`ring_fft.h`（归 T27）。
- 不改变任何积分器或热浴的数学定义；不改变 `mt19937_64` 路径的随机流。
- 不实现跨引擎迁移（归 T16b 续）。

## 文件所有权

- **独占：** `native/thermal.hpp`、`native/thermal.cpp`、新建 `native/philox.h`、
  `src/pyrpmd/native_ensemble.py`、`propagation.py` 与 `block_propagation.py` 中的回滚路径、
  `benchmarks/efficiency/thermal_throughput.py`、相关测试。
- **共享注册点：** `dynamics.py`（只追加两个私有方法）、`setup.py`（只在需要时调整 `_thermal` 扩展的编译依赖）、
  `input_tasks.py` 与 `engine.py` 的热浴参数表（只追加 `rng` 选项）。
- **移交说明：** 本轮 `native/thermal.*` 与 `native_ensemble.py` 从 T16b 移交本卡；T16b（续）在本卡合并后开始。

## 交付物

- 回滚令牌、OpenMP 化的热浴内核、`philox-v1` 可选随机流及其测试。
- 基准脚本与英文说明，追加到 `docs/native-ensembles.md`。
- 验证记录 `docs/validation/thermal-kernel-performance-<日期>/README.md`：
  剖析表、改动前后的同机基准、线程扩展表、逐位一致性证据（G16）。

## 验收标准

- **不改变现有结果。** 默认随机流下，所有原生热浴测试、`examples/` 中使用热浴的示例，
  以及用 `main@801953d` 生成的 checkpoint 的续算结果，在单线程下与改动前逐位一致。
  `mt19937_64` 路径在多线程下也必须逐位一致（并行只用于与随机数无关的部分），
  或者在该路径上保持串行，并在记录中说明。
- **回滚开销。** N=4096、P=32、PILE-L 下，`propagate(state, 2)` 中原生内核以外的耗时
  不超过原生内核耗时的 10%（目前约为 260%）。中途失败的整块回滚测试全部通过。
- **吞吐量。** 同机单线程下，PILE-L 吞吐量不低于同条件 NVE 融合内核的 50%（N ∈ {512, 4096}，P=32）。
- **线程扩展。** `philox-v1` 下，N=4096、P=32 时 4 线程相对 1 线程至少加速 2.5 倍；
  如果内存带宽是瓶颈，给出带宽测量与可达上限。
- **随机流正确性。** `philox-v1` 在 1、2、4、8 线程下的 `q,p` 轨迹逐位一致；
  中断续算与不中断逐位一致；谐振子 PIMD 的动能、势能与解析有限 P 值在统计误差内一致；
  PILE-L 的目标温度、正则模动能分布通过统计检验。
- 开启和关闭 OpenMP、开启 FMA 的三种构建都通过；`PYRPMD_NO_NATIVE=1` 路径不受影响。
- 小体系（N=3，P=1 与 P=32）单线程变慢不超过 2%。
- 满足通用规则 G5–G8、G15–G17。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
OMP_NUM_THREADS=1 python benchmarks/efficiency/thermal_throughput.py --baseline
for t in 1 2 4; do OMP_NUM_THREADS=$t python benchmarks/efficiency/thermal_throughput.py --rng philox-v1; done
OMP_NUM_THREADS=1 python -m pytest -q tests -k "native or thermal or ensemble or propagation or rollback"
python -m pytest -q -m "not campaign"
PYRPMD_NO_OPENMP=1 python -m pip install -e . --no-build-isolation && python -m pytest -q tests -k "thermal"
```
