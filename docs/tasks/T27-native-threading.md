---
orphan: true
---

# T27 原生内核多线程扩展与原生周期 LJ

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t27-native-threading` |

## 背景

2026-11-03 评审在 4 核容器上实测了原生块传播的性能（内置谐振势，Cayley-BCB，`loop_backend="auto"`）：

| 体系 | 线程数 | 每步耗时 |
| --- | --- | --- |
| N=1000，P=32 | 1 | 1.69 ms |
| N=1000，P=32 | 4 | 1.30 ms（加速 1.30 倍） |
| N=64，P=32 | 1 | Cayley-BCB 约 100 µs，normal-mode-BAB 约 183 µs |

- 单线程下每个原子·珠子·步约 45–60 ns；Cayley 的耗时随 P 线性增长。
- `native/cayley.cpp` 的自由环传播按坐标维度做 `#pragma omp parallel for`，
  但访问 `q[k*dims+j]` 时按珠子跨步，缓存利用率低。
- 内置 PES（`native/builtin_pes.hpp`）和 PILE 噪声生成是否串行，需要剖析确认。
- 另外，原生内置势只有 Free、Harmonic、MorsePair、DoubleWell、PeriodicCosine，
  缺少可用于材料、凝聚相测试的周期 Lennard-Jones。

## 目标

让原生块传播在中等规模体系上具有可用的多线程扩展性；normal-mode-BAB 的单线程开销
不高于 Cayley-BCB 的 1.3 倍；补一个带邻居表和维里的原生周期 LJ 内置势，
供 T33、T34、T35、T44 作测试体系。

## 范围

**要做：**

1. 剖析：用 `perf` 或计时插桩，分别测量自由环变换、PES、kick、热浴噪声、观测记录的耗时，
   结果写进验证记录。
2. 根据剖析结果优化内存布局与并行方式，例如：按坐标块转置以获得连续的珠子访问、
   并行化内置 PES、预生成 PILE 噪声并在线程之间确定性地划分。
3. 优化 normal-mode-BAB 路径（`ring_fft.h` 与精确谱传播）。
4. 新增原生周期 LJ 内置势：截断与平移、通用斜晶胞最小像（复用 `geometry.py` 的约定）、
   cell list 或 Verlet 邻居表、能量、力和物理维里。通过 C ABI 绑定暴露为 `provider="lj"`，
   参数带单位。

**不做：**

- 不修改 `native/thermal.hpp`、`native/thermal.cpp`（归 T16b）。
- 不改变任何积分器的数学定义，也不改变随机流定义。

## 文件所有权

- **独占：** `native/cayley.cpp`、`native/propagation.h`、`native/fused.hpp`、`native/ring_fft.h`、
  `native/builtin_pes.hpp`、`native/fixed_atoms.hpp`、`native/cv_core.hpp`（仅限性能改动，不改数学）、
  `src/pyrpmd/potentials/_builtin_native.py`、`potentials/builtin.py`（新增 LJ）、`setup.py`（编译选项）、
  `benchmarks/efficiency/`（新增脚本）、相关测试。
- **共享注册点：** `potentials/__init__.py`（注册 `lj`）。

## 交付物

- 剖析报告与优化后的基准表，放在 `docs/validation/native-threading-<日期>/`。
  基准表包含 N ∈ {64, 1000, 8000}、P ∈ {8, 32, 128}、线程数 ∈ {1, 2, 4, 8}，并写明硬件信息。
- 原生周期 LJ 及其英文说明（追加到 `docs/builtin-native-potentials.md`）。

## 验收标准

- 在 N=1000、P=32 的内置势基准上，4 线程相对 1 线程至少加速 2.5 倍；
  如果硬件内存带宽是瓶颈，需要给出带宽测量作为证据，并说明可达到的上限。
- N=3、P=1 的小体系单线程性能变慢不超过 2%（沿用 T16a 的标准）。
- normal-mode-BAB 单线程每步耗时不超过同条件 Cayley-BCB 的 1.3 倍（N=64，P ∈ {32, 128}）。
- 同一输入在线程数 1、2、4、8 下的 `q,p` 轨迹逐位一致；需要归约的标量（如能量）
  只允许几个 ulp 的差异，并说明原因。
- 原生与 NumPy 的一致性测试、开启 FMA 的构建、`PYRPMD_NO_NATIVE=1` 路径都通过。
- LJ：力与有限差分对照、维里与有限应变能量差分对照、截断处的能量连续性、
  FCC 晶格能量与解析晶格和对照、斜晶胞下与 ASE LennardJones 计算器在截断一致时逐帧对照。

## 验证命令

```bash
python -m pip install -e . --no-build-isolation
OMP_NUM_THREADS=1 python -m pytest -q tests -k "native or builtin or fused"
python -m pytest -q -m "not campaign"
PYRPMD_NO_NATIVE=1 python -m pip install -e . && python -m pytest -q tests/test_dynamics.py
```
