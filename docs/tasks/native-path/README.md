---
orphan: true
---

# 原生路径替换任务卡（第五轮）

更新日期：2026-11-05。基线：`main@ca093eb`（#116 合并后）。

本目录把"整条轨迹尽量留在 C++、只在 checkpoint/IO/事件处返回 Python"的评估结果，
拆成可以交给彼此独立的 agent 执行的任务卡。卡片编号用 `NP` 前缀（native path），
避免与 `T` 系列（已有 T50/T52 记录）冲突。

**用法：** 交给 agent 时，同时提供 `docs/tasks/README.md`（通用规则 G1–G15）、本页
（本轮规则 R1–R9）和对应卡片。agent 必须先完整阅读三者再动手。派发提示词模板见文末。

## 背景：当前各链路返回 Python 的频率

C++ 侧目前有两套互不相通的引擎：

- **确定性块引擎**：`native/fused.hpp`（`run_block`，供 `Dynamics.propagate`）与
  `native/propagation.h`（`advance`，供 `Dynamics.propagate_block`）。支持原生 PES、
  质心/珠均 CV、伞形与 flux-metric 偏置、质心 RATTLE、fragment COM 投影、固定原子、
  dRPMD 结果判定事件、端点力复用。**没有恒温器，也没有随机数。**
- **原生系综引擎**：`native/thermal.hpp`（`NativeEnsemble`）。自带 `mt19937_64` 随机流，
  支持 Andersen/Langevin/PILE/SVR/GLE/PIGLET/NHC 与 BZP/MTK/SCR 压浴。**没有偏置、CV 和约束。**
  它的 Andersen 用 middle splitting，与旧工作流的随机流和有限步样本不一致。

| 链路 | 入口 | 现状 |
| --- | --- | --- |
| 原生 PIMD（`sampling.engine="native"`） | `workflows/native_pimd.py:226` | 每个 checkpoint 块返回一次 |
| dRPMD 碰撞（内置分类器）、state-selected | `drpmd_trajectory.py:391`、`state_selected.py:205` | 每 `block_steps`（默认 256）返回一次 |
| 伞形/PMF/FES（固定偏置 + Andersen） | `biased_sampling.py:69` | 每次 Andersen 碰撞（默认 10 步）返回一次 |
| 再穿越父链（RATTLE + Andersen） | `rate_recrossing.py:131-145` | 同上 |
| `Dynamics.thermalize`、REUS 窗口 | `dynamics.py:806-813`、`replica_exchange.py:249-257` | 同上 |
| 再穿越子轨迹（RPMD） | `rate_recrossing.py:286-293` | 每 256 步；用 `propagate()`，每步 2 次 PES 调用 |
| PILE-L 采样、TRPMD 子轨迹 | `sampling.py:194-202`、`rate_recrossing.py:296-300` | 每步 |
| 元动力学/OPES/ABF/VES | `biased_sampling.py:127`（`_history_step`） | 每步 |
| materials/diffusion | `materials.py:657-680` | 每步（Andersen 路径是 `propagate(state, 1)`） |
| committor/TIS/FFS 射击 | `committor.py:258`、`transition_interface.py:316,327`、`path_sampling.py:430` | 每步 |

**关键事实（已实测，NumPy 2.5.3）：** 旧路径里的 Andersen 动量
（`dynamics.py:521`，`rng.normal(size)*sqrt(mP/β)`）、PILE-L 噪声（`dynamics.py:992`）和
geometric 碰撞调度（`sampling.py:97-102`，`rng.geometric`）都与当前状态无关
（带 bond constraints 时的动量投影除外）。对 `np.random.Generator`，一次抽出 k 组正态数
与逐次抽 k 次**逐位相同**，结束时 `bit_generator.state` 也相同；normal 与 geometric
交错抽样同样确定。因此 Python 可以按原顺序预抽一整块的随机数交给 C++，
而不改变随机流、checkpoint 中的 RNG 状态和物理指纹。

## 本轮规则（在 G1–G15 之上）

- **R1 等价性等级。** 每张卡写明等级，未经停止并报告不得降级：
  - **E0**：与替换前路径相比，`q`、`p`、时钟、`rng.bit_generator.state`、全部输出数组、
    checkpoint 内容（传播计数 `native_steps`/`python_steps`/`fallback_reasons` 除外）逐位一致。
  - **E1**：所有随机抽样与 RNG 状态逐位一致；依赖状态的投影（质心动量投影、fragment COM、
    固定原子）改在 C++ 中计算，`q`/`p` 与 Python 参考路径允许舍入级差异，必须给出容差和理由，
    容差不得宽于现有原生 RATTLE 对比测试。同一后端下"中断续算 = 不中断"仍须逐位一致（G5）。
  - **E2**：语义替换（新的随机流、分裂方式或热浴实现）。只能作为显式 opt-in，改变物理指纹，
    不改变任何已有输入的行为。
  - RPMD 轨迹是混沌的，E1 的舍入差异会在长时间后放大成宏观差异。因此 E1 测试分两层：
    短时程（≤ 200 步）逐帧比较 `q`/`p` 与容差；长时程比较统计量（PMF、κ(t)、FES、扩散系数等）
    在给定误差棒内一致。不得用长时程逐帧比较来"证明"或"否定"E1。
- **R2 随机数所有权。** E0/E1 路径上 NumPy `Generator` 仍是唯一随机源，C++ 不自行抽样。
  只预抽与状态无关的随机数，抽取顺序与旧路径完全一致（例如 geometric 调度：先 `normal`
  再 `geometric`）。块中途失败时，把 RNG 恢复到块起点，再按已提交的刷新次数重抽，
  或在每次刷新前保存 RNG 快照；不得出现"多消耗"或"少消耗"。凡依赖"整块抽 = 逐次抽"的卡，
  必须新增回归测试断言该性质，以便未来 NumPy 版本变化时立即暴露。
- **R3 能力标记。** 每个新的 C++ 能力导出一个整数模块属性（如 `MOMENTUM_REFRESH_API`），
  Python 先检查再使用。旧扩展：`auto` 带明确 `fallback_reason` 回退，`native` 在修改状态前报错。
  沿用 `DRPMD_OUTCOMES_API`、`BLOCK_ENERGY_CHECK_API` 的模式。
- **R4 后端一致。** `loop_backend="python"` 与无扩展（`PYRPMD_NO_NATIVE=1`）路径必须实现相同的调度
  语义，使 `auto`/`python`/`native` 在 E0/E1 意义下一致。
- **R5 执行参数不进指纹。** 块长、预抽内存预算等只进入 `execution_options`
  （`propagation.py` 的 `loop_options`，经 `block_propagation.py:856` 转出），不进入 `scientific_config`。checkpoint 格式不变；
  续算时更改块长不得改变结果。
- **R6 内存预算。** 预抽缓冲默认每块不超过 64 MiB（常量由 NP1 定义并导出，其他卡复用），
  超出时自动缩短块，而不是报错；与观测缓冲合计不得突破现有 256 MiB 上限。
- **R7 块内无回调。** 原生块内不调用 Python，释放 GIL。Python PES、ASE、Torch、i-PI、
  MPI bead 服务、自定义/子类化 CV/偏置/分类器保留原路径，不在本轮范围内。
- **R8 返回次数证据。** 每张接入卡都要加测试：在内置谐振势或 Morse 势上，用计数包装器统计
  原生 `advance`（或对应入口）调用次数，断言每 N 步的调用次数上界，并断言
  `propagation_stats["python_steps"] == 0`。
- **R9 性能声明。** 只报告实测数字，写明硬件与命令。真实体系用
  `benchmarks/h3/build_pes.py` 构建的 BKMP2；无法构建时写"未实测"。
- **R10 数值契约版本（落实 G15）。** 任何把已有运行从 E0 变成 E1 的接入，都是"带版本号的数值契约改动"：
  新运行在 checkpoint 中写入执行契约标记（例如 `native_path_contract: 1`，属于执行元数据，
  不进入物理指纹）；续算一个**没有**该标记的旧 checkpoint 时，继续使用旧调度，使旧运行逐位延续。
  E0 接入不需要契约标记。契约标记的读写由各接入卡在自己拥有的工作流文件中实现，并加旧 checkpoint 续算测试。

## 卡片总览

### 第 1 波：可立即开始，彼此可以并行

| 卡片 | 标题 | 规模 | 等级 | 主要独占文件 |
| --- | --- | --- | --- | --- |
| [NP0](NP0-scheduling-quick-wins.md) | 纯 Python 调度：再穿越子轨迹与 materials 分块 | S | E0 | `rate_recrossing.py`（子轨迹函数）、`materials.py`（旧 Andersen 循环） |
| [NP1](NP1-native-momentum-refresh.md) | C++：块内预抽 Andersen 动量刷新 | M | 无投影 E0，有投影 E1 | `native/propagation.h`、`block_propagation.py`、新 `momentum_refresh.py` |
| [NP2](NP2-native-sampling-block.md) | C++：PILE-L/TRPMD 随机采样块 | M | 无偏置 E0，有偏置 E1 | 新 `native/sampling_block.h`、新 `sampling_block.py` |
| [NP4](NP4-native-history-bias.md) | C++：元动力学/墙偏置的原生描述符 | M | E1 | `native/cv_core.hpp`（偏置部分）、`propagation.py`（`_biases`）、`bias.py`（追加描述符方法） |

### 第 2 波：前置卡合并后开始

| 卡片 | 标题 | 规模 | 等级 | 前置 |
| --- | --- | --- | --- | --- |
| [NP3](NP3-basin-stop-events.md) | 盆地停止事件与 committor/TIS 射击 | M | E1 | NP1 |
| [NP5](NP5-rate-chain-wiring.md) | 速率主链接入（伞形/PMF/FES、父链、thermalize、REUS、TRPMD 子轨迹） | L | E1 | NP0、NP1、NP2 |
| [NP6](NP6-materials-diffusion-route.md) | materials/diffusion 接入与原生系综 opt-in | M | E1 + E2 | NP0、NP1、NP2 |
| [NP7](NP7-history-bias-blocks.md) | 元动力学按 pace 分块 | M | E1 | NP1、NP4 |

### 第 3 波

| 卡片 | 标题 | 规模 | 前置 |
| --- | --- | --- | --- |
| [NP8](NP8-integration-benchmark.md) | 全链路返回次数审计、基准与文档整合 | S | 全部 |

## 依赖与冲突

- **与第四轮 T27 的重叠。** T27（未合并）独占 `native/cayley.cpp`、`native/propagation.h`、
  `native/fused.hpp`、`native/cv_core.hpp`（仅性能改动）。若 T27 仍未合并：NP1、NP4 只做功能追加，
  不重排、不改格式、不改已有函数的数学；后合并的一方 rebase。NP2 只在 `cayley.cpp` 的
  `methods[]` 表追加一行（共享注册点）。
- **`native/propagation.h`：** 第 1 波由 NP1 独占；NP3 在 NP1 合并后独占。
- **`native/cv_core.hpp`：** NP4 独占偏置部分（`struct Bias` 及新增偏置类型）；不改 CV 部分。
- **`biased_sampling.py`：** 函数级所有权。NP5 拥有 `_fixed_step` 和 `run_biased_sampling` 中的 `fused`
  判定；NP7 拥有 `_history_step` 及新增的历史偏置分块函数。两卡都在第 2 波，后合并者 rebase。
- **`rate_recrossing.py`：** NP0 拥有 `_child_task` 的 RPMD 分支（第 1 波）；NP5 拥有 `_parent_task` 循环和 `_child_task` 的 TRPMD 分支
  （第 2 波，在 NP0 合并后开始）。
- **`materials.py`：** NP0（第 1 波）后由 NP6 接手。
- **`dynamics.py`：** 函数级所有权。NP2 只允许把 `sample_step` 中的 PILE 系数计算抽成私有函数
  （不改数学和缓存键）；NP5 只修改 `thermalize`。其他改动写进"需要协调"。
- **共享注册点（只追加）：** `native/cayley.cpp` 的 `methods[]`；`docs/native-propagation.md`
  （每卡追加一个小节，NP8 统一整理）；`docs/conf.py` 的 `include_patterns`；
  `tests/` 新文件不受限。

## 派发提示词模板

把 `{…}` 换成卡片信息后发给对应 agent。

```text
你是 PyRPMD 仓库（chen-yu-hao/PyRPMD）的开发 agent，负责任务卡 {卡号}：{标题}。

开始前完整阅读以下三个文件，读完再动手：
1. docs/tasks/README.md —— 通用规则 G1–G15
2. docs/tasks/native-path/README.md —— 本轮规则 R1–R9、依赖与冲突
3. docs/tasks/native-path/{卡片文件名}

执行要求：
- 从最新 main 建分支 {建议分支}；一张卡一个分支、一个 PR。
- 只修改卡片"独占文件"、新建文件和登记的共享注册点。需要动其他文件时停下，
  写进 PR 的"需要协调"一节。
- 本卡的等价性等级是 {E0/E1/E2}。做不到时停止并报告，不要放宽容差或改测试来通过。
- 先读代码确认卡片中引用的行号和行为仍然成立；与卡片不符时，以代码为准并在 PR 中说明。
- 改动 native/ 后，开启与关闭 OpenMP 都要编译通过，PYRPMD_NO_NATIVE=1 路径可用。
- 运行卡片"验证命令"，把原始 pytest 汇总行写进 PR。本地无法编译或测试时停止并报告（G14）。
- 按 G11 写 PR 描述；不打 tag、不推 main、不发布。
```

集成者（合并本轮 PR 的人或 agent）使用：

```text
你负责集成 docs/tasks/native-path/ 第 {N} 波的 PR。按依赖顺序合并：{PR 列表}。
每合并一个 PR 后，在最新 main 上运行
  python -m pip install -e . --no-build-isolation
  python -m pytest -q -m "not campaign"
  OMP_NUM_THREADS=1 python -m pytest -q tests -k "native or fused or block or recross or drpmd or sampling"
任何失败都先回退该 PR 并在 PR 中说明，不在集成分支上修补他人代码。
处理各 PR 的"需集成者处理"一节；核对每个 PR 声明的等价性等级与其测试是否相符。
```
