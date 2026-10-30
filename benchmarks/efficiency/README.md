# 实现优化的基线与候选对照

`audit_optimization.py` 比较两个 checkout 的实际实现，覆盖生产工作流与局部
数值操作。它保留原始重复测量、源码和二进制指纹，以及用于核对科学结果的
数组。这些小工作量用于评估实现效率，不能证明自由能、速率或热力学采样已经
收敛，也不代表多节点或 GPU 性能。

## 准备与运行

需要同一个 Python 环境提供 NumPy、SciPy、h5py 和 threadpoolctl；两个 checkout
都必须已经拥有各自可导入的 `pyrpmd._native` 扩展。脚本不会自动构建、替换扩展
或修改任一 checkout。不要让候选版本的构建覆盖已经保留的基线扩展。

BKMP2 使用同一个绝对路径的真实 `libbkmp2.so`，以及相邻的
`libbkmp2.build.json`。构建方法见 `../h3/README.md`。脚本检查二进制与构建
provenance 中的 SHA256 一致；两个版本的实际势能适配器还会检查已固定的
BKMP2 源码指纹。

先运行缩小工作量的功能检查：

```bash
python benchmarks/efficiency/audit_optimization.py \
  --baseline /absolute/path/to/perf-baseline \
  --candidate /absolute/path/to/PyRPMD \
  --library /absolute/path/to/libbkmp2.so \
  --output /absolute/path/to/new-smoke-output \
  --smoke
```

`--smoke` 仍然启动基线、候选两个独立进程，验证导入隔离、工作量及数值结果，
但不调用计时器、不产生中位时间或加速比。它不应作为性能证据。正式对照使用：

```bash
python benchmarks/efficiency/audit_optimization.py \
  --baseline /absolute/path/to/perf-baseline \
  --candidate /absolute/path/to/PyRPMD \
  --library /absolute/path/to/libbkmp2.so \
  --output /absolute/path/to/new-measurement-output \
  --repeats 5
```

输出目录必须不存在，防止把已有 trajectory/checkpoint 恢复当作一次新的性能
测量。所有预热和正式生产运行都有独立目录，不覆盖旧审计结果。默认要求基线
HEAD 为 `6fde29d2039c01921f00393bf17830b939af131b`；评估另一个基线时，必须
通过 `--baseline-commit FULL_COMMIT_HASH` 明确指定，报告仍会记录实际文件内容。

默认选取当前允许 CPU 集合中的最小编号，并绑定每个 worker 到同一个逻辑 CPU。
可以用 `--cpu ID` 明确选择；正式计时要求操作系统支持 CPU affinity。
OMP、OpenBLAS、MKL、BLIS、Accelerate 和 NumExpr 的线程环境变量均设为 1，
并检查 threadpoolctl 可识别的已加载 BLAS 池确实为单线程。运行正式测量前应
停止同一机器上的构建、其他测试和 CPU 基准，以减少竞争。

## 工作量

默认生产配置保持原实现审计的三个代表性工作量。局部操作以相同输入和相同
调用次数比较两个 checkout，避免把减少科学采样量算成加速。

| case | 正式工作量 | 计时范围 |
| --- | --- | --- |
| `bkmp2_drpmd` | 真实 BKMP2，16 珠子，8 轨迹、2 parent；最大 200 fs；每步分类；每 10 步记录轨迹、每 500 步 checkpoint | 实际 `run_drpmd`，含初始化、热化、碰撞轨迹、I/O 和分析 |
| `bkmp2_fes` | 真实 BKMP2，16 珠子，4 个二维 umbrella；每窗 40 步平衡 + 1000 步生产；每 5 步采样；bootstrap=0 | 实际 `run_fes`，含采样、checkpoint、NPZ/HDF5 输出和 MBAR 分析 |
| `pimd_pile_l` | harmonic，32 原子 × 32 珠子；200 步平衡 + 800 步生产；每 10 步采样，1 replica | 实际 `run_materials`，含 PILE-L、checkpoint/trajectory 输出和估计量分析；`auto` 后端 |
| `com_gradient_n3` | 固定 3 原子配置，4000 次调用 | 实际 `COMDistance.centroid_value_gradient` |
| `com_gradient_n1000` | 固定 1000 原子配置，16 次调用 | 同一实际梯度 API，包含基线中原本的 Hessian 构造成本 |
| `nrpmd_mint_s16` | 32 珠子、16 电子态、2 原子，32 次固定 q 的 MInt 流 | 实际 `_mint`，含 eigensolve、密度变换收缩、核冲量、mapping 旋转和相同的逐次状态记录 |
| `fes_bootstrap` | 3 个二维 stream，样本数 `[1500,1300,1100]`；一个周期轴、重复 umbrella state、不同刚度；6 次 block bootstrap | 完整 `reconstruct_fes`，含 MBAR、相关性估计、重采样、直方图和不确定度 |
| `normal_mode_n32_p256` | 32 原子、256 珠子；固定随机 q/p、不等质量；32 次 `normal-mode-bab` 自由环推进 | 实际 `CayleyIntegrator(native=True).free_step`；不包含 PES、热浴、参考推进或能量核对 |

`nrpmd_mint_s16` 测量实际 MInt 局部路径，因此其时间同时包含其他 MInt 工作；
不能把这个加速比写成完整 NRPMD 轨迹或单独矩阵乘法的加速比。COM 的局部
加速比也不能直接用于推算总工作流加速。

`normal_mode_n32_p256` 是局部正常模自由环核。计时之外用已有
`CayleyIntegrator(native=False, scheme="normal-mode-bab")` 推进同一初态，独立
核对末态；没有用自写的 FFT 原型替换基线或候选。自由环能量为真实质量下的
动能与珠间弹簧能之和，候选和独立参考的相对能量漂移必须小于 `2e-12`。
固定旧基线的 dense 变换在这组 32 步输入上有约 `3.74e-12` 的累积能量舍入
漂移，因此单独给基线 `5e-12` 的界限，双方能量比较也使用 `5e-12`；候选的
独立能量门限保持 `2e-12`。每次运行均保留两者对参考的 q/p 误差和能量漂移，
不能用基线的较宽界限掩盖候选变差。
q 的比较界为 `rtol=2e-11, atol=2e-11`，p 为 `rtol=3e-11, atol=2e-9`，
采用 `tests/test_native_sampling.py` 中 40 步大环测试的既定界限；这里固定为
32 步，smoke 为 2 步。报告记录初态、质量、双方末态和参考末态、能量及误差。

可以用 `--cases bkmp2_fes pimd_pile_l` 等只选择需要的 case。每次报告包含
完整规模、随机种子、采样周期和工作计数；smoke 的缩小规模也会明确记录。
默认规模面向几十秒量级的单机对照，实际耗时依赖机器、文件系统和构建。
1000 原子的 COM 候选批次可能非常短；报告中的
`candidate_batch_shorter_than_5ms` 会标记此情况，应结合五次原始测量查看波动。

## 隔离、预热和重复

每个 checkout、每轮重复都使用全新 Python 子进程，并把该 checkout 的
`src` 和根目录写入独立的 `PYTHONPATH`。脚本断言 Python 包、native 扩展、
H3 helper 和动态 Python PES/classifier 插件确实来自所选 checkout，避免因
editable install 或模块缓存误测另一份实现。

正式计时默认进行五轮：第 0、2、4 轮为 baseline→candidate，第 1、3 轮
为 candidate→baseline。每个新进程中，每个 case 先完整预热一次，再从同一
初态在新目录执行正式工作量。预热与正式结果的科学数据 digest 必须相同；
同一版本在不同重复中的 digest 也必须相同。科学输入的生成、模块导入、配置
构建、数组提取和验证、报告序列化均在计时区间之外。

报告记录每次的 wall time、中位时间，以及相同轮次的 paired speedup。
最终 `speedup` 是基线中位时间除以候选中位时间。所有计时均来自没有
cProfile/tracemalloc 的运行；不得把单独 profile 的累计时间求和后代替它。

## 科学数据核对

结果数相等只是工作量核对的一部分。每轮基线与候选还必须通过以下检查：

- **dRPMD：** 每条轨迹的 status、channel、reason、停止步和时钟；全部发射
  q/p 与 thermal q/p；保存的 centroid、能量、时间；末态 q/p；每个 parent
  的末态、热化时钟和 RNG。
- **FES：** 每个记录下来的 CV 向量及对应能量、末态 q/p、window/replica
  身份、checkpoint step/time/RNG；每窗样本数与最终输出一致。再比较 FES、
  权重、重叠、mask 和其他分析数组。未保存的中间完整珠子坐标不在核对范围内。
- **PILE-L：** 全部保存的 centroid 位置、速度、量子估计量、回转半径、采样
  步和时间，以及完整末态 q/p；RNG 和终态时钟必须严格相同。
- **局部数值操作：** 输入数组严格相同，COM 值/梯度与逐次 MInt 核动量和
  mapping 变量在明确容差内一致。MInt 的 q 和时钟必须保持预期不变。
- **正常模自由环：** 初始 q/p、质量和调用次数严格相同；末态 q/p、初始与
  最终自由环能量在上述既定界限内一致，并分别通过未计时的 Python 参考核对。
  `free_step` 本身只更新 q/p，不推进 `State.step/time`，该时钟行为严格检查。
- **bootstrap FES：** 输入、窗口规模、每次 bootstrap 的有效性与拒绝原因、
  block 计划、有效 draw 数一致；自由能、标准误、权重、重叠、求解残差等
  数值数组在明确容差内一致；mask、计数和 sample clock 使用严格比较。

每个数值数组记录 shape、非有限值数量、是否逐位相同以及最大绝对误差。
浮点比较允许 NaN 出现在相同的掩蔽位置；不能把 NaN 随意填零后比较。使用的
绝对/相对容差保存在报告中，PILE-L 的不同变换次序允许小浮点差异。离散
决策、RNG、时钟、整型与布尔数组不适用这个放宽。

这些是**本次固定工作量的数值一致性检查**，并不替代独立物理验证、长期
稳定性或采样收敛研究。状态记录或工作量不匹配会使整次对照失败，而不是
继续报告一个表面上更好的加速比。

## 输出与可追溯性

顶层 `report.json` 包含所有原始重复、每轮比较和汇总，只有所有核对通过后
才标记 `passed: true`。异常时保留 `passed: false`、失败原因和已有证据。
每个 worker 目录还有自己的 `result.json`，每个 case 有无 pickle 的
`arrays.npz`，生产 case 保留正常运行产生的 checkpoint 和其他输出。

manifest 包含：

- git commit、工作区状态，以及所有 `src/pyrpmd` Python 源文件、native
  C/C++ 源文件和头文件、H3 helper/输入与构建入口的 SHA256；
- harness 自身和实际导入的 native 扩展 SHA256、扩展位置和能力标记；
- 相同 BKMP2 库的绝对位置、二进制/源代码/构建 provenance SHA256、编译器和 flags；
- Python、PyRPMD、NumPy、SciPy、h5py、threadpoolctl 版本，CPU affinity、
  线程环境与已加载线程池信息。

这些指纹在每个 worker 前后以及各轮之间检查，检测测量期间发生的源码或
二进制替换。工作区可以包含尚未提交的候选修改；因此只记录 git commit 不够，
实际源码和扩展指纹才是本次测量对象的身份。请把整个输出目录与对应源码
一起保留，不能只截取一个加速比。
