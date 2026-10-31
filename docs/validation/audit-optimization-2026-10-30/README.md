# 实现审计后的修复与优化验证

## 版本与结论

基线为已经合并的 `main`：
`6fde29d2039c01921f00393bf17830b939af131b`。
正式测量时，候选实现的 Git 暂存树固定为
`899cc0618ba92db03e891c236845dac671704c00`；测量前后相同。
测量时，候选尚未创建提交或上传到远端，因此报告中两个 checkout 的 HEAD 都仍是
基线提交。两者实际实现由完整源码 SHA256、native 二进制 SHA256 和
工作区状态区分，不能仅根据 HEAD 判断版本相同。该批测量的首次归档只补充
说明和验证证据；归档时所有生产源码、harness 和 native 二进制指纹均核对一致。
该批实现和证据现保存在提交 `a7bb586b5b5fb5072401d3b5b962b0b41886cf3d`。
后续安装验证、证据包恢复步骤和类级方法覆写兼容性补充单独记录在
[发布准备验证](../optimization-release-2026-10-30/README.md)，不改写本次原始计时
及其版本身份。

本批实现完成了表面速率不确定度修复、CV/MBAR/NRPMD 计算简化、检查点和
进程线程控制优化，以及 C++ PILE-L 半步–OU–半步核、正常模 FFT 和 FES
多观测块路径。最终本地完整测试 **1386 passed、3 skipped、153 subtests passed**；
四套实际四进程 MPI 检查均通过。八组性能对照完成五轮，全部通过科学数据核对。
这些检查验证实现和数值一致性，不证明化学速率、自由能或长时间采样已收敛。

## 实际改动

### 统计正确性与分析开销

- 表面 PMF 的相关块策略统一到一般 FES/气相 PMF 的 `ceil(5*g)` 请求与
  可用长度限制。`bootstrap.block_plan.status="unavailable"` 时，QTST 和主联合
  速率的标准误、置信区间留空。点估计、原始 draw 和配对协方差诊断仍可保留，
  保留诊断不等于误差可用。`limited_blocks` 保留有限块数的限定说明。
- 此修复会有意改变旧表面 PMF 的 bootstrap draws 和不确定度，不能要求与
  旧 `ceil(g)` 结果逐位一致；PMF/QTST 点估计不因块策略变化而改变。
- 圆形块重采样用广播构造索引，保持 RNG 调用和样本顺序；FES bootstrap
  按相同索引复用原 bias 矩阵列。已有一般 FES 的科学输出在本次基准中逐位相同。
- `solve_mbar` 的公共默认仍返回完整 `state_weights`；不消费该输出的内部调用
  使用 `return_state_weights=False`，省去扩展到原状态列表的冗余矩阵。
  计算 overlap 所需的唯一状态权重仍保留。

### Python 数值路径与 I/O

- COM、RPMDrateCoordinate、RateSurface 的普通值/梯度路径避免构造不用的
  Hessian。约束需要的 Hessian 和显式 `surfaces()` 接口仍可用。候选分支的
  校验、首个并列分支选择、自定义子类/实例/类覆写与嵌套 CV 行为有回归覆盖。
- NRPMD 的密度基变换改为两次批量矩阵乘法，收缩复杂度从
  `O(P*S^4)` 降到 `O(P*S^3)`；MInt 的 eigensolve、冲量和 mapping 演化保留。
- 检查点验证后复用已解压数组；CRC、SHA256、kind/schema、完整库存与
  `allow_pickle=False` 校验均保留。每次公开读取仍得到独立可写副本，缓存只在
  上下文内存活。代价是上下文打开时持有完整未压缩 payload；没有改成长历史分块存储。
- 材料采样避免状态未改变时重复写终点检查点，已完成任务恢复可直接重建输出。
  同版本的 q/p、时钟、RNG 与统计量保持原恢复契约。
- worker 用 `threadpoolctl` 限制已经加载的 BLAS 线程池，环境变量同时约束后续
  加载；任务作用域结束或异常时恢复。bead worker 在初始化时建立持久限制，
  不在每次 PES 请求中扫描。OpenMP 动力学线程和 MPI 路径不被重新配置。

### C++ 路径

- `sample_free` 在一次 Fourier 空间访问中融合自由环半步、OU 热浴、第二个
  半步。auto 仅对未覆写的内置实现以及 `P<=8` 或二次幂珠数启用；较大非二次幂
  使用 NumPy，避免直接 DFT 退化。NumPy 仍按原序列产生噪声，Python 仍执行两次
  势能力调用；该路径没有宣称整个采样长循环都在 C++ 中。
- 正常模确定性自由环对二次幂 `P>=32` 使用不可变 radix-2 FFT plan，复杂度
  从 `O(N*P^2)` 降到 `O(N*P*log(P))`。小环和非二次幂保留原 circulant 路径。
  默认 cyclic Cayley 的 `O(N*P)` 路径保持原算法。
- FES 及共享其采样器的 PMF 在支持 native 时使用多观测 `propagate_block`，
  保留采样 mask、连续 Andersen 时钟和 checkpoint 边界，块长不超过 4096。
  `check_energy=True` 检查初态和每一步，包括不保存观测的步；不会用较少检查
  换取速度。失败只提交成功前缀；Andersen 重抽后的首步失败恢复重抽前 p/RNG，
  避免恢复后重复消费随机数。自定义 Python 势的成功调用节奏保持兼容。

## 五轮性能结果

基线与候选使用同一 Python 环境、同一个真实且有固定来源的 BKMP2 动态库，
绑定同一逻辑 CPU 0，OMP/BLAS/MKL 等线程设置为 1。每轮每个版本使用全新
进程，先完整预热，再执行相同工作量；顺序交替为 B→C、C→B。计时不包含
模块导入、输入生成或结果核对，不使用 profiler。下表时间为五轮中位数，
加速比为基线中位时间除以候选中位时间。

| 工作量与计时范围 | 基线（ms） | 候选（ms） | 加速比 | 解释 |
| --- | ---: | ---: | ---: | --- |
| 真实 BKMP2 dRPMD，8 轨迹、16 珠子、10263 个 child 步；完整工作流 | 355.308 | 354.849 | 1.001× | 基本不变；不认定为有效提速 |
| 真实 BKMP2 FES，4 窗、4160 步、800 样本；完整工作流 | 237.176 | 233.891 | 1.014× | 变化落在重复波动内；不认定为有效提速 |
| harmonic PILE-L，32 原子 × 32 珠子、1000 步；完整材料工作流 | 669.563 | 481.258 | 1.391× | 中位耗时下降约 28.1% |
| COM 梯度，3 原子、4000 次调用；局部 API | 48.930 | 22.854 | 2.141× | 值和梯度逐位相同 |
| COM 梯度，1000 原子、16 次调用；局部 API | 702.900 | 0.331 | 约 2121× | 避免不需要的 Hessian；候选批次很短，不能外推完整模拟 |
| 实际 MInt，32 珠子、16 态、32 次调用；局部路径 | 99.277 | 38.259 | 2.595× | 含 eigensolve、密度变换、冲量和 mapping 旋转 |
| FES 重建，3900 样本、3 streams、6 次 bootstrap；完整分析函数 | 60.752 | 40.856 | 1.487× | 周期轴、重复 umbrella、不同长度 stream；输出逐位相同 |
| 正常模自由环，32 原子 × 256 珠子、32 次 free step；局部 C++ 核 | 467.920 | 29.760 | 15.723× | 不含 PES、热浴或完整动力学工作流 |

1000 原子 COM 的候选五次批次时间为 0.330–0.374 ms，均短于 5 ms；原始
结果明确标记 `candidate_batch_shorter_than_5ms=true`。巨大局部差异来自去掉
不需要的二阶导数工作，不能表述为整个 PyRPMD 提速数千倍。
PILE-L 的同轮加速比为 1.30–1.73×，MInt 为 1.97–3.22×，正常模核为
12.50–17.24×；报告保留全部波动，没有只挑最快一次。

环境为 Linux x86-64、Python 3.12.14、NumPy 2.5.4、SciPy 1.18.1、h5py 3.16.0、
threadpoolctl 3.6.0，native 扩展启用 OpenMP。不同机器、PES 成本、珠数、原子数
和采样/输出频率会改变实际收益；本次没有测 GPU 或多节点性能。

## 数值核对与兼容性

每轮都比较工作计数、离散决策、时钟、RNG、保存的科学数组和末态。每版本
预热与正式运行、五轮重复之间的科学数据 digest 也必须相同。相同舍入顺序的
dRPMD、COM、一般 FES bootstrap 数据逐位相同；BKMP2 FES 最大记录差异
为 `1.78e-15`。PILE-L 的末态 q/p 最大差异分别约 `6.22e-14`、`9.09e-14`，
保存的估计量最大差异约 `1.86e-10`，全部满足预先写入 harness 的容差；
RNG 和时钟严格相同。MInt 核动量最大差异约 `7.77e-16`，mapping 变量逐位相同。

正常模核分别与未计时的 Python/NumPy 参考实现比较。旧 dense 基线的 q/p
最大误差约 `5.10e-13` / `7.68e-10`；候选约 `1.72e-15` / `2.10e-12`。
相对自由环能量漂移分别约 `3.735e-12`、`8.88e-16`，参考自身为 `1.44e-15`。
固定旧基线采用明确记录的 `5e-12` 能量界限；候选和参考保留 `2e-12` 界限。
q/p 使用既有大环回归的界限，未用更宽的候选门限换取通过。

新增回归还覆盖独立模式公式、任意 cosine/sine 阻尼输入、噪声协方差、
固定原子、时间反演、能量守恒、失败重试、同版本精确恢复、跨对象绑定的方法
覆写、陈旧扩展 marker 以及 auto/native/Python 选择。FFT 的求和顺序不同，
跨后端或跨版本轨迹允许舍入差异，不承诺混沌长轨迹逐位相同。

## 验证记录

| 检查 | 结果 | 文件 |
| --- | --- | --- |
| 最终完整 pytest | 1386 passed；3 skipped；153 subtests passed；88.96 s | `pytest-final.txt` |
| 原生采样/大环专门回归 | 132 passed | `pytest-native-final.txt` |
| 实际四进程 MPI 调度/父链/恢复 | 通过 | `mpi-integration.txt` |
| 实际四进程 FES + rate | 通过 | `mpi-workflow.txt` |
| 实际四进程 PIMD + diffusion + site-rate | 通过 | `mpi-materials.txt` |
| 实际四进程表面工作流 | 通过；串行/MPI 的 27 个数组及检查点文件逐值一致 | `mpi-surface.txt` |
| 五轮八组性能/科学数据对照 | 全部通过 | `benchmark-report.json` |

三个跳过项分别为未配置外部固定来源的 NiAlH EAM 势文件（1 项）和未安装
可选 PyTorch runtime（2 项）。103 条 warning 来自已安装 ASE 使用的 NumPy
shape 设置弃用接口。没有把这些跳过项当作已通过的真实势或 PyTorch 验证。

最初受限沙箱全套运行有两个 i-PI 本机 TCP socket 测试被环境拒绝；
其日志保留为 `pytest.txt`（仅移除行尾空格；交付包另保留原始副本）。
获得本机 socket 执行能力后，这两个测试单独通过
（`pytest-socket.txt`），最终最新树完整运行也全部通过。最终结果不是把不同
轮次测试数量拼接得到的。四套 MPI 的命令、rank 数、返回码在 `mpi-results.json`。
MPI 的设备路由测试使用 mock，未验证实际 CUDA 设备。

## 复现

基准详细方法与全部 workload 位于
[`benchmarks/efficiency/README.md`](../../../benchmarks/efficiency/README.md)，
入口为 [`audit_optimization.py`](../../../benchmarks/efficiency/audit_optimization.py)。
必须为基线和候选分别构建扩展，不能让候选 `.so` 覆盖基线。

```bash
python benchmarks/efficiency/audit_optimization.py \
  --baseline /absolute/path/to/perf-baseline \
  --candidate /absolute/path/to/PyRPMD \
  --library /absolute/path/to/libbkmp2.so \
  --output /absolute/path/to/new-measurement-output \
  --repeats 5

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -m pytest -q -ra
```

本次 MPI 使用 MPICH、4 ranks、`MPIR_CVAR_CH4_CMA_ENABLE=0`、`UCX_TLS=tcp,self`，
以及同样的单线程环境；运行 `tests/mpi_integration.py`、
`tests/mpi_workflow_smoke.py`、`tests/mpi_materials_smoke.py` 和
`tests/mpi_surface_smoke.py`。后三个脚本需要各自新的输出目录。

[`benchmark-summary.json`](benchmark-summary.json) 便于读取原始时间和汇总；
[`benchmark-report.json`](benchmark-report.json) 保存所有重复、数值误差、输入规模、
源码/native/PES 指纹和环境。完整测量输出另以交付证据包保留，包含 1041 个文件、
42,300,624 字节的 worker 记录、科学数组、trajectory/checkpoint 和日志。
完整源码补丁以此基线为起点，包含实现、测试、文档及可复现 harness。

## 后续边界

长历史 checkpoint 的分块存储、周期势/压力 virial 的原生 ABI 与完整 PILE-L
多步循环仍是独立后续工作；本次没有暗中切换文件格式、物理积分器或外部
PES 契约。进一步迁移到 C++ 应先用相应实际工作流确认瓶颈和收益。
