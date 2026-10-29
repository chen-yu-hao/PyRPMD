# PyRPMD 0.1.0 验证记录

本记录对应初始实现的本地验证，日期 2026-10-29。程序测试通过不等于某一
化学体系的 RPMD 速率已经收敛；示例使用解析教学势或自由散射势。

## 完整测试

格式化后的最终源码运行结果：**70 tests passed，69 subtests passed**。
包括可选 ASE、Torch CPU 和本机 i-PI socket 测试，未因依赖缺失跳过这些路径。
环境为 Linux x86_64、Python 3.12.14、GNU C/C++、NumPy 2.5.4、SciPy 1.17.0、
h5py 3.16.0、mpi4py 4.1.2、MPICH 5.0.2、Torch 2.15.0+cpu、ASE 3.29.0。

| 验证项 | 实际执行内容 |
| --- | --- |
| Cayley | 独立 dense midpoint 解对照；P=1/2/3/5/8/32；零/正/负步长；反演；P=1 velocity Verlet；能量误差 |
| OpenMP | 原生 C++ 构建/加载；单线程与两线程一致；并发独立 context |
| 采样 | mP/beta 动量方差、centroid 方差、BCOCB harmonic 配置统计、持续母轨迹时钟、每次力计算的 fragment COM 约束 |
| 势能 | 解析力差分、shape/status/error、单位转换、梯度符号 |
| C `.so`/`.o` | 真正编译 C 共享库、对象文件和 manifest 链接，实际调用及 CLI force check |
| RPMDrate adapter | legacy Python fixture 的 f2py 数组布局、初始化和梯度协议；未编译真实 Fortran PES |
| Torch | 实际 CPU TorchScript/autograd、batch、eV/angstrom 转换、显式 forces、detach/低精度错误、无 CUDA 时明确拒绝 |
| ASE | 实际 Lennard-Jones calculator，batch/scalar、解析能量/力、AU 转换、有限差分 |
| i-PI | 真实本机 TCP loopback 驱动端，完整初始化/坐标/力协议 |
| CV 与偏置 | centroid/bead-average 完整 Jacobian、xi Hessian、flux metric 力差分、P*W 因子 |
| FES | 已知二维 harmonic 分布的 MBAR、非均匀 bin 体积、overlap、断连 mask、自相关、实际采样及恢复 |
| 标准 rate | 非线性 Cayley-RATTLE 约束与反演、±p child 初始 flux、实际完整 PMF/parents/children 流程 |
| dRPMD | Gamma(2) 与面积分布、内模保留、总数余数分配、双层并行、失败/未决界、模型文件变更、运行中断和终态中断恢复 |
| 输入/CLI | `.inp` 与 Python、不同 cwd 的插件路径、单位/同位素/整数拒绝、GPU 进程数解析、run/resume/analyze、C PES build/check |

Torch 对 TorchScript API 发出上游弃用提示；ASE 对 NumPy 2.5 的内部 shape
赋值发出弃用提示。测试仍通过。这些提示没有被静默抑制。

## 完整用户工作流

- `examples/drpmd/free_scattering.inp` 经 CLI 运行和恢复：5/5 nonreactive；
  两条母链累计热化步数 35、30，分别等于 `20+3*5` 和 `20+2*5`。
- Python PES、轨迹+珠子模式、最小自动分类 pilot 均完成实际运行。
  最小 pilot 明确标记分类未验证并保留 null 化学估计；默认短传播可产生 unresolved。
- `examples/rate/pilot.inp` 真正从 TS-only 准备进入 22 个窗口和 32 条 child：
  成功识别一个负模，完成 transmission 估计，恢复后 κ 相同；始终 `converged=false`。
  它不是物理 H3 势能面，也不是 H+H2 化学速率基准。
- 二维 harmonic `.inp` 和文件工厂自定义二维 CV，从输入目录之外启动：
  各完成 3600 个保存样本。
- CLI 二维 FES 完成、恢复与 `analyze --recompute --bins 12 12 --bootstrap 3`
  均通过；重新分析没有传播新轨迹。

## 真实 MPI 与硬件边界

同一台机器上使用 **4 个真实 MPI rank，G=2、B=2** 执行：

- 有序独立任务与珠子势能集合通信；
- PES 初始化错误向全部 rank 传播；
- 持续 dRPMD 母轨迹、准确总数及无重复恢复；
- FES 窗口与标准 rate 多阶段工作流及其恢复。

测试入口为 `tests/mpi_integration.py` 和
`tests/mpi_workflow_smoke.py /fresh/shared/output`。受执行环境限制，测试设置
`UCX_TLS=tcp,self` 与 `MPIR_CVAR_CH4_CMA_ENABLE=0`；这是本机通信设置，
不是在程序中硬编码的集群要求。

MPI GPU 节点本地 rank 到设备的分配使用 mock provider 验证。**没有真实
CUDA 硬件测试，没有多物理节点性能测试，没有具体 MACE/DeepMD/xTB 模型测试。**
随附 Slurm 模板经过 shell 语法检查，仍需站点填写分区、模块和模型路径。

## 分发验证

已通过 setuptools/PEP 517 构建 Linux x86_64、CPython 3.12 原生 wheel。
将 wheel 安装到独立目录后，离开源码目录调用安装的 `pyrpmd` 命令：
`info` 正确报告 native/OpenMP 可用，并实际完成上述 5 条双母链 dRPMD 任务。
wheel 包含 C ABI header、C 示例、MIT 与第三方通知。

源代码发行包与 GitHub Actions 定义随源码提供。Actions 的远程运行状态应以
GitHub 页面为准；此文档记录本地实际执行结果，不把尚未运行的远程 CI 当成通过。

## 后续真实 H₃ 体系与远程验证

上文保留初始实现的历史验证范围。后续已增加 BKMP2 H + H₂ 的真实 dRPMD、
二维 FES、伞抽样 PMF 和 recrossing；实际运行参数、数值、图表、GitHub 作业
与测试提交见 [2026-10-29 真实体系记录](validation/h3-bkmp2-2026-10-29/README.md)。
该记录明确区分数值门槛通过、PMF 估计器差异与尚未建立的物理收敛。
