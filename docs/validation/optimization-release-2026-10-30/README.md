# 优化批次的发布准备与兼容性补充

本记录补充 [原实现优化验证](../audit-optimization-2026-10-30/README.md)。
原八组五轮计时、原始数组与证据归档保留不变；其首批提交为
`a7bb586b5b5fb5072401d3b5b962b0b41886cf3d`，tree 为
`b790fbc0fb651ebd95f12050ff8ff344a50a3d66`。
本轮增加保守的原生分派兼容性检查、安装包验收与可执行的离线恢复补充。
最终构建和测试源码快照为 `ae5ac945d23413cf05ddfa9499b718555ff22898`；
随后只添加说明与证据，不改变该快照中的生产源码或测试脚本。

## 类级 Python 方法覆写

检查发现，普通两套原生传播接口已经排除自定义子类和实例方法覆写，但
类级方法替换仍可能被融合传播绕过。例如替换 `Dynamics._force`、
`CayleyIntegrator.free_step`、内置 CV/bias 的计算方法或 provider 的
`evaluate` 后，auto 仍可能按原内置公式推进。

修复把原方法身份保存在定义模块中，再由共享 guard 检查实例和类级覆写。
保存身份的时机很关键：若等 `propagation` 首次惰性导入后才记录，用户提前
替换的方法会被错误地当成原方法。嵌套 CV、CompositeBias、Python wrapper
与内层 provider 的检查均保留。

- `auto` 执行兼容的 Python 路径，实际调用用户覆写的数学方法。
- 严格 `native` 在推进前明确拒绝，q/p/clock 不变。
- 未修改的 CABI、Python wrapper 和 Fortran adapter 继续走 native。
- 仅处理受支持方法被原生 descriptor 跳过的情况；外部 provider 显式导出
  native binding 的契约保留。没有声称检测任意模块级全局函数的猴补丁。

新 `tests/test_native_override_compatibility.py` 实际改变力、能量、CV 或传播，
并核对 q/p、观测、能量、时钟与调用次数。覆盖 13 种覆写 × 两种 API ×
auto/strict 的 52 项、6 项原生正向对照，以及 6 项独立进程的首次导入顺序。

| 阶段 | 结果 | 证据 |
| --- | --- | --- |
| 修复前的新回归 | 58 failed、6 passed | `override-before.txt` |
| 修复后的新回归 | 64 passed | `override-after.txt` |
| 直接受影响的旧集合 | 383 passed、3 subtests、2 skipped；另 1 项本机 socket 被受限环境拒绝 | `override-targeted.txt` |
| 具备本机 socket 能力后的最终全套 | **1450 passed、3 skipped、153 subtests passed；92.10 s** | `pytest-final.txt` |

最终全套没有失败。3 个跳过项是未配置外部固定来源 EAM 势（1 项）和未安装
可选 PyTorch runtime（2 项）。103 条 warning 来自 ASE 与 NumPy 的 shape
设置弃用接口。回归文本的归档副本只去掉行尾空格；原始日志 SHA256 及转换
说明在 `override-log-provenance.json`，原始文件也保留在交付证据中。

## 独立源码包与 wheel 安装

在不含原工作区 `.so` 的隔离源码目录中运行 PEP 517
`python -m build --no-isolation`，先生成 sdist，再从该 sdist 构建平台 wheel。
随后安装到独立 Python 前缀，并从源码目录外运行 sdist 自带的
`tests/package_smoke.py`。

验收确认：

- sdist 包含 `ring_fft.h`、新增原生回归、兼容性回归及效率 benchmark。
- wheel 中所有 Python 源文件与冻结快照逐字节相同，包含真实编译扩展。
- METADATA 明确声明新增运行依赖 `threadpoolctl>=3.1`。
- 导入来自独立安装前缀，没有使用原 checkout 的 editable 包。
- 新 `SAMPLING_FREE_API`、`BLOCK_ENERGY_CHECK_API` 均为 1。
- 两种 32 珠子 PILE-L 方法与参考实现保持相同 RNG 和时钟，q 最大差约
  `6.66e-16`、p 最大差不超过 `6.93e-14`。
- 64 珠子正常模自由环 q/p 最大差约 `3.33e-16` / `1.38e-14`。
- network、adaptive FES、两种 metadynamics、PCET 以及两种积分器的 dRPMD
  示例均完成，并在完成后恢复时保持结果一致。

本地安装复用了预装 NumPy/SciPy/h5py/threadpoolctl 的逐包链接，项目本身是
独立安装；没有将本地检查说成全新联网解析所有依赖。GitHub 的安装矩阵
保留各平台新环境的独立安装流程，并新增上述原生能力与 sampler/FFT 检查。

`final-package-source.json` 保存冻结源码指纹；`final-installation.json` 保存
sdist/wheel SHA256 和内容检查；`final-installed-smoke.json` 保存完整安装
验证结果。`initial-*` 文件保留补充兼容性修复前、首批实现的独立安装记录。

## 无 OpenMP 构建

从首批 `b790fbc…` 源码快照单独执行 `PYRPMD_NO_OPENMP=1` 的 setuptools
构建，没有覆盖原来参与性能测量的扩展。串行能力、两个新 API marker 和
拒绝 `threads>1` 的行为均得到检查；动态依赖中没有 OpenMP runtime。

新 sampler/大环数值测试和每步能量检查相关回归为 **132 passed、2 skipped**；
两项跳过都是要求 OpenMP 的线程一致性测试。源码和构建命令见
`serial-build.json`，结果见 `serial-report.txt`、`serial-pytest.txt` 和
`serial-capabilities.json`。本机没有 CMake，因此没有实跑可选 CMake 路线。
后续兼容性补充只改变 Python 分派，原 C++ 源码及测量用 `.so` 均保持原指纹。

## 默认原生路径的五轮复测

类级身份 guard 会参与 BKMP2 dRPMD/FES 的默认分派，因此对这两套原工作量
单独复测。复用原 harness、CPU 0、单线程 BLAS、原基线与同一 BKMP2/PES、
两份 native 二进制，五轮交替新进程、预热和数值核对。全部通过。

| 工作量 | 原基线中位时间 | 最终候选中位时间 | 比值 |
| --- | ---: | ---: | ---: |
| BKMP2 dRPMD，8 trajectories、10263 child steps | 0.328088 s | 0.298730 s | 1.098× |
| BKMP2 FES，4 streams、4160 steps、800 samples | 0.268765 s | 0.199627 s | 1.346× |

两个版本的上述生产步数全部是实际 native steps。dRPMD 的 84 个数组字段
逐位相同；FES 的 41 个数组字段通过核对，仅 stream 能量最大差
`1.7763568394002505e-15`，其他数组逐位相同。RNG、时钟、离散决策相同。
各版本的科学 digest 也与原八组报告一致。

比较对象仍是 `6fde29d…` 原基线与最终候选，不能把此处时间比值归因于身份
guard，也不能单独量化其增量成本。本轮相对原基线未观察到性能退化；
初轮完整化学工作流的结果接近不变，短工作量的收益幅度会受运行波动影响。
原八组计时不会被这一轮较好的数值替换。全部原始重复、输入、数组误差与
源码身份见 `dispatch-benchmark-report.json`。

## 原证据包的离线恢复补充

原 source tar 不含 `.git`，原 harness 又要求真实 baseline HEAD，导致原
恢复说明不足以直接启动基准。独立补充包加入了仅含 `6fde29d…` 及其祖先的
42 提交 Git bundle、恢复脚本与精确校验。它不包含其他 refs 或本地配置。

隔离恢复核对了原归档的 1041 个 raw 文件、完整 baseline/candidate 源树、
原补丁应用、源码/native/PES 指纹和各自导入位置。原 harness 的全部 8 个
`--smoke` case 通过，未采集计时。原正式证据包 SHA256 仍为
`00daad7cf6e48ecbc17625500dcbf0a2a18eb30d732f22f58d5be123a7493cae`；
恢复补充包 SHA256 为
`bc0b48925ce9ce868a514750bfc112f6ea368122408c94563742a68fa27b74b8`。

`restore-instructions.md` 和 `restore.py` 是补充包说明/脚本的审计副本；
实际离线执行请使用交付的 `perf-audit-optimization-restore-supplement.tar.gz`，
其中同时提供脚本所需的 `baseline.bundle`。仓库不重复保存嵌套的 Git 历史
与测量二进制。`restore-validation.json`、`restore-smoke-summary.json` 保存
恢复验证摘要；原始正式性能数据不被修改。

所有本地构建、测试和性能测量的目的都是验证实现、兼容与可复现性，不构成
化学体系采样收敛、GPU、多节点或任意平台 wheel 兼容性的声明。
