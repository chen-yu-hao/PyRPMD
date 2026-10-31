# 优化批次的最终整合与力缓冲区兼容性修复

本记录承接 [首批实现与八组性能测量](../audit-optimization-2026-10-30/README.md)
和 [发布准备与分派兼容性补充](../optimization-release-2026-10-30/README.md)。
两轮已有日志、数值结果和性能报告均保留；本目录单独记录后续整合的证据。

## 整合来源

PR #14 在另一项并发操作中合并为
`3dc82c399a7701d19d636ee78b43f2e8ba7659f9`，其中包含首批优化
`a7bb586b5b5fb5072401d3b5b962b0b41886cf3d` 和并发补充
`2b1f45c4e9f4b7a821ccd2ab1b6b53cb885ac059`。
另一份已验证补充 `38a16e4263ccd28909f99c3b73c16c605928150b` 尚未进入
该主分支。本次以更新后的主分支为第一父提交整合，保留两份补充的历史与实现。
PR #14 合并时尚有 CI 未完成；不能将它描述为本次流程在全部 CI 通过后合并。
后续 PR 将单独核对其最终提交的 CI，并在全部通过后合并。

验证前的冻结源码树为 `1dc54d50c2362c4d9bc8156b37078e47847f3bb3`。
随后只增加本目录的证据和说明；`source.json` 逐文件记录生产实现、测试、
benchmark、打包配置和 workflow 的 SHA256，区分工作树源码与其原 HEAD。
C++ 源码未因本次整合改变，原测量扩展 SHA256 仍为
`f2b9dea09296246648c1765297afe740526950f56b47f839c4f71afdb6687bd1`。
独立安装验证另行编译，不覆盖原二进制。

## 已修复的问题

### 自定义 Python 方法与 provider

保留双方对 Dynamics、integrator、内置及嵌套 CV/bias 的原方法身份检查，
统一使用定义模块保存的身份表，并保留 integrator `_check` 的保护。
补入 C ABI、Python wrapper 和 BKMP2 provider 的类级 `evaluate` 检查，
防止原生 binding 绕过用户替换后的数学方法。`auto` 使用 Python 实现，
严格 `native` 在推进前拒绝；未覆写的默认对象仍可走 native。

### 力数组的所有权

并发补充在没有 bias 或固定原子时直接返回 provider 的力数组。
这对内置只读 kick 是安全的，但破坏了自定义 `_force` 和 `kick`
可以原地修改独立可写数组的既有行为。若 provider 重用可写数组，
用户方法会污染其缓存；若数组只读，则直接报错。

可复现的常力例子中，覆写 `_force` 并将 `super()._force(...)` 原地乘以
0.5，两个 dt=0.2 的步骤应得到 p=0.2；旧借用路径把 provider 的初始
力 1 反复缩至 0.0625，得到 p=0.09375。只读 provider 则在首步失败。

修复恢复 `_force` 始终返回独立、可写副本。在内部 `_kick` 中，只有
精确内置 Dynamics/integrator、原 `_force` 与原 `kick` 均未覆写、没有
bias 或固定原子时，才借用已验证的 PES 缓冲同步完成只读 kick。
其他路径仍经过用户的方法及独立副本。没有引入跨 PES 调用的力缓存。

新的 `_kick` 同样登记到原生 block 的方法身份表；覆写该 helper 时，两套
原生传播 API 都会回退或在严格模式拒绝。PILE-L 的两端仍调用 Python
helper，C++ free/OU/free 内核不会绕过它。

| 检查 | 结果 | 日志 |
| --- | --- | --- |
| 修复前的力所有权回归 | 74 failed、20 passed | `force-before.txt` |
| 修复后的所有权与 helper 回归 | 100 passed | `force-after.txt` |
| 相关 dynamics、integrator、sampling、dispatch 集合 | 391 passed、36 subtests passed | `force-targeted.txt` |

覆盖可写/只读复用缓冲、子类/类级/实例覆写、普通推进和两种采样路径。
12 个组合通过观察原 kick 的实参身份，直接确认默认路径仍无复制。
日志归档只去掉行尾空格，原始 SHA256 与副本 SHA256 在
`force-log-provenance.json`；原始日志保留。

## 安装矩阵的 macOS 迁移

将唯一的 macOS 安装 job 从 `macos-14` 改为 `macos-15`，保留 Python 3.12、
独立 wheel/sdist 安装和全部 smoke 验收；Linux Python 3.11/3.12 不变。
两份分支的安装测试取并集，继续实测 sampler、5/32/64/256 珠子自由环、
block 能量/观测及应用工作流，并要求包内包含新增回归和效率 harness。

[GitHub 官方退役公告 #13518](https://github.com/actions/runner-images/issues/13518)
列出 macOS 14 于 2026-11-02 退役，且本次执行时间处于
2026-10-30 14:00 UTC 至 2026-10-31 00:00 UTC 的停用演练窗口。
[官方 runner 镜像清单](https://github.com/actions/runner-images/blob/main/README.md)
仍支持 `macos-15`；普通 `macos-14` 与 `macos-15` 均列为 arm64 标签。
本次观测的旧 macOS job 尚处于无 runner 的排队状态，因此没有将公告
与时间吻合推断写成该 job 已输出的失败原因，也没有跳过该平台验收。

## 最终源码验证

完整 pytest 在具备本机 socket 能力的环境下运行，结果为 **1566 passed、
3 skipped、153 subtests passed，0 failed**。三项跳过分别是未配置外部
固定来源 EAM 势的一项检查与未安装 PyTorch 的两项检查；远端保留独立
EAM 和 PyTorch 工作流。103 条 warning 来自 ASE 使用已弃用的 NumPy
shape 赋值接口。完整输出和执行范围见 `pytest.txt`、`pytest.json`。

独立打包从上述冻结 tree 导出源码，在不含原扩展的目录构建 sdist，再由
sdist 构建 wheel；wheel 与 sdist 分别安装到两个新 venv，sdist 安装时再次
编译扩展。两套安装均在源码目录外完整通过组合后的 smoke 检查。

- 52 个 Python 包源码文件在 wheel 和两个安装目录中均逐字节一致。
- 39 项 CI 必需文件、136 项包/测试 Python 与 native 源文件在 sdist 中
  存在且逐字节一致；运行依赖包含 `threadpoolctl>=3.1`。
- 两套安装的三个 native API marker 均为 1，`pip check` 通过。
- 两种 PILE-L 方案、非等质量、RNG/时钟、normal-mode FFT 和三个确定性
  积分器通过；C ABI block 的能量和观测、五组应用运行/续跑、两组 dRPMD
  母子轨迹流程均通过。两套 smoke 的全部数值与工作流结果相同。
- 原仓库和整合目录的测量扩展保持原 SHA256；两次新构建的扩展也互相一致。

本地明确复用了逐包链接的第三方依赖；PyRPMD 本身是独立安装，没有使用
editable 项目。这里不将本机检查称为全新联网依赖解析；远端矩阵继续执行
各平台的新环境独立安装。详情、构建指纹和导入路径见
`package-report.txt`、`package-metadata.json`、`wheel-smoke.json`、
`sdist-smoke.json`。

四套真实 4-rank MPI 检查（integration、workflow、materials、surface）全部
通过。16 份 rank 导入记录均指向本次整合目录，已加载 BLAS 池保持单线程；
surface 与独立串行参考的 27 个数组/checkpoint 文件中全部数组精确相等。
前后源码和扩展指纹一致。完整记录见 `mpi-results.json` 与四份 `mpi-*.txt`。

首轮默认 CMA transport 在进入测试前因本环境不支持 `process_vm_readv`
而失败，随后按既有验证配置设置 `MPIR_CVAR_CH4_CMA_ENABLE=0`、
`UCX_TLS=tcp,self`，在新的输出目录完整重跑四套并通过。初次启动失败
日志保留为 `mpi-transport-startup.txt`，未修改程序来绕过测试断言。

## 最终热路径的五轮完整工作流复测

新增 `_kick` 位于 Python 每步热路径，因此在全部测试和构建结束后，复用
原 harness 对三个受影响的完整工作流做五轮复测。仍使用原基线
`6fde29d…`、同一 CPU 0、单线程 BLAS、原测量扩展和固定来源 BKMP2；
每个 worker 先预热，基线/候选交替在独立进程运行。原有性能记录未改动。

| 完整工作流 | 基线中位时间 | 整合候选中位时间 | 中位时间比 |
| --- | ---: | ---: | ---: |
| BKMP2 dRPMD：8 trajectories、10263 child steps | 398.757 ms | 376.210 ms | 1.060× |
| BKMP2 FES：4 windows、4160 steps、800 samples | 263.213 ms | 241.039 ms | 1.092× |
| harmonic PILE-L：32 atoms × 32 beads、1000 steps | 744.571 ms | 488.055 ms | 1.526× |

dRPMD 的 84 个数组字段在每轮均逐位相同。FES 的 41 个数组字段全部满足
既定核对，只有四条 stream 的能量最大差为 `1.7763568394002505e-15`。
PILE-L 的 8 个数组字段全部通过，末态 q/p 最大差分别约 `6.22e-14`、
`9.09e-14`，观测最大差 `1.8553691916167736e-10`；RNG、时钟、离散决策
与工作量均严格相等。

这些比值比较的是整批优化后的源码和最初基线，不能单独归因于本次 guard
或缓冲区修复。dRPMD/FES 的短工作量仍有明显轮间波动，不据此承诺稳定的
6%/9% 化学工作流提速；PILE-L 的五个配对比值均在 1.44× 至 1.69× 范围。
原八组测量和原分派复测继续独立保留，不以本轮数字替换。

完整原始时间、逐数组误差、源码/native/PES 指纹和科学输出 digest 在
`workflow-benchmark-report.json`；`workflow-benchmark-source.json` 说明
实际源码树与当时 HEAD 的区别。991 个原始文件合计 20,824,725 bytes
保留在独立工作目录。本轮是实现一致性与固定工作量性能验证，不证明化学
采样的物理收敛。
