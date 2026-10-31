# FES PILE-L / REUS 验证

基线 main `3dc82c399a7701d19d636ee78b43f2e8ba7659f9`。
本次复用积分器、CV、伞势、MPI/进程任务、MBAR 和校验和 checkpoint，新增同温固定窗口
REUS、FES PILE-L 入口，以及交换链内同步 bootstrap。不包含其他独立 PR 的更改。

## 当前版本实际本地验证

Python 3.13.5；从与 main 一致的 C++ 源码重新编译 OpenMP 扩展。
相关套件 **151 passed、1 skipped、10 subtests passed**；唯一跳过是可选 PyMBAR 独立参考。
扩大套件 **1428 passed、21 skipped、153 subtests passed**，4 个既有警告，无失败。
扩大套件显式排除了 `test_hho2_channels.py` 和 `test_hho2_scan.py`：用于恢复工作区的 sdist
遗漏 `benchmarks/hho2` 依赖，不能声称完整仓库全套已本地通过。本次增加 MANIFEST 规则修复
这一分发遗漏。其他跳过为可选真实 BKMP2、ASE/EAM 或独立参考依赖，跳过不计作通过。

检验覆盖：

- 12 个 P=1/4/16 × Cayley/normal-mode × NumPy/native 组合：由实际采样单步构建漂移/噪声
  矩阵并解离散 Lyapunov 方程；二维耦合谐势质心协方差与解析值相符，rtol=2e-10。
- 独立完整环 Hamiltonian 交换前后比值和离散详细平衡；周期/非周期 CV、交叉刚度、无额外 P。
- 4 个真正编译 C ABI 势能的 REUS 测试：原生/NumPy 数值一致，交换决定、walker 和 RNG 一致。
- 半窗口/半轮交换故障、原子写入前后故障、有效校验和下的非法形状/时钟/排列/计数拒绝。
- 独立交换链的实际双进程/串行 checkpoint 一致；组内同步 bootstrap、组间独立；未分组时旧随机流不变。
- 固定周期 slab、adaptive 的 PILE-L pilot/production，以及完成后恢复/离线分析不调用 PES。

二维 CLI 示例实际运行 **P=4、9 窗口 × 2 链、9000 样本**，overlap 连通、118/256 个 bin
受支持。完成后恢复不改变 checkpoint SHA256，离线重建所有数组一致；见 `example.json`。
`converged=false`，这些是实现测试和短谐势检查，不代表真实化学收敛或性能加速。

本地无 MPI runtime，新增四 rank harness 仅串行参考部分已运行；现有 GitHub CI 调用的
`mpi_workflow_smoke.py` 会接续实际 G2×B2 REUS 比较。远端结果必须以对应 commit 的 CI 为准。
完整原始日志由本次提交/PR 的验证记录提供；未完成的远端任务不计为成功。

## 复现

```bash
python -m pip install -e '.[test,analysis,ase,mpi]'
python setup.py build_ext --inplace --force
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
mpiexec -n 4 python tests/mpi_exchange_smoke.py /tmp/fresh-reus-mpi
pyrpmd run examples/fes/replica_exchange_2d.inp --output /tmp/fresh-reus-example
pyrpmd resume /tmp/fresh-reus-example
pyrpmd analyze /tmp/fresh-reus-example --recompute
```

方法及限制见 [副本交换指南](../../replica-exchange.md)。没有修改或合并其他 PR，没有向 main
直接提交。当前窗口在单条交换链内顺序推进，不宣称已实现跨节点的窗口级交换。
