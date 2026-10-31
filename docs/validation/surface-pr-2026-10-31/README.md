# 表面催化功能：主分支整合验证

本次 PR 以 `3dc82c399a7701d19d636ee78b43f2e8ba7659f9` 为基线，保留其原生执行、
偏置分派和 PMF 不确定度修复，并整合周期表面三维 MetaD、NHC、软墙、
非线性键长约束、条件 FES 到 reactive-flux 以及面积／覆盖度换算功能。
新增的四进程测试已接入 `.github/workflows/tests.yml`。

## 整合后的结果

- 全量测试：**1441 passed、153 subtests passed、11 skipped**，零失败。
  跳过项为 10 个需要可选 BKMP2 的测试和 1 个未配置的真实 EAM 测试；Torch 测试实际执行。
- 新表面工作流的 19 个 NPZ、已有表面工作流的 27 个数组／检查点文件，
  在四进程 G2/B2 与独立串行执行之间逐数组完全一致；已完成任务的续算检查通过。
- 原生扩展重新编译，`pyrpmd info` 的 `native.available/openmp` 均为 true，
  `pip check` 通过；quickstart 输入的 PES 检查通过。
- HTML 与单页 Sphinx 手册均以 `-W --keep-going` 构建成功；wheel 与 sdist 构建成功，
  两份分发包中的全部 55 个 Python 源文件均与本次源文件逐字节一致。
- 真实 39 原子 CO/O/Pt(111) 在 600 K、P=1、0.2 fs 下完成 100 步热化和 800 步生产，
  得到 3D FES、40 个样本、2 个 hills。续算输出完全一致；重分析保留所有原摘要字段，
  仅额外记录 `analysis_settings`。高度投影也执行成功。
- 源 FES 不足以支持反应盆地及分割面，返回 `basin_lower_outside_histogram_centers`，
  正式／候选速率均为空，没有启动子轨迹，完成态续算结果相同。
- 真实 SP1 构型的 P=1/P=4 约束母轨迹和配对 NVE 子轨迹再次执行，冻结位置误差为零，
  最大约束残差约 2.22e-15 bohr。P=4、0.2 fs 的最大每珠能量漂移为 0.18055 kBT；
  改用 0.1 fs 后为 0.04503 kBT。这不是时间步或物理速率的收敛证明。

## 复现入口与证据

使用 [Pt(111) 示例](../../../examples/surface/pt111/README.md) 的模型下载、短算、
投影、速率和 `check_dynamics.py` 命令；完整测试为 `python -m pytest -q`。
所有本地计算设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
四进程命令分别运行 `tests/mpi_surface_extensions_smoke.py` 和
`tests/mpi_surface_smoke.py`，并为每次运行指定新的输出目录。

[results.json](results.json) 汇总本次结果；[source_sha256.json](source_sha256.json)
冻结 153 个源文件、测试、构建配置及示例文件。整合验证前的源树为
`bc7f9a3bf06792a1001c52b28f8522fed2d5939f`，此后仅添加本目录的验证记录。
完整日志和紧凑计算摘要与本文件一起提交；原始轨迹和模型权重不纳入 Git。
原先基于 `6fde29d` 的 [2026-10-30 验证记录](../surface-features-2026-10-30/README.md)
保持原样，不能把它的源哈希当成本次整合后的哈希。

## 科学范围

本文验证程序功能和数值行为，**没有完成论文 40 ns 采样与定量速率复现**。
长算输入只做配置检查，未执行。直接 Gaussian 求和的成本随 hills 数增长；
源 FES 投影没有完整的 PMF 不确定度，只能在支撑充分时报告候选速率。
公开双精度 TorchScript 与原 float LAMMPS／旧 Fortran 实现、恒温器自由度和
静态墙协议的差别仍见示例说明。GitHub CI 状态另见 PR Checks，不能用本地通过替代。
