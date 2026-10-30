# 材料计算实现验证：2026-10-30

本记录对应 `feat/materials-diffusion` 的固定晶胞/PIMD/RPMD 扩散与位点速率实现。
开发基于 `4115580fcd6f54aa5f74613772dc052f32e5481a`；真实势能 pilot
另在机器可读报告中记录执行时各源码文件的 SHA256，保留实际来源而不回填后来的提交号。

## 验证结果

| 检查 | 实际结果 |
| --- | --- |
| 最终完整 pytest，启用本地已验证的 Ni–H 模型 | **421 passed, 12 skipped, 105 subtests passed**，76.97 s |
| 四个真实 MPI 进程，`G=2, B=2` | PIMD、diffusion、site_rate、精确续跑、离线分析全部通过 |
| 三个公开材料示例 | PIMD complete；free diffusion 正确报告 ballistic；site rate 完成 PMF→RATTLE→回穿 |
| Source distribution 与本机平台 wheel | 构建通过，检查包含新模块、示例和材料基准脚本 |
| 独立有限 P 谐振子平衡检查 | 所有物理能量均值与精确目标相差不超过 1.43 个副本间 SE |
| 源码固定的真实 Ni–H EAM 四组 pilot | H/D × P=1/4 全部通过数值链路检查，未报告有效扩散系数 |

本地跳过的 12 项为可选 BKMP2/Fortran 模型 10 项和未安装 Torch 的 2 项。
它们不计作通过；仓库的独立 GitHub 工作流继续覆盖真实 BKMP2 和 Torch CPU。
MPI 检查发生在同一台机器的四个进程中，不构成跨节点或 GPU 性能验证。

```bash
PYRPMD_NIALH_POTENTIAL=/absolute/path/NiAlH_jea.eam.alloy \
  python -m pytest -q

mpiexec -n 4 python tests/mpi_materials_smoke.py /fresh/shared/output
python -m build --no-isolation
```

本次受限容器上的 MPICH 需要 `MPIR_CVAR_CH4_CMA_ENABLE=0 UCX_TLS=tcp,self`
以使用可用的传输机制；这不是 PyRPMD 或其他集群的一般配置要求。

## 量子平衡与约束测度

[`harmonic_validation.json`](../../benchmarks/materials/harmonic_validation.json)
保存四个独立 BCOCB/PILE-L 副本的实际结果；复现脚本为
[`harmonic_validation.py`](../../benchmarks/materials/harmonic_validation.py)。
这是 m=k=1、β=4、P=16 的三维解析振子数值检查，不是金属/氢的物理预测。

| 物理量（hartree） | 样本均值 | 副本间 SE | 有限 P 精确值 |
| --- | ---: | ---: | ---: |
| 势能 | 0.776092575 | 0.002681954 | 0.772272526 |
| Centroid-virial 动能 | 0.773887990 | 0.003214987 | 0.772272526 |
| Primitive 动能 | 0.782771557 | 0.011215906 | 0.772272526 |
| Centroid-virial 内能 | 1.549980565 | 0.005624685 | 1.544545053 |

连续量子极限内能为 1.555972081 hartree，刻意与有限 P 参考区分。
这项随机平衡 spot check 不作为容易受随机波动影响的 CI 门槛。

site-rate 还使用独立有限 P 高斯条件协方差检验约束母链：质量 2/20、
β=2、P=4、两个质心的相对位置约束。由实际线性传播矩阵求离散
Lyapunov 方程，并与解析 Schur 补条件协方差比较；时间步
0.1/0.05/0.025 时相对误差为 9.27e-5/2.32e-5/5.79e-6。
该无采样噪声测试确认二阶步长收敛，且独立检查质量度量和 P 因子。

## 真实 Ni–H 势能与轨迹

完整来源、复现命令和物理限制见
[`benchmarks/materials/README.md`](../../benchmarks/materials/README.md)，
实际数值见 [`ni_h_eam_validation.json`](../../benchmarks/materials/ni_h_eam_validation.json)。
势能数据不随仓库分发；下载必须显式选择，并验证固定 commit、长度和 SHA256。

模型为 27 个可移动 Ni 加一个 H/D 的 Ni(100) slab，600 K、0.1 fs，
64 步 PILE-L 热化与 160 步无恒温生产，每组 81 帧。

| 算例 | 最大 NVE 能量偏差 / 珠子（hartree） | 扩散筛选 |
| --- | ---: | --- |
| H, P=1 | 1.6704324e-7 | ballistic_or_drifting |
| D, P=1 | 1.1438048e-7 | ballistic_or_drifting |
| H, P=4 | 1.8282463e-7 | ballistic_or_drifting |
| D, P=4 | 1.6073011e-7 | ballistic_or_drifting |

独立直接 ASE 对照的能量误差为 4.44e-16 hartree，力误差为
6.94e-18 hartree/bohr；所选力分量的最大有限差分误差为
4.44e-10 eV/angstrom。晶胞平移后的力差小于 1.35e-17 hartree/bohr。
H 与 D 保留不同核质量、相同电子元素；P=1 的环回转半径严格为零，P=4 非零。

这些短轨迹只有 16 fs 生产时间，尚未证明充分平衡或长时扩散。
四组 accepted D 均为 `null`，没有独立扩散误差条，`physical_converged=false`。
不从这些数值推断量子增强因子、同位素输运比或复现 2012 年 Ni(100) 文献结果。
后续正式材料研究还需势模型、晶胞、层数、覆盖度、珠子数、步长和采样收敛。
