# PySCF 参考数据与 Caracal 势能生成

可选模块 `python -m pyrpmd.pesgen` 提供离线流程：端点与给定反应路径 → PySCF 能量、
梯度、Hessian、Hirshfeld 电荷和 Mayer 键级 → 原版 Caracal QMDFF/dE-EVB/TREQ →
原生势能包 → 现有 PyRPMD JobSpec。这里不是把简谐势或普通插值改名为 QMDFF/TREQ。

## 构建与运行

首版限定 Linux，需要 C++17、GNU Fortran/OpenMPI、BLAS/LAPACK、FFTW 与 Python 开发配置。
PySCF 是可选依赖，普通 PyRPMD 运行不需要安装。

```bash
python -m pip install -e '.[pesgen,test]'
git clone https://github.com/Trebonius91/Caracal.git external/Caracal
git -C external/Caracal checkout 7f95ce9818b7f6d81d7cb3b634f360373c090820
python -m pyrpmd.pesgen build-native external/Caracal runs/caracal-native
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes
python -m pyrpmd.pesgen generate examples/pesgen/water.toml --output runs/water-pes --resume
python examples/pesgen/nh3_inversion.py --native-build runs/caracal-native --output runs/nh3-treq
```

构建目录必须是新目录且位于上游源码目录之外。构建复制并校验固定版本的源码，保留安全补丁、
编译命令、二进制散列和许可证。水分子和 NH₃ 示例使用 RHF/STO-3G 检查接口，不代表生产级精度。
NH₃ 是九点约束反转扫描，不是优化 IRC；加 `--method de_evb` 可运行 dE-EVB 路线。

## 参数和科学边界

`method` 支持 `qmdff`、`de_evb`、`treq`，分别需要一个或两个 `endpoints` XYZ 文件；后两者还需
多帧 `path`。所有帧必须使用相同原子顺序、总电荷和自旋。`optimize_endpoints=true` 只进行极小点
优化，不进行 TS 搜索。端点检查梯度与负曲率。当前拟合流程限定中性、非周期、单电子势能面；
独立参考数据 API 可以计算带电分子。

`[qm]` 支持 RHF/UHF/RKS/UKS，显式全电子基组、DFT `xc`、`spin=Nα−Nβ`、SCF 和网格收敛控制。
内部坐标为 bohr，能量 hartree，正梯度 hartree/bohr，Hessian hartree/bohr²；XYZ 使用 Å。
默认解析 Hessian，显式 `hessian="finite_difference"` 才切换为梯度差分。SCF 不收敛则拒绝缓存。
冷启动不等于电子态连续性或波函数稳定性已验证，这些限制写入来源记录。

Hirshfeld 参考是同一 AO 基组中的中性球平均 HF 原子密度；这不是声称复制了 ORCA 的原子表。
独立 `population_grid_level` 和电荷守恒检查控制积分。Mayer 键级含非限制自旋密度项。
传给 Caracal 的是 Hirshfeld 电荷，它只做一次 CM5 转换与经验缩放。兼容 ORCA 读取器的文件明确
标注数据来自 PySCF，并保留完整散列 JSON；不会伪造 ORCA 计算来源。

`de_evb` 用原始 `evbopt.x` 拟合，`[fit] coupling="sp"` 或 `"1g"`。
TREQ 要求显式 `coordinates`：恰好 `3N−6` 个非冗余内坐标；键 `i j`、角 `i j k`、
二面角 `0 i j k l`、面外坐标 `1 i j k l`，原子编号从 1 开始。线性几何、冗余内坐标、自动选择
内坐标尚未覆盖；数量正确也不保证条件数良好。`hessian_indices` 是从 0 开始的路径帧索引，
默认全部取样，手选必须包括两端、最高能量帧及至少一个中间点。

## TREQ 力与原生传播

固定版本的 CF₃O + CH₄ 示例中，原始解析力的一个分量没有通过自身能量的有限差分检查。
因此公共 TREQ 流程使用 **Fortran 内部四阶中心能量差分力**，保留原始 TREQ 能量表达式。
`force_step` 默认 `1e-4 bohr`，每构型需要 `1+12N` 次上游调用，不能称为快速解析力。
该模式标记为实验性，需检查差分步长、时间步长与能量守恒，不能直接宣布真实反应速率已验证。

动力学调用 `libpes.so` 的 C ABI v1，每个上下文使用一个持久 Fortran 子进程，隔离全局状态；
不是每步启动程序。C++ 连续传播直接调用编译接口，没有逐步 Python 势能回调。批次内逐帧计算，
首版不提供周期晶胞、virial、GPU 或批次内珠子并行。上下文关闭时回收子进程，异常退出和超时
会显式报错。TREQ 每次使用全路径搜索，避免上一珠子的邻域历史污染下一次求值。

## 误差、续跑和计算衔接

参考缓存按输入、方法、版本和代码散列校验；不同进程不能同时写同一目录。完成后的模型、worker
和摘要被校验，输入变化或文件损坏会拒绝续跑。失败尝试保留日志。

`summary.json` 分开记录训练误差和可选独立验证误差。`[validation]` 同时设置 `geometries`、
`energy_rmse_max` 和 `force_rmse_max`，单位分别是 hartree 与 hartree/bohr。重复训练帧会被拒绝。
达到自定阈值不等于全局可迁移性或 RPMD 速率收敛；报告仍保留 `validated_physical_accuracy=false`。

```bash
python -m pyrpmd.pesgen run-job runs/water-pes existing-job.inp --output runs/water-dynamics --allow-unvalidated
```

这使用已有完整 JobSpec 的反应坐标、归一化与采样设置，不会根据一条路径自动编造速率输入。
无通过的独立验证时，以及实验性 TREQ 模式下，必须显式用 `--allow-unvalidated` 允许试算。
Python API 是 `generate`、`build_native`、`run_generated_job`。

构建补丁修复全零 Hirshfeld 电荷误判、未初始化累加器、缺失电荷处理、CM5 缩放电荷守恒及
`ff_clean` 的大栈数组；不改变势能能量方程。

文献：QMDFF **10.1021/ct500573f**；dE-EVB **10.1039/C5CP02580J**；
TREQ **10.1063/1.5092589**；Caracal **10.1021/acs.jctc.3c00568**。
