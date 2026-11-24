# 示例索引与学习路线

每类任务都有基于解析模型的可运行示例，单核几秒到约一分钟即可完成，便于在接入自己的势能之前看到完整的工作流与输出。所有命令从仓库根目录执行：先运行 `pyrpmd validate INPUT`（加 `--check-pes` 以构造 PES），再用新的 `--output` 目录运行
`pyrpmd run`。外部模型、MPI 和 GPU 需要[安装](installation.md)中列出的依赖。

第一次运行无需查找仓库文件：使用 `pyrpmd init my-first-run`，然后按[快速入门](quickstart.md)操作。
下表中的路径均相对于源码仓库。

## 含实测验收的解析教程

下表中的运行耗时来自 2026-11-10 单核本地验证批次（Python 3.11.18，OMP/OpenBLAS/MKL 线程设为 1）；对比脚本耗时另计。
`pending` 表示尚无已完成的批次计时记录。每个教程的 `compare.py` 在不改写生产摘要的前提下报告实测偏差、标准误（SEM）和适用的离散误差上界。

| 教程输入 | 独立解析参照 | 实测运行耗时 | 验收对照 |
| --- | --- | --- | --- |
| [有限珠数谐振子 PIMD](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/harmonic_pimd/README.md) | 有限 $P$ 谐振子能量与回转半径 | 10.20 s | 通过；势能偏差 1.565 SEM |
| [周期势垒位点逃逸速率](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/periodic_site_rate/README.md) | 有向经典 TST | 176.80 s | 通过 |
| [双势阱冻结 WT-MetaD](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/double_well_metad/README.md) | 分箱积分精确 Boltzmann 密度 | 93.95 s | 通过 |
| [带摩擦自由扩散](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/free_diffusion/README.md) | $k_BT/(m\gamma)$ | 31.07 s | 通过 |
| [谐振子光谱](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/harmonic_spectroscopy/README.md) | $\sqrt{k/m}$ 质心频率 | 95.36 s | 通过；偏差 0.225 cm⁻¹（半格界 2.545 cm⁻¹ 内） |
| [D + H₂ 反应交换速率对照](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/tutorials/exchange_rate/README.md) | 同一 C ABI 势能面上的 dRPMD、UI-BC 与 MBAR-BC 热速率 | pending | 3-SEM 跨方法一致性检验（`compare.py`） |

使用新的输出目录并运行各教程的对比命令。这些固定模型检验不能替代具体研究体系的物理收敛验证；见
[如何使结果收敛](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/convergence.md)。

## 无需外部模型的示例

“耗时”是单个 CPU 核心、`OMP_NUM_THREADS=1` 下 `run` 的参考墙钟时间；未单独计时的条目记为 `pending`。

| 输入/程序 | 内容 | 耗时 | 章节 |
| --- | --- | --- | --- |
| `examples/manual/quickstart.inp` | 五个伞窗口、10 条采样流的一维谐势 FES，可与精确解 $\tfrac12 kx^2$ 对比 | 约 4 s | [快速入门](quickstart.md) |
| `examples/manual/low_level.py` | 使用 `Dynamics` / `State` 的 100 步程序 | <1 s | [API](api.md) |
| `examples/fes/harmonic2d.inp` | 两个质心 CV、二维 MBAR 与 mask | 约 6 s | [气相与自由能](workflows.md) |
| `examples/fes/replica_exchange_2d.inp` | PILE-L、副本交换伞形采样与同步 bootstrap | 约 8 s | [副本交换](../replica-exchange.md) |
| `examples/fes/custom_input.py` | 带完整 Jacobian 的自定义二维 CV | 约 7 s | [势能与插件](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | pilot 插入窗口后冻结并运行生产段 | 约 4 s | [气相与自由能](workflows.md) |
| `examples/drpmd/free_scattering.inp` | 两条母链在 V=0 上共享 5 次碰撞 | 约 2 s | [气相与自由能](workflows.md) |
| `examples/state_selected/ne_rpmd.inp`、`qct.inp` | Morse 模型上的选定振动态 NE-RPMD 及其单珠子 QCT 对照；`state-check`、`extend` | 各约 1 s | [气相与自由能](workflows.md) |
| `examples/nonadiabatic/exchange.inp` | 在耦合双态模型上从指定态出发的 NRPMD 散射 | 约 15 s | [非绝热 NRPMD](nonadiabatic.md) |
| `examples/nonadiabatic/population.inp` | 从谐振 spin-boson 初态出发的 NRPMD 布居与相干性 | pending | [非绝热动力学](nonadiabatic.md) |
| `examples/nonadiabatic/rpsh_population.inp` | Tully I 上的质心 RPSH，含可选的能量退相干修正 | pending | [非绝热动力学](nonadiabatic.md) |
| `examples/nonadiabatic/extensions/` | SM-NRPMD、CA-RP-MASH、BA/DM RPSH、DBOC 与复数 SOC 输入 | pending | [扩展非绝热方法（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-workflow.md) |
| `examples/nonadiabatic/extensions/nah2_sm_nrpmd.inp` | 已发表 NaH₂ 分子双态 PES（含笛卡尔梯度） | pending | [分子多态势能（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/molecular-nonadiabatic.md) |
| `examples/excited_state/toy_rpsh.py` | 即时绝热激发态表面跳跃与相位追踪 NAC | 约 1 s | [激发态接口（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/excited-state-interface.md) |
| `examples/rate/research.inp` | 显式反应、独立 TS 检查、分区窗口与分开的母链/子轨迹 | pending | [速率输入指南](../rate-input-v2.md) |
| `examples/rate/subspace.inp`、`examples/subspace_correction/eckart_bath_2d.py`、`molecular_ts_to_cs.py` | 低温隧穿区含 `[rate.subspace_correction]` 的 `rate`（支持 `--stage subspace`）、二维 Eckart 基准及免伞抽样 9D 笛卡尔 TS $\to C_S, B_P$ | 约 20–50 s | [子空间速率修正](../subspace-correction.md) |
| `examples/rate/pilot.inp` | 保留 schema 1 的 TS 准备、PMF 与回穿 | 约 15 s | [气相与自由能](workflows.md) |
| `examples/rate/general_coordinate.inp` | 沿光滑非线性质心坐标的 `site_rate` | pending | [一般坐标速率（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/general-coordinate-rates.md) |
| `examples/rpmdrate_import/` | 转换 RPMDrate Python 输入（`import-rpmdrate`）并校验解析设置 | 约 1 s | [RPMDrate 导入（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rpmdrate-import.md) |
| `examples/studies/site_rate.json`、`drpmd.json` | `pyrpmd study` 温度/同位素/收敛扫描；用 `--plan` 查看算例 | pending | [速率研究（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-studies.md) |
| `examples/network/two_entrances.py` | 两个入口、四条 TS/PES 路径 | 约 25 s | [气相与自由能](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 二维 well-tempered metadynamics | 约 2 s | [气相与自由能](workflows.md) |
| `examples/metadynamics/standard.inp` | 四次势模型上的标准（恒定高度）metadynamics | 约 1 s | [气相与自由能](workflows.md) |
| `examples/opes/harmonic_2d.inp` | OPES 自适应、冻结偏置再平衡与直方图重加权 | 约 3 s | [OPES（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/opes.md) |
| `examples/tempering/double_well.inp`、`hremd_umbrella.inp` | 并行回火（`mode="temperature"`）与 Hamiltonian 副本交换（`mode="hamiltonian"`） | 约 2 s | [并行回火（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tempering.md) |
| `examples/plumed/centroid_metadynamics.py` | 可选的 PLUMED 质心 MetaD，显式更新 kernel | pending | [PLUMED（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md) |
| `examples/sampling/transition_interface.py` | 受限双势阱上的四个界面系综 | 约 7 s | [TIS（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md) |
| `examples/instanton/eckart.py` | Eckart 势垒上的环聚合物瞬子与量子对照 | 约 1 s | [瞬子（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md) |
| `examples/quantum_statistics/harmonic_validation.py` | 高阶 PIMD 估计量与谐振解析结果对照 | 约 13 s | [量子统计（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-statistics.md) |
| `examples/path_integral/bosonic.inp`、`fermionic.inp`、`bosonic_winding_npt.inp` | 玻色/费米交换递推、周期缠绕数与 PIGS | 约 2–5 s | [路径积分作用量（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-path-integral.md) |
| `examples/spectroscopy/pacmd.inp`、`mixed_trpmd.inp`、`fqcmd_water.py`、`harmonic_properties.py` | PA-CMD、混合 TRPMD、f-QCMD 水与 IR/Raman/SFG 属性光谱 | 约 2–10 s | [光谱（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-spectroscopy.md)、[QCMD（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/qcmd.md) |
| `examples/ml_lifecycle/committee.inp`、`delta_rpc.inp`、`closed_loop.py` | 委员会不确定度监控、$\Delta$-learning + RPC 与主动学习闭环 | 约 2 s | [ML 生命周期（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-ml-lifecycle.md) |
| `examples/anharmonic_quantum/`、`examples/anharmonic_3d_validation/` | 受控 RPMD 响应修正与三维非谐量子子空间基准 | pending | [量子修正（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-correction.md) |
| `examples/nonadiabatic/method_models.py`、`soc_and_rpsh.py`、`thermal_rates.py` | KC-/MV-/同构 RPMD、SOC-NRPMD 与 RPSH、热 NRPMD 速率曲线 | 1–3 s | [非绝热](nonadiabatic.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

`examples/rate/minimal.inp` 显式选择从 TS 推断通道的便捷模式；研究输入应像 `research.inp` 一样声明片段和成断键。
`examples/drpmd/minimal.inp` 使用两个反应物与间距，其余设置取 pilot 缺省值。

`examples/plumed/centroid_metadynamics.py` 演示带显式 kernel 更新的可选 PLUMED 质心 metadynamics。
它需要可用的 PLUMED 安装，运行前请按 [PLUMED 英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md)准备环境。

## 材料、表面与 PCET

| 输入 | `task` | 内容 | 耗时 |
| --- | --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | 周期模型中的平衡能量与环尺寸 | 约 2 s |
| `examples/materials/fixed_slab.inp` | `pimd` | 带固定层的周期 slab | 约 1 s |
| `examples/materials/npt_free.inp` | `pimd` | 理想气体上的 Python BZP 恒压及解析体积检验 | 约 4 s |
| `examples/ensembles/pimd_native.inp`、`pimd_native_npt.inp` | `pimd` | 原生 C++ NVT/BZP NPT 引擎，使用单独的恒温/恒压及检查点约定 | 各约 1 s |
| `examples/metadynamics/grid_fixed_slab.inp` | `fes` | 固定 slab 上的网格加速偏置 | 约 1 s |
| `examples/nonadiabatic/friction_surface.inp` | `diffusion` | 固定 slab 上的质心电子摩擦 | 约 3 s |
| `examples/materials/diffusion.inp` | `diffusion` | 自由粒子；分析标出弹道运动 | 约 3 s |
| `examples/materials/site_rate.inp` | `site_rate` | 周期模型、显式盆地与单一有向出口 | 约 25 s |
| `examples/surface/fes2d.inp` | `fes` | 固定基底上的二维表面 CV | 约 1 min |
| `examples/surface/diffusion.inp` | `surface_rate` | 单一有向表面跃迁速率 | 约 1.5 min |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 18 个条件的电势/温度/同位素扫描 | 约 2 s |
| `examples/electrochemistry/fes2d.inp`、`sample_grand_canonical.py`、`her_scan.py` | `fes` / API | 恒电位有效势、电子数采样、HER 电流 | ≤5 s |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

其他输入：`examples/notebooks/`（[六个交互式 Jupyter Notebook 案例](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/notebooks/README.md)，涵盖 PES、反应速率、周期材料、Torch GPU 加速、C++ ONNX JaxPIP 与 JAX DLPack GPU 动力学）、`examples/pesgen/`（[势能生成](pes-generation.md)、[JaxPIP](jaxpip-generation.md)）、`examples/potentials/`
（周期 Torch 模型、批处理 ML 势能、即时 tblite/PySCF）、`examples/drpmd/*.py`（Python、Torch、珠子、MPI 与 GPU 布局）、`examples/slurm/`
（集群作业脚本），以及 `examples/analysis/reference-rates.csv`（`pyrpmd report --reference` 的格式）。

表面与位点速率例子包含伞抽样、受约束母链和回穿，因此耗时更长。输出说明见[材料与表面](periodic.md)和 [PCET](pcet.md)。

## 使用真实模型的基准

[H + H₂ / BKMP2 基准](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md)在已发表的 BKMP2 势能面上
运行 dRPMD、FES 和 PMF/回穿，固定了源码版本、SHA256 并做能量/梯度参考检查。构建需要 GNU Fortran；构建过程会下载
第三方源码，其许可与来源列在基准说明中。

[Ni–H 材料 pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)在固定来源、带校验值的
EAM 模型上运行镍中的 H 和 D（P=1 与 4），模型需单独下载。

[ASE 表面例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/surface/README.md)用 ASE EMT 连接 Cu(111) 上的 Cu 吸附原子。

[PCET 对照](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)在谐振子模型上
与固定版本的 pyPCET 偏差不超过 0.062%。

## 从示例到研究计算

1. 把最接近的输入复制到自己的项目中，设定任务、坐标与归一化。
2. 用自己的势能替换模型势；检查单位、原子顺序、力和适用范围。
3. 运行短 pilot 并阅读诊断：失败、mask、重叠、弹道区和回穿平台。
4. 逐项增加珠子数、减小时间步、延长平衡与生产、增加独立副本并检查坐标域。
5. 固定最终方案，运行独立的生产计算，保存完整的来源与不确定度信息。

`preset="production"` 检查必填字段和任务限制，省略的参数仍可使用缺省值。固定方案前用
`pyrpmd validate INPUT --write-resolved resolved.json` 检查完整配置；第 3、4 步确立收敛，见[科学约定](science.md)。
