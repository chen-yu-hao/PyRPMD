# 示例索引与学习路线

每类任务都有基于解析模型的可运行示例，单核几秒到约一分钟即可完成，便于在接入自己的势能之前看到完整的工作流与输出。所有命令从仓库根目录执行：先运行 `pyrpmd validate INPUT`（加 `--check-pes` 以构造 PES），再用新的 `--output` 目录运行
`pyrpmd run`。外部模型、MPI 和 GPU 需要[安装](installation.md)中列出的依赖。

第一次运行无需查找仓库文件：使用 `pyrpmd init my-first-run`，然后按[快速入门](quickstart.md)操作。
下表中的路径均相对于源码仓库。

## 无需外部模型的示例

“耗时”是单个 CPU 核心、`OMP_NUM_THREADS=1` 下 `run` 的墙钟时间。

| 输入/程序 | 内容 | 耗时 | 章节 |
| --- | --- | --- | --- |
| `examples/manual/quickstart.inp` | 三个伞窗口、6 条采样流的一维谐势 FES | 约 2 s | [快速入门](quickstart.md) |
| `examples/manual/low_level.py` | 使用 `Dynamics` / `State` 的 100 步程序 | <1 s | [API](api.md) |
| `examples/fes/harmonic2d.inp` | 两个质心 CV、二维 MBAR 与 mask | 约 6 s | [气相与自由能](workflows.md) |
| `examples/fes/replica_exchange_2d.inp` | PILE-L、副本交换伞形采样与同步 bootstrap | — | [副本交换](../replica-exchange.md) |
| `examples/fes/custom_input.py` | 带完整 Jacobian 的自定义二维 CV | 约 7 s | [势能与插件](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | pilot 插入窗口后冻结并运行生产段 | 约 4 s | [气相与自由能](workflows.md) |
| `examples/drpmd/free_scattering.inp` | 两条母链在 V=0 上共享 5 次碰撞 | 约 2 s | [气相与自由能](workflows.md) |
| `examples/nonadiabatic/exchange.inp` | 在耦合双态模型上从指定态出发的 NRPMD 散射 | 约 15 s | [非绝热 NRPMD](nonadiabatic.md) |
| `examples/rate/pilot.inp` | 解析反应势上的 TS 准备、PMF 与回穿 | 约 15 s | [气相与自由能](workflows.md) |
| `examples/network/two_entrances.py` | 两个入口、四条 TS/PES 路径 | 约 25 s | [气相与自由能](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 二维 well-tempered metadynamics | 约 2 s | [气相与自由能](workflows.md) |
| `examples/opes/harmonic_2d.inp` | OPES 自适应、冻结偏置后的平衡与直方图重加权 | — | [OPES 英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/opes.md) |
| `examples/sampling/transition_interface.py` | 受限双阱模型上的四个界面系综 | — | [TIS 英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

`examples/rate/minimal.inp`（只给 TS）和 `examples/drpmd/minimal.inp`（两个反应物与间距）展示了最小的有效输入，
其余设置均取 pilot 缺省值。

`examples/plumed/centroid_metadynamics.py` 演示带显式 kernel 更新的可选 PLUMED 质心 metadynamics。
它需要可用的 PLUMED 安装，运行前请按 [PLUMED 英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md)准备环境。

## 材料、表面与 PCET

| 输入 | `task` | 内容 | 耗时 |
| --- | --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | 周期模型中的平衡能量与环尺寸 | 约 2 s |
| `examples/materials/diffusion.inp` | `diffusion` | 自由粒子；分析标出弹道运动 | 约 3 s |
| `examples/materials/site_rate.inp` | `site_rate` | 周期模型、显式盆地与单一有向出口 | 约 25 s |
| `examples/surface/fes2d.inp` | `fes` | 固定基底上的二维表面 CV | 约 1 min |
| `examples/surface/diffusion.inp` | `surface_rate` | 单一有向表面跃迁速率 | 约 1.5 min |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 18 个条件的电势/温度/同位素扫描 | 约 2 s |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

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

`preset="production"` 检查所有设置都已显式给出；第 3、4 步确立收敛。见[科学约定](science.md)。
