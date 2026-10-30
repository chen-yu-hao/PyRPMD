# 示例索引与学习路线

所有命令从源码仓库根目录执行。先运行 `pyrpmd validate INPUT`，再用新的 `--output`
目录运行；想检查实际 PES 时添加 `--check-pes`。外部模型、MPI 和 GPU 的环境要求
见[安装](installation.md)。表中的输入属于演示或有限验证，不是自动收敛的生产配方。

## 无需外部模型的起点

| 输入/程序 | 内容 | 推荐阅读 |
| --- | --- | --- |
| `examples/manual/quickstart.inp` | 三个伞窗口的一维谐势 FES，6 条独立流 | [快速入门](quickstart.md) |
| `examples/manual/low_level.py` | `Dynamics` / `State` 的完整 100 步程序 | [API](api.md) |
| `examples/fes/harmonic2d.inp` | 两个质心 CV、二维 MBAR 与 mask | [气相与自由能](workflows.md) |
| `examples/fes/custom_input.py` | 自定义完整二维 CV Jacobian | [势能与插件](potentials.md) |
| `examples/fes/adaptive_harmonic.inp` | pilot 插窗后冻结独立 production | [气相与自由能](workflows.md) |
| `examples/drpmd/free_scattering.inp` | 两条持续母链分配共 5 次直接碰撞 | [气相与自由能](workflows.md) |
| `examples/rate/pilot.inp` | 教学反应势的 TS 准备、PMF 与回穿 | [气相与自由能](workflows.md) |
| `examples/network/two_entrances.py` | 两入口、四个独立 TS/PES 路径 | [气相与自由能](workflows.md) |
| `examples/metadynamics/well_tempered_2d.inp` | 二维 well-tempered 历史偏置 | [气相与自由能](workflows.md) |

```bash
pyrpmd validate examples/fes/harmonic2d.inp
pyrpmd run examples/fes/harmonic2d.inp --output runs/tutorial-fes2d
pyrpmd analyze runs/tutorial-fes2d --recompute --bins 20 20 --bootstrap 50
```

自由散射例子使用零势能；它验证发射、分类、输出和恢复机制，不代表 H + H₂ 的化学反应。
`minimal.inp` 中出现的外部势能库路径是用户需要提供的依赖；完整入门优先选上表可独立运行的例子。

## 材料、表面与 PCET

| 输入 | 实际 `task` | 模型与结果定位 |
| --- | --- | --- |
| `examples/materials/pimd.inp` | `pimd` | 有限 P 谐振子平衡能量与环尺寸 |
| `examples/materials/diffusion.inp` | `diffusion` | 自由粒子弹道区反例，不接受为收敛扩散系数 |
| `examples/materials/site_rate.inp` | `site_rate` | 周期教学势、显式盆地和单一有向出口 |
| `examples/surface/fes2d.inp` | `fes` | 固定基底的二维表面 CV |
| `examples/surface/diffusion.inp` | `surface_rate` | 单一有向表面跃迁率；不是 MSD diffusion 工作流 |
| `examples/pcet/harmonic_scan.inp` | `pcet` | 局域谐势井的 18 条件电势/温度/同位素扫描 |

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/tutorial-pimd
pyrpmd resume runs/tutorial-pimd

pyrpmd validate examples/pcet/harmonic_scan.inp
pyrpmd run examples/pcet/harmonic_scan.inp --output runs/tutorial-pcet
pyrpmd resume runs/tutorial-pcet
```

表面例子和 site-rate 例子需要检查采样长度与回穿，不承诺与最短 quickstart 相同的运行成本。
以[周期任务指南](periodic.md)与 [PCET 指南](pcet.md)解释输出。

## 外部模型与真实体系基准

[H + H₂ / BKMP2 基准](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md)
包含势能来源固定与 SHA256、能量/梯度参考检查、dRPMD、FES、PMF/recrossing。
构建需要 GNU Fortran；下载第三方源码后应阅读其许可与来源说明。

[Ni–H 材料 pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)
固定实际 EAM 模型来源与校验值，比较 H/D、P=1/4 的短轨迹。
模型表不随仓库分发，需要显式下载。短 NVE 能量检查不代表长时扩散已形成。

[ASE 表面例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/surface/README.md)
说明 Cu(111)/Cu adatom 的 EMT 接入。这里验证的是特定 calculator 与短工作流，
不能推广为所有 ASE 机器学习模型均已验证。

[PCET 独立比较](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/pcet-model-2026-10-30/README.md)
与固定外部 pyPCET 提交比较同一局域模型，不等于真实电极实验复现。

## 一次正式研究怎样从示例开始

1. 复制最贴近问题的输入到自己的项目目录，明确任务、坐标和归一化。
2. 用真实势能替换教学模型，先检查单位、原子顺序、力和适用范围。
3. 运行小型 pilot，查看失败、mask、overlap、弹道区或回穿平台等诊断。
4. 分别增加珠子数、缩小步长、延长平衡/生产、增加独立副本并检查定义域。
5. 固定最终方案后运行独立 production，并保存完整来源和不确定度信息。

各步的物理判断不能由 `preset="production"` 代替。参考[科学约定](science.md)。

