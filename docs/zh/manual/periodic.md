# 周期材料、扩散与表面反应

PyRPMD 支持周期体系、带冻结层的 slab，以及各向同性 NPT PIMD。本章介绍平衡 PIMD、无恒温 RPMD 输运，以及盆地归一化的局部跃迁速率。

## 任务与边界条件

| 任务 | 用途 | 周期晶胞 | `fixed_atoms` | 主要结果 |
| --- | --- | --- | --- | --- |
| `pimd` | 平衡量子统计，可选 BZP 恒压 | 支持 | 固定晶胞时支持 | 能量估计量、回转半径 |
| `diffusion` | 连续质心轨迹与输运 | 支持 | 支持 | MSD、VACF、D 与诊断 |
| `site_rate` | 全原子可动材料中的有向出口 | 支持 | 不支持 | 沿单一恒定梯度 CV 的盆地速率 |
| `fes` | 一维/二维几何自由能 | 支持 | 支持 | 位置密度 FES、重叠、mask |
| `surface_rate` | 表面上的有向出口，可含固定层 | 支持，也可为有限团簇 | 支持 | 以可动原子质量度量计算的一阶速率 |

`pimd`、`diffusion`、`fes`、`metadynamics` 和 `surface_rate` 支持固定原子；固定原子在所有珠子上保持
参考坐标和零动量。材料量子动能估计量只计入可动原子。`site_rate` 仍要求全部原子可动。
气相 `rate`、`drpmd` 和 `network` 用于不含 PBC 和固定原子的分子体系。

所有可动原子使用同一珠子数。晶胞默认固定，`pimd` 可显式增加 `[barostat]` 做恒压采样。

## 晶胞、坐标与同位素

一个倾斜晶胞的 slab：

```toml
[system]
geometry = "slab.xyz"
geometry_unit = "angstrom"
cell = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
```

晶胞每一行是一个晶格矢量，笛卡尔坐标满足 `r = s @ cell`。矩阵须有限、可逆且条件良好；slab 的真空方向同样需要
非零矢量。PBC 标志对应晶格矢量。给出晶胞而省略 `pbc` 时三个方向都是周期的，因此请显式声明 `pbc`。

`cell_unit` 缺省等于几何单位：内联数组为 bohr，XYZ 文件为 angstrom。显式设置 `geometry_unit` 和 `cell_unit`，
可在更换几何来源时保持输入含义不变。

传播坐标 `q[P,N,3]` 和导出的质心轨迹保持展开，使环弹簧、质心和 MSD 连续。周期势能和周期 CV 自行处理最小镜像；
在倾斜晶胞中，这与逐个分数坐标取整不同。原子核是可区分的闭合环，不采样交换和缠绕。

H、D、T 在电子 PES 上都对应氢，差别在于核质量。可用 `symbols=["D"]`、`isotopes` 或 `masses_amu` 指定。
同位素沿用同一 Born–Oppenheimer PES；对每种同位素分别收敛珠子数和平衡时间。

### 周期 CV

| CV 类型 | 几何 | 用途 |
| --- | --- | --- |
| `distance`、`bond_difference`、`angle` | 展开坐标，不取最小镜像 | 分子定义、显式展开的几何 |
| `periodic_distance`、`periodic_bond_difference`、`periodic_angle` | 按体系晶胞与 PBC 取最近镜像 | 周期键长、键角与 FES |
| `projection`、`fractional` | 连续质心投影或分数坐标 | 区分相邻晶胞的有向位点跃迁 |
| `surface_height` | 吸附原子组 COM 沿表面法向 | 高度自由能、平面分割面 |
| `surface_lateral` | 展开 COM 沿面内方向 | 表面迁移自由能与有向出口 |

`periodic_bond_difference` 也接受别名 `periodic_difference`。最近镜像 CV 在镜像切换处有导数分支，偏置范围应避开该处。
`site_rate` 接受其支持的线性质心 CV；`surface_rate` 还接受具有恒定端点质量度量的展开 `distance`。
`periodic_distance` 的分割面须小于周期晶格基最小奇异值的一半，以远离最小镜像切换边界。
键差和变质量度量坐标不自动适用这一表面速率公式。

## 周期势能

ASE 接收真实晶胞、PBC 和每个珠子的坐标，并通过 calculator 转换单位。calculator 提供 stress 时，适配器以
`PESResult.virial[B,3,3]` 返回构型 virial `-volume * stress`（hartree，不含动能与弹簧项）；分块执行也保留该输出。

Python 周期插件使用以下两种契约之一：

- 声明 `metadata["periodic"] = True`，在 `evaluate(q, *, cell, pbc)` 中接收当前晶胞；坐标与晶胞使用 provider 声明的长度单位。
- 声明 `metadata["supports_pbc"] = True` 并实现 `bind_system(system)`，在 `evaluate(q)` 之前调用。绑定的 System
  使用原子单位，因此以 angstrom/eV 工作的插件自行转换保存的晶胞。

对新的 calculator 先运行 `validate --check-pes`，再独立检查力和周期平移。

i-PI 在 POSDATA 中发送真实晶胞及其逆矩阵，采用协议的列向量约定。POSDATA 不含 PBC 标志，因此 slab 需设置与体系一致的
`potential.params.init.pbc`，服务端需读取这一 INIT 字段。周期模型请通过 ASE、Python 或 i-PI 接口连接；C ABI v1、
RPMDrate/f2py 和 Torch 适配器用于分子体系。

周期与固定原子体系在 Python 时间步循环中运行，并使用编译的确定性子步骤；使用 `loop_backend="auto"` 或 `"python"`。

## PIMD：平衡性质

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/manual-pimd
pyrpmd analyze runs/manual-pimd
pyrpmd resume runs/manual-pimd
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/pimd.inp)使用周期余弦势。
`sampling.equilibration` 是初始平衡时间，`production` 是采样时间，`interval` 是保存间隔，`replicas` 是独立随机流数。

| `dynamics.integrator` | PILE-L 时的 `sampling.integrator` | Andersen 名称 |
| --- | --- | --- |
| `cayley-bcb` | `bcocb` | `cayley-bcb-andersen` |
| `normal-mode-bab` | `baoab` | `normal-mode-bab-andersen` |
| `velocity-verlet` | （无 PILE-L） | `velocity-verlet-andersen` |

`sampling.thermostat="pile-l"` 时，`gamma_centroid="0.01 fs^-1"` 设置质心摩擦，内部模按正常模热化。
采样与生产使用相同的时间步。`loop_backend="auto"` 时，PILE-L 对 `P<=8` 和 2 的幂次珠子数使用 C++ 自由环
半步–OU–半步内核，势能调用和 NumPy 随机数保留在 Python 中；其他珠子数使用 NumPy 实现。PILE-L 要求
`loop_backend="auto"` 或 `"python"`。

摘要中的 `estimators` 包括 `potential_energy`、`kinetic_primitive`、`kinetic_centroid_virial`、
`internal_energy_centroid_virial` 和 `ring_polymer_hamiltonian`。前四项是物理能量估计量，最后一项是传播诊断。
每项有 `mean` 和 `standard_error_between_replicas`，只有一个副本时后者为 `null`。
`radius_of_gyration_squared_bohr2` 给出每个原子的平均环回转半径平方。

`samples/materials_XXXX/manifest.json` 索引不可变 NPZ 分块，保存质心坐标、速度、采样步数、能量估计量与
回转半径。`checkpoints/materials_XXXX.npz` 保存状态和已提交的轨迹前缀。默认
`output.trajectory_storage="stream"`、`output.chunk_frames=256`；`"memory"` 保留旧的单 NPZ 格式。
已完成的旧任务可在不加载 PES 的情况下迁移。离线相关分析仍加载标量或所选原子的历史，需要单独预算内存。
详见[流式存储](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/streaming-trajectories.md)。输运性质使用 `diffusion`。

### 恒压 PIMD

```toml
[barostat]
method = "bzp"
pressure = "1 bar"
time_constant = "1000 fs"
```

各向同性 Bussi–Zykova–Parrinello barostat 耦合质心与晶胞，保持内部环模不随晶胞缩放。要求三维完全周期、
全部原子可动及能够更新晶胞并返回物理构型 virial 的 PES。支持 ASE、i-PI、兼容的 Python 模型和独立副本
进程并行；暂不支持 NPT 的珠子进程池、MPI 或 GPU 模式。压力可用 Pa/kPa/MPa/GPa/bar/atm/原子单位；
可选 `friction` 默认取时间常数的倒数。逐帧保存晶胞、压力和体积，检查点保存 barostat 动量。
见[算法与示例](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials-npt.md)。RPMD 扩散生产段仍固定晶胞；默认使用无恒温器传播，可显式选择物理电子摩擦浴。

## 扩散：从连续轨迹得到 MSD 与 D

`task="diffusion"` 在热化后保存 t=0，默认在没有恒温器、伞势或 metadynamics 的情况下传播 RPMD。它保存每个原子的
质心位置和速度，被测原子、参考系和拟合都可以事后更改。

显式指定 `[electronic_friction]` 可使用 Markovian 质心 EF-RPMD 生产浴，包括固定基底。
这一路径交换能量，不应用 NVE 能量守恒判据；它没有非 Markovian 记忆或虚时间分布的浴重整化。
方法和限制见[电子摩擦](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md)。

```bash
pyrpmd run examples/materials/diffusion.inp --output runs/manual-diffusion
pyrpmd analyze runs/manual-diffusion --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40 --bootstrap 100
pyrpmd resume runs/manual-diffusion
```

[例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/diffusion.inp)是一个自由粒子，用作弹道运动的
测试：分析给出 `analysis_status="ballistic_or_drifting"` 和 `diffusion_m2_s=null`。

| `diffusion` 字段 | 含义与单位 |
| --- | --- |
| `atom_indices` | 被测原子的 0 起始索引；省略时选 H/D/T |
| `axes` | 被测笛卡尔分量，如 `[0,1]` 表示 xy |
| `reference_atoms` | 每帧扣除其质量加权 COM 的原子；不与被测原子重叠 |
| `max_lag` | 最大延迟，以保存帧数计 |
| `fit_range` | 拟合的延迟时间范围，须在保存范围内 |
| `bootstrap` | 对独立副本的重采样次数；0 表示关闭 |
| `block_length` | 单副本分块的保存帧数；至少 `4*max_lag` |

对 d 个分析分量，位移张量与扩散张量为

$$
M_{\alpha\beta}(t)=\langle\Delta\bar r_\alpha(t)\Delta\bar r_\beta(t)\rangle,
\qquad D_{\alpha\beta}=\frac12\frac{dM_{\alpha\beta}}{dt},
\qquad D=\frac{\operatorname{tr}D_{\alpha\beta}}d,
$$

因此面内 MSD 斜率除以 4，三维斜率除以 6。PyRPMD 还保存对称 VACF 张量及其累积积分。用 `reference_atoms` 去除晶格漂移。

`analysis_status` 和 `results/diffusion.json` 报告拟合结果：

| 状态 | 含义 |
| --- | --- |
| `insufficient_data` | 帧数、拟合点或时间原点不足 |
| `no_observed_motion` | 记录区间内没有运动 |
| `ballistic_or_drifting` | 弹道运动或漂移 |
| `non_diffusive_window` | 拟合区间不是线性的 |
| `indefinite_diffusion_tensor` | 拟合张量不是半正定的 |
| `unstable_fit_window` | 区间前后两半给出的斜率不同 |
| `diffusive_regime_candidate` | 处于线性扩散区；报告 D 与张量 |

所有状态都保存 `raw_fit`、MSD 和 VACF；只有 `diffusive_regime_candidate` 时才填入 D 和张量。误差来自对独立副本或
单副本分块的 bootstrap。用拟合区间、生产长度、副本数和体系尺寸确认 D。

## 位点与表面速率

两个局部速率工作流都使用恒定质量度量的线性质心坐标 s，并从同一位置密度得到分割面值与盆地概率：

$$
k_{\mathrm{QTST}}=
\sqrt{\frac{g^TM^{-1}g}{2\pi\beta}}
\frac{\rho(s^\ddagger)}{\int_A\rho(s)\,ds},
\qquad k(t)=k_{\mathrm{QTST}}\kappa(t).
$$

g 是坐标对原子质心的梯度，A 是显式反应物盆地。`surface_rate` 从质量度量中去除固定原子。盆地和分割面需要连通且
未 mask 的采样，每次 bootstrap 抽样同时计算密度和盆地积分以保留其协方差。

两者都给出一个有向出口的一阶速率，单位 `s^-1`。总逃逸率和宏观扩散系数由多个出口结合跳跃距离、位点数和跳跃相关性得到。

### `site_rate`：材料位点

```bash
pyrpmd prepare examples/materials/site_rate.inp --pilot
pyrpmd run examples/materials/site_rate.inp --output runs/manual-site-rate
pyrpmd analyze runs/manual-site-rate
```

[例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/site_rate.inp)使用展开的 `fractional`
质心坐标。支持的 CV 为 `cartesian`、`projection` 和 `fractional`，后两者可相对于一组晶格原子。`site_rate.cv`
指定伞抽样 CV，`basin=[lo,hi]` 定义有限盆地；`direction=1` 时分割面在 hi，`direction=-1` 时在 lo。

窗口与分析范围覆盖整个盆地并越过出口。窗口和受约束母链使用 Andersen 恒温器，±p 子轨迹不恒温。有限样本的 κ 保留其符号。
输出为 `site_qtst.json` / `.npz`、`pmf/fes.csv` 和 `recrossing.csv`；摘要字段为 `k_qtst_s`、`kappa`、`rate_s`、
`rate_estimate_s`、`rate_standard_error_s`、`rate_percentile_95_ci_s` 和 `rate_uncertainty`。

### `surface_rate`：带固定层的表面

```bash
pyrpmd validate examples/surface/diffusion.inp --check-pes
pyrpmd run examples/surface/diffusion.inp --output runs/manual-surface-rate
pyrpmd resume runs/manual-surface-rate

pyrpmd run examples/surface/fes2d.inp --output runs/manual-surface-fes
pyrpmd analyze runs/manual-surface-fes --recompute --bins 24 12 --bootstrap 40
```

第一个例子 `diffusion.inp` 运行的是 **`surface_rate`**（不是 MSD 的 `diffusion` 任务）：在可分离解析表面上离开一个
局部势阱向右的速率。单核约需 1–2 分钟；二维表面 FES 约需 1 分钟。

`system.fixed_atoms` 冻结所列原子的全部珠子，动量为零；它们仍参与 PES 并作用于可动原子。速率坐标是一个质心
`cartesian`、`surface_height` 或 `surface_lateral` CV。`surface.reactant_basin`、`surface.dividing_surface` 和
`surface.product_side="higher"/"lower"` 定义盆地与产物一侧；整个盆地位于分割面的反应物一侧。这些长度带单位书写。

输出为 `surface_pmf.csv` / `.npz` / `.json`、`recrossing.csv`、`umbrella/` 和摘要，速率字段为 `rate_s_inv`、
`rate_estimate_s_inv`、`rate_standard_error_s_inv` 和 `rate_percentile_95_ci_s_inv`（单位与 `site_rate` 相同，
字段名不同）。若部分 bootstrap 抽样中盆地缺少支持，联合区间留空。

表面 `fes` 支持自适应伞抽样；`surface_rate` 使用固定窗口。运行真实的 ASE calculator 需要安装 ASE：

```bash
python -m pip install -e '.[ase]'
pyrpmd validate examples/surface/ase_emt.py --check-pes
pyrpmd run examples/surface/ase_emt.py --output runs/manual-surface-emt
```

该输入用 ASE EMT 计算 Cu(111) 上的 Cu 吸附原子，检验 calculator、倾斜晶胞和固定层的连接。模型与来源见
[材料说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials.md)、
[表面说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/surface-catalysis.md)和
[Ni–H 材料 pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)。
