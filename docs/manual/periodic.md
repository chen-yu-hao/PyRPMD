# 周期材料、扩散与表面反应

周期结构描述几何边界，势能模型决定材料相互作用，工作流定义要估计的物理量。固定晶胞并不自动意味着可以测得扩散；一个表面自由能势垒也不直接给出扩散系数或催化速率。本章分别介绍平衡 PIMD、无恒温 RPMD 输运，以及盆地归一化的局部速率。

## 先选择任务和边界

| 任务 | 目的 | 周期晶胞 | `fixed_atoms` | 核心结果 |
| --- | --- | --- | --- | --- |
| `pimd` | 平衡量子统计 | 支持 | 当前拒绝 | 能量估计器、回转半径 |
| `diffusion` | 连续质心轨迹与输运分析 | 支持 | 当前拒绝 | MSD、VACF、D 与诊断 |
| `site_rate` | 全原子可动材料中的选定局部出口 | 支持 | 当前拒绝 | 一个光滑标量质心 CV 的盆地速率，含非线性质量度量修正 |
| `fes` | 一维/二维几何自由能 | 支持 | 支持 | 位置密度 FES、overlap、mask |
| `surface_rate` | 可含固定层的表面局部出口 | 支持，也可使用有限簇 | 支持 | 活动自由度质量度量的一阶速率 |

材料能量与输运估计器目前假定全部原子可动，因此合并后的程序明确拒绝在 `pimd`、`diffusion`、`site_rate` 中设置固定原子。需要固定底层时，选择表面 `fes` / `surface_rate`。气相 `rate`、`drpmd`、`network` 和当前 `metadynamics` 不接受周期或固定原子体系。

所有活动原子使用同一珠子数，没有“吸附物多珠子、活动基底单珠子”的混合分辨率入口。当前没有 NPT/变胞、电子摩擦、非绝热跃迁、集体电导率或自动 kMC 构建。

## 晶胞、连续坐标与同位素

下面是一个结构表的写法示意；`slab.xyz` 应替换为目标体系的实际文件：

```toml
[system]
geometry = "slab.xyz"
geometry_unit = "angstrom"
cell = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
```

晶胞每一行是一个晶格矢量，笛卡尔位置满足 `r = s @ cell`。矩阵应有限、可逆且条件良好；slab 的非周期真空方向也必须有非零矢量。PBC 标志对应晶格矢量，而不一定是笛卡尔 x/y/z 轴。提供 cell 而省略 pbc 时缺省为三个方向周期，建议总是显式声明。

`cell_unit` 缺省继承几何输入使用的单位：内联数组缺省 bohr，XYZ 缺省 angstrom。明确给出 `geometry_unit` 和 `cell_unit` 可以使输入在改换几何来源后仍保持原意。

传播中的 `q[P,N,3]` 和导出的质心轨迹保持连续、未包装的坐标。不要在每步后将每个珠子独立 wrap 到晶胞：这会改变环弹簧、质心和 MSD。周期势能及显式周期 CV 负责镜像处理；倾斜晶胞的最近像不应简单等同于逐分数坐标四舍五入。此路径积分表示使用局域核的闭合环，不采样核交换或非零绕胞拓扑。

H/D/T 在电子 PES 上都对应氢元素，区别在于核质量。可以使用 `symbols=["D"]` 或显式 `isotopes` / `masses_amu`。替换质量无需重新定义同一 Born–Oppenheimer PES，但珠子数、平衡时间和结果的收敛必须分别检查。

### 显式选择周期 CV

| CV 名称 | 几何含义 | 适合的用途 |
| --- | --- | --- |
| `distance`、`bond_difference`、`angle` | 连续展开坐标上的普通几何，不自动做最小镜像 | 原有分子定义、明确展开的几何 |
| `periodic_distance`、`periodic_bond_difference`、`periodic_angle` | 按系统 cell/PBC 求最近像几何 | 周期键长、角度与 FES |
| `projection`、`fractional` | 连续 centroid 投影或分数坐标，不取模 | 区分相邻晶胞的定向位点跃迁 |
| `surface_height` | 吸附原子组 COM 沿表面法向的投影 | 高度自由能、线性表面分割面 |
| `surface_lateral` | 指定方向的连续展开 COM 投影 | 面内迁移自由能与选定出口 |

`periodic_bond_difference` 也接受别名 `periodic_difference`。普通几何 CV 不会因任务类型或是否固定底层而隐式改变定义。最近像 CV 在像切换处可有导数分支，需检查所选范围是否适合平滑偏置和采样。`site_rate` 支持满足光滑标量质心合同的非线性坐标，并对约束母轨迹施加质量度量权重；`surface_rate` 保留其原有恒定质量度量限制。周期像切换或坐标奇点不能直接当作光滑分割面，详见[通用坐标速率](../general-coordinate-rates.md)。

## 周期势能接口

ASE 会收到真实 cell、PBC 和每珠子坐标，并按 calculator 的单位转换。若 calculator 支持 stress，适配器可返回构型 virial `-volume * stress`，单位转为 hartree。`PESResult.virial[B,3,3]` 不含动量或环弹簧项，也不自动等于量子压力估计器。分块执行保留这一可选输出。

Python 周期插件支持两种明确契约：

- 声明 `metadata["periodic"] = True`，以 `evaluate(q, *, cell, pbc)` 接收当前晶胞；坐标和 cell 使用 provider 声明的长度单位。
- 声明 `metadata["supports_pbc"] = True`，实现 `bind_system(system)` 绑定几何后再执行 `evaluate(q)`。绑定得到的 System 使用原子单位；若 evaluate 使用 angstrom/eV，插件应自行转换保存的晶胞。

已经绑定的周期模型仍须与当前 System 一致。仅写一个能力标志而忽略 cell/PBC 不满足物理契约。对任一新 calculator，先运行 `validate --check-pes`，再做独立的力与周期平移检查。

i-PI 在 POSDATA 发送真实晶胞和逆矩阵，使用协议的列向量约定。POSDATA 不单独传部分 PBC；slab 需要 `potential.params.init.pbc` 与系统匹配，且服务端实现这一 INIT 约定。大真空层本身不能替代部分 PBC 语义。周期 C ABI v1、RPMDrate/f2py 和 Torch 入口当前不支持晶胞契约，应通过适合的 ASE/Python/i-PI 接口连接模型。

周期/固定原子体系使用兼容的 Python 时间步循环，并可利用编译的确定性积分子步；使用 `loop_backend="auto"` 或 `"python"`。不能据此宣称整个周期传播已在原生融合循环或 GPU 上常驻执行。

## PIMD：平衡物理量

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/manual-pimd
pyrpmd analyze runs/manual-pimd
pyrpmd resume runs/manual-pimd
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/pimd.inp)使用周期余弦解析势。`sampling.equilibration` 是首次平衡时间，`production` 是随后的平衡采样时间，`interval` 控制保存间隔，`replicas` 生成独立随机流。

| `dynamics.integrator` | PILE-L 的 `sampling.integrator` | Andersen 的显式名称 |
| --- | --- | --- |
| `cayley-bcb` | `bcocb` | `cayley-bcb-andersen` |
| `normal-mode-bab` | `baoab` | `normal-mode-bab-andersen` |
| `velocity-verlet` | 不支持 | `velocity-verlet-andersen` |

选择 `sampling.thermostat="pile-l"` 时，可用 `gamma_centroid="0.01 fs^-1"` 设置质心摩擦；内部模按正常模热化。材料任务采样和确定性生产步长必须一致。PILE-L 的 `auto` 对 `P<=8` 或二次幂珠数使用 C++ 自由环半步–OU–半步核，Python 保留势能力调用与 NumPy 随机数；较大非二次幂珠数保留 NumPy 参考路径。它不接受强制 `loop_backend="native"` 的完整原生循环请求。

摘要的 `estimators` 包括 `potential_energy`、`kinetic_primitive`、`kinetic_centroid_virial`、`internal_energy_centroid_virial` 和 `ring_polymer_hamiltonian`。前四项描述物理能量估计，最后一项用于环传播诊断。每项有 `mean` 和 `standard_error_between_replicas`；只有一个副本时，副本间误差是 `null`。`radius_of_gyration_squared_bohr2` 给出每个原子的平均环回转半径平方。

`samples/materials_XXXX.npz` 导出 `positions`、`velocities`、`times`、`observables` 和 `radii_gyration_squared`，其中位置/速度是连续质心量。`checkpoints/materials_XXXX.npz` 是恢复状态的权威来源。PIMD 采样时间不应直接用于计算扩散或反应寿命。

## 扩散：从连续轨迹到 MSD 和 D

`task="diffusion"` 在热化后保存 t=0，再进行没有恒温器、umbrella 或 metadynamics 的 RPMD 生产传播。它保存所有原子的质心坐标和速度，以便离线改变被测原子、参考系和拟合设置。

```bash
pyrpmd run examples/materials/diffusion.inp --output runs/manual-diffusion
pyrpmd analyze runs/manual-diffusion --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40 --bootstrap 100
pyrpmd resume runs/manual-diffusion
```

这个[自由粒子负对照](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/diffusion.inp)应识别弹道运动，并保留 `diffusion_m2_s=null`。成功执行但不发布 D，正是这一例子的预期行为。

| `diffusion` 字段 | 含义与单位 |
| --- | --- |
| `atom_indices` | 被测原子的 0-based 索引；省略时选择 H/D/T |
| `axes` | 被测笛卡尔分量，如 `[0,1]` 是 xy；不是分数晶格轴 |
| `reference_atoms` | 每帧扣除其质量加权 COM 位置/速度；不得与被测原子重叠 |
| `max_lag` | 最大延迟，以保存帧间隔数计 |
| `fit_range` | 拟合的物理延迟时间范围，须在实际保存范围内 |
| `bootstrap` | 独立副本重采样次数；0 表示不估计该误差 |
| `block_length` | 单副本分块时的保存帧数；至少 `4*max_lag`，且需足够独立块 |

对 d 个分析分量，位移张量和扩散张量满足

$$
M_{\alpha\beta}(t)=\langle\Delta\bar r_\alpha(t)\Delta\bar r_\beta(t)\rangle,
\qquad D_{\alpha\beta}=\frac12\frac{dM_{\alpha\beta}}{dt},
\qquad D=\frac{\operatorname{tr}D_{\alpha\beta}}d.
$$

因此二维面内 MSD 斜率除以 4，三维除以 6。程序同时保存对称 VACF 张量和有限时间积分供比较；它不自动认证 VACF 积分平台。固定材料参考系可消除晶格整体漂移，但不能通过单独扣掉每个扩散原子的拟合漂移来制造线性扩散。

先查看 `analysis_status` 和 `results/diffusion.json`：

| 状态 | 应如何理解 |
| --- | --- |
| `insufficient_data` | 帧数、拟合点或时间起点不足 |
| `no_observed_motion` | 当前记录看不到运动，不证明真实 D 为零 |
| `ballistic_or_drifting` | 仍处于弹道区或有漂移 |
| `non_diffusive_window` | 拟合区间不是近似线性扩散区 |
| `indefinite_diffusion_tensor` | 拟合张量未达到半正定要求 |
| `unstable_fit_window` | 拟合前后两半的斜率不稳定 |
| `diffusive_regime_candidate` | 通过有限记录的启发式筛选，仍需物理收敛研究 |

非候选状态仍保存 `raw_fit`、MSD 和 VACF，顶层接受的 D/tensor 保持 `null`。一个副本的许多重叠时间起点不是许多独立实验；拟合 OLS 残差也不替代采样误差。改变拟合时间、增加生产长度、独立副本和体系尺寸后，应比较结果的稳定性。

## 位点与表面速率的共同物理定义

对于恒定质量度量的线性 centroid 坐标 s，两个局部速率工作流使用同一位置密度的分割面值与盆地概率。非线性 `site_rate` 进一步使用约束母轨迹的质量度量平均，详见[通用公式](../general-coordinate-rates.md)：

$$
k_{\mathrm{QTST}}=
\sqrt{\frac{g^TM^{-1}g}{2\pi\beta}}
\frac{\rho(s^\ddagger)}{\int_A\rho(s)\,ds},
\qquad k(t)=k_{\mathrm{QTST}}\kappa(t).
$$

g 是坐标对原子质心的梯度，A 是显式反应物盆地。`surface_rate` 在质量度量中剔除固定原子的分量。完整盆地及分割面都需要连通、未 mask 的采样支持；密度和盆地积分必须在每个 bootstrap draw 内一起计算，以保留协方差。

两者输出都是所选有向出口的一阶速率，单位 `s^-1`，不含气相体积、表面积、压力、覆盖度或位点简并因子。一个出口速率不是自动汇总的总逃逸率，也不自动换算成宏观 D；后者还需要跳距、邻居数和连续跳跃相关性。

### `site_rate`：材料位点

```bash
pyrpmd prepare examples/materials/site_rate.inp --pilot
pyrpmd run examples/materials/site_rate.inp --output runs/manual-site-rate
pyrpmd analyze runs/manual-site-rate
```

[示例](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/materials/site_rate.inp)使用未包装的 `fractional` centroid 坐标。`cartesian`、`projection` 或 `fractional` 保持原有恒定度量路径；光滑的距离、键长差、角度及兼容的 Python centroid CV 也可使用非线性度量路径，详见[通用坐标速率](../general-coordinate-rates.md)。`projection` 和 `fractional` 可参考指定晶格原子组。`site_rate.cv` 指定同一个 umbrella CV，`basin=[lo,hi]` 定义有限盆地，`direction=1` 要求分割面为 hi，`direction=-1` 要求分割面为 lo。

窗口与分析范围需覆盖整个盆地，并在出口另一侧提供所需支持。窗口/受约束母链用 Andersen，±p 子轨迹不恒温；负的有限样本 κ 不会被裁为零。结果含 `site_qtst.npz`、`pmf/fes.csv`、`recrossing.csv`，摘要字段是 `rate_s`、`rate_estimate_s`、`rate_standard_error_s` 及 `rate_uncertainty`。

### `surface_rate`：带固定层的表面

```bash
pyrpmd validate examples/surface/diffusion.inp --check-pes
pyrpmd run examples/surface/diffusion.inp --output runs/manual-surface-rate
pyrpmd resume runs/manual-surface-rate

pyrpmd run examples/surface/fes2d.inp --output runs/manual-surface-fes
pyrpmd analyze runs/manual-surface-fes --recompute --bins 24 12 --bootstrap 40
```

虽然示例名为 `diffusion.inp`，其任务是 **`surface_rate`**，计算一个局部井的右侧出口；不要将它与 MSD 的 `diffusion` 任务混淆。解析势是可分离教学模型，不能验证真实热化位点跳跃或材料扩散系数。

使用 `system.fixed_atoms` 固定底层的所有珠子，动量保持为零；这些原子仍进入真实 PES，作用于活动原子。支持一个 centroid `cartesian`、`surface_height` 或 `surface_lateral` 坐标。`surface.reactant_basin`、`surface.dividing_surface` 和 `surface.product_side="higher"/"lower"` 指定归一化区域和产品侧；整个盆地须在分割面的反应物侧。这里的长度推荐写带单位字符串。

表面输出为 `surface_pmf.csv` / `.npz` / `.json`、`recrossing.csv` 和摘要。表面速率字段使用 `rate_s_inv`、`rate_estimate_s_inv`、`rate_standard_error_s_inv`，与 `site_rate` 的字段名称不同，但物理单位相同。若 bootstrap 中盆地支持失败，主联合区间可能被保留为空；仅对成功 draw 的条件统计不能冒充完整误差。

表面 `fes` 支持自适应 umbrella；`surface_rate` 当前要求固定声明的窗口。若要检验实际 ASE 连接，可安装 ASE 后运行：

```bash
python -m pip install -e '.[ase]'
pyrpmd validate examples/surface/ase_emt.py --check-pes
pyrpmd run examples/surface/ase_emt.py --output runs/manual-surface-emt
```

该输入使用 Cu(111) / Cu adatom 与 ASE EMT，验证 calculator、倾斜晶胞及固定层连接；它不证明 EMT 适用于任意催化反应。更详细的模型和来源见[材料说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/materials.md)、[表面说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/surface-catalysis.md)和 [Ni–H 材料 pilot](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/materials/README.md)。
