# 输入规范 v1

`.inp` 和 `.toml` 使用 TOML 语法；`.py` 执行模块内的 `build()`，返回 `JobSpec` 或字典。两种入口经同一个解析器产生规范配置。输入中相对路径以**输入文件所在目录**为基准；命令行 `--pes`、`--pes-manifest`、`--output` 以启动目录为基准。`module:object` 和 `file.py:object` 均可作为插件入口。

## 通用字段

| 字段 | 类型、单位与含义 |
| --- | --- |
| `schema_version` | 当前为整数 1 |
| `task` | `rate`、`drpmd`、`fes`、`surface_rate`、`network`、`metadynamics`、`pcet`、`pimd`、`diffusion`、`site_rate` |
| `preset` | `pilot`（缺省）或 `production` |
| `temperature` | 正温度，例如 `"300 K"`；数值也按 K 解释 |
| `beads` | 正整数；`auto` 只在 pilot 中使用 P=32 并标记未收敛 |
| `seed` | 非负整数，任务的随机流由稳定任务 ID 派生 |
| `dynamics.timestep` | 例如 `"0.1 fs"`；纯数值按原子时间 |
| `dynamics.integrator` | `cayley-bcb`（缺省）、`normal-mode-bab`、`velocity-verlet`；参见[积分器规范](integrators.md) |
| `dynamics.native` | `true` 优先编译内核并允许串行 NumPy 回退；`false` 用 NumPy；`"required"` 必须有编译内核 |
| `output.directory` | 运行目录；新任务不覆盖已有非空目录 |
| `output.checkpoint_interval` / `checkpoint_steps` | 时间间隔或正整数步数；步数优先 |
| `output.trajectory_stride` | dRPMD 轨迹保存的正整数步数间隔 |
| `output.trajectory_storage` / `chunk_frames` | 材料任务默认 `"stream"`、256 帧；`"memory"` 使用旧格式 |

`production` 要求显式指定温度、珠子数、dynamics 和 sampling；dRPMD 还要求显式 `bmax`。这只是输入完整性要求，科学收敛仍需用户用系列计算确认。未知顶层和常用 section 字段会报错，不会自动忽略。

`pcet` 使用下述独立模型输入，不使用分子系统、PES、珠子数、传播与采样
预设；其 `temperature` 也接受温度列表以运行扫描。

时间数量必须是步长的整数倍。需要非整除的时间时请先选择实际整数步数，例如 `equilibration_steps=333`。纯数值的内部单位是 bohr、hartree、原子时间、电子质量；XYZ 坐标缺省为 angstrom。力常数支持 `"10 kcal/mol/angstrom^2"` 等组合单位。所有原子索引均为 **0-based**。

## 系统

`[system]` 可使用 `transition_state="ts.xyz"`、`geometry="system.xyz"`，或 `reactants=["A.xyz","B.xyz"]` 加 `separation="12 angstrom"`。dRPMD 的 separation 是两个反应物 centroid COM 的初始**径向距离** R0；发射时使用 `z=sqrt(R0^2-b^2)` 保持径向距离，需 `bmax<R0`。

Python 或 TOML 也可提供 `symbols`、`geometry=[[...], ...]` 与 `masses`（电子质量）或 `masses_amu`。`geometry_unit` 可覆盖坐标单位。同位素可用 `isotopes=[{atom=0, mass="2.01410177812 amu"}]`。质量在反应物居中和放置前解析。电荷 `charge` 为整数，`multiplicity` 为正整数，默认 0 和 1；须与所选电子势能面一致。软件不从几何自动推断自旋统计权重。

固定晶胞使用 `cell=[[...],[...],[...]]`（每行一个矢量）、`cell_unit="angstrom"`
及 `pbc=[true,true,false]`。有 cell 时默认全周期，无 cell 时默认非周期。
晶胞、部分 PBC、连续质心坐标及支持的势能接口见[材料计算](materials.md)。
## 周期表面

周期表面新增 `system.cell`（3×3 行向量）、`cell_unit`（缺省继承 geometry_unit；inline 默认 bohr，XYZ 默认 angstrom）、
`pbc=[true,true,false]` 和 `fixed_atoms=[...]`。这些字段参与恢复的科学指纹。
`task="surface_rate"` 还要求 `[surface]` 的 `reactant_basin` 和 `dividing_surface`；
`product_side` 为 `higher`（默认）或 `lower`。速率使用位置密度 PMF 和一阶盆地
归一化，单位 s^-1，仅支持内置线性 centroid 坐标。已有气相 `rate`／`drpmd`／
`network` 会拒绝周期／固定原子系统。表面 FES 可使用自适应伞抽样；当前
`surface_rate` 不接受 `[adaptive_umbrella]`。`metadynamics` 支持固定周期晶胞和固定
原子，偏置网格设置见 [网格 metadynamics](grid-metadynamics.md)。表面速率的
完整配置、公式与限制见 [表面催化](surface-catalysis.md)。

## 热化与持续母轨迹

```toml
[sampling]
integrator = "cayley-bcb-andersen"
thermostat = "andersen"
equilibration = "5 ps"
interval = "0.2 ps"
production = "20 ps"
andersen_interval = 10
replicas = 2
parents = 8
```

`equilibration` 是每条链首次平衡时间。dRPMD 的 `interval`/`interval_steps` 是**同一母轨迹两次抽取之间推进的时间/步数**；不是温度，也不会每次从初始几何重新热化。`parents` 控制母链数，省略时使用轨迹工作进程数。`collision.trajectories=N` 是总碰撞数，按 `N//parents` 加余数分配。若要改变进程数量后续跑，首次就显式固定 `parents`。

FES 使用 `replicas` 指定每个窗口的独立链数，`production` 是每条链的生产采样时间，`interval` 是保存 CV 样本的间隔。FES、气相和位点速率工作流使用所选动力学积分器 + Andersen；省略 `sampling.integrator` 时自动匹配，显式指定时须为所选 canonical 名称加 `-andersen`。`pimd`/`diffusion` 还支持与 Cayley 匹配的 BCOCB/PILE-L、与正常模匹配的 BAOAB/PILE-L，`sampling.gamma_centroid` 设置质心摩擦。采样和子轨迹步长须相同。真实碰撞、回穿子轨迹和 `diffusion` 生产段使用无温控传播。

## 材料任务

`pimd` 和 `diffusion` 使用上述 sampling 时间配置，不接受 reaction、umbrella 或
metadynamics 偏置。`diffusion` 增加 `[diffusion]`：`atom_indices`、`axes`、
`reference_atoms`、`fit_range`、`max_lag`、`bootstrap`、`seed`、`block_length`。
`max_lag`/`block_length` 的单位是保存帧，`fit_range` 是两个时间数量；所有选择索引
均从零起始。`site_rate` 使用一个线性 centroid CV、固定 umbrella 和 `[recrossing]`，
并显式设置 `[site_rate]` 的 `cv`、`basin`、`dividing_surface`、`direction`。
三种任务的完整配置与结果解释见[材料计算](materials.md)。

`pimd` 和 `diffusion` 可固定部分原子；扩散选择的测量原子必须可动，参考原子可固定。
`pimd` 可添加 `[barostat]`：`method="bzp"`、`pressure="1 bar"`、
`time_constant="1000 fs"`，可选 `friction`。它要求三维完全周期、全部原子可动、
支持变晶胞及物理 virial 的 PES；各向同性压力不改变晶胞形状。详见 [NPT](materials-npt.md)。
材料轨迹默认流式保存，见 [分块与续跑](streaming-trajectories.md)。

## dRPMD

```toml
[collision]
trajectories = 10000
sampling = "thermal_flux"
bmax = "5 angstrom"
max_time = "2 ps"

[outcomes]
exit_distance = "12 angstrom"
persistence = "20 fs"
check_interval = 5
bond_cutoffs = { "O-H" = "1.3 angstrom", "H-H" = "0.9 angstrom" }
channels = { water = ["H2O", "O2"] }
```

示例 channel 名/组成只是语法展示，必须与实际反应原子守恒一致。`thermal_flux` 使用 E~Gamma(2,kBT) 的通量分布；`maxwell` 使用普通相对速度分布并逐轨迹按 v 加权；`fixed_energy` 另需 `energy`，只给该能量散射截面，不给热速率。冲量参数按面积均匀采样，整体空间取向各向同性。

最小 pilot 输入省略分类设置时，程序按共价半径和 1.25 倍系数推断连通 cutoff，并输出暂定候选标签、推断值和假设。它不推断电荷、键级或全部化学身份；热速率、散射截面和化学结果上下界均暂不报告。正式 production 输入必须提供显式 cutoff 或自定义分类器。

显式内置分类器由 centroid 连通图、分子式、远离条件与持续时间分类。它不能可靠区分所有相同组成异构体或同元素交换通道；这些情况必须给 `outcomes.factory="classify.py:build"`。插件返回 `reactants`、产品通道名或未决状态，并应自行定义必要的物理判据；实际协议见 [势能和插件说明](potentials.md)。未决和失败样本单独计数，存在它们时热速率点值留空并给出观察下界/可能上界。统计置信区间与这些截断界不是同一概念。

## 标准 RPMD 速率

最小 `rate` 输入从 TS 和 PES 自动准备。无法唯一选择入口时，填写：

```toml
[reaction]
fragments = [[0], [1, 2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
rinf = "12 angstrom"
electronic_weight = 1.0
symmetry_factor = 1.0
```

单个 `task="rate"` 使用一个 TS 参考几何和等长、成对的成键/断键列表。不同入口或非等价 TS 结构使用下述 `task="network"`。通过
`equivalent_paths` 可声明由原子置换生成的等价过渡态。例如 H + H₂ 的
两个交换出口可写为：

```toml
[reaction]
fragments = [[0], [1, 2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
equivalent_paths = [[0, 2, 1]]
rinf = "10 bohr"
symmetry_factor = 1.0
electronic_weight = 1.0
```

置换列表使用 0-based 的 `旧原子索引 → 新原子索引` 映射；原始 TS 已包含，
不要再列恒等置换。程序检查完整双射、元素、同位素质量和入口片段划分，
同时置换参考几何与成断键。多个不同 TS 结构由 `network.transition_states` 声明。
组合方式严格沿用 RPMDrate 的 `min_bond(max_TS(s1))`，其分支交界处是
分段可微的。等价出口已经显式包含时，不应再用 `symmetry_factor` 重复
乘一次路径数。`electronic_weight` 和额外的 `symmetry_factor` 仍由用户
根据势能和通道定义给出；`reaction.entry_side` 可选择数值下坡两端之一。

`[recrossing]` 的主要字段是 `parent_replicas`、`parent_samples`、`equilibration`、`parent_interval`、`child_pairs`、`child_time`、`dividing_surface`、`plateau_fraction`、`plateau_tolerance`、`bootstrap`。时间也可用相应 `_steps` 字段。每个 child pair 以相反动量传播，禁止以未计算的 κ=1 替代回穿。完整可运行语法见 `examples/rate/pilot.inp`。

### 速率 PMF 后端与分割面

默认 `rate.pmf_method = "ui"` 保留 Gaussian umbrella integration。
选择 histogram MBAR 并固定分割面可写：

```toml
[rate]
pmf_method = "mbar"
dividing_surface = 1.00925
compare_ui = true

[analysis]
bins = 170
ranges = [[-0.30, 1.40]]
min_count = 5
bootstrap = 40
seed = 12345
```

这里的分割面、bin 范围是 H₃ 示例参数，需根据其他反应的坐标和采样支撑设置。
MBAR 下 `compare_ui` 默认开启，UI 使用相同样本与相同 block-bootstrap draw，
可以计算配对差值，且不重复运行 MBAR 重采样。
`bootstrap = 0` 可用于单纯曲线诊断；不能据此报告完整速率误差。
`bootstrap = 1` 不足以估计方差，会在输入阶段报错。

省略 `dividing_surface` 或写 `"auto"` 时自动选峰，
可用 `rate.pmf_search_range = [0.8, 1.1]` 限定搜索区间。
MBAR 使用未平滑的支撑 bin 中心；参考点到分割面的全部区间须有连续支撑。
Bootstrap 保持选定分割面固定，不包含自动选峰的不确定度。
旧的 `recrossing.dividing_surface` 仍可用；两处同时指定不一致会报错。
改变 PMF 方法或分割面不能复用另一设置的母/子轨迹 checkpoint。

结果保存在 `rate_pmf.npz` 和 `rate_pmf.json`，包含密度、mask、误差、
端点支撑、求解残差和逐次重采样状态。断连、缺失端点/中间 bin 或求解失败
会给出独立失败状态，并停止启动 recrossing；不会跨越 mask 补出势垒。
进一步说明见 [MBAR 速率 PMF](mbar-rate-pmf.md)。

## 一维与二维 FES

```toml
[[cvs]]
name = "r1"
kind = "distance"
atoms = [0, 1]
estimator = "centroid"
unit = "angstrom"

[[cvs]]
name = "r2"
kind = "distance"
atoms = [1, 2]
estimator = "centroid"
unit = "angstrom"

[umbrella]
cvs = ["r1", "r2"]
grid = [{start=0.8, stop=2.0, count=9}, {start=0.8, stop=2.0, count=9}]
stiffness = ["40 kcal/mol/angstrom^2", "40 kcal/mol/angstrom^2"]

[analysis]
method = "mbar"
measure = "cv_density"
bins = [40, 40]
bootstrap = 100
min_count = 5
```

内置 CV 包括 distance、bond_difference、angle、cartesian、com_distance、rpmdrate_xi、surface_height、surface_lateral、periodic_distance 和 python。内置几何 CV 可选择 `centroid` 或 `bead_average`；自定义 CV 返回自己定义的完整 estimator 和导数。`stiffness` 可为标量、对角数组或对称正定矩阵。数值 centers 使用声明的 CV 输入单位；`centers_au` 与数值 stiffness 使用原子单位，带单位字符串按量纲验证。

输出 FES 是相对于所选 CV 测度的密度：例如距离 CV 的自由能自然包含径向熵，不自动把它变成去除 Jacobian 的两体势。没有样本或连通重叠支持的区域 mask，不外推。`analyze --recompute --bins ... --bootstrap ...` 使用保存的样本重新估计而不调用 PES。

## 多入口与不同 TS

`task="network"` 使用 `[network]` 中的 `entrances`、`channels`、
`transition_states` 记录列表。每个 TS 显式引用一个入口和通道标签，提供
自己的结构、成断键，按全局 → 入口 → TS 继承或覆盖 PES 与采样设置。
入口指定自己的两个片段、`rinf` 和电子统计权重；所有路径共享有序原子与
同位素身份。每条路径经过普通 `rate` 的严格验证并运行完整 PMF/recrossing。

当前 `network.aggregation` 仅允许 `"separate"`。结果保留独立入口/TS
归一化和误差；通道名称不会被当作通量不重叠的证明。完整 TOML/Python
语法与可运行四路径例子见[多入口与多 TS](reaction-networks.md)。

## 自适应伞抽样

为 `fes` 或 `rate` 添加 `[adaptive_umbrella]`：

```toml
[adaptive_umbrella]
enabled = true
pilot_steps = 1000
pilot_equilibration_steps = 200
max_rounds = 4
max_windows = 64
target_overlap = 0.03
min_effective_samples = 20.0
min_spacing = 0.000001
```

初始窗口、范围与刚度仍由 `[umbrella]` 给出；适应阶段按实际 MBAR
overlap 与有效样本数插窗、延长 pilot。窗口冻结后使用独立种子重新平衡并
采集 production；pilot 不进入最终 FES/PMF。预算耗尽明确失败，不静默跳过
重叠要求。参数单位、阶段恢复及适用边界见[增强采样](enhanced-sampling.md)。

## Metadynamics

`task="metadynamics"` 使用普通 `[[cvs]]` 和 `[sampling]`，替换 umbrella 为：

```toml
[metadynamics]
cvs = ["x"]
mode = "well_tempered"
sigma = ["0.2 bohr"]
height = "0.02 kcal/mol"
pace = 50
bias_factor = 8.0

[analysis]
method = "bias_estimate"
measure = "cv_density"
bootstrap = 0
```

`mode="standard"` 使用固定 hill 高度并省略 `bias_factor`。`sigma_au`、
`height_au` 可指定原子单位，但不得与对应未带 `_au` 的字段重复。
`pace` 是生产步数间隔；1D/2D、周期 CV 与独立 replicas 均支持。
`loop_backend="native"` 会预先报错，因为偏置历史随时间改变；`auto` 的
兼容传播仍能使用所选积分器的编译子步。有限时间 FES 估计及 HILLS/恢复
时序见[增强采样规范](enhanced-sampling.md)。

## 电化学 PCET 模型

`task="pcet"` 计算固定局域复合物的一维单质子 golden-rule 速率，输出
还原/氧化的 s⁻¹ 速率和 H/D KIE；它不执行 RPMD 轨迹。输入不需要
`[system]` 或 `[potential]`，而使用 `[pcet]`、`[pcet.electrode]`、
`[pcet.proton]` 及两电子态的质子势：

```toml
task = "pcet"
temperature = ["250 K", "300 K", "350 K"]

[pcet]
method = "golden_rule"
normalization = "local_first_order"
isotopes = ["H", "D"]
driving_force = "0.0 eV"
reorganization_energy = "0.7 eV"
electronic_coupling = "0.001 eV"

[pcet.electrode]
model = "wide_band"
density_of_states = "1 eV^-1"
potential_reference = "SHE"
reference_potential_V = 0.0
potentials_V = [-0.2, 0.0, 0.2]

[pcet.proton]
grid_min = "-1.5 angstrom"
grid_max = "1.5 angstrom"
grid_points = 401
states = 12

[pcet.proton.oxidized]
kind = "harmonic"
center = "-0.2 angstrom"
force_constant = "4 eV/angstrom^2"

[pcet.proton.reduced]
kind = "harmonic"
center = "0.2 angstrom"
force_constant = "5 eV/angstrom^2"
```

这些是无真实材料归属的教学参数。`driving_force` 是参考电势下 reduced
减 oxidized 的最低点自由能差，不含转移质子的零点能与热振动自由能；
两个质子势分别去除自身最低值，由本征态求解器提供质子量子贡献。
电势增加时按 `+e*(U-U_ref)` 增大还原自由能。`potential_reference` 只记录
标尺，不自动转换 SHE/RHE 或加入 pH 依赖。

`[pcet.numerics]` 可设置 `quadrature_rtol`、`quadrature_atol`、
`convergence_check`、`relative_tolerance` 和 `absolute_tolerance_s`。
数值收敛与弱耦合/宽带/经典环境的物理适用性分开报告。完整可运行示例、
表格势字段、速率公式和输出解释见 [PCET 规范](pcet.md)与
[教学示例](../examples/pcet/README.md)。

## 运行与重启记录

`manifest.json` v2 记录程序版本、环境、并行配置、单位、科学指纹和本地输入/PES/plugin 内容 SHA256。`input/resolved.json` 是解析后的规范输入，`input/system.json` 保存原子和质量。新 NPZ checkpoint 使用 v2 容器与逐数组校验，始终以 `allow_pickle=False` 读数组；RNG/进度采用同一容器内的 JSON 字符串或明确的工作流元数据。旧 v1 文件的兼容读取与 manifest 迁移见[恢复规范](checkpoints.md)。任意 Python 间接依赖和远程 PES 服务内部状态不属于自动依赖闭包，应另行固定运行环境。

PCET 用 `input/model.json` 保存一维模型，不创建分子 `system.json`；
其扫描点 checkpoint 是带模型身份与校验的 JSON，不包含 q/p 或 RNG。

## 连续原生传播

`[dynamics]` 新增 `loop_backend="auto"|"native"|"python"` 和
`block_steps=256`（1..65536）。`auto` 对已支持的原生 PES/CV 使用 C++17
连续传播，其余情况保留 Python；`native` 在请求的 block 路径不支持时明确报错。
`python` 保留逐步调度，不必关闭所选积分器的 C++ 单步内核。
受约束 recrossing 母轨迹、dRPMD 碰撞与支持原生执行的 FES/PMF 采样使用
`propagate_block()`，其单次调用
上限为 4096 步；工作流自动把较大的 `block_steps` 限制到这个上限，并继续
遵守原有热浴、分类、采样和 checkpoint 时间轴。输入范围与默认值仍为
1..65536 和 256；recrossing child 与普通 Andersen 热化继续使用 `propagate()`。
支持范围及两种 API 的故障语义见 [连续传播说明](native-propagation.md) 和
[分段 API](block-propagation.md)。
