# 输入、体系与命令行

## TOML 或 Python

输入文件可以是 TOML（`.inp` 或 `.toml`），也可以是定义 `build()` 并返回 `JobSpec` 或字典的 Python 模块
（`.py`）。两者经过同一解析器。TOML 适合简洁地记录一次计算；Python 适合参数扫描、批量任务和可导入插件。

```python
from pathlib import Path
from pyrpmd import JobSpec


def build():
    return JobSpec.from_mapping(
        {
            "task": "pimd",
            "temperature": "300 K",
            "beads": 8,
            "system": {
                "symbols": ["H"],
                "geometry": [[0.0, 0.0, 0.0]],
                "geometry_unit": "bohr",
            },
            "potential": {"provider": "harmonic", "params": {"k": 0.002}},
            "dynamics": {"timestep": "0.1 fs"},
            "sampling": {
                "thermostat": "andersen",
                "equilibration_steps": 100,
                "production_steps": 500,
                "sample_interval": 5,
                "replicas": 2,
            },
        },
        base_directory=Path(__file__).resolve().parent,
    )
```

保存为 `job.py` 后运行 `pyrpmd validate job.py`。Python 输入作为普通的受信任 Python 代码执行。
正式 PIMD 的设置见[材料与表面](periodic.md)。

## 相对路径

| 路径出现在 | 相对于 |
| --- | --- |
| 输入文件中（XYZ、模型、`file.py:factory`、输出目录） | 输入文件所在目录 |
| `JobSpec.from_mapping(..., base_directory=...)` | `base_directory`；省略时为当前目录 |
| 命令行 `--pes`、`--pes-manifest`、`--output` | 运行命令的目录 |
| `.o` 链接 manifest | manifest 所在目录 |

例如，入门输入 `my-first-run/quickstart.inp` 中的 `output.directory="run"` 解析为 `my-first-run/run`，
而命令行 `--output runs/custom` 从当前目录解析。`validate INPUT --output runs/custom` 显示解析后的输入与输出路径，
不创建运行目录。移动输入时，连同其引用的文件一起移动并保持原子顺序。

## 通用字段

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 输入版本，当前为整数 1 |
| `task` | [任务表](../index.md)中的一项；每个任务接受各自的 section |
| `preset` | `pilot`（缺省）或 `production`；`production` 要求显式设置 |
| `temperature` | 正温度，如 `"300 K"`；纯数值按 K |
| `beads` | 正整数；pilot 中 `auto` 取 P=32 |
| `seed` | 非负整数；独立随机流由稳定任务 ID 派生 |
| `dynamics.timestep` | 正步长，如 `"0.1 fs"`；纯数值按原子时间 |
| `dynamics.integrator` | `cayley-bcb`、`normal-mode-bab` 或 `velocity-verlet` |
| `sampling` | 平衡、采样间隔、生产长度、恒温器、副本数或母链数 |
| `parallel` | 进程/MPI 拓扑、珠子分块与线程 |
| `output.checkpoint_steps` | 以步数计的检查点间隔；`checkpoint_interval` 接受时间 |
| `output.trajectory_storage` / `chunk_frames` | 材料轨迹默认 `"stream"` 分块保存，256 帧缓冲；`"memory"` 保留旧格式 |
| `barostat` | 三维完全周期、全部原子可动的 `pimd` 可设置 BZP 压力、时间常数与摩擦 |

每个任务只接受自己的 section。PCET 使用独立的 `[pcet]` 模型，没有分子 `system` 或珠子，见 [PCET](pcet.md)。
未知字段和无效组合会报错。

时间必须是步长的整数倍。需要精确计数时使用 `equilibration_steps`、`production_steps`、`interval_steps`
等步数字段；同时给出时，步数优先。间隔字段的含义随任务而定：dRPMD 中是同一母链两次发射之间的时间，
FES 中是两次保存样本之间的时间。

### 缺省值

`pilot` 输入可以省略下表字段，PyRPMD 填入缺省值并在 `input_warnings` 中记录提示。`preset="production"`
要求显式给出 `temperature`、`beads`、`[dynamics]` 和 `[sampling]`。它检查输入完整性，不检验科学收敛，
也不会自动选择精度更高的一套参数。这些表中省略的子项仍使用缺省值。正式研究应显式给出物理模型和采样预算。

| 字段 | 缺省值 | 说明 |
| --- | --- | --- |
| `temperature` | `300 K` | 记录在 `input_warnings` 中 |
| `beads` | 32 | `"auto"` 也取 32 并给出警告；production 拒绝 `auto` |
| `seed` | 0 | 独立随机流由稳定任务 ID 派生 |
| `system.charge` / `multiplicity` | 0 / 1 | 须与所选 PES 一致 |
| `dynamics.timestep` | `0.1 fs` | 将步长减半后重新比较要报告的物理量 |
| `dynamics.integrator` | `cayley-bcb` | |
| `dynamics.native` / `loop_backend` / `block_steps` | `true` / `auto` / 256 | 见[执行与并行](execution.md) |
| `sampling.thermostat` | `andersen` | `pimd` / `diffusion` 为 `pile-l`（velocity-Verlet 时为 Andersen） |
| `sampling.equilibration` / `production` / `interval` | `100 fs` / `200 fs` / `1 fs` | 启用自适应伞或 metadynamics 时，生产段缺省为 `1000 fs`；显式长度优先 |
| `sampling.andersen_interval` | 10 步 | 两次 Andersen 动量重抽之间的步数 |
| `sampling.replicas` | 1 | `diffusion` 为 4 |
| `output.directory` | `runs/<task>` | 相对于输入文件 |
| `output.checkpoint_interval` | `10 fs` | |
| `analysis.bins` | FES 40；速率 PMF 100 | |
| `analysis.bootstrap` | 20 | 仅为较短的分析预算，不是精度目标；metadynamics 和 OPES 为 0；不接受 1 |
| `analysis.min_count` | 1 | 样本更少的 bin 被 mask |
| `collision.trajectories` / `max_time` | 4 / `100 fs` | 仅 `drpmd` |
| `collision.sampling` / `bmax` | `thermal_flux` / `auto` = R0/4 | `auto` 时不报告热速率 |
| `outcomes.exit_distance` / `persistence` | R0 / `1 fs` | 仅 `drpmd` |

`pyrpmd validate INPUT --write-resolved resolved.json` 写出前端解析后的输入。部分工作流设置，例如自动生成的
速率窗口和自适应计划，在运行阶段解析并记录到相应采样输出中。

气相 `rate` 工作流还会为省略的参数补充下列起始值：

| 字段 | 缺省值 | 含义 |
| --- | --- | --- |
| `rate.pmf_method` | `ui` | 高斯 umbrella integration；可显式选择 `mbar` |
| 初始伞中心 | 从 -0.05 到 1.05 的 23 个等距点 | 无量纲反应坐标步长 0.05 |
| 伞刚度 | `50 / beta` | 无量纲反应坐标的物理谐和刚度，即 50 kBT |
| `rate.pmf_points` | 401 | UI 积分网格数，不是实际采样窗口数 |
| `recrossing.parent_replicas` / `parent_samples` | 2 / 4 | 若提供 `sampling.parents`，它替代默认母构型数 |
| `recrossing.equilibration` / `parent_interval` / `child_time` | `20 fs` / `5 fs` / `50 fs` | 均需针对实际反应做收敛检查 |
| `recrossing.child_pairs` / `bootstrap` | 2 / 100 | 每个母构型的子轨迹对数 / 重采样次数 |

这些值不等同于 RPMDrate 手册的生产示例。初始窗距和刚度需要针对具体体系检查，尤其要验证默认 UI 的高斯近似。
自适应选窗可以增加中心，但保持所给定的刚度不变。

启用 `[adaptive_umbrella]` 后，下列缺省值是 **pilot 选窗判据与资源上限**，不保证生产 PMF 或速率已收敛：

| 字段 | 缺省值 | 含义 |
| --- | --- | --- |
| `pilot_steps` | 1000 | 每个窗口、每个副本的长度；默认步长下为 100 fs |
| `pilot_equilibration_steps` | `sampling.equilibration_steps` | 其他设置也取默认时为 1000 步 / 100 fs |
| `max_pilot_steps` | `16 * pilot_steps` | 每个窗口、每个副本的上限；默认步长下为 1.6 ps |
| `max_rounds` | 8 | Pilot 轮数上限 |
| `max_windows` | 128 | 插窗上限；达到该窗数后仍可通过延长 pilot 满足 ESS/覆盖目标 |
| `target_overlap` | 0.03 | 新 schema-3 计划采用归一化重叠分数；旧计划沿用原指标 |
| `min_effective_samples` | 20 | 每窗经相关性修正的 ESS，独立副本之间求和 |
| `min_spacing` | `1e-6` | 无量纲度量 `sqrt(beta * delta.T @ K @ delta)` 下的距离 |
| `coverage_bins` / `coverage_ranges` | 省略 | 只有同时提供两项才检查空间覆盖 |
| `min_bin_effective_samples` / `target_coverage` | 5 / 0.95 | 仅在显式启用覆盖网格后生效 |

默认每 1 fs 保存一次，因此首轮 100 fs pilot 每窗、每副本仅约 100 帧，且这些帧还存在时间相关性。
通过 ESS 20 只是筛选结果。应依据独立生产诊断和[收敛流程](science.md#评估自适应伞抽样结果)，
为实际体系选择长度、副本数、珠子数与不确定度精度。

### 各任务使用的 section

| Section | 用于 | 主要字段 |
| --- | --- | --- |
| `[system]` | 除 `pcet` 外全部 | `geometry`、`transition_state`、`reactants`、`separation`、`symbols`、`masses`、`masses_amu`、`isotopes`、`geometry_unit`、`fragments`、`charge`、`multiplicity`、`cell`、`cell_unit`、`pbc`、`fixed_atoms` |
| `[potential]` | 除 `pcet` 外全部 | `provider`、`factory`、`library`、`manifest`、`params`、`units`、`derivative`、`device` 等，见[势能](potentials.md) |
| `[dynamics]`、`[sampling]`、`[parallel]` | 除 `pcet` 外全部 | 见上文与[执行与并行](execution.md) |
| `[[cvs]]`、`[umbrella]`、`[analysis]` | `fes`、`rate`、`network`、`site_rate`、`surface_rate`；`metadynamics` 和 `opes` 使用 `cvs` 与 `analysis` | 见下文 |
| `[reaction]` | `rate`（`drpmd` 可用 `electronic_weight`） | `fragments`、`forming_bonds`、`breaking_bonds`、`rinf`、`symmetry_factor`、`equivalent_paths` |
| `[rate]` | `rate` | `pmf_method`、`dividing_surface`、`pmf_search_range`、`compare_ui` |
| `[recrossing]` | `rate`、`network`、`site_rate`、`surface_rate` | 母链数、母链间隔、child pairs、child 时间、bootstrap |
| `[collision]`、`[outcomes]` | `drpmd` | 发射分布、`bmax`、总轨迹数、最大时间；产物分类 |
| `[nonadiabatic]` | `drpmd`（实验性 NRPMD） | 见[非绝热 NRPMD](nonadiabatic.md) |
| `[electronic_friction]` | `diffusion` | `method = "centroid"`、`provider`、`gamma` 或 `tensor`，或 Python `factory` 与 `params`；见[电子摩擦](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electronic-friction.md) |
| `[initial_ensemble]` | `drpmd`（实验性核初态选择） | `method = "ne_rpmd"` 或 `"qct"`、NE 显式 `beta`、片段 Hessian/模式和 `vibrational_quanta`；见[直接碰撞](workflows.md) |
| `[adaptive_umbrella]` | `fes`、`rate` | 见[气相与自由能](workflows.md) |
| `[metadynamics]` | `metadynamics` | `grid.ranges` / `grid.bins` 可选偏置网格 |
| `[opes]` | `opes` | 概率估计宽度、物理能垒、更新间隔和可选冻结/再平衡步数 |
| `[replica_exchange]` | `fes` | 同温固定窗口的交换间隔与可选连通图，见[副本交换](../replica-exchange.md) |
| `[barostat]` | `pimd` | `method="bzp"`, `pressure`, `time_constant`, `friction` |
| `[diffusion]`、`[site_rate]`、`[surface]` | 同名任务（`[surface]` 属于 `surface_rate`） | 见[材料与表面](periodic.md) |
| `[network]` | `network` | 入口、通道、过渡态 |
| `[pcet]` | `pcet` | PCET 只接受 `schema_version`、`task`、`preset`、`description`、`temperature`、`pcet`、`output` |
| `[output]` | 全部 | `directory`、`checkpoint_interval` / `checkpoint_steps`、`trajectory_stride` |

全部字段与组合规则见[输入规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)。

## 原子、同位素与几何

典型的 `[system]`：

```toml
[system]
geometry = "molecule.xyz"
geometry_unit = "angstrom"
charge = 0
multiplicity = 1
isotopes = [{atom = 0, mass = "2.01410177812 amu"}]
```

也可以用内联的 `symbols` 和 `geometry=[[x,y,z], ...]`。内联长度缺省为 bohr，XYZ 文件缺省为 angstrom，
`geometry_unit` 可显式指定。`masses` 以电子质量计，`masses_amu` 以原子质量单位计。H、D、T 各自携带核质量，
传给电子势能时都对应氢。`charge` 与 `multiplicity` 须与势能模型一致。

直接碰撞使用 `reactants=["A.xyz", "B.xyz"]` 并显式给出 `separation`；速率计算可从
`transition_state="ts.xyz"` 开始。反应通道和统计简并度在输入中定义，不从 XYZ 文件推断。原子索引从 **0**
开始，fragments 须无重叠地划分全部原子。

周期晶胞、PBC、固定原子和展开坐标见[材料与表面](periodic.md)。新输入请始终显式写出 `geometry_unit`、
`cell_unit` 和 `pbc`。

## 集体变量与偏置

`[[cvs]]` 定义带名字的 CV，`umbrella.cvs` 选择要偏置的 CV。`centroid` 在环质心上求坐标；`bead_average`
逐珠子求值后平均，对非线性 CV 两者不同。自定义 CV 返回完整 Jacobian，见[势能与插件](potentials.md)。

数值伞中心使用 CV 声明的输入单位；带单位字符串携带自身单位。数值 stiffness 以原子单位计。`distance`、
`angle` 和 `bond_difference` 作用于展开几何；最小镜像几何使用 `periodic_distance`、`periodic_angle` 或
`periodic_bond_difference`。连续的 `fractional` 和 `surface_lateral` 坐标不取模，可区分相邻晶胞。

### CV 类型

`kind` 缺省为 `distance`，`estimator` 缺省为 `centroid`。`unit`（或逐分量的 `units`）设定**数值**伞中心
与分析范围的单位；CV 内部以原子单位计算。

| `kind` | 必需字段 | 规范单位 | 说明 |
| --- | --- | --- | --- |
| `distance` | `atoms=[i,j]` | bohr | 展开几何上的原子间距离 |
| `angle` | `atoms=[i,j,k]`（j 为顶点） | radian | 三个不同原子；直线构型处导数无定义 |
| `bond_difference`（别名 `difference`） | `atoms=[i,j,k]` 或 `bonds=[[i,j],[k,l]]` | bohr | $r_{ij}-r_{jk}$ 或 $r_{ij}-r_{kl}$，如质子转移坐标 |
| `cartesian` | `atom`，`axis` 为 `x`/`y`/`z` 或 0/1/2 | bohr | 模型体系与受约束采样 |
| `com_distance` | 可选 `fragments`，缺省使用体系片段 | bohr | 两片段质心距离；仅分子体系 |
| `rpmdrate`（别名 `rpmdrate_xi`） | `forming_bonds`、`breaking_bonds`、`rinf` | 无量纲 | 速率工作流的 RPMDrate 反应坐标；仅分子体系 |
| `periodic_distance`、`periodic_angle`、`periodic_bond_difference` | 同非周期版本 | 同上 | 最小镜像；仅周期体系 |
| `projection` | `atom`、`direction`；可选 `origin`、`reference_atoms` | bohr | 连续质心沿方向的投影 |
| `fractional` | `atom`、`axis`；可选 `origin`、`reference_atoms` | 无量纲 | 不取模的分数坐标 |
| `surface_height` | `atoms`；可选 `normal`、`origin`、`reference_atoms` | bohr | 吸附原子组 COM 沿表面法向 |
| `surface_lateral` | `atoms`；可选 `direction`、`origin`、`reference_atoms` | bohr | 面内连续 COM 投影 |
| `python` | `factory`；可选 `params` | 由插件声明 | 返回完整 Jacobian 的自定义 CV，见[势能与插件](potentials.md) |

恰好两个周期方向时，`surface_height` 的缺省法向为两晶格矢量的叉积，`surface_lateral` 的缺省方向为第一个
周期晶格矢量；否则分别为 z 与 x。

### 伞窗口

用以下任一形式给出窗口中心，`cvs` 指定被偏置的 CV：

```toml
[umbrella]
cvs = ["x"]
centers = [-0.5, 0.0, 0.5]              # 显式列表；二维时为 [[x, y], ...]
# start = -1.0, stop = 1.0, count = 9   # 一维等间距
# grid = [{start = -1.0, stop = 1.0, count = 3},
#         {start = -1.0, stop = 1.0, count = 3}]  # 二维张量网格；每轴也可直接给列表
stiffness = "0.002 hartree/bohr^2"      # 也可写 kappa
```

`centers_au` 以原子单位给出中心，忽略 CV 的 `unit`。`stiffness` 可以是标量、每个 CV 一项的对角列表或完整的
对称正定矩阵；量纲不同的 CV 组合时，每一项写成带单位字符串。

## 命令行参考

| 命令 | 用途 | 常用选项 |
| --- | --- | --- |
| `pyrpmd init DIR` | 向新目录或空目录写入内置谐振子 FES 入门输入 | 打印输入路径和下一步命令 |
| `pyrpmd validate INPUT` | 解析配置并检查任务 | `--check-pes`、`--output DIR`、`--write-resolved FILE` |
| `pyrpmd prepare INPUT` | 显示准备信息 | `--pilot` 执行数值准备；`--output FILE` 写出 JSON |
| `pyrpmd run INPUT` | 开始运行，或按输入继续 | `--output DIR`、`--resume` |
| `pyrpmd resume DIR` | 按保存的配置续跑 | `--workers N`、`--loop-backend auto`、`--block-steps N` |
| `pyrpmd extend DIR` | 为已有 dRPMD 系综中未结束的轨迹提高总时间上限 | 必填 `--max-time '20 ps'`，从发射时刻计时 |
| `pyrpmd state-check INPUT` | 无碰撞赋速演化已准备核初态，检查谐振模式稳定性 | `--time '100 fs'`、`--samples 8`、`--stride 10`、`--output report.json` |
| `pyrpmd analyze DIR` | 读取结果或离线重分析 | `--recompute`、`--bins`、`--bootstrap`；扩散选项见下文 |
| `pyrpmd report RUN [RUN ...]` | 从保存结果生成图形与 HTML 报告 | `--output DIR`、`--style`、`--formats`、`--overwrite`；见[自动报告](analysis-reports.md) |
| `pyrpmd project DIR` | 对保存的 FES 积分消去未保留的 CV 轴 | 必填 `--axes` 与 `--output`；可选 `--selection` |
| `pyrpmd rate-postprocess DIR` | 从保存的 PMF/FES 与带符号通量重新计算速率 | `--spec FILE`、`--output FILE`、`--bootstrap`；见[速率后处理](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-postprocessing.md) |
| `pyrpmd study INPUT` | 规划或运行温度、同位素及数值收敛扫描 | `--plan`、`--output DIR`、`--resume`；见[速率研究](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-studies.md) |
| `pyrpmd checkpoint FILE` | 校验检查点容器 | `--require-integrity` 拒绝没有校验值的旧容器 |
| `pyrpmd info` | 显示版本与原生能力 | |
| `pyrpmd pes build/inspect/check` | 构建与测试 C ABI 势能 | 见[势能](potentials.md) |

`validate`、`prepare` 和 `run` 接受适用于该任务的覆盖选项 `--pes`、`--pes-manifest`、`--temperature`、
`--beads` 和 `--workers`；PCET 只接受 `--temperature` 和 `--output`。`pyrpmd COMMAND --help` 列出当前安装版本的选项。

- `validate --check-pes` 与 `prepare --pilot` 执行同样的数值准备：构造 PES、TS 准备或 PCET 本征态求解。
  不加该选项时只检查输入。
- `validate --output DIR` 选择验证报告中显示的目标路径，不创建运行。`prepare --output FILE` 则写出准备 JSON 文件；
  `validate --write-resolved FILE` 显式写出解析后的配置。
- `--workers N` 设置 `parallel.trajectory_workers`；非 MPI 输入在 N>1 时使用 `process` 后端。
- `run` 不写入已有的非空输出目录。继续运行使用 `pyrpmd resume DIR` 或 `pyrpmd run INPUT --resume`。
- `--bins` 与 `--bootstrap` 隐含 `--recompute`。离线重分析适用于 `fes`、`metadynamics`、`opes` 和 `diffusion`；
  其他任务的 `analyze` 读取保存的摘要。

扩散重分析接受 `--fit-range '10 fs' '100 fs'`、`--axes 0 1`、`--atoms 0` 和 `--max-lag N`，其中延迟以保存帧数计。
拟合范围须在保存的时长之内，见[输出与分析](results.md)。

### 终端输出与调试

`python -m pyrpmd` 与 `pyrpmd` 可执行命令接受相同参数。`pyrpmd --help` 列出命令，`pyrpmd COMMAND --help`
列出子命令选项。命令结果以 JSON 写到标准输出；预期的输入或运行错误写到标准错误，退出码为 2。
命令执行成功时返回 0。退出码只表示执行状态；解读科学结果还需检查 `ready`、`warnings`、`converged` 与任务特有诊断。

普通命令显示简洁的错误说明。报告意外故障时添加 `--debug` 保留 traceback，例如：

```bash
pyrpmd validate my-first-run/quickstart.inp --check-pes --debug
pyrpmd info > environment.json
```

`--debug` 可以放在子命令之前或之后。请连同错误保留命令、输入与环境报告；恢复方法见[常见问题](troubleshooting.md)。

完整的 section 参考见 [docs/input.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/input.md)。
