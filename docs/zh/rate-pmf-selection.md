# 为 RPMD 速率选择 PMF 方法

**中文** · [English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-pmf-selection.md)

标准气相 `task="rate"` 现在用 `[rate.pmf]` 集中选择 PMF 的获取方式。
随后仍由同一次速率任务运行受约束母链和成对回穿子轨迹。

| `rate.pmf.method` | `rate.pmf.estimator` | 执行内容 |
| --- | --- | --- |
| `"umbrella"` | `"ui"`（默认）或 `"mbar"` | 固定或自适应伞窗口采样，再重构 PMF。 |
| `"metadynamics"` | `"reweighted_histogram"`（默认） | 学习偏置、冻结偏置、重新平衡，再重加权固定偏置下的生产样本。 |
| `"opes"` | `"reweighted_histogram"`（默认） | 学习概率偏置、冻结偏置、重新平衡，再重加权固定偏置下的生产样本。 |
| `"external"` | 不填写 | 读取符合要求的已保存 `FESResult`，不再生成 PMF 轨迹。 |

`method` 表示采样方法或已有来源，`estimator` 表示重构方法。
UI 和 MBAR 都使用伞形采样的数据。原始 MetaD 偏置估计和 OPES KDE 不能直接充当速率 PMF。

## 伞形采样不再选择速率 PMF 估计器

UI 速率只需在一处选择估计器：

```toml
[rate.pmf]
method = "umbrella"
estimator = "ui"

[rate.pmf.analysis]
bootstrap = 40
```

不要用 `analysis.method="mbar"` 或 `analysis.method="ui"` 选择速率 PMF。
新建的伞形采样阶段不再注入 `analysis.method`。历史上显式填写的 `"mbar"`
仍可读取，但不会覆盖 `rate.pmf.estimator`；已保存的阶段输入也不会被改写。
旧的 `rate.pmf_method` 选择器继续支持。普通 `task="fes"` 的
`analysis.method="mbar"` 接口不变。

伞形采样器只产生可续算的样本流，随后由速率分析阶段执行所选估计器和端点配对
bootstrap。UI 仍保留一次 MBAR 拟合来检查重叠和实际直方图支撑；这个拟合不是
UI 的 PMF 估计器，也不再运行额外的 FES bootstrap。MBAR 速率直接复用其
最终拟合生成诊断，不再先求解一遍生产阶段的预备 FES。自适应 pilot 仍需评估
重叠以选择窗口，只有冻结后的独立生产样本进入最终速率分析。

`umbrella/fes.h5`、标准 `umbrella/fes.*` 和 `umbrella/summary.json` 继续保留，
改由速率分析阶段写出。这些文件保存样本和 MBAR 点估计诊断，**不是第二份误差
估计**：bootstrap 次数为零，误差不可用。元数据记录 `analysis_owner="rate"`
和所选 `rate_estimator`。所选 UI/MBAR 曲线及势垒误差应读取 `rate_pmf.*`。
普通续算和固定窗口生产扩展采用相同的职责划分；采样时钟、checkpoint 标识、
flux 测度以及重叠断连时拒绝速率的规则保持不变。

### UI 高斯模型偏差检查（#170）

UI 继续作为兼容性默认估计器，但不能将它视作无需验证的定量结果。
当偏置窗口过宽或分布偏斜时，高斯近似可能高估自由能势垒；UI bootstrap
的标准误差只度量采样误差，**不包含**这种模型偏差。定量研究宜使用
`rate.pmf.estimator = "mbar"`，或在保证相邻窗口重叠的情况下收窄窗口，
验证势垒随窗口宽度收敛。

每次完成的 UI 分析会复用重叠/支撑诊断中**已经求出的、无 bootstrap 的 MBAR
直方图点估计**。仅当参考端点、分界面和两者之间的直方图都受采样支撑，且
bin 宽度小于最窄窗口谐振热宽度的一半时，才在**同一固定分界面**比较势垒。
如果差值超过 `max(UI 势垒标准误差, 0.1 k_B T)`，输出包含两个势垒和差值
的具体警告。这只是保守的模型诊断，不是统计假设检验或第二套误差条；
有限 bin 的 MBAR 本身也有离散化和采样误差。为保留现有速率语义，
警告不会自动替换或撤销 UI 速率。如果直方图不足或分辨率过低，则明确警告
无法比较，绝不外推无支撑区间。

`rate_pmf.json` 中的 `ui_mbar_comparison` 记录阈值、支撑原因、比较结果
及估计的 UI/MBAR QTST 比值；`window_moments` 记录各窗的均值、标准差、
偏度、超额峰度和热宽度，供定位偏离高斯的窗口。不需要额外的 MBAR 求解、
bootstrap 或改变采样测度。

## 从完整示例开始

四个示例方案使用项目已有的解析教学势能面和 TS：

```bash
pyrpmd validate examples/rate/pmf_umbrella.inp --check-pes
pyrpmd validate examples/rate/pmf_umbrella_mbar.inp --check-pes
pyrpmd validate examples/rate/pmf_metadynamics.inp --check-pes
pyrpmd validate examples/rate/pmf_opes.inp --check-pes

pyrpmd run examples/rate/pmf_umbrella.inp --output runs/rate-umbrella
```

在相同的 `run` 命令中换用另外两个输入文件即可运行对应方法。
这些短 pilot 用于展示输入和工作流，不是物理 H + H2 基准，也不是收敛参数推荐。
增强采样的短轨迹可能尚未覆盖反应物参考点或势垒；程序会保留覆盖不足的诊断，
而不会外推出缺失的 PMF 并报告速率。

## 将 PMF 参数与回穿参数分开

顶层动力学和采样设置保留原有含义。PMF 的局部子表继承共享默认值，
再仅对 PMF 阶段覆盖指定的设置：

```toml
[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"

[sampling]
thermostat = "andersen"

[rate.pmf]
method = "umbrella"
estimator = "mbar"

[rate.pmf.dynamics]
integrator = "normal-mode-bab"
timestep = "0.2 fs"

[rate.pmf.sampling]
thermostat = "pile-l"
gamma_centroid = "0.02 fs^-1"
equilibration = "10 fs"
production = "200 fs"
interval = "1 fs"
replicas = 2

[rate.pmf.umbrella]
start = -0.05
stop = 1.10
count = 47
force_constant = 0.1

[rate.pmf.analysis]
bins = 80
bootstrap = 40
```

此处 PMF 使用正常模 BAB + PILE-L（BAOAB 采样），步长为 0.2 fs；
受约束母链和 RPMD 子轨迹使用顶层 Cayley 设置，步长为 0.1 fs。
顶层采样恒温器仍保留 Andersen；其他 canonical 恒温器填入 `rate.pmf.sampling`。
省略 `sampling.integrator` 时程序选择匹配的分裂方式。例如，PILE-L 可搭配
Cayley/BCOCB 或正常模/BAOAB，不能搭配 velocity-Verlet。

| 设置 | 输入位置 | 作用范围 |
| --- | --- | --- |
| 温度、珠子数、PES、体系、反应定义 | 原有顶层字段或子表 | 两阶段共用，不能在 PMF 子表另行覆盖。 |
| 回穿积分器和步长 | `[dynamics]` | 受约束母链与子轨迹。 |
| 母链／子轨迹数量和时长 | `[recrossing]` | 回穿阶段。 |
| PMF 积分器和步长 | `[rate.pmf.dynamics]` | 仅 PMF 阶段。 |
| PMF 恒温器、时间预算、replicas | `[rate.pmf.sampling]` | 仅 PMF 阶段，未填写的值继承共享默认设置。 |
| 偏置参数 | `[rate.pmf.umbrella]`、`[rate.pmf.metadynamics]` 或 `[rate.pmf.opes]` | 仅当前选中的方法。 |
| 自适应窗口 | `[rate.pmf.adaptive_umbrella]` | 仅伞形采样。 |
| 直方图和统计设置 | `[rate.pmf.analysis]` | PMF 分析，受所选估计器的兼容性检查约束。 |
| 分界面／搜索范围 | 原有 `[rate]` 字段 | PMF 势垒端点与回穿分界面。 |

时间量按各自阶段的步长换算；显式 `*_steps` 表示该阶段的步数。
局部时间选项会替换对应的继承时间／步数选项。PMF 自身的 sampling 和 dynamics
步长仍须一致。

显式 `centers` 可以不等间隔。[分区窗口接口](rate-input-v2.md)也可写在
`[rate.pmf.umbrella]` 内：`regions` 或 `windows` 能选择不同的力常数和采样预算。
刚度列表仍表示 CV 维度；逐窗口差异应写在对应的窗口／区域记录中。
新的 `force_constant`、`force_matrix` 保留按温度归一化的单位语义。
此气相速率接口不开放 REUS。

操作型 `[rate.pmf.umbrella.sampling]`、`[rate.pmf.umbrella.analysis]`
分别替代 `[rate.pmf.sampling]`、`[rate.pmf.analysis]`，同一阶段只选一个位置。
嵌套分析中的 `estimator` 替代 `rate.pmf.estimator`，`integration_points`
替代 `rate.pmf_points`；重复声明会在预检时报错。独立选择 PMF 方法后，继承的顶层
`umbrella.sampling` 也按 PMF 步长解析。每个窗口的 `equilibration`、`production`、
`sample_every` 和 Andersen 刷新时长同样使用 PMF 时钟；`recrossing.parent`、
`recrossing.children` 使用顶层 dynamics 时钟。

不同力常数、共同采样预算的固定窗口支持生产延长。每窗口／每区域采样预算覆盖会在
获取延长锁前被拒绝；普通运行、分析和续跑仍可用。自适应伞形采样会拒绝不同窗口
力常数或窗口局部采样设置。

## MetaD／OPES：明确各阶段的时间预算

两种增强方法都需要在对应的方法子表中填写 `freeze_after_steps` 和
`freeze_equilibration_steps`。学习偏置应在一次沉积／概率更新后结束，
因此 `freeze_after_steps` 必须是 `pace` 的正整数倍。

例如，假设 PMF 步长为 0.2 fs：

```toml
[rate.pmf]
method = "metadynamics"

[rate.pmf.sampling]
equilibration_steps = 50
production_steps = 1000
interval_steps = 5
replicas = 2

[rate.pmf.metadynamics]
mode = "well_tempered"
sigma = 0.05
height = "0.02 kcal/mol"
pace = 20
bias_factor = 10.0
freeze_after_steps = 200
freeze_equilibration_steps = 100

[rate.pmf.analysis]
bins = 40
ranges = [[-0.15, 1.15]]
bootstrap = 0
```

初始平衡单独计时，`production_steps` 包含后面的三个阶段：

| 阶段 | PMF 步数 | 本例时间 | 是否进入重加权 PMF？ |
| --- | ---: | ---: | --- |
| 初始平衡 | 50 | 10 fs | 否 |
| 学习偏置 | 200 | 40 fs | 否 |
| 最终固定偏置下重新平衡 | 100 | 20 fs | 否 |
| 固定偏置生产采样 | 700 | 140 fs | 是 |

每 5 步保存一次，每条独立历史保留 140 个生产样本。
只保留严格晚于“冻结 + 重新平衡”边界的样本；最后一次更新偏置所在步的观测也不使用，
即使额外平衡步数为零。估计器要求每条历史至少保留两个样本，这只是输入有效性检查，
不代表统计精度已经足够。

若选择 OPES，将方法改为 `method="opes"`，并用以下子表替换 MetaD 子表：

```toml
[rate.pmf.opes]
sigma = 0.05
barrier = "0.5 kcal/mol"
bias_factor = 10.0
pace = 20
freeze_after_steps = 200
freeze_equilibration_steps = 100
```

完整示例还包含 PMF 步长、顶层回穿设置、体系和势能面。
切换方法时需要移除原方法的参数子表；不匹配的子表会明确报错。

### 重加权去掉了什么？

实际采样的力包含 RPMDrate 通量度量势。固定偏置生产样本使用
`exp(beta * V_frozen)` 去掉学习得到的历史偏置，并保留 `rate_flux` 测度。
这是一种平稳采样的重加权协议；偏置轨迹的时间不用于计算真实反应时间。
此速率协议暂不支持 MetaD 静态墙和共享偏置 walkers。

这两种估计器的 PMF `bootstrap` 必须为零。独立历史之间的离散程度和只考虑权重的
有效样本数是诊断量，不能当作校准后的标准误。PMF 势垒误差和完整速率误差会标为不可用；
如果回穿误差可估计，则它表示固定 PMF 因子条件下的回穿误差。增加历史数不会改变这一误差报告规则。

`analysis.ranges` 只定义直方图区间，不会施加限制动力学的墙。
RPMDrate xi 在反应走廊外可能出现奇点，因此全局探索范围和平稳采样区域仍需针对具体体系检查。

## 支撑、权重与窗口门槛

直方图 PMF 只在 bin **中心**有定义。`analysis.ranges=[[-0.1,1.1]]`、`bins=240`
时最后一个中心是 1.0975，而不是 1.1。输入预检会拒绝落在
`[first_center, last_center]` 之外的参考点 ξ=0、`rate.dividing_surface` 或
`rate.pmf_search_range` 端点，同时报告 edge/center 范围和需要的边界，
绝不把 1.1 静默截断为 1.0975。端点恰在中心或内部 edge 上可以接受。
省略 `analysis.ranges`（按数据决定 bin）时，在分析阶段给出同样的分类。

`rate_pmf.json` 保存解析后的分箱与支撑：`support.histogram`
（edges、centers、首末中心、宽度）、`support.search_endpoints`
（`on_center`、`between_centers`、`outside_histogram_centers` 或 `masked`，
含相邻 bin 和被屏蔽区间）、`support.reference_endpoint` 与
`support.dividing_surface_endpoint`。

MBAR 速率直方图由归一化 log 权重以 log-sum-exp 累积 bin 质量，
极小但非零的权重不再因下溢变成屏蔽/NaN。`weight_diagnostics` 报告全局
Kish ESS、每条流的 ESS 与权重占比、端点 bin ESS、最大归一化权重和
log 权重跨度。log 域运算只能恢复**诊断**曲线，最终速率仍受门槛约束：

| 门槛 | 默认 | `[rate]` 选项 | 失败状态 |
| --- | --- | --- | --- |
| 全局 Kish 权重 ESS（坍缩报警） | ≥ 2 | `pmf_min_weight_ess` | `weight_ess_collapse` |
| 参考点/分割面 bin 的 ESS | ≥ 2 | `pmf_min_endpoint_ess` | `endpoint_weight_ess_collapse` |
| 流均值距中心不超过 N 个偏置宽度 `1/sqrt(beta K)` | N = 20 | `pmf_max_window_displacement` | `umbrella_stream_anomaly` |
| 末态质心位于 RPMDrate ξ 的正常分支 `s0-s1>0` | 始终 | — | `umbrella_stream_anomaly` |
| 自适应伞形：冻结生产的重叠、每窗 ESS 与覆盖 | 自适应目标 | `[adaptive_umbrella]` | `adaptive_production_targets_not_met` |

即使数据健康，无偏伞形权重的全局 ESS 也很小：边缘窗口的权重按
`exp(beta K (xi-c)^2/2)` 增长，因此该下限只用于拒绝坍缩到一两帧的情形；
端点 bin 的 ESS 才衡量进入势垒的支撑。仅在有意做诊断时把门槛设为 `"off"`。任一门槛失败时：`ready=false`、
`status="unsupported_pmf"`、`curve_role="diagnostic_only"`、`k_qtst=null`，
不启动 recrossing；潜在势垒只写入 `diagnostic_free_energy_barrier_hartree`。
`rate_pmf.npz` 含 `diagnostic_only=true` 与 `log_density`，`gates` 列出全部检查。

偏离中心或位于错误分支的流还会写出 `umbrella_stream_anomalies.json`
（中心、刚度、均值、宽度、以 σ 计的偏移、均值处偏置能 kcal/mol、
由末态质心重算的 ξ、s0、s1 与分支，以及检查清单）。越过奇异面
`s0=s1` 后，RPMDrate ξ 跳到 −∞，再沿产物通道回升到约 1/3；因此中心为
ξ=1.1 的窗口可能报告 ξ≈0.4–0.7、偏置能巨大而回复力很弱。
继续该流不是收敛手段。

自适应 pilot 只负责选择窗口。即使所有 pilot 目标都满足，速率仍要求
冻结生产满足 `target_overlap`、`min_effective_samples` 及声明的覆盖。

### 失败的偏置 history

伞形窗口、WT-MetaD 与 OPES history 在传播或 PES 失败时（BKMP2 中任一
原子对距离小于 0.2 bohr）写出 `failures/<history>.json`，记录错误、最后一个
已提交状态的 step/time/时钟、步长、珠子与质心最小原子对距离、质心 CV、
最后记录的 CV、偏置状态（hills 或 kernels）、native/Python 传播计数和
checkpoint。此时 `rate_pmf.json` 以 `status="sampling_incomplete"` 列出全部失败
history，运行抛出错误；不会挑选已完成的 history 拼接 PMF。

## 复用已有 PMF

```toml
[rate.pmf]
method = "external"
directory = "../compatible-fes"
allow_non_umbrella = false
```

目录相对于输入文件解析，须包含保存的 `FESResult` 以及输入、体系和 manifest 记录。
来源应具有相同的温度、珠子数、PES、体系／参考几何、一维质心 RPMDrate 反应坐标，
能量单位为 hartree，测度为 `rate_flux`。非伞形来源需要
`allow_non_umbrella=true` 和有效的重加权记录；打开此开关本身不会执行重加权。

外部来源不填写估计器，也不填写局部 PMF 动力学、采样和分析子表。
母链和子轨迹仍然运行。来源逐点误差不能确定反应物参考点与势垒之间的协方差，
因此仍遵循已有外部 PMF 接口的误差报告限制。

## 旧输入与续跑

没有 `[rate.pmf]` 的输入保持原来的伞形采样路径、默认行为和检查点身份。
原有 `rate.pmf_method` 和 `rate.pmf_source` 仍可使用。两种写法选一种；
旧选择字段与新表同时出现会报错。同一方法参数子表也不要在顶层和 `[rate.pmf]` 内重复填写。

续跑沿用保存的协议、时钟、偏置历史和恒温器状态。改变方法、步长、恒温器、反应坐标
或冻结计划会改变科学计算，应开始新的运行。普通续跑和增加生产预算是不同操作；
速率生产延长有单独的方法适用限制。

先执行 `validate --check-pes`，再执行 `run`。解释速率前，应检查解析后的 PMF 阶段设置、
采样支撑和误差诊断；输入检查通过不代表物理结果已经收敛。

`validate` 和 `prepare` 输出中的 `pmf_plan` 会列出所选采样器、估计器、各自独立的
PMF／回穿时钟、阶段时长及保留样本数。新的本地 PMF 运行将有效设置保存到
`RUN/input/pmf-resolved.json`；`RUN/rate_pmf.json` 会同时报告计划、端点支撑、
势垒及其误差状态。
