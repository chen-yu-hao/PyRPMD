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

## 从完整示例开始

三个示例使用项目已有的解析教学势能面和 TS：

```bash
pyrpmd validate examples/rate/pmf_umbrella.inp --check-pes
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
kappa = "0.16 hartree"

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
