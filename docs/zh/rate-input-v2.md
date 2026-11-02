# 显式反应输入、分区窗口与独立采样阶段

**中文** · [English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate-input-v2.md)

推荐以 [`examples/rate/research.inp`](../../examples/rate/research.inp) 的结构组织研究输入：
明确给出反应物片段及成断键，独立检查 TS，按阶段设置采样。文件中的解析教学势与短 pilot 预算
用于检查工作流，不是实际体系的生产参数。旧 `pilot.inp` 作为 schema 1 兼容示例保留。
`schema_version = 2` 选择新的显式反应与独立母链默认规则。

## 反应定义和 TS 检查是两项设置

```toml
schema_version = 2
task = "rate"

[system]
transition_state = "ts.xyz"

[reaction]
definition = "explicit"
reactant_fragments = [[0], [1, 2]]
product_fragments = [[0, 1], [2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
rinf = "5 angstrom"

[preparation]
ts_action = "validate"
check_channel = true
```

所有原子索引使用全体系 TS 的同一个顺序，从 0 开始；片段不得重复或遗漏原子。
TS 起步的 `rate` 不要求另给两份反应物 XYZ。schema 2 默认 `definition="explicit"`，
需要推断时明确写 `definition="infer"`，见 [`minimal.inp`](../../examples/rate/minimal.inp)。
推断要求通道检查；不唯一或不符合声明的候选不能成为已准备好的运行。

`ts_action="validate"` 不改坐标，检查驻点与一阶鞍点；`"refine"` 允许先做 P-RFO 细化；
`"none"` 明确跳过 TS 检查，仅适用于给定 TS、完整显式反应且 `check_channel=false`。
原子映射一致的反应物/产物端点搜索仍使用原有端点输入。

显式反应定义不会自动证明 TS 正确。`check_channel` 独立控制双向下降及与片段、成断键、
可选产物片段的对照。这是连接性检查，不等于已收敛的量化 IRC。准备结果分别报告
`saddle_valid`、`channel_consistent` 和 `refinement_performed`。
普通输入校验不调用 PES；数值检查使用 `validate --check-pes` 或 `prepare --pilot`。

## 力常数默认采用 RPMDrate 的 k/T 约定

新字段 `force_constant` 的裸数字表示温度归一化系数 `a = k/T`。T 使用 Kelvin，
k 使用原子单位。对于无量纲反应坐标，a 的物理单位为 hartree/K，界面简记为 `1/T`：

$$
W_i(\xi)=\tfrac12 k_i(\xi-\xi_i)^2,\qquad k_i=a_i T.
$$

来源是 RPMDrate 示例中的
[`kforce = 0.1*T`](https://github.com/chen-yu-hao/RPMDrate/blob/710670c6f753b1159c0147c817e82998c2cc8486/examples/H%2BH2/input.py)；
其 [Fortran 偏置实现](https://github.com/chen-yu-hao/RPMDrate/blob/710670c6f753b1159c0147c817e82998c2cc8486/rpmdrate/_main.f90)
直接使用这个物理力常数。解析时系数乘以 T，动力学不再额外除以温度。

| 写法 | 无量纲 xi 对应的物理力常数 |
| --- | --- |
| `force_constant = 0.00016` | `0.00016*T` hartree |
| `force_constant = "0.00016 1/T"` | 同上，显式简写 |
| `force_constant = "0.00016 hartree/K"` | 同上，显式物理单位 |
| `force_constant = "0.16 hartree"` | 固定 0.16 hartree，不随 T 改变 |
| 旧 `stiffness = 0.16` 或 `kappa = 0.16` | 保留原来固定原子单位力常数的含义 |

因此教学模板在 1000 K 下用 `0.00016` 得到原来的 0.16 hartree。
直接照抄 RPMDrate 示例系数 `0.1` 会得到 100 hartree，强度并不等价。
省略速率力常数时仍保留原来的 `50 k_B T` 强度，不会因更换输入约定而突然改变默认势。

对于有量纲 CV，显式使用相应的刚度/K 单位，例如 `hartree/bohr^2/K`；
`1/T` 简写只适用于无量纲 CV。`force_matrix` 的裸系数同样表示原子刚度单位/K。
已有多维 FES 的标量、对角及完整 `stiffness` 矩阵继续支持。
程序负责珠子数 P 的 Hamiltonian 缩放，用户不再手动乘 P。

## 分区设置密度和力常数

```toml
[umbrella]
coordinate = "xi"
force_constant = 0.00016

[[umbrella.regions]]
id = "entrance"
start = 0.0
stop = 0.8
spacing = 0.05
include_endpoints = "left"

[[umbrella.regions]]
id = "barrier"
start = 0.8
stop = 1.05
spacing = 0.01
include_endpoints = "both"
force_constant = 0.00032
```

第一段生成 0 至 0.75 的 16 个中心，第二段生成 0.8 至 1.05 的 26 个中心，共 42 个。
端点包含方式可选 `left`、`right`、`both`、`neither`。间距必须整除区间，
不静默追加一个较短尾间隔；重复中心和冲突定义会被拒绝。
禁止的是分区定义冲突，采样分布之间的统计重叠仍然必要。

窗口来源只选一种：`centers`、`grid`、`regions`、`windows` 或旧 `start/stop/count`。
新输入不依赖多个来源之间的隐含优先级。分区用于一维；多维可用网格或逐窗口记录与正定矩阵。
`[[umbrella.windows]]` 可以显式填写 `id`、`center`、`force_constant`，并在它的
`sampling` 子表覆盖生产时长等预算。分区和逐窗口设置继承公共参数；它们是不同输入方式，
不同时构成多级覆盖。

每窗实际 K 会进入传播、样本元数据、UI/MBAR 分析和检查点。
固定窗口独立采样支持不同 K；仍要求公共 K 的自适应加窗与副本交换组合会明确报错。
副本交换也保留既有恒温时钟接口：显式新 Andersen 调度（包括 `schedule="fixed"`）暂不支持。
本页按操作命名的 Andersen 示例用于独立窗口和气相速率采样。
普通运行、分析和续跑支持逐窗口采样预算。独立的 `extend --production-steps` 延长采样接口
目前要求公共预算，遇到分区/窗口的 `sampling` 覆盖会在修改运行前报错；公共预算下的不同力常数可正常延长。
不同区间的力常数与间距需通过真实样本重叠、有效样本和误差检查确定。

## 用操作名称区分各阶段

```toml
[umbrella.sampling]
equilibration = "5 fs"
production = "20 fs"
sample_every = "0.2 fs"
chains_per_window = 1

[umbrella.sampling.thermostat]
type = "andersen"
scheme = "global_refresh"
schedule = "fixed"
refresh_every = "2 fs"

[recrossing.parent]
chains = 2
equilibration = "5 fs"
draw_every = "1 fs"
configurations_per_chain = 4

[recrossing.parent.thermostat]
type = "andersen"
scheme = "global_refresh"
schedule = "fixed"
refresh_every = "2 fs"

[recrossing.children]
method = "rpmd"
pairs_per_configuration = 2
duration = "10 fs"
```

在 0.1 fs 步长下，`sample_every` 表示每 2 步记录生产 CV；`refresh_every` 表示
每 20 步刷新全环动量。`chains_per_window` 是每窗口随机链数，与 worker 数无关。
时长接受带单位字符串或 `{steps=N}`，新旧字段重复声明会报冲突。
随机等待模式写 `schedule="geometric"` 和 `mean_refresh_interval="2 fs"`，
与固定周期的 `refresh_every` 互斥。

Andersen 时钟从 step 0 开始，贯穿平衡和生产。首次观测或构型抽取位于平衡结束后一个完整
记录/抽取间隔。原始样本数不等于有效独立样本数。

受约束母链有自己的热浴时钟，新接口不会继承伞采样的刷新参数；未给母链热浴时，其独立默认
为每 10 个 MD 步固定刷新。示例得到 8 个构型、16 对正负动量子轨迹，共 32 条相关轨迹。
标准 RPMD 子轨迹不继承平衡热浴、伞弹簧或分割面约束。需要 TRPMD 时明确选择
`method="trpmd"` 及其内部模摩擦参数。

`[umbrella.analysis]` 分别设置 `estimator`、`integration_points`、`bins` 和
`bootstrap_replicates`；`[recrossing.analysis]` 单独控制回穿 bootstrap 与平台检查。
输入重构不改变速率通量测度、归一化和成对估计器。

## 检查、预览、运行和续跑

```bash
pyrpmd validate examples/rate/research.inp --check-pes
pyrpmd prepare examples/rate/research.inp --pilot --output preparation.json
pyrpmd run examples/rate/research.inp --output runs/research-layout
pyrpmd analyze runs/research-layout
pyrpmd resume runs/research-layout
```

研究前查看 `umbrella_preparation.window_plan` 中的窗口计划、实际物理力常数及分别记录的鞍点/通道状态；
`recrossing_preparation` 列出计划母构型数、轨迹对数和单条轨迹数。
纯结构预览可根据显式反应展开窗口，同时保留 `ready=false` 表明数值 TS 检查尚未执行。
采样时 `umbrella-plan.json` 和逐窗口检查点记录解析后的偏置身份。
窗口、力常数、采样或热浴等科学设置改变时，不能用它们重新解释旧检查点；
执行资源改变仍遵守原有续跑规则。旧运行继续使用其保存的 schema 与输入。
若手动迁移旧输入以保持行为，旧物理力常数要除以 T 得到新系数，并把共享的旧 Andersen 时钟
分别复制到伞采样和母链两处。不要直接修改已有运行目录里的保存输入。
