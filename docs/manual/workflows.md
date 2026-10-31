# 气相反应、自由能与增强采样

本章按计算任务介绍 `rate`、`drpmd`、`fes`、`network` 和 `metadynamics`。所有命令从已安装 PyRPMD 的仓库根目录运行；每个新计算使用新的输出目录。示例以解析势或简化模型演示流程，替换成真实化学体系前，需要重新定义势能、反应和收敛参数。

周期材料和固定表面请转到[周期任务](periodic.md)，电化学局域模型请转到 [PCET](pcet.md)。各方法使用的单位、统计系综与归一化见[科学概念](science.md)。

## 气相 RPMD 速率：`rate`

当问题具有两个可定义的远区反应物片段和一个过渡态参考结构时，`rate` 执行 Bennett–Chandler 流程：准备反应坐标，采样 umbrella PMF，生成受约束母构型，再发射成对的无偏回穿子轨迹。最终以 $k_{\mathrm{RPMD}}=k_{\mathrm{QTST}}\kappa$ 组合结果。

### 先运行完整教学例子

```bash
pyrpmd validate examples/rate/pilot.inp --check-pes
pyrpmd prepare examples/rate/pilot.inp --pilot
pyrpmd run examples/rate/pilot.inp --output runs/manual-rate
pyrpmd analyze runs/manual-rate
pyrpmd resume runs/manual-rate
```

[完整输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/rate/pilot.inp)使用 `toy_exchange.py` 解析势，并从 `ts.xyz` 开始准备。该例的 4 珠子、短热化、短 PMF 和短回穿适合学习文件组织，不代表真实 H + H₂ 速率。`prepare --pilot` 会做数值准备，而不仅是打印输入。

只提供 TS 时，准备程序检查力、数值 Hessian 的不稳定模式和双向下降后的连接关系。只有能够唯一识别入口时才继续；若通道或模式有歧义，应显式给出 `reaction.fragments`、`forming_bonds` 和 `breaking_bonds`。这一步是计算准备工具，不是成熟量子化学程序的完整 IRC 替代品。

### 从试算进入目标体系

| 输入部分 | 决定的内容 | 实际检查 |
| --- | --- | --- |
| `system.transition_state` | TS 参考几何与原子顺序 | 与 PES、同位素及成断键一致 |
| `reaction.fragments`、成断键 | 两个入口片段和反应坐标 | 0-based 索引、原子守恒、通道定义 |
| `reaction.rinf` | 远区归一化距离 | 改变距离，检查片段解耦与长程势 |
| `umbrella` | 窗口中心、范围、刚度 | 反应物参考点至分割面的连通支持 |
| `sampling` | 每窗口平衡、生产、保存间隔和副本 | 分布形状、有效样本数、独立重复 |
| `rate.pmf_method` | `ui` 或 `mbar` | UI 高斯窗口假设；MBAR 的 bin 与支持 |
| `recrossing` | 母链数、抽样间隔、child pairs、传播时间 | 独立母构型、稳定平台、能量误差 |

气相速率使用包含质量通量度量的 `rate_flux` 测度。它与普通 `fes` 的 `cv_density` 不同。`rate.pmf_method="mbar"` 改变同一速率测度下的 PMF 估计器，不是把一个任意距离 FES 直接代入双分子公式。固定分割面通过 `recrossing.dividing_surface` 指定；否则由工作流按 PMF 选择，仍需考虑 bin 宽和峰选择偏差。

母链在分割面上受约束采样。每个 child pair 重新抽取未约束 Maxwell 动量并使用 `+p` 和 `-p` 发射；子轨迹不加 thermostat 或 umbrella。将 child 动量投影为切向会消去初始通量，因此不能照搬母链的动量约束。

### 读取 PMF、κ 和速率

| 文件 / 摘要字段 | 解释 |
| --- | --- |
| `rate_pmf.csv`、`rate_pmf.npz` | 专用速率 PMF、支撑区域及相关数组 |
| `recrossing.csv`、`recrossing.npz` | 完整 $\kappa(t)$ 曲线、误差与 bootstrap 信息 |
| `umbrella/` | 各窗口采样、checkpoint 和中间分析 |
| `results/summary.json` | 公共运行摘要、警告和诊断 |
| `rate_estimate_cm3_s` | 可审查的有限时间候选估计 |
| `rate_cm3_s` | 满足当前平台与能量诊断时填写的估计，否则 `null` |
| `rate_standard_error_cm3_s` | PMF 和回穿不确定度均可用时的联合采样误差 |
| `conditional_rate_standard_error_cm3_s` | 把 QTST 因子固定后的条件性回穿误差 |

速率单位为 `cm^3 molecule^-1 s^-1`。平台被检测到仍不代表珠子数、步长、远区距离和统计长度全部收敛。PMF overlap 断连或分割面缺少支持时，不应通过平滑、填洞或忽略警告制造一个速率。

完整数学定义见[速率科学约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md)和 [UI / MBAR 说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md)。真实势能接口与有限采样基准见 [H₃/BKMP2](https://github.com/chen-yu-hao/PyRPMD/blob/main/benchmarks/h3/README.md)。

## 直接碰撞：`drpmd`

`drpmd` 从两个反应物结构开始，持续采样母链，并发射独立的相对碰撞运动。它适合显式检查碰撞、散射和产物分类。初始分布决定输出究竟是热速率还是固定能量截面。

如需指定入射 diabatic 电子态并使用完整耦合矩阵势，可选择实验性的
[NRPMD 扩展](nonadiabatic.md)。该扩展仍属于 `task="drpmd"`，电子态贡献有符号，
并不把下文的标量势流程推广为电子热平衡 PMF 或非绝热 Bennett–Chandler 速率。

```bash
pyrpmd validate examples/drpmd/free_scattering.inp --check-pes
pyrpmd run examples/drpmd/free_scattering.inp --output runs/manual-scattering
pyrpmd analyze runs/manual-scattering
pyrpmd resume runs/manual-scattering
```

[自由散射例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/free_scattering.inp)使用 `V=0`，用于检验发射、传播、完成条件和恢复。它没有反应势，因此完成的碰撞应为非反应事件。

`collision.trajectories` 是全部母链共享的总碰撞数。例子用两个母链产生五个任务，分配为 3+2。`sampling.equilibration` 是母链首次热化；`sampling.interval` 是同一母链相邻两次抽取之间的推进时间。固定 `sampling.parents` 有助于保持改变执行进程数前后的科学任务身份。

| `collision.sampling` | 发射分布 | 输出含义 |
| --- | --- | --- |
| `thermal_flux` | 相对能量按 $E\sim\mathrm{Gamma}(2,k_BT)$ 采样 | 热通量平均速率 |
| `maxwell` | 普通相对速度 Maxwell 分布 | 按每条轨迹速度加权的热速率 |
| `fixed_energy` | 显式 `collision.energy` | 该碰撞能的散射截面，不给热速率 |

`system.separation` 是初始 COM 径向距离 R0，包含冲量参数 b 的贡献；`collision.bmax` 必须小于 R0。面积均匀采样意味着均匀采样 $b^2$。正式结论需要分别扩大 R0、bmax 和最大传播时间，检查截断误差。

产物判断需要明确规则。内置连通图分类器使用键长 cutoff、远离条件和持续时间；相同分子式异构体或同元素交换通道通常需要 `outcomes.factory` 自定义分类器。缺省推断 cutoff 只能给暂定候选标签，不能代替经检查的化学通道定义。

在 `results/summary.json` 中先看 `counts`、`all_trajectories_resolved` 和 `classification`，再看 `total_reaction` / `channels`。通道项的热速率字段是 `rate_cm3_molecule_s`，固定能截面字段是 `cross_section_bohr2`。未决或失败轨迹单独保留；相应观察下界/可能上界不等同于置信区间。同一母链内存在相关性，应结合 `parent_cluster_bootstrap95` 和更多独立母链判断误差。

运行目录还保存 `states/initial-*.npz`、`trajectories/` 和 checkpoint，便于复查每条轨迹的初态和分类。详见 [dRPMD 示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/drpmd/README.md)。

## 一维与二维自由能：`fes`

`fes` 对一个或两个 CV 做 umbrella 采样，并用 MBAR 去偏置。它计算指定坐标的概率密度自由能，不自动产生反应速率。

```bash
pyrpmd validate examples/fes/harmonic2d.inp --check-pes
pyrpmd run examples/fes/harmonic2d.inp --output runs/manual-fes
pyrpmd analyze runs/manual-fes --recompute --bins 20 20 --bootstrap 50
```

[二维谐振子输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/harmonic2d.inp)定义质心 x/y、3×3 umbrella 网格和每窗口两个独立副本。`umbrella.stiffness` 可以是标量、对角数组或完整对称正定矩阵。窗口范围要覆盖研究区域；更大的分析范围不会自动产生缺失样本。

结果保存在 `fes.csv`、`fes.npz` 和 `fes.h5`，含 `free_energy`、`probability_mass`、`density`、bin 边界、计数与 `mask`。检查 overlap 的连通性、各时间序列的有效样本数和加权 ESS。无支持区域保持 masked/NaN；断连窗口组之间的自由能常数不能由现有数据确定。

`analyze --recompute` 从保存的窗口 checkpoint 重建分析，可以改变 bins、计数门槛和 bootstrap，而不重新调用 PES。减小 bin 数有时提高每个 bin 的样本数，但不会修复未采样的盆地或缺少窗口 overlap。

### 自适应 umbrella

自适应工作流在给定域内，根据 pilot 的 overlap 和有效样本数插入窗口、增加 pilot 长度，然后冻结窗口，并以新的独立随机流重新热化和生产：

```bash
pyrpmd run examples/fes/adaptive_harmonic.inp --output runs/manual-adaptive
pyrpmd resume runs/manual-adaptive
pyrpmd analyze runs/manual-adaptive --recompute --bins 30 --bootstrap 20
```

`adaptive-plan.json` 保存每轮决定、窗口、随机种子和校验信息。pilot 样本不会混入最终 production 估计。`max_rounds` 或 `max_windows` 用尽仍未达到要求时，任务明确停止；它不会将不达标 pilot 直接作为生产结果。

默认允许 8 轮、128 窗，只有 ESS 不足才延长 pilot，`max_pilot_steps` 限制单窗
pilot 长度。预算耗尽后可运行
`pyrpmd resume runs/manual-adaptive --adaptive-max-rounds 12 --adaptive-max-windows 192`
继续，保留已完成的 pilot。生产阶段的 ESS 另行报告；pilot 达标不意味着生产
达标。未显式指定长度时，增强采样生产默认 1000 fs，仍需按实际统计检查调整。

自适应 umbrella 可用于普通 `fes`、表面 `fes` 和气相 `rate` PMF。它细化已给定区域，不自动发现任意远处未知盆地；`surface_rate` / `site_rate` 当前需要预先声明固定窗口。完整参数见[增强采样规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)。

## 多入口、多 TS：`network`

`task="network"` 将多条明确的气相双分子速率问题组织为一个可恢复计算。每条路径仍完整执行 PMF 和 recrossing，各自保留入口归一化、PES、TS 几何和误差。

```bash
pyrpmd prepare examples/network/two_entrances.py --pilot
pyrpmd run examples/network/two_entrances.py --output runs/manual-network
pyrpmd analyze runs/manual-network
pyrpmd resume runs/manual-network
```

[完整例子](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/network/two_entrances.py)包含两个入口、每入口两个独立解析 TS/PES。所有路径共享有序元素、同位素质量、电荷和多重度；入口可有自己的片段划分与 `rinf`，TS 可有自己的结构、成断键、PES 和对称因子。不同化学计量或同位素库存应建立独立任务。

PES 按全局→入口→TS 继承；更具体的 `potential` 表完整替换上层，不混合不同 provider 的残留字段。采样和速率等参数表可以逐层覆盖。`parallel` 和执行分块设置放在共享配置中。

结果索引为 `matrix[entrance][channel]["transition_states"][ts_id]`。每条路径的目录为 `paths/<entrance-id>/<ts-id>/`，`network-progress.json` 记录已完成路径。恢复会复用已完成结果，并继续未完成路径。

`aggregation="separate"` 是当前唯一模式；`total_rate_cm3_s` 和 `summed_rate_cm3_s` 保持 `null`。通道名称只是标签，不能证明不同分割面通量互不重叠；把多 TS 的估计直接相加可能重复计数。它也不是 kMC、微观动力学或多 PES 非绝热跳跃模拟。详见[反应网络定义](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/reaction-networks.md)。

## Metadynamics

`task="metadynamics"` 支持 standard 和 well-tempered 模式、1D/2D CV，以及独立偏置历史的多个副本。CV 本身可以具有声明的周期；这与原子系统的周期晶胞是不同概念。当前该工作流不接受带 PBC 或固定原子的 System。

```bash
pyrpmd validate examples/metadynamics/well_tempered_2d.inp
pyrpmd run examples/metadynamics/well_tempered_2d.inp --output runs/manual-meta
pyrpmd analyze runs/manual-meta --recompute --bins 30 30
pyrpmd resume runs/manual-meta
```

| 参数 | 含义 |
| --- | --- |
| `metadynamics.cvs` | 一个或两个已声明 CV 的名称 |
| `mode` | `standard` 或 `well_tempered` |
| `sigma` | 各 CV 的 Gaussian 宽度，推荐显式单位 |
| `height` | 物理 hill 高度 |
| `pace` | 两次 deposition 间的已完成生产步数 |
| `bias_factor` | WT 模式的 $\gamma>1$；standard 模式省略 |
| `analysis.method` | `bias_estimate` |
| `analysis.bootstrap` | 必须为 0 |

初始平衡阶段没有 hills。第一次 hill 在生产第 `pace` 步成功完成后加入；同一步内的力评估看到同一份历史。恢复时以 `checkpoints/metadynamics_XXXX.npz` 中的偏置历史为准，`replicas/XXXX/HILLS.csv` 可由 checkpoint 重建。

standard 的估计为 $F\approx -V+C$，WT 为 $F\approx-\gamma V/(\gamma-1)+C$。`fes.npz` / `fes.csv` 保存有限时间偏置估计和支持 mask。多个副本先按共同支持点对齐，再在共同区域比较；不要拼接独立 HILLS 文件伪装成一个共享偏置轨迹。`replica_spread` 是副本离散程度，不是校准后的置信区间。

历史偏置使用 Python 时间步循环，可使用兼容的编译子步；强制 `loop_backend="native"` 会被拒绝。采样成本也会随直接求和的 hill 数增长。当前没有实现时间依赖重加权、infrequent-metadynamics 速率或自动物理收敛认证，因此不能把有偏越垒时间当作反应速率。方法与恢复时序见[增强采样规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)。
