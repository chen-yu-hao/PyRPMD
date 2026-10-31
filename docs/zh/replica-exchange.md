# FES PILE-L 与副本交换采样

`task="fes"` 现在支持 PILE-L 恒温采样，以及固定伞窗的同温副本交换（REUS）。
支持一维和二维 CV，保留默认 Andersen 行为。这些都是平衡构型采样，**交换后的轨迹时间
不是物理 RPMD 时间**，不能直接用于速率、回穿因子或碰撞分支比。

## PILE-L

```toml
[dynamics]
integrator = "cayley-bcb"
loop_backend = "auto"
timestep = "0.1 fs"

[sampling]
thermostat = "pile-l"
gamma_centroid = "0.02 fs^-1"
equilibration = "20 fs"
production = "100 fs"
interval = "0.2 fs"
replicas = 2
```

Cayley 对应 BCOCB，normal-mode-bab 对应 BAOAB；直接复用 `Dynamics.sample_step`
与现有原生 free/OU/free 内核，没有重新实现积分器。固定窗口和自适应 FES 的 pilot / production
均可使用。可选 `sampling.integrator` 必须与对应方法匹配。摩擦系数默认 `0.01 fs^-1`，
直接 Python API 的无单位数值采用原子单位。

PILE-L 不与 velocity-Verlet 或严格 `loop_backend="native"` 组合。`auto` 仍可调用原生
子步，外层编排与 NumPy 随机数保留在 Python。`rate`、`surface_rate` 和 metadynamics 的恒温
限制没有改变。周期晶胞、固定原子的普通 FES 继续使用已有势能/CV 接口。

## 固定窗口同温 REUS

在普通 FES 输入中增加：

```toml
[replica_exchange]
enabled = true
interval_steps = 50
# pairs = [[0, 1], [1, 2]]  # 可选，零起始窗口编号；必须连接全部窗口
```

所有窗口使用相同温度、珠子数、质量、物理 PES 与边界条件。仅固定窗口 `cv_density`
测度可用；不能同时启用自适应插窗。不支持异温、异珠子数、不同 PES 或历史依赖偏置交换。
显式交换图拒绝孤立窗口、自环、重复/反向重复、非整数和越界编号。省略 `pairs` 时，采用
`beta * delta.T @ K @ delta` 度量下的确定性最小生成树，周期 CV 使用最短回绕位移。
超过 2048 个窗口需要显式提供图。图连通不代表实际分布已经重叠。

每轮按与构型无关的随机顺序遍历所有边各一次，交换整条环的 q/p。物理伞势为 W 时：

\[
\log R_{ij}=-\beta[W_i(s_j)+W_j(s_i)-W_i(s_i)-W_j(s_j)],\quad
A_{ij}=\min(1,\exp(\log R_{ij})).
\]

总珠子 Hamiltonian 中的偏置是 P W，采样逆温是 beta/P，因而接受指数**不额外乘 P**。
动力学、弹簧与同一物理 PES 项抵消。同温时动量不缩放；窗口的时钟和随机流不交换。
接受判定使用对数概率。

`sampling.replicas` 在 REUS 中表示独立的**完整交换链**。复用现有 process/MPI 任务调度和
珠子势能分解；一条交换链内的窗口目前顺序推进，不宣称窗口级分布式交换或固定加速比。
样本按照采样时的窗口保存，在与交换重合的时刻先保存样本再交换。`fes.h5` 额外保存
`exchange_group` 和 `walker_ids`，`exchange.json` 记录每条边尝试/接受数、接受率及 walker
访问矩阵（包含热化阶段的交换）。这些诊断不自动证明充分混合。

## 误差与恢复

同一交换链各窗口并不独立。MBAR 点估计不变；`reconstruct_fes(..., bootstrap_groups=...)`
使同链窗口共用循环分块重采样的时钟索引，不同链独立抽取。组内必须等长、等间隔、同步采样；
以组内最大 CV 统计低效因子选择既有保守块长，并保留 unavailable / limited_blocks 判据。
不提供分组时，旧 bootstrap 的索引和随机流逐位不变。CV 相关性仍可能漏掉隐藏慢自由度，
这些有限样本误差不是覆盖率保证；独立初态族、块长敏感性及珠子数/步长/采样长度检查仍必需。

每条链以 `checkpoints/exchange_0000.npz` 原子保存全部窗口 q/p、时钟、窗口/交换 RNG、
样本、能量、walker 历史、交换决定及计数。复用校验和封装并检查数组形状、同步时钟、固定
原子、排列、样本归属和计数。推进到下一事件并完成整轮交换是一个事务；任一窗口或半轮
交换失败都会回滚**整个同步事件区间**。写盘故障保留旧或新完整状态，不接受混合时刻。
完成后 resume 和离线分析不创建 PES。

直接 Python 工作流对支持的、未覆盖方法的内置 CV 对象使用数学参数建立 checkpoint 身份。
自定义、覆盖方法、周期或表面 CV 对象需要改用可序列化的 CV 字典或工厂配置；通过字典配置的
周期/表面 FES 支持不变。这样恢复不会因对象复制或跨进程后的内存地址变化而失效。

## 运行与范围

```bash
pyrpmd validate examples/fes/replica_exchange_2d.inp
pyrpmd run examples/fes/replica_exchange_2d.inp --output runs/reus2d
pyrpmd resume runs/reus2d
pyrpmd analyze runs/reus2d --recompute --bins 12 12 --bootstrap 10
```

示例为 P=4 的谐势、9 窗口和 2 条交换链，不是生产收敛配方。Python 仍调用
`run_fes` / `reanalyze_fes`，按配置分派。无采样支持的格子和断连的 MBAR 重叠仍被 mask；
`converged=false` 保留。本方法不提供 ABF、MetaD 多 walker 或新的化学势能面。OPES 是独立的采样工作流。

## 方法文献

实现独立编写，没有复制下列工作的源码。

- Sugita, Kitao, Okamoto, *Multidimensional replica-exchange method for free-energy calculations*,
  JCP 113, 6042 (2000), DOI: 10.1063/1.1308516；[作者稿](https://arxiv.org/abs/cond-mat/0009120)。
- Ceriotti et al., *Efficient stochastic thermostatting of path integral molecular dynamics*,
  JCP 133, 124104 (2010), DOI: 10.1063/1.3489925；[作者稿](https://arxiv.org/abs/1009.1045)。
