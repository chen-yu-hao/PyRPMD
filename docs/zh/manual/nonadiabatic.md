# 实验性非绝热碰撞：NRPMD

NRPMD 是 `task="drpmd"` 的实验性模式。它在每个核珠子上加入电子 mapping 变量，在完整的 diabatic 势能矩阵上同时
推进电子与核，并把电子布居与化学产物分类器结合，给出电子态分辨的贡献。

碰撞从一个指定的入射 diabatic 电子态出发，该态在远区解耦。初始条件、碰撞参数和分类沿用普通 dRPMD，见[气相工作流](workflows.md)。

## 运行示例

在仓库根目录执行：

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

[输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)使用 4 个珠子、两个电子态、
两条母链和 8 条固定碰撞能的轨迹。[矩阵势](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)
是带局域耦合的人工双态模型；通道按 H + HD → H₂ + D 标记。

`collision.sampling="fixed_energy"` 时结果为截面，单位 `bohr^2`。改为 `thermal_flux` 并删除 `energy`，即得到从所选入射态出发的热碰撞分布。

## 配置与矩阵 PES

在普通 dRPMD 输入中加入：

```toml
[potential]
provider = "diabatic_python"
factory = "matrix_pes.py:make"
derivative = "gradient"

[nonadiabatic]
method = "nrpmd"
nstates = 2
initial_state = 0
state_labels = ["entrance", "excited"]
initialization = "focused"
asymptotic_coupling_tolerance = "1e-8 hartree"
```

`initial_state` 从 0 开始，标签对应固定 diabatic 基中的态。factory 返回带 `evaluate(q)` 的对象或可调用对象，
返回 `pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`：

| 数据 | 形状与单位 |
| --- | --- |
| 核坐标 q | `[B,N,3]`，bohr |
| 能量矩阵 | `[B,K,K]`，hartree |
| 矩阵的梯度 | `[B,K,K,N,3]`，hartree/bohr |

B 是批大小，K 是电子态数。两个数组均为实数、有限，并在电子指标上对称，每个对角元和耦合都有梯度。矩阵接口使用
**正梯度**（标量势返回的是力）。provider 声明坐标或能量单位时，适配器负责转换。

与态无关的能量放在每个对角元中；该矩阵就是完整的势能。`validate --check-pes` 检查矩阵并对每个矩阵元的梯度做有限差分。

## Mapping 变量与有符号估计量

核变量为 `q,p[P,N,3]`，电子变量为 `mapping_q,mapping_p[P,K]`。母链在入射态的对角势上采样。每次发射为每个珠子和
每个电子态抽取独立相位；focused 初始化使入射态布居为 1、其余为 0。碰撞过程不加恒温器。

记电子变量为 x、y，一条轨迹的布居估计量为

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

各态布居之和在舍入误差内为 1，单个值可以落在 $[0,1]$ 之外。对轨迹求平均得到态分辨的贡献。

化学通道 α 与电子态 s 的联合贡献为 `w * I_alpha * n_s`，以发射轨迹总数归一化；对态求和即得化学贡献。所有通道共用
母链 bootstrap 抽样，从而得到它们的联合协方差。

## 停止条件、输出与续跑

发射时，所有珠子的非对角耦合须低于 `asymptotic_coupling_tolerance`。产物或返回的反应物在几何上分离且电子解耦时
才算已决；否则轨迹运行到时间预算。耦合阈值、分离距离和传播时间需要分别收敛。

| 输出 | 内容 |
| --- | --- |
| `results/summary.json` 中的 `nonadiabatic` | `final_estimates_available`、态标签、布居和误差、通道结果 |
| `results/electronic_channels.csv` | 有符号观察贡献及热速率或截面 |
| `nonadiabatic.channels` | `signed_observed_contributions`、`rate_cm3_molecule_s` 或 `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | 按 `covariance_columns` 排列的联合误差信息 |

若有轨迹失败或未决，最终的电子态速率或截面留空，仍报告观察到的贡献。摘要中带有 `experimental=true` 和
`physically_validated=false`。

检查点保存核与 mapping 数组、时钟、分类持续状态和 RNG 状态，单步失败时电子与核一起回滚。`resume` 在原时间预算内
完成中断的工作；更长的最大碰撞时间需要新的运行。

## 支持的配置

NRPMD 在 CPU 上用 NumPy 运行，可串行或使用本机轨迹进程；核自由环可使用编译内核。使用 `loop_backend="auto"` 或
`"python"`。此模式不支持 MPI、珠子 worker 拆分、GPU、融合原生电子传播、周期或固定原子体系，以及复 Hermitian
自旋–轨道耦合。

用于真实体系时，提供含态间耦合的 diabatic 基、完整矩阵及其梯度，并收敛远区态、能量与 mapping 范数、珠子数、
时间步、母链相关性、碰撞参数和产物判定。方法与来源见 [NRPMD 专题](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md)
和[示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md)。
