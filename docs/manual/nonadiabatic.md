# 实验性非绝热碰撞：NRPMD

NRPMD 是 `task="drpmd"` 的实验性扩展，**不是新的 task**。它在每个核珠子上加入电子 mapping 变量，用完整 diabatic 势能矩阵同时推进电子与核运动，并结合已有化学产物分类器估计电子态分辨的贡献。

当前问题是：从远区解耦、指定的入射 diabatic 电子态发射碰撞。它不建立电子热平衡系综，也不提供非绝热 PMF 或 Bennett–Chandler 平衡速率。普通 dRPMD 的初态、冲量参数和分类约定仍适用，见[气相工作流](workflows.md)。

## 运行完整示例

从仓库根目录执行，使用新的输出目录：

```bash
pyrpmd validate examples/nonadiabatic/exchange.inp --check-pes
pyrpmd run examples/nonadiabatic/exchange.inp --workers 2 --output runs/manual-nrpmd
pyrpmd resume runs/manual-nrpmd --block-steps 32
pyrpmd analyze runs/manual-nrpmd
```

[完整输入](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/exchange.inp)采用四珠子、两个电子态、两条母链及八条固定碰撞能轨迹。H + HD → H₂ + D 只是分类标签；[矩阵势](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/matrix_pes.py)是带非零局域耦合的教学模型，不是拟合的真实 H₃ PES。少量轨迹可能没有产物，或在时间预算内仍未决。

该例设置 `collision.sampling="fixed_energy"`，符合发布条件时输出截面 `bohr^2`，不输出热速率。改为 `thermal_flux` 并删除 `energy` 可设置指定入射电子态的热碰撞分布；这仍不等同于电子热平衡速率。

## 配置与矩阵 PES

在普通 dRPMD 输入中使用以下势能与电子设置：

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

`initial_state` 从 0 开始；标签对应固定 diabatic 基，不是按每步能量排序的绝热态。factory 返回带 `evaluate(q)` 的对象或 callable，结果使用 `pyrpmd.nonadiabatic.DiabaticResult(matrix, gradients)`：

| 数据 | 规范形状与单位 |
| --- | --- |
| 输入核坐标 q | `[B,N,3]`，bohr |
| 完整能量矩阵 | `[B,K,K]`，hartree |
| 矩阵正梯度 | `[B,K,K,N,3]`，hartree/bohr |

B 是本次 batch 大小，K 是电子态数。两组输出都必须实数、有限，并在电子指标上对称；每个对角元和耦合项都需要梯度。这里要求**正梯度**，与标量 PES 常用的负梯度力不同。若声明 provider 的坐标/能量单位，适配层会统一转换。

共同的态无关能量要放在每个对角元中；不要再叠加第二份标量势，也不要乘除珠子数。绝热能量梯度或导数耦合不能直接塞入这个 diabatic 能量/梯度接口。`validate --check-pes` 会检查矩阵并做逐矩阵元的梯度有限差分诊断。

## Mapping 变量与有符号估计

核变量为 `q,p[P,N,3]`，电子变量为 `mapping_q,mapping_p[P,K]`。母链在指定入射态的对角势上采样；每次发射为每珠子、每电子态抽取独立相位，focused 初始化使入射态布居为 1，其余为 0。碰撞生产段不恒温。

令电子变量为 x、y，轨迹的态布居估计是

$$
n_s(t)=\frac1P\sum_b\frac{x_{bs}(t)^2+y_{bs}(t)^2-1}{2}.
$$

总和应在舍入误差内为 1，但单个态估计可以小于 0 或大于 1。它不是给单条轨迹分配势能面的概率：不能截断到 `[0,1]`，也不能用 argmax 替代。

化学通道 α 与电子态 s 的联合贡献为 `w * I_alpha * n_s`，使用全部原始发射轨迹数归一化；各电子态贡献之和应恢复相应化学贡献。所有通道共享母链 bootstrap 抽样以保留联合协方差，有符号估计不使用二项/Wilson 区间。

## 停止条件、输出与恢复

发射时所有珠子的非对角耦合都必须低于 `asymptotic_coupling_tolerance`。即使几何上已经分离，产物或返回反应物也要满足电子解耦条件才被接受；否则继续到原始时间预算。耦合门槛、分离距离和传播时间均需收敛检查。

| 输出 | 应检查的内容 |
| --- | --- |
| `results/summary.json` 的 `nonadiabatic` | `final_estimates_available`、态标签、布居总和误差、通道结果 |
| `results/electronic_channels.csv` | 有符号观察贡献与可用的热速率/截面 |
| `nonadiabatic.channels` | `signed_observed_contributions`、`rate_cm3_molecule_s` 或 `cross_section_bohr2` |
| `parent_cluster_bootstrap_covariance` | 按 `covariance_columns` 排列的联合误差信息 |

存在 failed/unresolved 轨迹时，不发布最终电子速率或截面，但保留观察贡献；未决的有符号贡献不能套用普通 0–1 概率上下界。`experimental=true`、`physically_validated=false` 明确保留模型限制。

检查点保存核与 mapping 数组、时钟、分类持续状态和 RNG。单步失败同时回滚电子与核状态。`resume` 恢复中断工作，不延长已经到达原始时间预算的未决轨迹；改变最大碰撞时间应新建计算。

## 当前能力边界

矩阵传播使用 CPU NumPy，支持串行或本机轨迹进程；核自由环可使用已有编译内核。当前不支持 MPI、珠子 worker 拆分、GPU、融合原生电子传播、周期/固定原子体系，以及一般复 Hermitian 自旋–轨道耦合。`loop_backend` 使用 `auto` 或 `python`。

独立 singlet/triplet PES 不能在缺少适当耦合时产生系间窜越。真实应用需要有依据的 diabatic 基、完整矩阵及梯度，并验证远区态、能量和 mapping 范数、珠子数、步长、母链相关性、碰撞参数与产物承诺。完整方法和来源见 [NRPMD 专题](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic.md)及[示例说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/nonadiabatic/README.md)。
