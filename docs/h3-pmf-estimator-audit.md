# H3 的 UI / MBAR 势垒差异审查

审查日期：2026-10-29。本页只讨论本地 **coarse** 数据：BKMP2 H + H₂，
1500 K、16 珠子、23 个窗口、每窗 2 条 replica，各生产 **300 fs**，
合计 **27,600 个样本**，固定已保存的 **ξ*=1.023**。
[完整诊断 JSON](validation/h3-bkmp2-2026-10-29/local-coarse-pmf-diagnostics.json)
保留曲线、窗口矩、端点支撑、配对 bootstrap 和来源校验信息。
当前 GitHub 生产测试使用每窗 **1 ps**，应另看其最终报告；两套数字不能混用。

源诊断记录三条 checkpoint 比完整 HDF5 短，但均为精确前缀；本页统计使用
完整 HDF5，没有将较短序列混入。长度差异的原因不由本审查推断，重放及文件
完整性结论应另看对应报告。

**结论：未发现能够解释差异的明显符号、单位、P 因子、metric 或参考对齐错误。
现有证据支持 Gaussian UI 的近似与有限样本敏感性，但不能把全部差异唯一归因于
平衡分布非高斯。此次没有独立速率验证，也没有证明物理收敛。**

## 1. 两种方法针对同一 flux 测度

设 β=1/(kBT)，P 为珠子数，Hₚ 是未加偏置的总环哈密顿量，
g=∇ξ 为相对 centroid 的梯度。
[bias.py](../src/pyrpmd/bias.py) 加入的能量为

$$
P W_i(\xi)-\frac{P}{\beta}\ln f_\xi,
\qquad f_\xi=\sqrt{\frac{g^T M^{-1}g}{2\pi\beta}}.
$$

[dynamics.py](../src/pyrpmd/dynamics.py) 按 β/P 采样，因此窗口分布满足

$$
\pi_i(q)\propto e^{-\beta H_P(q)/P}\,f_\xi(q)\,e^{-\beta W_i(\xi(q))}.
$$

MBAR 使用 βWᵢ 去掉窗口间不同的 umbrella，公共的物理势与 metric 留在目标测度中；
不应再除去 metric 或多乘一次 P。centroid 梯度的 1/P 与偏置能的 P 抵消，
力的实现与能量导数一致。该符号约定与核实的
[RPMDrate add_bias_potential](https://github.com/chen-yu-hao/RPMDrate/blob/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78/rpmdrate/_main.f90)
一致。

## 2. 分界面、参考点与归一化

[cv.py](../src/pyrpmd/cv.py) 的 ξ=s₀/(s₀−s₁) 与约束面满足

$$
S_{\xi_*}=(s_0-s_1)(\xi-\xi_*),\qquad
\delta(S_{\xi_*})f_{S_{\xi_*}}=\delta(\xi-\xi_*)f_\xi.
$$

因此 ξ 的 flux PMF 与约束面的 recrossing 测度匹配。
[rate.py](../src/pyrpmd/workflows/rate.py) 和
[独立诊断](../benchmarks/h3/pmf_diagnostics.py) 都比较 W(ξ*)−W(0)，
没有把不同位置的峰值相减。直方图范围为 [−0.325641, 1.448656]；
两端点均有未 masked 的相邻 bin 中心支撑，没有外推。

直方图使用密度而非单个 bin 的概率质量。公共归一化和能量零点在势垒中抵消；
有限样本的归一化常数不能解释为双分子全空间配分函数，但这不改变局部密度比。
无需再为其中一种方法单独添加径向 Jacobian。

## 3. coarse 的数值排错

以下代数复算使用同一完整 HDF5，不产生新轨迹或新采样。
UI 公式及权重与核实的
[RPMDrate computePotentialOfMeanForce](https://github.com/chen-yu-hao/RPMDrate/blob/9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78/rpmdrate/main.py)
一致；实现见 [analysis/fes.py](../src/pyrpmd/analysis/fes.py)。

| 检查 | 结果 |
|---|---:|
| 实际 βK | 99.9999999961 |
| 按原始 MBAR 方程重算权重，最大差 | 2.78×10⁻¹⁷ |
| 保存的 MBAR 方程残差 | 8.60×10⁻¹⁰ |
| 所有权重乘 10⁸ 后，势垒变化 | 2.18×10⁻¹⁵ kcal/mol |
| UI：401 点 / 4001 点 / 自适应积分 | 15.85830656 / 15.85858141 / 15.85858419 kcal/mol |
| MBAR：50 / 100 / 150 bins | 19.36188 / 19.54292 / 19.60038 kcal/mol |

401 点 UI 的积分误差仅约 0.000278 kcal/mol，三个 bin 分辨率的中心估计相差
约 0.24 kcal/mol，均远小于 100-bin 的 **MBAR−UI=3.68461 kcal/mol**。
40 次逐流连续块 bootstrap 全部有效；同次抽样同时计算两法，保留协方差，
该差值的 95% 百分位区间为 **[2.02982, 5.08213] kcal/mol**。
这与两法各自边际误差直接相加不同。

UI 以每条流的均值和方差近似其完整分布，再减去已知 umbrella 的平均力。
MBAR 则利用逐样本的相对偏置能求权重，不要求窗口分布为 Gaussian。
两者复用同一批轨迹，因此比较提供的是估计方法诊断，不能作为彼此独立的验证。

## 4. 可以和不能作出的归因

完整样本最大绝对偏度为 **1.61704**、最大绝对超额峰度为 **2.85984**，
提示 Gaussian 拟合需要警惕。不过短相关链的矩估计、慢自由度及未充分平衡，
也可能影响这些数值，不能把偏斜直接认定为已收敛的平衡性质。
相关性也意味着有效样本数少于原始计数；增加 bootstrap 复制次数只改善不确定度
估计的数值稳定性，不会增加独立物理信息或消除近似偏差。

另有一个有限样本设计差别：本实现 UI 对每条 replica 单独拟合 Gaussian，
上游通常先按同一窗口累积矩。仅合并相同 center 的两条 replica，UI 势垒变为
**15.52095 kcal/mol**；MBAR 对相同 Hamiltonian 的合并不变，权重最大差
**7.09×10⁻¹⁴**。这约 0.34 kcal/mol 的敏感性并非单位错误，也无法消除主要差异。

MBAR 权重本身不需将能量分箱，但此处重建 ξ 密度仍使用直方图，保留 bin 宽偏差。
[Shirts 与 Chodera 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2671659/)
讨论了平衡样本与相关性条件；本诊断的 block bootstrap 不能替代平衡性检查。
保留同一 ξ*、同一 κ 得到的次级速率，其区间仅条件于该 κ，不含 κ 自身误差。
珠子数、时间步长、R∞、热化及统计收敛，仍需分别验证。
