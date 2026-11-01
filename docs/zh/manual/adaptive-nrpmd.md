# 自适应反应路径、恒电势与热非绝热 RPMD

本实验性 Python 接口把已有 `ThermalNRPMDSampler`、电子库恒电势求解器和
`ThermalNRPMDDynamics` 接起来，提供可运行的完整流程：

1. 在选定集体变量中，用有符号热初态和固定 umbrella 探索路径。
2. 固定端点，以垂直受限平均力更新路径，并等弧长重分布。
3. 冻结最终路径；使用新的随机流、重新平衡和独立生产阶段取样。
4. 对绝对 SEO 权重做 MBAR，再携带原始符号计算物理平均。
5. 按正 MBAR 权重抽取完整初态，沿同一个恒电势矩阵无偏推进核与电子。

完整参数、公式和参考文献见 [英文技术说明](../../adaptive-nrpmd.md)。

## 直接运行

安装 PyRPMD 后，在仓库根目录执行：

```bash
python examples/electrochemistry/adaptive_constant_potential.py \
  --beads 2 --potential-V -0.3 --output runs/adaptive-electrode.json
```

`--quick --beads 1` 可用于廉价的经典极限冒烟测试；`--quick --beads 2` 的
短预算可能正确触发符号／有效样本不足诊断。JSON 记录初始与最终路径、窗口符号、
MBAR 重叠、电子化学势残差，以及允许发射时的无偏短轨迹和 Hamiltonian 漂移。
该解析两态弯曲势谷使用模型原子单位，没有标定真实电极。
`physical_converged` 始终为 `false`；它不输出已收敛的化学速率。

## 恒电势的具体含义

新增 `ConstantPotentialDiabaticPotential` 组合的是可明确分离的模型：

$$
V_\mu(R)=V_{\mathrm{bare}}(R)+\min_N[E_{\mathrm{charge}}(R,N)-\mu N]I.
$$

电子充电项对所有 diabatic 电子态相同，且不能已包含在裸矩阵势中。
它改变核力和路径；在固定核构型下，各电子态能隙和耦合保持原值。
这不是一般的电子态分辨充放电、金属连续能带电子交换或法拉第电流模型。
那类问题需要额外的、物理上一致的多态矩阵势和电子库动力学。
如果已有完整的、含电位依赖能隙／耦合的有效矩阵势，可直接传给路径入口，
替换公共充电项适配器。适配器本身保留复 Hermitian 耦合，但当前 SEO 热采样
与路径流程仍要求实对称矩阵；尚未实现复相位重加权。

这里每个构型／珠子独立求解 BO 电子极小值，并要求电子数域内严格凸、
驻点位于边界内部、化学势残差收敛。它与已有全环共享电子数的
`GrandCanonicalSampler` 是不同近似，初态不可静默混用。
电位标尺和参考化学势均由 `ElectronReservoir` 显式设置，没有默认 SHE/RHE 标定。

## Python 接口

```python
from pyrpmd import AdaptivePathOptions, run_adaptive_nrpmd_path
from pyrpmd.electrochemistry import ConstantPotentialDiabaticPotential
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics

potential = ConstantPotentialDiabaticPotential(bare_matrix, electrode)
options = AdaptivePathOptions(
    max_iterations=5, pilot_samples=200, pilot_burn_in=200,
    production_samples=1000, production_burn_in=500,
    min_average_sign=0.1, min_effective_samples=20,
)
result = run_adaptive_nrpmd_path(
    potential, masses, beta, images, nstates, cv, stiffness,
    options=options, cv_scales=scales, seed=42,
)
print(result.summary())
mean_cv = result.reweighted.expectation(result.reweighted.cv_values)
parents = result.reweighted.resample_parents(200, seed=23)
dynamics = ThermalNRPMDDynamics(
    potential, masses, beta, images.shape[1], dt=0.01,
    nstates=nstates, native=False,
)
```

`images` 形状为 `[K,P,N,3]`，至少三个路径图像。`cv_scales` 定义路径度量，
混合单位必须显式设置。当前路径只支持非周期且可固定身份的内置分子 CV。
端点的 CV 窗口中心固定，端点构型仍然在 umbrella 中采样。

这是有限刚度、指定缩放度量下的受限路径探索；不自动发现所有通道，
不等同于严格的最小自由能路径或承诺概率面。
`max_step` 限制等弧长重分布之前的垂直力更新步。
窗口符号不足时停止路径更新并记录原因，不使用绝对权重均值冒充物理均值。

## 统计权重与实时传播

已有偏置约定是总珠子 Hamiltonian 加入 `B=P*W`，因此热采样的对数权重
增加 `-beta/P*B=-beta*W`。符号和电子投影估计器原样保留。
生产窗口必须共享同一势对象、温度、质量、珠子数及电子表示，偏置身份
也要与记录一致，才能在 MBAR 中消去公共势。

物理观测量使用

$$
\langle A\rangle=\frac{\sum_n w_n\sigma_n A_n}{\sum_n w_n\sigma_n},
\qquad w_n\ge0,\quad\sigma_n=\pm1.
$$

MBAR 不收敛、窗口重叠断开、平均符号未分辨或有效样本诊断不足时，
`expectation` 和 `resample_parents` 会拒绝给出可用结果。
绝对测度下的窗口自由能偏移不会被标成物理 PMF。
有效样本量和符号误差是启发式诊断，不是对源链及 MBAR 拟合做 bootstrap
得到的可靠置信区间。
`min_effective_samples` 检查正的绝对测度有效样本量；另行报告的
`signed_ess_diagnostic` 可更小，不作为该阈值的含义。

抽样父态保存全部核珠子、动量、mapping 变量、SEO 符号和来源索引。
实时阶段使用已有 `ThermalNRPMDDynamics` 的标量／无迹矩阵分解，
避免把公共充电力乘上涨落的 mapping 范数。此阶段没有 umbrella 力。
重采样会重复父态，不能视为新增独立平衡样本；已有速率摘要会阻止把重排后的
父态直接做普通 block error 或据此宣称速率平台收敛。

MC sweep、路径迭代和电子数求解均不是物理时间。
后续仍须检查珠子数、时间步、独立链、符号问题、路径尺度／刚度、初态覆盖
和真实动力学平台。当前不添加 `.inp` task，也没有整个自适应外循环的
断点续跑接口；单个热采样器继续支持包含偏置身份的 snapshot/restore。
