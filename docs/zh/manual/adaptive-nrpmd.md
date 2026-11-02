# 自适应反应路径、恒电势与热非绝热 RPMD

本实验性 Python 接口把已有 `ThermalNRPMDSampler`、电子库恒电势求解器和
`ThermalNRPMDDynamics` 接起来，提供可运行的完整流程：

1. 在选定集体变量中，用有符号热初态和固定 umbrella 探索路径。
2. 固定端点，以垂直受限平均力更新路径，并等弧长重分布。
3. 冻结最终路径；使用新的随机流、重新平衡和独立生产阶段取样。
4. 对绝对 SEO 权重做 MBAR，再携带原始符号计算物理平均。
5. 按正 MBAR 权重抽取完整初态，沿同一个恒电势矩阵无偏推进核与电子。

完整参数、公式和参考文献见 [英文技术说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/manual/adaptive-nrpmd.md)。

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

## 多条独立链的完整计算

较完整的计算覆盖三个电势、三个珠子数，并与独立的有限珠子数网格参考比较：

```bash
python examples/electrochemistry/adaptive_nrpmd_study.py --preset full --plan
python examples/electrochemistry/adaptive_nrpmd_study.py \
  --preset full --workers 3 --output runs/adaptive-study
python examples/electrochemistry/adaptive_nrpmd_reference.py \
  --convergence --output runs/adaptive-study/reference.json
python examples/electrochemistry/analyze_adaptive_nrpmd_study.py \
  runs/adaptive-study --reference runs/adaptive-study/reference.json --plot
```

绘图需要安装 `analysis` 扩展依赖。完整预设为 `U=-0.6,0,+0.6 V`、
`P=1,2,4`，每点有五个窗口、每窗口三条独立生产链、每链 512 个保存样本，
九个参数点共 **69,120 个生产构型**。路径最多学习三轮，pilot 数据不进入
生产估计。电势仍使用显式模型参考，没有真实 SHE/RHE 标定。

64 次 bootstrap 在原始链／窗口内抽取时间块，并在每次重复中重新拟合 MBAR。
置信区间要求每条源序列至少有八个完整时间块，而且所有重复都通过重加权与
符号归一化检查。区间以冻结路径和选定时间块为条件，不证明热化充分或尾部
完全覆盖。平均符号、正测度有效样本数与有符号有效样本诊断分别保存；
不通过门槛的生产结果不会输出一个貌似可靠的数值。

独立参考解析积分 mapping 变量，保留不对易电子 Boltzmann 矩阵的有序乘积，
在坐标网格上计算同一有限珠子数的平衡量。珠子二阶矩与质心二阶矩使用不同
估计式，后者保留虚时交叉相关。网格加密、两次计算域扩大以及 `P=8,16`
的额外参考分别记录，从而区分采样误差、有限域误差与珠子离散误差。

实时传播使用同样的 16 个完整父状态，以 `dt=0.02,0.01,0.005` 推进至
相同的 `t=2` 原子时间，比较完整核／mapping 相空间、带符号观测量与能量漂移。
这是给定父状态下的积分误差检查，不是化学速率的统计收敛证明。

每点保存 `report.json`、完整源样本 `source.npz` 及可用时的 `dynamics.npz`。
`--resume` 检查配置、相关源码和数据哈希后复用完整完成的参数点；尚不支持
从一个中断参数点的自适应迭代内部续跑。`--config` 接受 JSON 字段覆盖，
可以显式增大预算而不修改驱动源码。

## 与已发表论文的对比

独立基准使用 Richardson 与 Thoss（2013），
[JCP 139, 031102，DOI:10.1063/1.4816124](https://doi.org/10.1063/1.4816124)
的式 (15) 与图 1(c)：`m=hbar=omega=kappa=Delta=beta=1`、`alpha=0`。

```bash
python examples/nonadiabatic/published_nrpmd_benchmark.py \
  --workers 3 --output runs/published-nrpmd \
  --published-data examples/nonadiabatic/data/richardson_thoss_2013_fig1c.csv
```

默认运行 `P=1,4,8`，每个珠子数四条独立链、每链 512 个父状态，比较论文的
核质心位置 Kubo 自关联和电子布居 Kubo 自关联。电子初始端使用 SEO 投影
估计器，传播端使用 MMST 布居，沿轨迹保留初始符号。误差分析同时重采样
分子、分母，保留原始链的时间相关。

对比文件明确区分论文图中的数值、独立重算的 sinc-DVR 参考，以及本次 NRPMD
采样结果。论文数据附来源、坐标标定和读图误差。DVR 自身也进行网格／域检查，
不会将独立重算值冒充论文原始数据。

论文采用最低绝热能面的哈密顿量分拆，因此示例中的 `Paper2013Dynamics`
在复用已有 MInt 和自由环传播器时明确实现该分拆，并包含对应的
`-Tr(V_mapping)/2` 能量项。库默认的迹／无迹分拆仍保留，可通过
`--potential-split trace` 比较，并明确标记为不同近似。

这项论文对比检验已有非绝热核量子动力学部分。论文没有使用新增的公共充电
电极模型或自适应路径算法，因此不能据此宣称组合方法已通过真实电化学验证。

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

## 原始方法出处与复现任务

分别引用热映射、受限路径/string、MBAR 以及选用的电极模型。这个自适应带符号流程是包内组合，不能称为完整 Γ_W SM-NRPMD 或完整巨正则 DFT 方法的实现。

- [J. O. Richardson and M. Thoss (2013), Communication: Nonadiabatic ring-polymer molecular dynamics](references.md#na-nrpmd2013). [DOI](https://doi.org/10.1063/1.4816124).
- [L. Maragliano; A. Fischer; E. Vanden-Eijnden; G. Ciccotti (2006), String method in collective variables: Minimum free energy paths and isocommittor surfaces](references.md#samp-stringcv2006). [DOI](https://doi.org/10.1063/1.2212942).
- [M. R. Shirts; J. D. Chodera (2008), Statistically optimal analysis of samples from multiple equilibrium states](references.md#samp-mbar2008). [DOI](https://doi.org/10.1063/1.2978177).
- [M. Sun; B. Jin; X. Yang; S. Xu (2025), Probing nuclear quantum effects in electrocatalysis via a machine-learning enhanced grand canonical constant potential approach](references.md#gc2025). [DOI](https://doi.org/10.1038/s41467-025-58871-7).

逐项计算：[na-adaptive-thermal](reproduction.md#task-na-adaptive-thermal) · [na-thermal-nrpmd](reproduction.md#task-na-thermal-nrpmd) · [samp-string](reproduction.md#task-samp-string) · [gc](reproduction.md#task-gc).
