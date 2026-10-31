# 恒电位电极模型与表面电流

`pyrpmd.electrochemistry` 将电子数相关势能接入现有的周期体系、固定基底、
FES 和传播器。它包含三个独立组件，使用前先选择对应的物理模型。

| 计算内容 | 入口 | 结果含义 |
| --- | --- | --- |
| 电子即时绝热松弛后的有效势 | `AdiabaticElectrodePotential` + `task="fes"` | 每个核构型独立最小化的巨势与量子核自由能 |
| 环聚合物和一个共享电子数的联合平衡 | `GrandCanonicalSampler` | HMC/MC 平衡样本；步数没有真实时间含义 |
| Volmer、Heyrovsky、Tafel 速率给定后的表面行为 | `steady_state_her` | 稳态 H 覆盖度、H₂ TOF、有符号电流密度 |

已有的 `task="pcet"` 仍用于[弱耦合局域 golden-rule 速率](pcet.md)。
新组件与它互补；局域复合物的 s⁻¹ 速率需要先匹配表面位点、活度和标准态，
才能作为微观动力学系数。

## 运行示例

在仓库根目录运行：

```bash
pyrpmd validate examples/electrochemistry/fes2d.inp --check-pes
pyrpmd run examples/electrochemistry/fes2d.inp --output runs/electrode-fes
pyrpmd resume runs/electrode-fes
python examples/electrochemistry/sample_grand_canonical.py
python examples/electrochemistry/her_scan.py
```

示例使用解析、带充电项的周期势，未拟合任何电极材料。FES 为软件测试用
短采样；HER 扫描中的速率是另行指定的教学系数，不从该 FES 自动提取。

`ElectronReservoir` 要求显式给出电位参考与参考电子化学势：

$$
\mu_e(U)=\mu_{e,\mathrm{ref}}-e(U-U_\mathrm{ref}).
$$

程序不猜测 SHE/RHE 换算或电子结构能量零点。电子数相关 PES 接口为
`evaluate(q, electron_numbers)`，返回 `ElectronicPESResult` 中的能量、核力和
正的偏导数 `denergy_dnumber`。全部使用原子单位；电子数可以是模型定义的连续超额电子数。

## 两种电子数模型的区别

绝热适配器对每个构型定义
\(\Omega(q)=\min_N[E(q,N)-\mu_eN]\)。每颗核珠子独立调用这个有效势，
因此可以复用既有 PES 批处理、FES、周期 CV 与并行执行路径。电子数方向必须是
光滑、严格凸的有效模型，且极小值位于指定域内；域外或求解失败会报错。

共享电子数采样器针对另一种平衡分布：

$$
\rho(q,N)\propto\exp\left[-\frac{\beta}{P}H_P(q;N)+\beta\mu_eN\right].
$$

它复用 `Dynamics` 生成固定 N 的核 HMC 提案，再执行电子数 MC。
电子数更新的能量项是珠子平均能量，核 HMC 的接受指数为
\(-\beta\Delta H_P/P\)。所有采样诊断使用 Monte Carlo 次数。
这不是 TP-RPMD 的扩展电子实时动力学，也不直接给出再穿越传输系数。

## 从表面速率到电流

`HERRates` 接收六个正反向速率系数。`steady_state_her` 还要求质子活度、
氢气活度和位点摩尔面密度。电流单位为 A/m²，阳极为正、阴极为负。
稳态满足 \(j=-2F\Gamma_{\mathrm{site}}\mathrm{TOF}_{\mathrm H_2}\)；
非稳态还必须保留吸附氢积累项。

周期质子键长差可以使用已有的 `periodic_bond_difference` 计算 FES。
`surface_rate` 支持其规定的线性质心与恒定质量度量的距离坐标；周期距离还须
满足最小镜像光滑性条件。新适配器不会放宽这些限制，参见[材料与表面](periodic.md)。

完整公式、Python 调用、参考源码提交及适用范围见仓库的
[电化学接口说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/electrochemistry.md)。
