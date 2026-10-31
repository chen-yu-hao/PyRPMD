# 入口归一化：H + H₂ / BKMP2 的双键长 2D QTST

## 已实现的统计量

反应物 H(0)+H(1)H(2) 的入口球面取
`R=|q̄_0-(m1 q̄_1+m2 q̄_2)/(m1+m2)|=Rinf`。
在非相互作用的自由入口渐近极限：

```math
k^{(2)}_\mathrm{in}=4\pi R_\infty^2
  \sqrt{\frac{1}{2\pi\beta\mu}}\;g_\mathrm{el}g_\mathrm{sym},
\quad \mu=\frac{m_0(m_1+m_2)}{m_0+m_1+m_2}.
```

在 1500 K，16 beads，Rinf=10 bohr，三个中性 H 质量分别取
1837.1526474396187 electron masses 时，
`k_in=0.9873012308830207 bohr^3/au_time`。
这只是**入口几何通量前因子**，不等于反应速率。

双键长分界面的两个原子标记为
`sigma_1=r01-r12=0`, `sigma_2=r02-r12=0`。
利用完整构型的第三键长进行互斥通道分类。法向热速度使用
`sqrt(grad(sigma)^T M^-1 grad(sigma)/(2*pi*beta))`，
不是第二次把平均速度乘入旧的 `rate_flux` FES。
纯二维 `cv_density` PMF 必须按正确曲线质量度量积分。

**入口比必须从同一无偏正则分布获得：**

```math
k_i^{(2),\mathrm{QTST}}=k_\mathrm{in}^{(2)}
\frac{\langle\delta(\sigma_i)\dot\sigma_i^+\,h_i\rangle}
{\langle\delta(R-R_\infty)\dot R_\mathrm{in}^+\rangle}.
```

`entrance_flux_ratio` 使用一个共享 MBAR 权重集合和矩形核
`delta_h(s)=1(|s|<=h)/(2h)`，要求原始采样关联身份、连通的
reference-to-target overlap、入口和目标各自的最小有效样本数。单个 PMF
的任意零点或接触复合物归一化不能替代这个比。对使用不同偏置重建的
两份 FES **不能分别归一后相除**。

## Python

```python
from pyrpmd.analysis import (
    free_h3_entrance_flux, h3_entrance_from_centroid_samples,
    bond_pmf_rate_from_entrance_bridge,
)
entrance = free_h3_entrance_flux(10, beta_au, masses_au)
# 跨入口和过渡态的同一系综无偏联合采样，并正确使用 MBAR 计算 per-configuration 权重
bridge = h3_entrance_from_centroid_samples(
    sampled_centroids, joint_mbar_weights, masses_au, beta_au,
    radius_bohr=10, reference_width_bohr=0.2,
    target_width_bohr=0.1, canonical_identity=sampled_physics_sha,
    measure="joint_unbiased_canonical", connected=True,
)
# 以该入口桥接对有限的 F(r01,r12) PMF 同一键长对角线锚定；只为第一标记通道
qtst = bond_pmf_rate_from_entrance_bridge(
    F2d, [edges_r01, edges_r12], beta=beta_au,
    curve=target_curve, positive_speed=target_speed,
    anchor_curve=bond_diagonal_curve, anchor_positive_speed=diagonal_speed,
    bridge=bridge, canonical_identity=sampled_physics_sha, mask=masked_bins,
)
```

离线命令：`python benchmarks/h3/entrance_normalization.py --samples bridge.npz --output normalized.json`。
`bridge.npz` 应具有 `centroids`, `unbiased_weights`, `beta_au`,
`masses_au`, `canonical_identity`, `measure='joint_unbiased_canonical'`,
`overlap_connected`, `sample_set_sha256`，必须保持真实原始样本来源。
`sample_set_sha256` 仅用于追踪来源，不能替代物理审核。

## 真实数据检查与剩余问题

先前 307200 个 CV 样本的双键长 PMF 来自有限范围
`1.15–5.55 bohr`。其 128 个保存的 checkpoint 端点中，反应物
质心 COM 距离 `R` 为 `0.5674–7.7343 bohr`，在
`Rinf=10±0.2 bohr` 内的 checkpoint 为 **0**。更重要的是，
原文件没有覆盖 **全反应物入口和过渡态的联合无偏 MBAR 构型权重**，
因此仅检查 checkpoint 端点不能用来计算真实通量比。

现在能从真实数据确定的是解析入口前因子，以及**双键长绝对 QTST
仍不可用**；没有伪造速度、体积因子、比例或把旧的 1D kappa 硬套到新面。
新增加的接口是可运行的后处理器；完整的入口桥接窗口、额外 CV 原始流
和与 2D 面匹配的无偏约束父链 / 子轨迹仍需采样并检验。

物理收敛检查还包括 shell 带宽外推、独立链 bootstrap、入口距离、
第三键长与两出口互斥性、珠数/时间步以及真实长时间产物承诺。
所有接口返回 `converged=false`，不把候选速率升级为正式预测。

参考：[RPMDrate rate theory](https://doi.org/10.1021/acs.jpca.6b07140)。
