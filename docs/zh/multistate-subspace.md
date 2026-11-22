# 量子子空间中的多电子态耦合

**中文** · [English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/multistate-subspace.md)

> **状态：research。** 目前只有 Python API（`pyrpmd.analysis.multistate_subspace`），
> `subspace-correct` 命令和 `[rate.subspace_correction]` 仍然只构造单势能面模型。这里的
> “精确”只针对给定的子空间哈密顿量；一个 1–3 维、保留 $S$ 个电子态的子空间能否代表真实分子，
> 需要逐个体系检验。

[子空间速率修正](subspace-correction.md) 只处理一个 Born–Oppenheimer 势能面。本页把其中精确量子
那一半推广到经过**多个耦合电子态**的反应：避免交叉与锥形交叉、自旋–轨道耦合引起的系间窜越
（复数耦合），以及反应路径上的电荷转移交叉。

## 1. 子空间哈密顿量与精确速率

模型是定义在 1–3 维张量 sinc-DVR 网格上的 $S\times S$ 厄米透热矩阵：

$$
H_S = \sum_a T_a \otimes \mathbb{1}_S + V(\mathbf{s}), \qquad V_{ij} = V_{ji}^* .
$$

速率沿用 Miller–Schwartz–Tromp 公式，在第 0 轴两端加吸收势 $\varepsilon\otimes\mathbb{1}_S$，
与单面稠密解法完全一致：

$$
k\,Q_r = \operatorname{Re}\operatorname{Tr}[F_\beta X],\qquad H^\dagger X - XH = iF,\qquad
F = i[H_S, h],\qquad h = \theta(s_0 - s_0^\ddagger)\otimes\mathbb{1}_S .
$$

分界面取在核坐标上，对所有电子态求和。产物端吸收的通量按电子**通道** $c$ 分辨（按能量排序的
绝热态，或透热态）：

$$
\int_0^\infty e^{iH^\dagger t}Fe^{-iHt}dt = -h + \sum_c Y_c,\qquad
H^\dagger Y_c - Y_cH = iB_c,\qquad B_c = 2\,\varepsilon_P^{1/2}P_c\,\varepsilon_P^{1/2},
$$

于是 $k_cQ_r = \operatorname{Re}\operatorname{Tr}[F_\beta Y_c]$。所有通道共用同一个复 Schur
分解；各通道之和与总通量相差 $\operatorname{Tr}[F_\beta h]$，这个残差记在
`channel_sum_residual` 里。

$Q_r$ 对**模型中所有电子态**做热平均：`reactant = "asymptotic"` 时为
$\sqrt{m_0/2\pi\beta}\,\operatorname{Tr}_{\perp,\mathrm{el}}e^{-\beta H_\perp(s_\mathrm{ref})}$，
`reactant = "well"` 时为 $\operatorname{Tr}[e^{-\beta H_S}(1-h)]$。如果某个不参与反应的渐近电子态
（例如某个自旋分量）没有放进模型，需要自己把它的配分函数加上。

## 2. 与 RPMD 速率结合

`nonadiabatic_subspace_factor` 给出

$$
\eta_\mathrm{NA} = \frac{k_\mathrm{QM}(H_S^{\,\mathrm{multi}})}{k_\mathrm{QM}(H_S^{(0)})},
$$

其中 $H_S^{(0)}$ 是同一子空间在最低绝热面上的模型（不含对角 Born–Oppenheimer 修正）。
$\eta_\mathrm{NA}$ 包含非绝热跃迁、对角修正、锥形交叉附近的几何相位效应，以及模型中其他电子态的
热布居。把它乘进现有的单面修正：

$$
k \approx k^{(0)}_\mathrm{RPMD}(\text{全维})\times
\frac{k_\mathrm{QM}(H_S^{(0)})}{k_\mathrm{RPMD}(H_S^{(0)})}\times\eta_\mathrm{NA}
= k^{(0)}_\mathrm{RPMD}(\text{全维})\times
\frac{k_\mathrm{QM}(H_S^{\,\mathrm{multi}})}{k_\mathrm{RPMD}(H_S^{(0)})} .
$$

全维 RPMD 仍然只在基态势能面上运行，不需要多态环聚合物方法。这个分解的**前提**是：电子耦合
只在子空间坐标内起作用，其余模式跟随最低绝热面。$H_S^{\,\mathrm{multi}}$ 要和单面修正使用相同的
子空间坐标、网格和吸收势。如果耦合强烈依赖子空间外的某个模式，应当把它并入子空间，而不是
依赖这个分解。

## 3. 最小示例

```python
import numpy as np
from pyrpmd.analysis import (exact_multistate_rate, multistate_model_from_function,
                             nonadiabatic_subspace_factor)

def crossing(points, height=0.04, width=2.0, coupling=0.004):
    s = points[..., 0]
    out = np.zeros(s.shape + (2, 2))
    out[..., 0, 0] = 0.5 * height * (1 + np.tanh(s / width))   # 反应物透热态
    out[..., 1, 1] = 0.5 * height * (1 - np.tanh(s / width))   # 产物透热态
    out[..., 0, 1] = out[..., 1, 0] = coupling * np.exp(-(s / 4.0) ** 2)
    return out

model = multistate_model_from_function(crossing, [np.linspace(-12, 12, 241)],
                                       masses=2000.0, dividing_surface=0.0)
beta = 1.0 / (1000.0 * 3.166811563e-6)
result = exact_multistate_rate(model, beta)        # 默认 channels="adiabatic"
print(result.rate, result.branching)               # 总速率、产物通道分支比
print(nonadiabatic_subspace_factor(model, beta)["factor"])
```

复数耦合（例如自旋–轨道矩阵元）直接以复厄米矩阵传入。`model.adiabatic_surface(k)` 和
`model.diabatic_surface(i)` 返回普通的 `SubspaceModel`，可以用来和单面结果对比。

## 4. 从电子结构数据构造透热模型

在一维路径上，`propagation_diabatization(energies, overlaps)` 由绝热能量 `energies[N, S]` 和
相邻帧的原始重叠 `overlaps[N-1, S, S]` 构造 $V(s)$，其中
$O_n[i,j]=\langle\phi_i(s_n)|\phi_j(s_{n+1})\rangle$。这与
[激发态接口](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/excited-state-interface.md) 中 `BackendResult.overlap` 的约定相同：行按上一帧的
原始顺序，列按本帧的原始顺序。每个重叠矩阵先取极分解的幺正部分 $W_n$，再按
$A_{n+1}=W_n^\dagger A_n$ 传播绝热–透热变换，并在 `anchor` 帧取 $A=\mathbb{1}$。根的重排和
符号、相位变化都会被精确吸收。当保留的电子态构成完备子空间时，结果是精确的。奇异值低于
`min_singular_value`（默认 0.9）时会被拒绝，因为这说明有布居泄漏到未保留的态，或者步长太大。
锚点应放在透热耦合应当为零的地方，通常是反应物渐近区。

在二维、三维网格上，导数耦合含有不可消除的旋度部分，沿单条路径传播并不构成透热化。这种情况
请直接提供透热模型，例如拟合的线性或二次振动耦合模型，或者正则化透热化的结果。

## 5. 诊断与收敛

| 诊断量 | 含义 / 处理 |
| --- | --- |
| `channel_sum_residual` | 各通道通量之和减去总通量和 `flux_side_t0`，相对于总通量；应在舍入误差量级 |
| `flux_side_t0` | $\operatorname{Tr}[F_\beta h]$；精确热通量算符下为零 |
| `electronic.adiabatic_gaps` | 网格上及分界面所在网格线上相邻绝热态的最小能隙；接近零说明存在锥形交叉或交叉缝，网格必须能分辨 |
| `minimum_adiabatic_gap_in_product_absorber`、`channel_resolution_ambiguous` | 产物吸收区内绝热态简并时，单个绝热通道的投影不唯一（总速率不受影响）；改用 `channels="diabatic"` |
| `partition.reference_energy_levels` | $Q_r$ 中的电子和横向能级；检查保留的态是否覆盖了热布居的反应物能级 |
| `solver_plan.dense_memory_mb` | 稠密解法的内存估计；基函数数超过 `dense_limit` 时抛出 `MultistateRouteUnavailable` |

需要收敛的量：网格间距（交叉附近上绝热面上的动能可能很大）、盒子大小和吸收势（`cap_width`、
`cap_strength`），以及**保留的电子态数**。`dense_limit` 按网格点数 × 电子态数计算，默认 1600
时，两态二维网格大约上限为 $28\times28$。

## 6. 测试中的验证

`tests/test_multistate_subspace.py` 检验了以下各项：

- 单态时与单面稠密速率一致，误差 $10^{-12}$（一维渐近、一维势阱、二维）；
- 无耦合的各态按通道逐一叠加单面通量；
- 常数实/复电子基旋转，以及耦合上的常数相位，不改变总速率和绝热通道速率；
- 弱耦合满足黄金规则标度 $k_{0\to1}\propto|V_{01}|^2$（耦合加倍时比值为 4.002）；
- 强耦合收敛到“最低绝热面 + 对角 Born–Oppenheimer 修正”的精确速率（Massey 参数约 6 时偏差
  $1.0\times10^{-3}$）；
- 二维模型锥形交叉能被能隙诊断定位，速率在电子基旋转下不变，且通道求和闭合；
- 对打乱顺序、翻转符号后的绝热数据做传播透热化，可在 $10^{-12}$ 精度内还原透热模型（相差锚点帧的
  旋转），速率保持不变。

## 7. 尚未提供

- 超出稠密规模的多态迭代解法（Chebyshev 玻尔兹曼算符 + 矩阵分裂算符传播）；
- 与 `subspace-correct`、`[rate.subspace_correction]` 的集成：沿反应路径调用激发态 provider
  计算多个态，并对能隙和重叠完备性设置有效性检查；
- 二维、三维透热化，以及按反应物电子态选择的速率。目前速率对模型中的所有态做热平均；
- 束缚体系的凝聚相电荷转移。这需要耗散环境而不是吸收边界，例如把
  `pyrpmd.analysis.dvr_heom` 中的 DVR-HEOM 参考推广到多态，并与
  [非绝热数据与速率](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-nonadiabatic.md) 中的黄金规则、Marcus 和热 NRPMD 工具对照。
