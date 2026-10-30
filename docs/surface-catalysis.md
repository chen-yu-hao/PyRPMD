# 表面催化：固定晶胞、表面自由能与一阶 RPMD 速率

PyRPMD 的表面入口复用环聚合物传播、umbrella/MBAR、受约束母轨迹和回穿统计。
当前版本可以计算周期 slab 或有限表面簇上的 1D/2D 自由能，并针对**一个线性
centroid 反应坐标、一个明确反应物盆地、一个选定出口**计算一阶 RPMD 速率。
结果单位为 `s^-1`。势能模型须适用于目标材料、吸附物和反应。

## 已支持的范围

| 能力 | 输入／接口 | 说明 |
| --- | --- | --- |
| 固定晶胞 | `system.cell`, `cell_unit`, `pbc` | 3×3 行向量；支持倾斜晶胞、部分周期边界 |
| 固定底层 | `system.fixed_atoms` | 0-based 索引；所有珠子坐标固定，动量恒为零 |
| 表面高度 | `surface_height` | 吸附原子组 COM 对平面的线性投影，可相对于参考原子组 |
| 表面迁移 | `surface_lateral` | 沿指定方向的连续展开 COM 投影，不取周期模 |
| 周期键长 | `periodic_distance` | 一般倾斜晶胞下的最近像距离；显式启用 |
| 表面 FES | `task="fes"` | 1D/2D，MBAR、相关块 bootstrap、未采样区域 mask、离线重分析 |
| 表面速率 | `task="surface_rate"` | 位置密度 PMF → 盆地 QTST → 约束母轨迹 → ±p 回穿 |
| 势能 | ASE；显式支持 PBC 的 Python 插件 | 每个 worker 收到完整 System 和固定晶胞 |
| 执行与恢复 | 现有窗口／轨迹进程及珠子势能执行器 | 保存 cell、PBC、fixed atoms、q/p、RNG、科学指纹 |

固定原子仍参与真实 PES 的能量和对活动原子的作用力，只是不参与动力学自由度。
活动表层与吸附物按同一 bead 数和物理质量传播；当前不提供“仅吸附物量子化、
活动基底经典化”的混合 bead 模型。NVT 采样使用已有 Andersen；回穿子轨迹无热浴。

## 三个可运行示例

从仓库根目录运行：

```bash
python -m pip install -e '.[test,ase]'

# 解析周期模型：一个局域吸附井向右侧出口的迁移速率。
pyrpmd validate examples/surface/diffusion.inp --check-pes
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
pyrpmd resume runs/surface-diffusion

# 迁移坐标与高度的二维自由能密度。
pyrpmd run examples/surface/fes2d.inp --output runs/surface-fes2d
pyrpmd analyze runs/surface-fes2d --recompute --bins 24 12 --bootstrap 40

# ASE 构建 Cu(111)+Cu adatom，使用真实 ASE EMT calculator。
pyrpmd validate examples/surface/ase_emt.py --check-pes
pyrpmd run examples/surface/ase_emt.py --output runs/surface-ase
```

`toy_surface.py` 是可解析的周期起伏势，不是氢／铜的真实反应势。
其势能可分离且所有基底原子固定；越过势垒后的连续多位点运动不能用于验证
真实位点跳跃的热化、长跳统计或扩散系数。它只验证所选分割面的有限时间通量链路。
`ase_emt.py` 验证实际 ASE calculator、倾斜 Cu(111) 晶胞、固定底层和活动表层的连接，
也不代表 EMT 能描述某个目标催化反应。两个例子都没有完成物理收敛研究。

## 晶胞与固定原子

```toml
[system]
geometry = "adsorbate_slab.xyz"
geometry_unit = "angstrom"
cell = [[10.0, 0.0, 0.0], [5.0, 8.660254, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
fixed_atoms = [0, 1, 2, 3]
```

数值 `cell` 默认 **bohr**，与 XYZ 默认 angstrom 独立。建议始终写 `cell_unit`。
PBC 必须是三个布尔值；周期系统须有有限、非奇异的晶胞。示例中的两个周期方向
沿前两个晶格向量，第三方向为非周期真空。cell/PBC 的方向不是简单的笛卡尔轴开关。

`fixed_atoms` 在每颗珠子上固定于 `system.geometry`，使用有限的真实质量。
热化不生成固定动量，umbrella 初态 steering 和 RATTLE 都只使用活动自由度。
改变 cell、PBC、固定原子或参考几何会改变科学指纹，不能作为同一计算恢复。

状态 **始终使用展开坐标**。不要在时间步之间逐原子、逐珠子 wrap：环弹簧使用
同一展开图像中的相邻珠子距离，任意 wrap 会改变弹簧能量和质心。初始分子须保持
连通，参与 COM 的原子组及珠子须放在一致图像中。程序不自动修复已经包裹断裂的
分子；此版也不采样跨周期晶胞的量子绕行、粒子交换或 Bose/Fermi 置换。

## 表面 CV 的定义

```toml
[[cvs]]
name = "height"
kind = "surface_height"
atoms = [12, 13]
reference_atoms = [8, 9, 10, 11]
normal = [0.0, 0.0, 1.0]
origin = "0 angstrom"
estimator = "centroid"
```

设吸附组和参考组的质量加权 COM 分别为 R 和 Rref，单位方向为 n，
则高度为 `n·(R−Rref)−origin`。没有 `reference_atoms` 时，参考点是坐标原点。
法向量自动归一化；只有两个周期方向且未给 `normal` 时，取相应晶格向量叉积。
法向是固定方向，不是随表面起伏变化的局部拟合法向。

`surface_lateral` 使用相同线性形式，将 `normal` 改为 `direction`。未指定方向且
恰有两个周期方向时，沿第一个周期晶格向量；其余情况沿 x。它跟踪展开路径，不把相邻吸附位点
折叠回同一坐标。用户应选择实际的反应路径；方向与表面相容由输入负责。

`distance`、`bond_difference`、`angle` 等已有 CV 保留原展开几何语义。
跨周期边界的局域键长请显式用：

```toml
[[cvs]]
name = "bond"
kind = "periodic_distance"
atoms = [12, 8]
estimator = "centroid"
```

最近像选择在 Wigner–Seitz 边界存在分支；在像切换点导数不唯一，不能把这些
位置用作光滑约束。`periodic_distance`、非线性键差和自定义 CV 可以用于 FES，
当前表面速率只接受内置 `surface_height`、`surface_lateral` 或 `cartesian`
的线性 **centroid** 形式。非线性表面反应坐标的质量度量／约束测度尚需另行实现。

## 表面速率的物理归一化

表面反应物与基底耦合，不能套用气相双分子的远区球面面积与相对平动前因子。
本实现使用单分割面的反应物盆地归一化 [1]。对线性坐标
`s = Σ_i a_i·q̄_i + c`，仅包含活动原子的常数质量度量为：

\[
g=\sum_{i\in\mathrm{mobile}}\frac{|\mathbf a_i|^2}{m_i},\qquad
\sigma_{\dot s}^2=\frac{g}{\beta}.
\]

固定参考组的梯度不贡献 g；活动参考组的负梯度贡献必须计入。环聚合物按
`exp(-β H_P/P)` 采样，单珠动量方差为 `mP/β`，质心速度方差仍是 `g/β`，
因此公式中没有额外 bead 数因子。

定义位置密度自由能 `F(s) = −β⁻¹ log ρ(s) + const`，则

\[
k_{\mathrm{QTST}}=
\sqrt{\frac{g}{2\pi\beta}}
\frac{\exp[-\beta F(s^\ddagger)]}
{\int_{s\in A}\exp[-\beta F(s)]\,ds},\qquad
k_{\mathrm{RPMD}}=k_{\mathrm{QTST}}\,\kappa.
\]

这要求 `analysis.measure="cv_density"`，不能使用气相 rate 的 `rate_flux` PMF。
算法在相邻受支持 bin 中心之间线性插值 log 密度，对其指数逐段精确积分；
盆地内部、盆地到分割面的任何缺样本区间都会阻止速率报告。不会跨空洞插值或
在直方图范围之外外推。改变 bin 数和采样范围仍需要收敛检查。

```toml
task = "surface_rate"

[surface]
reactant_basin = ["-3 bohr", "3 bohr"]
dividing_surface = "3 bohr"
product_side = "higher"
```

`higher` 指 `s>s‡` 为产物侧，`lower` 指 `s<s‡`。盆地必须完全位于相反侧；
其边界可与分割面相同。`[surface]` 中纯数值总按 bohr，不继承 CV 的显示单位。
umbrella 中心须覆盖完整盆地到分割面，实际样本还须覆盖相应插值点。

盆地边界不能仅为数值方便而截断一部分反应物种群，否则分母及速率都改变。
需要确认边界足以涵盖目标亚稳态，或明确这是对指定有限区间条件化的速率。
该入口不自动寻找过渡态，也不自动寻找盆地。

### 回穿与不确定度

约束母轨迹使用 mobile-only 线性梯度的 Cayley–RATTLE。线性约束的质量度量
为常数，约束测度与目标 delta-coordinate 测度只差一个会消去的常数。
每组子轨迹重新抽取**未约束** Maxwell 动量，再分别传播 `p` 与 `−p`。
所有子轨迹均保留初始法向速度权重：

\[
\kappa(t)=\frac{\sum_j v_j\,[h(s_j^+(t)-s^\ddagger)-h(s_j^-(t)-s^\ddagger)]}
{\sum_j |v_j|}.
\]

`product_side="lower"` 时使用反向的有符号分割面。禁止把子轨迹动量投影回
切平面，也不能用无权重产物比例替代 κ。所有 child pairs 先按母构型分组，
再对母链及链内相关块 bootstrap。PMF 每个 bootstrap draw 同时重算分割面
密度和盆地积分；最终联合区间组合独立 PMF 与回穿抽样分布。
这个独立性假定要求约束母轨迹的初始热化足以消除它与 umbrella 起始末态的相关性，
程序不会自动证明这一点。若 PMF bootstrap 有缺支撑／失败 draw，主联合误差与
区间留空；`supported_draw_rate_*` 只报告条件于有效支撑 draw 的分布，不能把它
解释为完整的 95% 联合置信区间。

在周期迁移中，应寻找微观回穿衰减后、重复跳跃／有限盒返回前的时间平台。
`h(s>s‡)` 表示所选出口方向，不是全局化学产物分类器；它不自动追踪其他出口。
示例中左右两个出口只统计右侧。位点简并、二维双分子面积、覆盖度、压力和
微观动力学 TOF 需要明确的额外物理模型，程序不会隐式补上。

## 输出与恢复

- `surface_pmf.csv/.npz`：位置密度 PMF 和 mask；NPZ 包含 QTST bootstrap draws。
- `surface_pmf.json`：盆地、质量度量、归一化、样本支持及逐 draw 失败原因。
- `recrossing.csv/.npz`：κ(t)、母构型通量、误差及 bootstrap。
- `results/summary.json`：`rate_estimate_s_inv` 为有限时间估计；通过平台与能量
  检查时 `rate_s_inv` 才有值。`converged` 仍为 false，必须另做物理收敛研究。
- `umbrella/` 与 `checkpoints/`：窗口样本、约束母轨迹和已完成子轨迹结果。

`pyrpmd resume` 支持同一物理设置下的续跑，已完成结果保持一致。现有 FES 可
`analyze --recompute`；表面速率目前只能读取已存结果，不提供整条速率管线的
独立离线重分析入口。未完成的 child pair 在重启后按稳定 seed 重算；其轨迹中途
状态不做增量 checkpoint。母轨迹和窗口则从保存的 q/p 与 RNG 继续。

## 周期 PES 契约和性能

ASE 接口会设置 `Atoms.cell` 和 `Atoms.pbc`，坐标仍按 bead batch 逐帧送入
calculator。固定晶胞在同一 run 中不变；此版不请求应力、virial 或压强。

自定义周期 Python PES 必须显式声明并绑定系统：

```python
class SurfacePES:
    metadata = {"supports_pbc": True}

    def bind_system(self, system):
        self.cell_bohr = system.cell.copy()
        self.pbc = tuple(system.pbc)

    def evaluate(self, q):
        # Use the supplied periodic cell; return physical energies and forces.
        ...
```

`bind_system` 始终接收原子单位。若 `evaluate` 配置为 angstrom/eV，插件应自行把
保存的 cell 转为它内部使用的单位。不要只声明 `supports_pbc` 却忽略晶胞。
当前 C ABI v1、原 RPMDrate/f2py、Torch 与 i-PI socket 工厂未支持本周期契约，
请求周期系统时明确报错。可通过符合契约的 Python/ASE wrapper 接入适合的实际
机器学习或电子结构 calculator，但每个模型仍须做能量／力和周期边界验证。

固定原子及周期 PES 使用 Python 时间步循环，保留已编译 Cayley 微内核；
`loop_backend="auto"` 选择兼容路径，固定原子路径会记录回退原因；强制原生融合循环不受支持。
现有过程级并行可用，不声称完整 GPU 常驻传播或新增周期原生 ABI。

## 与 PyRPMD 0.2 工作流的组合

固定原子约束同时覆盖 `cayley-bcb`、`normal-mode-bab` 和 `velocity-verlet`
实时积分器；独立 `Dynamics.sample` 接口中的 BCOCB/BAOAB 采样也保持基底约束。
表面 FES 和速率工作流采用与所选积分器对应的 Andersen 采样。

`task="fes"` 可通过已有的 adaptive umbrella 流程进行试探和冻结后的生产采样；
`surface_rate` 当前要求预先声明固定窗口集合，不接受 adaptive umbrella。
新增的 `metadynamics` 工作流尚未实现本表面约束契约，带周期边界或固定原子的
输入会明确报错。`network` 保持气相双分子归一化，也拒绝此类系统；它不能用来
汇总表面出口。单个表面出口请使用 `surface_rate`。

## 后续可扩展项

当前交付不包含气体／表面入射初态与粘附概率、非线性通用表面速率、二维
双分子浓度归一化、NPT/变胞、电子摩擦、非绝热跃迁、电极恒电势或微观动力学。
这些需要各自的统计定义和验证。现有周期 System、固定约束和 PES 契约为其提供基础。

## 参考

1. C. Li, L. Zhang, B. Jiang, H. Guo, *Unified ring polymer molecular dynamics rate
   calculations for reactions with separable and non-separable reactants*,
   J. Chem. Phys. **163**, 024105 (2025), [doi:10.1063/5.0273941](https://doi.org/10.1063/5.0273941),
   [作者预印本](https://arxiv.org/abs/2503.22964)。此处仅实现恒定质量度量的线性坐标子集。
2. [ASE Atoms cell/PBC 文档](https://docs.ase-lib.org/ase/atoms.html)；
   [ASE surface builders](https://docs.ase-lib.org/ase/build/surface.html)。
