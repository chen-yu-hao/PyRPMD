# 表面催化：固定晶胞、表面自由能与一阶 RPMD 速率

PyRPMD 的表面入口复用环聚合物传播、umbrella/MBAR、受约束母轨迹和回穿统计。
当前版本可以计算周期 slab 或有限表面簇上的 1D/2D 伞抽样及 1D–3D MetaD 自由能，并针对**一个恒定质量度量的
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
| 表面 MetaD | `task="metadynamics"` | 1D–3D，周期 slab、固定底层、NHC/PILE-L/Andersen、二次墙和精确恢复 |
| 非线性键长 | `distance` | 连续展开的质心键长；真实曲面 RATTLE，恒定活动端点质量度量 |
| 条件速率 | `surface.free_energy` | 已保存 FES → 条件 Boltzmann 投影 → 同域母构型 → 配对回穿 |
| 表面速率 | `task="surface_rate"` | 位置密度 PMF → 盆地 QTST → 约束母轨迹 → ±p 回穿 |
| 势能 | ASE、i-PI、显式支持 PBC 的 Python 插件 | 每个 worker 收到完整 System 和固定晶胞 |
| 执行与恢复 | 现有窗口／轨迹进程及珠子势能执行器 | 保存 cell、PBC、fixed atoms、q/p、RNG、科学指纹 |

固定原子仍参与真实 PES 的能量和对活动原子的作用力，只是不参与动力学自由度。
活动表层与吸附物按同一 bead 数和物理质量传播；当前不提供“仅吸附物量子化、
活动基底经典化”的混合 bead 模型。伞窗口和母轨迹使用 Andersen；MetaD 还可用
NHC 或 PILE-L；回穿子轨迹无热浴。

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

`cell_unit` 缺省继承 `geometry_unit`；未指定 geometry_unit 时，inline 坐标默认 bohr、XYZ 默认 angstrom。有 cell 时 PBC 缺省全周期。建议始终显式写 `cell_unit` 和 `pbc`。
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
表面速率接受内置 `surface_height`、`surface_lateral`、`cartesian`，以及连续展开的
`distance` 或满足下述光滑性条件的 `periodic_distance`，均要求 **centroid** 形式。距离分割面采用实际曲面上的解析梯度和 Hessian，
不在线性切面上近似传播。`periodic_distance` 也可用于速率，但分割面必须小于周期晶格基矩阵最小奇异值的一半；这个保守界保证分割面不触碰最近像切换边界。程序会拒绝不满足条件的输入，任意变质量度量的非线性 CV 仍不适用该公式。

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

对键长 `s=|q̄_a−q̄_b|`，质量度量为两个**活动**端点的 `1/m_a+1/m_b`；
固定端点不贡献。它虽然是非线性坐标，度量仍为常数，因此现有约束采样测度可直接使用。
RATTLE 每一步重新计算曲面的法向，固定端点的位置、动量和导数保持约束。
键长盆地和分割面必须严格大于零；重合端点的导数没有定义。

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
# Reporting target for the joint relative standard error; default 0.2.
max_relative_standard_error = 0.2
```

`higher` 指 `s>s‡` 为产物侧，`lower` 指 `s<s‡`。盆地必须完全位于相反侧；
其边界可与分割面相同。`[surface]` 中纯数值总按 bohr，不继承 CV 的显示单位。
umbrella 中心须覆盖完整盆地到分割面，实际样本还须覆盖相应插值点。

盆地边界不能仅为数值方便而截断一部分反应物种群，否则分母及速率都改变。
需要确认边界足以涵盖目标亚稳态，或明确这是对指定有限区间条件化的速率。
该入口不自动寻找过渡态，也不自动寻找盆地。

### 回穿与不确定度

约束母轨迹使用仅包含活动原子梯度的 Cayley–RATTLE。支持坐标的质量度量
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
程序不会自动证明这一点。若 PMF bootstrap 有缺支撑／失败 draw，主 QTST 和
联合速率的误差与区间留空；`supported_draw_qtst_*`、`supported_draw_rate_*`
只报告条件于有效支撑 draw 的分布，不能把它解释为完整的 QTST 或联合速率
95% 置信区间。

PMF 相关块长度采用与一般 FES、气相 PMF 相同的策略：请求 `ceil(5*g)`，
在每条流内按可用样本数限制，并在 `bootstrap.block_plan` 中报告块长度、
状态与警告。流过短或没有可解析的独立变化时，状态为 `unavailable`，
QTST 和主联合速率的标准误与置信区间留空；点估计、逐 draw 记录以及
密度–盆地积分的配对协方差诊断仍可保留，但不代表误差已经可用。
`limited_blocks` 表示只能提供受有限块数限制的估计。当前统一的 `ceil(5*g)`
策略与早期表面 PMF 的 `ceil(g)` 会给出不同 bootstrap draw 和误差；本次报告
门槛修复沿用当前策略，不改变其抽样序列或 PMF/QTST 点估计。

回穿统计与气相及 site-rate 共用母构型/相关块实现。单母构型不能通过增加
bootstrap draw 数量获得误差支持；零通量母构型也不增加有效样本数。
`rate_uncertainty.recrossing_scope` 区分独立母链与
`conditional_on_single_chain_mixing`。后者至少需要两个有效相关块，且条件于
单链相关时间估计足以描述混合；surface-rate 保留允许该条件误差的既有政策，
不会把它解释为独立母链之间已经一致，且此时不填充 `rate_s_inv`。
零通量 draw 被拒绝时主误差留空。

surface-rate 复用前置 FES 的 MBAR 点估计权重，避免相同数据的第二次求解。
复用前校验有序 reduced-potential 矩阵、stream counts 和 overlap threshold；
直方图、支持 mask、盆地积分仍按 surface-rate 自己的 bins/min_count 重建，
bootstrap 也仍逐次重新求解。不将不同 binning 的 FES 直接当作速率 PMF。

在周期迁移中，应寻找微观回穿衰减后、重复跳跃／有限盒返回前的时间平台。
`h(s>s‡)` 表示所选出口方向，不是全局化学产物分类器；它不自动追踪其他出口。
示例中左右两个出口只统计右侧。位点简并、二维双分子面积、覆盖度、压力和
微观动力学 TOF 需要明确的额外物理模型，程序不会隐式补上。

### 平台与速率报告门槛

`plateau_diagnostics` 明确给出选定尾段的失败原因。共享的气相、site-rate 和
surface-rate 判据同时要求：尾段至少 10 帧、正的 κ、可用的母构型相关块误差，
以及以下两项均小于 `recrossing.plateau_tolerance * abs(kappa)`：

1. 尾段两半均值差的绝对值，加两倍该差的 bootstrap 标准误；
2. 尾段相对其均值的最大绝对偏离，加中心化 bootstrap 误差的 95% 最大偏离半径。

每个 bootstrap 曲线先减去自己的尾段均值，因此 κ 高度的不确定度不会被误当成
尾段形状的不确定度。半径由整条相关尾段的最大偏离计算，未把相邻时间帧当成
独立样本。这是有限样本诊断，条件于母链 bootstrap 能描述混合；仍需改变子轨迹
时长和尾段来排除延迟回穿。关闭 bootstrap 时保留 κ 点估计，但不认证平台。
误差很大不会放宽允许漂移，两个半段均值相等也不会掩盖持续振荡。

surface-rate 另外在 `rate_quality` 中要求完整 PMF 区间、
`pmf.bootstrap.block_plan.status="finite_block_estimate"`、至少两条非零通量
独立母链和 20 个非零通量母构型、完整的联合标准误，以及该标准误／正速率
不超过 `surface.max_relative_standard_error`（默认 0.2，必须为有限正数）。
还必须通过平台与每珠子 0.1 kBT 的端点能量误差检查。

这些是报告前提，不是物理收敛证明；`converged` 仍为 false。前提未通过时
`rate_s_inv=null`、`status="pilot_requires_convergence"`，
`rate_estimate_s_inv` 和可用的有限样本误差保留。`rate_quality.reasons` 可用于
决定继续增加采样还是检查积分器/回穿。旧版只根据平台和端点能量填充
`rate_s_inv`，下游脚本应使用新质量字段，并明确区分有限时间估计。

提高有效采样应优先于增加 bootstrap 次数：逐窗检查相关块、独立 replica 与
时间前后半段的一致性，比较 Andersen 间隔的有效样本／计算成本，必要时先在
FES 入口进行自适应 pilot、冻结窗口后再独立生产。随后分别收敛 bins、dt、P、
slab、盆地和回穿时长。教学势的独立有限 P QTST 基准见
[表面精度 benchmark](../benchmarks/surface/README.md)。

## 输出与恢复

- `surface_pmf.csv/.npz`：位置密度 PMF 和 mask；NPZ 包含 QTST bootstrap draws
  以及逐样本 MBAR `weights`。
- `surface_pmf.json`：盆地、质量度量、归一化、样本支持及逐 draw 失败原因；
  `format_version=2`，不再内嵌逐样本权重数组。
- `recrossing.csv/.npz`：κ(t)、母构型通量、误差及 bootstrap。
- `results/summary.json`：`rate_estimate_s_inv` 为有限时间估计；通过上述
  `rate_quality` 门槛时 `rate_s_inv` 才有值。`plateau_diagnostics` 给出尾段检验。
  `converged` 仍为 false，必须另做物理收敛研究。
- `umbrella/` 与 `checkpoints/`：窗口样本、约束母轨迹和已完成子轨迹结果。

**权重读取迁移：** 旧脚本中的 `pmf["weights"]` 应改为
`np.load(pmf["sample_weights_file"])[pmf["sample_weights_key"]]`，当前键名为
`"weights"`。最终 summary 中嵌套的 PMF 也使用相同版本。该变更减少随样本数
增长的重复 JSON 序列化；低层 `surface_rate_pmf()` 数组 API 仍返回 `weights`。

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
同时兼容材料接口的 `metadata["periodic"]=True` 与 `evaluate(q, *, cell, pbc)`；
后者的坐标和晶胞均为插件声明的长度单位，详见[材料计算](materials.md)。
材料 i-PI adapter 支持真实 cell/PBC 和物理 virial；部分周期必须配置匹配的
`potential.params.init.pbc` 并由 driver 解释。C ABI v1、原 RPMDrate/f2py 与 Torch
仍不支持周期契约，请求周期系统时明确报错。可通过符合契约的 Python/ASE wrapper 接入适合的实际
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
`metadynamics` 支持 1–3 个 CV、周期晶胞、固定底层、二次软墙，以及 Andersen、
Nosé–Hoover 链和 PILE-L 采样。其 checkpoint 同时保存热浴和偏置历史，
详见[增强采样](enhanced-sampling.md)及[真实 Pt(111) 示例](../examples/surface/pt111/README.md)。
`network` 保持气相双分子归一化，也拒绝此类系统；它不能用来
汇总表面出口。单个表面出口请使用 `surface_rate`。

## 面积、覆盖度和路径数

`rate_estimate_s_inv` 始终是原始有限晶胞中、每个指定盆地反应物的速率。
需要二维双分子系数时，显式声明参考晶胞中的共反应物数目：

```toml
[surface.normalization]
reaction_order = 2
reference_adsorbates = 1
site_count = 9
coverage = 0.04
path_multiplicity = 1
# area = "62.318 angstrom^2"  # 省略时用周期 cell[0] × cell[1] 的面积
```

令面积为 A、参考共反应物数为 n，则 `k2=k_cell*A/n`，单位 `cm²/s`。
指定 θ ML 和 N 个表面位点后，`k(θ)=k2*θ*N/A`，单位 `s⁻¹`。
例如 n=1、N=9、θ=0.04 时，覆盖度换算因子为 0.36。所有换算结果与
标准误、置信区间一起保存在 `normalized_rates` 中，不覆盖原始速率。
默认一阶、路径数为 1；已在 FES 采样中包含的等价通道不能再次乘简并度。
换算采用稀释、等价共反应物假设，不能修正有限覆盖度下 PES/自由能的变化。
若基准速率未通过支持或平台检查，换算后的正式速率仍保留为空。

## 从三维自由能接入回穿

除重新进行一维伞抽样，也可以读取已有 `fes` 或 `metadynamics` 的位置密度 FES：

```toml
[surface]
reactant_basin = ["2.05 angstrom", "5.8 angstrom"]
dividing_surface = "1.98 angstrom"
product_side = "lower"

[surface.free_energy]
directory = "../runs/pt111-production"
axis = 1                          # 三维 CV 的第二个分量，零起始
# selection = "co_channel.npy"   # 可选：与完整三维网格同形的布尔积分域

[analysis]
method = "source"
bootstrap = 0
```

这个模式不再声明 `[umbrella]`，CV 从源输入读取；`system`、`potential`、温度和
珠子数仍需在当前任务声明并与源计算一致。活动原子的初始位置可更换；固定原子
位置、质量、晶胞和 PBC 必须一致。源输入及模型哈希会检查；源 FES、配置、
selection 也进入当前运行的 manifest，改变网格后应开始新的速率任务。

程序按源网格和 selection 对 Boltzmann 密度积分，任何盆地到分割面的未支持区间
都会阻止子轨迹启动。母链进行真实曲面 RATTLE 和 Andersen 采样，仅保留落在
相同积分域内的构型。`parent_samples` 是每条链的**尝试抽样数**，接受数在
`source_parent_sampling` 中；若全被拒绝，需要改善初态、平衡、通道支持或采样长度。
子轨迹使用新抽取的非约束 Maxwell ±p，并在源静态墙势下进行 NVE 传播；
MetaD hills 和伞势均不进入母／子传播。墙很远也必须检查它是否影响有限时间平台。

这保持 PMF、条件母系综及哈密顿量的一致性，定义的是选定分割面的通量侧相关函数，
不是多个化学出口的自动总和。通道边界及网格分辨率也需要收敛检查。
源 FES 的完整统计误差目前不随投影传播，因此只给 `rate_estimate_s_inv` 和
条件回穿误差，正式 `rate_s_inv` 保持空值，状态为 `projected_pmf_pilot`；
有偏 PMF 不能仅凭一个回穿平台被标成物理收敛。

## 后续可扩展项

当前交付不包含气体／表面入射初态与粘附概率、非线性通用表面速率、
浓度变化导致的相互作用修正、NPT/变胞、电子摩擦、非绝热跃迁、电极恒电势或微观动力学。
这些需要各自的统计定义和验证。现有周期 System、固定约束和 PES 契约为其提供基础。

## 参考

1. C. Li, L. Zhang, B. Jiang, H. Guo, *Unified ring polymer molecular dynamics rate
   calculations for reactions with separable and non-separable reactants*,
   J. Chem. Phys. **163**, 024105 (2025), [doi:10.1063/5.0273941](https://doi.org/10.1063/5.0273941),
   [作者预印本](https://arxiv.org/abs/2503.22964)。此处实现恒定质量度量的线性坐标及连续键长子集。
2. [ASE Atoms cell/PBC 文档](https://docs.ase-lib.org/ase/atoms.html)；
   [ASE surface builders](https://docs.ase-lib.org/ase/build/surface.html)。
