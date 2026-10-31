# 固定晶胞中的氢扩散、PIMD 与位点跃迁

此开发分支增加三个独立任务：`pimd`、`diffusion` 和 `site_rate`。
它们使用真实原子质量、固定晶胞和连续未包装坐标；材料的相互作用仍由用户提供的
势能模型决定。`periodic_cosine` 与 `free` 是算法验证模型，不能预测特定金属或质子导体。

## 当前范围

| 任务/接口 | 已实现的含义 | 需要单独验证的内容 |
| --- | --- | --- |
| `pimd` | 固定晶胞平衡路径积分采样；物理能量估计器和环聚合物回转半径 | 平衡时间、采样误差、珠子数、步长 |
| `diffusion` | PIMD 热化后无恒温 RPMD；质心 MSD、扩散张量、VACF 积分和独立副本误差 | 长时扩散区间、有限尺寸、势模型、H/D 收敛 |
| `site_rate` | 一个光滑标量质心坐标、一个有限反应物盆地、一个有向分割面的 QTST 与回穿 | PMF 支持、窗口重叠、回穿平台、盆地定义、非线性度量 |
| 周期结构 | 正交/倾斜固定晶胞；三维体相或部分周期 slab | 各材料的晶格、表面、缺陷、浓度与电荷态 |
| 周期 PES | ASE、显式声明周期能力的 Python 插件、带真实晶胞的 i-PI socket | 每个 calculator/服务实际支持的 PBC、元素、力与应力 |

当前所有原子均可运动，使用同一个珠子数。没有 NPT/barostat、固定原子约束、
机器学习模型训练、任意全环坐标约束速率、长时 kMC 网络、集体电导率或电子摩擦模型。
光滑非线性质心坐标的 `site_rate` 已提供质量度量修正，见[通用坐标速率](general-coordinate-rates.md)。
周期 Torch/C ABI v1/f2py 入口会明确拒绝；不能因为这些入口支持分子计算就假定
它们接收晶胞。旧的气相 `rate`/`drpmd` 也拒绝周期系统。

## 结构、晶胞和坐标约定

```toml
[system]
geometry = "slab.xyz"
cell = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.0, 0.0, 25.0]]
cell_unit = "angstrom"
pbc = [true, true, false]
```

晶胞的**每一行是一个晶格矢量**，笛卡尔坐标满足 `r = s @ cell`。
三维矩阵必须有限、可逆且数值条件良好；slab 的非周期方向也要给出非零晶胞矢量。
提供晶胞而省略 `pbc` 时默认三个方向都周期；无晶胞时默认都不周期。
`pbc` 必须是三个布尔值。建议显式声明 `cell_unit`；缺省继承 `geometry_unit`，
否则内联数组为 bohr、XYZ 输入为 angstrom。内部单位统一为 bohr/hartree/原子时间。

环聚合物 `q[P,N,3]` 和保存的质心坐标始终连续，穿越晶胞后不会取模。
势能和几何 CV 自己处理周期镜像；不能在传播器中逐珠子 wrap，否则会破坏弹簧、
质心和 MSD。`minimum_image` 对倾斜晶胞求最近晶格像，不能用简单的逐分数坐标
四舍五入替代。环弹簧使用同一连续坐标覆盖，适用于可区分、局域核的闭合路径；
不采样核交换置换或非零绕胞拓扑扇区。

内置周期距离、键差和角度 CV 在构造时保存只读晶胞/PBC 快照，复用固定晶格的
正交性判断和 QR 分解。`cv.cell` 或数组形式的 `cv.pbc` 不支持原地修改；需要
重新绑定时使用 `cv.cell = new_cell`、`cv.pbc = new_flags`，这会重新验证并更新缓存。
旧的 `pyrpmd.cv_periodic.PeriodicDistance` 导入路径仍可用，现在与
`pyrpmd.cv.PeriodicDistance` 指向同一个实现。普通值/梯度计算不构造稠密 Hessian；
显式调用 Hessian 接口仍返回原来的 `[N,3,N,3]` 数组。

H/D/T 共用电子势能上的氢元素身份，但动力学质量不同。可写 `symbols=["D"]`，
也可用 `isotopes=[{atom=0, mass="2.01410177812 amu"}]`；不需要为氘训练另一套
Born–Oppenheimer 势。缺少内置精确质量的元素须显式提供 `masses_amu`。

### 周期势能契约

ASE adapter 向 calculator 传入真实 angstrom 晶胞、PBC 和每个珠子的坐标。
若 calculator 声明 `stress`，返回的可选物理构型 virial 为 `-volume * stress`，
转换为 hartree。`PESResult.virial[B,3,3]` 不包含动量或环弹簧贡献，也不自动
构成量子压力估计器。进程/MPI 珠子分块保留 virial；不同块能力不一致会报错。

Python 周期模型可声明 `metadata["periodic"] = True` 并实现
`evaluate(q, *, cell, pbc)`；坐标和晶胞均采用插件声明的长度单位。另一种
显式契约是 `metadata["supports_pbc"] = True` 与 `bind_system(system)`，
后者总是接收原子单位 System，然后由插件保存晶胞并在 `evaluate(q)` 中使用。
直接传入的已绑定 provider 也必须显式声明周期能力且 cell/PBC 与系统一致。
自定义 CV 的 `state` 包含只读晶胞、PBC 与 fixed_atoms。

i-PI adapter 在 POSDATA 传真实晶胞和逆矩阵，使用 i-PI 的列向量约定。
POSDATA 本身不携带部分 PBC 标志，因此 slab 必须显式提供匹配的
`potential.params.init.pbc`，并由服务端实现此 INIT 契约；仅传一个大真空层
不等于服务端理解了部分周期边界。

## PIMD：平衡采样

```bash
pyrpmd validate examples/materials/pimd.inp --check-pes
pyrpmd run examples/materials/pimd.inp --output runs/materials-pimd
```

`sampling.equilibration` 是平衡期，`sampling.production` 是平衡期后的采样期，
`sampling.interval` 是保存间隔，`sampling.replicas` 控制独立随机流。

| 确定性积分器 | PILE-L 采样分裂 | 另一可选采样方式 |
| --- | --- | --- |
| `cayley-bcb` | `sampling.integrator="bcocb"` | `cayley-bcb-andersen` |
| `normal-mode-bab` | `sampling.integrator="baoab"` | `normal-mode-bab-andersen` |
| `velocity-verlet` | 不支持 | `velocity-verlet-andersen` |

前两者默认 PILE-L，Verlet 默认 Andersen；显式名称必须匹配。
`gamma_centroid="0.01 fs^-1"` 设定 PILE-L 质心摩擦，内部模使用正常模热化。
PILE-L 的 `auto` 路径对 `P<=8` 或二次幂珠数使用 C++ `sample_free`，融合
自由环半步、OU 热浴与第二个半步；Python 保留两次势能力调用和原 NumPy
随机数序列。较大非二次幂珠数保留 NumPy 路径。它仍不是完整原生长块循环，
因此拒绝强制 `loop_backend="native"`；`native=false` 或 `loop_backend="python"`
可选择参考采样路径。固定原子保持精确固定。
材料任务暂要求采样和生产步长一致。

采用 sum-beads Hamiltonian、逆温度 `beta/P` 约定，输出：

\[
\langle V\rangle_P=\frac1P\sum_b V(q_b),\qquad
K_{\mathrm{prim}}=\frac{3NP}{2\beta}-\frac{H_{\mathrm{spring}}}{P},
\]

\[
K_{\mathrm{cv}}=\frac{3N}{2\beta}
-\frac1{2P}\sum_{bi}(q_{bi}-\bar q_i)\cdot F_{bi},\qquad
E_{\mathrm{cv}}=\langle V\rangle_P+K_{\mathrm{cv}}.
\]

物理内能不使用虚拟环动量的瞬时动能。`ring_polymer_hamiltonian` 单独用于
传播诊断。独立副本均值之间的标准误不包含珠子数或步长系统误差；PIMD
恒温采样时间本身不作为材料的真实动力学时间。[1,2]

`sampling_diagnostics` 另外逐副本、逐观测量报告保存序列的统计低效因子、
有效样本数和相关块长度。至少四个完整块时才提供 `conditional_standard_error`；
它条件于估计块长足以覆盖相关时间，不替代独立副本误差或混合性验证。原有
`estimators` 均值和副本间标准误保持原义；不能把保存帧数当成独立样本数。

## 扩散：无恒温 RPMD 与保存后再分析

`task="diffusion"` 在热化末端保留 `t=0`，随后使用真实原子质量进行
**无恒温器、无 umbrella/metadynamics 偏置**的生产传播。保存所有原子的
连续环质心位置、质心速度、物理能量估计器与回转半径。

```toml
[sampling]
thermostat = "pile-l"
equilibration = "2 ps"
production = "20 ps"
interval = "2 fs"
replicas = 4

[diffusion]
atom_indices = [27]
axes = [0, 1]
reference_atoms = [0, 1, 2, 3]  # 实际应选定合适的完整晶格参考组
max_lag = 2500
fit_range = ["2 ps", "5 ps"]
bootstrap = 200
```

上面的参数只是语法示例；每个材料需要确定自己的时间尺度。未提供
`atom_indices` 时选取所有 H/D/T；没有氢且未显式选择时会报错。
`axes=[0,1]` 分析 xy 面内，`[0,1,2]` 分析三维。任意倾斜表面应先将
模型旋转到所需笛卡尔分析坐标系，目前 `axes` 不表示晶格分数轴。

`reference_atoms` 在每一帧扣除指定原子组的质量加权 COM 位置/速度。
它用于定义随晶格运动的参考系，不会把每个扩散原子的拟合漂移单独消掉；
参考组与被测原子不允许重叠。未指定时报告实验室参考系结果。

对选定的 `d` 个分量计算所有时间起点的位移张量：

\[
M_{\alpha\beta}(t)=\langle\Delta\bar r_\alpha(t)
\Delta\bar r_\beta(t)\rangle,\quad
D_{\alpha\beta}=\frac12\frac{dM_{\alpha\beta}}{dt},\quad
D=\frac{\operatorname{tr}D_{\alpha\beta}}d.
\]

因此 xy 面的标量扩散是 MSD 斜率除以 4，三维除以 6。
同时保存对称速度自相关张量及其有限时间积分以供交叉检查；不自动认定
VACF 积分已达平台。质心 Einstein/RPMD 关系参考 [1,3]。

`max_lag` 的单位是**保存帧间隔数**，缺省为记录跨度的四分之一。
拟合区间必须在 `max_lag * sampling.interval` 内；缺省取有效 lag 的后半段。
Python API 允许 `fit_range=(None, stop)` 或 `(start, None)`。
拟合至少需要四个正 lag 点；增加重叠时间起点不增加独立误差样本数。

误差默认在完整独立副本间 bootstrap，不能将不同 H 原子或重叠起点
当作独立实验。只有一个副本时可显式给出 `block_length`（保存帧数），
要求至少 `4*max_lag`，且至少四块才给条件性的分块误差；块长是否超过
物理相关时间仍需检查。曲线拟合的 OLS 残差不作为扩散采样误差。

| `analysis_status` | 解释 |
| --- | --- |
| `insufficient_data` | 帧数、时间起点或拟合点不足 |
| `no_observed_motion` | 没有可分辨位移；不表示已确定 `D=0` |
| `ballistic_or_drifting` | MSD 增长过快，可能处于弹道区或存在整体漂移 |
| `non_diffusive_window` | 拟合区间不符合近似线性扩散增长 |
| `indefinite_diffusion_tensor` | 拟合张量不是半正定，采样/拟合尚不可靠 |
| `unstable_fit_window` | 拟合区间前后两半斜率不稳定 |
| `diffusive_regime_candidate` | 通过有限记录筛选，仍未认证物理收敛 |

候选筛选要求至少 16 帧、最大 lag 至少 8 个起点、log-MSD 斜率在
`[0.7,1.3]`、线性拟合 `R²>=0.8`、半正定张量，以及可估计时两半斜率差
不超过拟合 D 的 75%。这些是公开的启发式诊断。非候选状态保留 `raw_fit`、
MSD、VACF 和副本数据，但顶层 `diffusion_m2_s`/tensor 为 `null`。
所有状态均保留 `physical_converged=false`；模型准确性和参数收敛不能由
一次线性拟合自动证明。

扩散工作流还提供 `nve_energy_diagnostics`，以首个生产保存帧为参考，计算
`(H_P(t)-H_P(0))/(P*kBT)` 的最大绝对误差、终点误差和每 fs 的线性漂移。
默认诊断阈值为每珠子 `0.1 kBT`，超过时添加 warning。该诊断只覆盖已保存帧，
可能漏掉帧间偏离；不适用于恒温 PIMD，也不自动改变 MSD 的候选状态或认定
步长已经收敛。能量零点的常数平移不影响这个差值诊断。

```bash
# 解析自由粒子负对照：应报告弹道运动，并不提供有效 D。
pyrpmd run examples/materials/diffusion.inp --output runs/materials-free
pyrpmd resume runs/materials-free
pyrpmd analyze runs/materials-free --recompute --axes 0 1 \
  --fit-range '10 fs' '20 fs' --max-lag 40 --bootstrap 100
```

再分析读取经过完整性校验的保存 checkpoint，不调用 PES：

```python
from pyrpmd import analyze_run, analyze_diffusion

report = analyze_run("runs/materials-real", recompute=True,
                     diffusion={"atom_indices": [27], "reference_atoms": list(range(27)),
                                "axes": [0, 1], "fit_range": ["1 ps", "4 ps"]})
# 已有自己的连续质心轨迹时，也可调用独立数组 API：
# report = analyze_diffusion(positions, times, velocities=velocities, axes=(0, 1))
```

数组 API 接受 `[replica,frame,atom,3]` 或单副本 `[frame,atom,3]`；所有输入
采用原子单位，输出同时给出 bohr²/原子时间和 m²/s。

## 位点速率：盆地归一化与回穿

`site_rate` 不使用气相平移配分函数、远区距离或双分子 cm³/s 前因子。
对恒定质量度量的线性质心坐标 `s`，计算

\[
k_{\mathrm{QTST}}=
\sqrt{\frac{g^TM^{-1}g}{2\pi\beta}}\,
\frac{\rho(s^\ddagger)}{\int_A\rho(s)\,ds},\qquad
k(t)=k_{\mathrm{QTST}}\,\kappa(t).
\]

其中 `g` 是 `s` 对原子质心的梯度，`A=[lo,hi]` 是显式反应物盆地。
分子和分母使用同一 MBAR 密度，任意归一化常数相消。结果单位为 s⁻¹。
当前密度是有限 bin 估计，需要 bin/采样收敛；整个盆地和分割面必须有
连通且未 mask 的支持，不能跨未采样区积分。[3,4]

支持 `cartesian`、`projection` 和连续 `fractional` 的 centroid 坐标。
`projection` 支持质量加权晶格参考组；`fractional` 使用 `r @ inv(cell)`
的一个分量。fractional 坐标**不取模**，没有周期 bias period，确保能够
区分相邻单胞和一次有向跳跃。最小镜像 `periodic_distance`、`periodic_bond_difference`、
`periodic_angle` 可以用于 FES，但不是本版本 site-rate 的恒定梯度约束。
原 `distance`、`bond_difference`、`angle` 始终使用展开几何，不会因 System 有晶胞而自动切换。

```toml
[[cvs]]
name = "site_x"
kind = "fractional"
atom = 27
axis = "x"
estimator = "centroid"
reference_atoms = [0, 1, 2, 3]

[site_rate]
cv = "site_x"
basin = [-0.5, 0.5]
dividing_surface = 0.5
direction = 1
```

`direction=1` 时分割面必须是盆地上端点，`-1` 时是下端点。
实际窗口/分析范围要覆盖完整盆地并延伸到分割面另一侧。
位点任务的窗口/约束母链使用 Andersen；回穿子轨迹成对使用 `+p`、`-p`，
无恒温传播，并按有符号初始通量统计。负的有限样本 κ 不会静默裁成零。
输出完整 κ(t)、有限时间/平台诊断；没有稳定平台时不声明物理速率收敛。
这是所选边界的有向通道，product 指示函数是其未包装坐标半空间；盆地的另一
出口可能竞争，须识别亚稳态的中间时间平台。结果不自动代表总盆地逃逸率。
没有将一个局部跳跃率自动换算成宏观 D，因为跳距、邻居简并度和连续
跳跃相关性仍需明确。[4,5]

回穿误差以母构型和母链为统计单位。增加同一母构型的子轨迹或 bootstrap draw
数量不能补足独立构型支持；只有一个有效母构型时，误差与联合区间为 `null`，
而非零误差。`site_rate` 的主联合误差要求至少两条独立的非零通量母链；单链
具备至少两个有效相关块时，只提供明确标记为
`conditional_on_single_chain_mixing` 的条件误差。零通量母构型不增加有效支持，
出现零通量 bootstrap draw 时主误差也留空，原因记录在 `rate_uncertainty`。

site PMF bootstrap 复用同一偏置矩阵并重抽样其列，估计器和随机流不变。
`pmf_uncertainty` 中的 `attempted`、`draws`、`failures`、`failure_counts` 记录
每次重采样是否得到有效 QTST，以及 MBAR 不收敛、重叠断开或盆地支持不足的原因。

```bash
pyrpmd prepare examples/materials/site_rate.inp --pilot
pyrpmd run examples/materials/site_rate.inp --output runs/materials-site
```

氧化物质子导体可以在合适周期 PES 上做 PIMD、直接 RPMD 和几何 FES，
但 O–H 重取向、质子转移与掺杂捕获常需要多坐标机制分析；本版本不把
任意质子转移 CV 视作已经实现了完整约束速率，也不从 self-D 自动生成
集体电导率或 Haven 比。[6]

## 保存、并行与验证

每个副本单独保存 `checkpoints/materials_XXXX.npz`，原子提交 q/p、RNG、
阶段时钟、采样时钟与全部观测数据。`samples/materials_XXXX.npz` 是可重建的
导出数据；物理状态以上述 checkpoint 为准。完整性校验、科学指纹和原子
替换沿用 checkpoint v2 容器，内部类型为 `PyRPMD.materials.v1`。
长轨迹当前将全部保存帧保留在内存/checkpoint，尚无分块流式存储后端。

serial/process/MPI 使用相同稳定副本 ID。允许改变执行分块/进程布局续跑；
温度、质量、晶胞、势模型和科学时间参数不能静默改变。分析参数可离线
调整，生产轨迹不因此重新计算。单步传播或观测失败时恢复至上一次已提交状态。

材料传播在每个单步事务内复用相同坐标的 PES 结果，供力、能量和保存帧观测量
共同使用。缓存拥有坐标及返回数组的副本，失败或事务结束即清空，不跨步或续跑
复用。Python 确定性传播且该步保存观测量时，原来的五次 provider 调用减少为
两次；PILE-L 对应步骤由三次减为两次。原生传播仍使用 provider 原有绑定。
轨迹导出统一在分析阶段执行一次，checkpoint 格式和逐步提交/回滚语义保持不变。
这些改动没有解决上述全历史 checkpoint 的长轨迹存储开销。

实际材料验证见 [`benchmarks/materials`](../benchmarks/materials/README.md)：
使用可核对来源和 SHA256 的 Ni–Al–H EAM，运行 Ni(100) slab 上 H/D 的
短 P=1/P>1 pilot。该检查覆盖电子势/质量分离、力、PBC、能量和完整工作流；
短 pilot 不复现文献扩散系数，也不表示势模型已验证适用于所有材料。

```bash
python -m pytest -q tests/test_periodic_materials.py tests/test_materials_input_cv.py \
  tests/test_materials_workflows.py tests/test_diffusion_analysis.py tests/test_site_rate.py
mpiexec -n 4 python tests/mpi_materials_smoke.py /fresh/shared/output
```

## 方法来源

1. I. R. Craig and D. E. Manolopoulos, *J. Chem. Phys.* **121**, 3368 (2004),
   [10.1063/1.1777575](https://doi.org/10.1063/1.1777575)：RPMD 时间相关函数。
2. M. Ceriotti, M. Parrinello, T. E. Markland and D. E. Manolopoulos,
   *J. Chem. Phys.* **133**, 124104 (2010),
   [10.1063/1.3489925](https://doi.org/10.1063/1.3489925)：路径积分正常模热化。
3. Y. Kataoka et al., *Phys. Rev. Research* **6**, 043224 (2024),
   [10.1103/PhysRevResearch.6.043224](https://doi.org/10.1103/PhysRevResearch.6.043224)：
   Pd(111) 氢扩散与盆地归一化 QTST。
4. I. R. Craig and D. E. Manolopoulos, *J. Chem. Phys.* **122**, 084106 (2005),
   [10.1063/1.1850093](https://doi.org/10.1063/1.1850093)：凝聚相 RPMD 速率。
5. Y. V. Suleimanov, *J. Phys. Chem. C* **116**, 11141 (2012),
   [10.1021/jp302453z](https://doi.org/10.1021/jp302453z)：Ni(100) 氢扩散。
6. Q. Zhang et al., *Phys. Rev. Lett.* **101**, 215902 (2008),
   [10.1103/PhysRevLett.101.215902](https://doi.org/10.1103/PhysRevLett.101.215902)：
   BaZrO₃ 质子转移/重取向的核量子效应；该文的 PIMD/centroid-TST 与完整
   RPMD 回穿计算应分别表述。


### 合并前周期 CV 记录的恢复

材料开发分支曾让 `distance`、`bond_difference`、`angle` 隐式使用最小镜像，
表面分支对相同字段使用展开几何。整合版为这类周期配置的科学指纹增加了
`explicit-periodic-v1` 约定，旧的歧义记录会拒绝恢复或再分析，避免改变已采样的数学定义。
请保留旧代码与原始记录，在新目录中用明确的 `periodic_*` 或展开 CV 重新运行。
不要手动改 checkpoint 哈希。非周期记录和已经使用明确周期 CV 的记录不受此变更影响。
