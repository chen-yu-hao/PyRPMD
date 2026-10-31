# 科学概念、单位与收敛

本章定义 PyRPMD 计算的物理量、单位和数组约定，以及使结果收敛的流程。各任务的输入和命令见
[工作流指南](workflows.md)、[材料与表面](periodic.md)和 [PCET](pcet.md)。

## 任务与结果

| 目标 | 任务 | 结果与归一化 |
| --- | --- | --- |
| 两个气相反应物的热速率 | `rate` | PMF、传输系数与双分子速率，单位 `cm^3 molecule^-1 s^-1` |
| 显式碰撞与产物分类 | `drpmd` | 热速率或固定能量截面，由发射分布决定 |
| 所选 CV 上的平衡自由能 | `fes` | 相对于 CV 体积元的概率密度及其自由能 |
| 多入口、多 TS 的独立速率 | `network` | 按入口、通道和 TS 索引的速率矩阵 |
| 一维/二维/三维 CV 上的历史偏置 | `metadynamics` | 有限时间偏置自由能估计与 HILLS |
| 固定晶胞中的平衡核量子统计 | `pimd` | 物理能量估计量、环回转半径 |
| 无恒温 RPMD 输运 | `diffusion` | 质心 MSD、VACF、扩散张量与诊断 |
| 材料位点或表面盆地的选定出口 | `site_rate` / `surface_rate` | 盆地归一化的一阶速率，单位 `s^-1` |
| 局域电化学质子–电子转移 | `pcet` | 还原/氧化速率与同位素效应，单位 `s^-1` |

`pcet` 求解一维质子本征态并计算 golden-rule 速率，不做 RPMD 传播。其余动力学任务共用同一套使用真实原子
质量的环聚合物约定。`network` 分别报告每个 TS（`aggregation="separate"`）。

## 原子单位与数组约定

PyRPMD 以原子单位计算。低层 Python API 接受原子单位数值；输入文件也接受带单位的字符串，并做量纲检查。

| 量 | 内部单位 / 形状 | 推荐输入方式 |
| --- | --- | --- |
| 位置、晶胞 | bohr；`q[P,N,3]`，`cell[3,3]` | `geometry_unit="angstrom"`、`cell_unit="angstrom"` |
| 能量 | hartree | `"0.7 eV"`、`"1 kcal/mol"` |
| 力 | hartree/bohr，物理势能的负梯度 | PES 输出 `[B,N,3]` |
| 质量 | 电子质量 | `masses_amu` 或带 `amu` 的同位素质量 |
| 时间 | 原子时间 | `"0.1 fs"`、`"2 ps"` |
| 温度 | 输入用 K；内部用 `beta=1/(k_B T)` | `temperature="300 K"` |
| 动量 | 原子单位，`p[P,N,3]` | 由质量和温度采样 |
| PES 批次 | 坐标 `[B,N,3]`，能量 `[B]` | B 是每次调用的帧数 |
| D 维 CV | `value[D]`，`jacobian[D,P,N,3]` | 自定义 CV 返回完整的环坐标 Jacobian |

原子索引从 0 开始。XYZ 坐标缺省为 angstrom，内联坐标缺省为 bohr。晶胞每一行是一个晶格矢量，`r = s @ cell`，
见[周期结构](periodic.md)。

势能返回每个珠子的**物理势能和物理力**。环弹簧、珠子缩放和偏置由传播器加入。单位、错误处理和可选 virial 见
[势能接口规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md)。

## 环聚合物 Hamiltonian

对 P 个珠子、N 个原子，PyRPMD 使用求和形式的 Hamiltonian

$$
H_P=\sum_{j=0}^{P-1}\left[
\sum_i\frac{|p_{ji}|^2}{2m_i}+V(q_j)
+\sum_i\frac{m_i\omega_P^2}{2}|q_{ji}-q_{j-1,i}|^2
\right],\qquad \omega_P=\frac{P}{\beta}.
$$

取 $\hbar=1$ 时平衡分布为 $\exp[-(\beta/P)H_P]$，单个动量分量的 Maxwell 方差为 $m_iP/\beta$。P=1 时弹簧消失，
确定性传播即经典分子动力学。增大 P 使路径积分离散化收敛。

质心为 $\bar q_i=P^{-1}\sum_j q_{ji}$。对非线性函数 s，两种估计量

$$
s_{\mathrm{centroid}}=s(\bar q),\qquad
s_{\mathrm{bead\ average}}=\frac{1}{P}\sum_j s(q_j)
$$

一般不同；按所需物理量选择 `estimator="centroid"` 或 `"bead_average"`。质心 CV 的 Jacobian 已含因子 $1/P$。

对物理伞势 $W(s)=\tfrac12(s-c)^TK(s-c)$，求和 Hamiltonian 加入 $P W$，环上的偏置力为 $-P J^T K(s-c)$，
因此通用 FES 的 MBAR 约化偏置为 $\beta W$。组合不同量纲的 CV 时，中心和刚度矩阵使用相匹配的单位。

## 采样与实时传播

PIMD、伞窗口和受约束母链在恒温器下采样平衡分布。dRPMD 碰撞、RPMD 回穿子轨迹和 `diffusion` 生产段使用
无恒温、无偏置的确定性传播，其时间就是物理时间。

| 确定性积分器 | 自由环 | 需要收敛的参数 |
| --- | --- | --- |
| `cayley-bcb` | Cayley 映射（缺省） | 时间步：物理力与分裂误差 |
| `normal-mode-bab` | 精确正常模转动 | 时间步：物理力与分裂误差 |
| `velocity-verlet` | 显式弹簧与力 | 时间步，且须满足环最高频率的稳定条件 |

每种方法都要对时间步收敛。Verlet 的自由环在 $|\Delta t|\max\omega_k<2$ 时稳定；刚性键或刚性偏置会收紧这一界限。
见[积分器说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/integrators.md)。

反应与 FES 工作流默认使用与积分器匹配的 Andersen 采样；FES 也支持 PILE-L。材料 `pimd` / `diffusion`
默认使用 PILE-L：Cayley 对应 `bcocb`，正常模对应 `baoab`，Verlet 使用 Andersen。`native=true` 启用编译子步骤；`loop_backend="native"`
要求整个循环融合执行。周期与固定原子体系的规则见[周期任务指南](periodic.md)。

## 自由能测度与速率归一化

通用 FES 是所选 CV 体积元上的密度自由能：

$$
F(s)=-\beta^{-1}\log\rho(s)+C.
$$

PyRPMD 用 bin 体积除直方图概率质量得到密度。坐标变换会改变密度测度，因此距离 FES 包含其几何简并度。

气相 `rate` 使用带通量质量度量的专用 PMF 和两个远离片段的双分子归一化。局部位点与表面速率使用位置密度
和显式盆地概率，给出一阶速率。每种速率使用各自工作流的 PMF、分母和单位，见[气相速率](workflows.md)和
[盆地速率](periodic.md)。

`pimd` 报告物理能量估计量。全部原子可动时，势能为物理势能的珠子平均，centroid-virial 动能为

$$
K_{\mathrm{cv}}=\frac{3N}{2\beta}
-\frac{1}{2P}\sum_{ji}(q_{ji}-\bar q_i)\cdot F_{ji}.
$$

内能估计为 $\langle V\rangle_P+K_{\mathrm{cv}}$。`ring_polymer_hamiltonian` 作为传播诊断报告。材料估计量要求全部原子可动。

## 使结果收敛

分五层建立结果，并保存每一步的完整输入与输出：

1. **模型与接口。** 检查元素、同位素、质量、电荷、自旋与 PES 定义，并对力做有限差分。周期体系还要检查晶胞、
   周期平移和部分 PBC。
2. **数值参数。** 逐一改变珠子数、时间步和网格/bin 设置，比较要报告的能量、自由能、速率或 D。
3. **平衡与覆盖。** 延长热化、增加独立副本并延长生产段；检查 CV 慢模、窗口重叠、有效样本数以及盆地和分割面的采样支持。
4. **任务特有参数。** 气相：`rinf` 或初始距离、`bmax`、分类规则与最大碰撞时间。局部速率：盆地与回穿平台。
   扩散：长时线性区、参考系、体系尺寸与拟合区间。PCET：坐标域、网格、态数与弱耦合模型。
5. **统计误差。** 使用独立副本，或保留重叠时间原点、同一母链的多条轨迹和同一环上各珠子之间相关性的
   block 与母链 bootstrap。

误差条给出已采样问题的统计不确定度。缺失的误差报告为 `null`，未采样的 bin 保持 mask，dRPMD 未决轨迹给出的
上下界与置信区间分开报告。

### 评估自适应伞抽样结果

[输入缺省值](input.md#缺省值)是较短的起始预算。P=32、0.1 fs 步长、1 ps 自适应生产段、ESS 20 和
20 次 bootstrap 都不能为未知反应或材料指定精度。`preset="production"` 也不表示已经收敛。
自适应选窗保持伞刚度固定，只细化所给定的范围；选窗后丢弃 pilot 数据，换用独立种子重新平衡并生产采样。
它不能发现未指定的盆地，也不能证明所选 CV 之外的慢变量已经采样充分。

在 `RUN/results/summary.json` 中，FES 读取 `adaptive_umbrella`，气相速率读取 `pmf.adaptive_umbrella`。
气相运行的 `RUN/rate_pmf.json` 和 `RUN/umbrella/summary.json` 也保留 `adaptive_umbrella` 诊断。
分三层读取结果：

1. **Pilot 选窗。** `adaptive_umbrella.pilot_target_met` 表示 pilot 通过了所要求的局部重叠、ESS 和可选覆盖判据。
   旧字段 `target_met` 是该 pilot 结果的兼容别名。
2. **生产采样。** `production_target_met` 用独立生产数据重新检查这些判据。查看
   `production_overlap_connected_at_target`、`production_effective_samples_target_met` 和 `production_coverage`；
   未启用覆盖判据不等于整个范围均有支持。检查失败会输出警告，但不会自动增加采样或删除已计算的点估计。
3. **科学收敛。** 比较独立种子与初态、更长平衡和生产段、P 与 2P，以及原步长与减半步长。细化窗口和直方图
   分箱，按任务检查参考点到势垒的区间、渐近距离与回穿平台。只有差异落在明确的误差预算内才可接受。

气相速率默认 `rate.pmf_method="ui"`，用高斯分布拟合各窗口。窗口偏斜、多峰和缺少实测支持的高斯尾部，
可在均值与方差看似稳定时仍引入 PMF 系统偏差。UI 在端点缺少直方图支持时保留模型估计并警告；模型值不是
实测支持。采用 `rate.pmf_method="mbar"` 并保留 `compare_ui=true`（MBAR 下的默认值）可在同一批生产样本上
进行配对比较。直方图 MBAR 检查反应物参考点至分割面全区间的 bin 支持，但仍需分箱收敛和充分采样。
两个估计器在同一条轨迹上吻合，不能排除共同遗漏的盆地。通用 `fes` 使用 CV 密度测度下的 MBAR，不能代替
气相速率工作流的 `rate_flux` PMF。

不确定度需要另行检查。增加独立样本，验证 block-bootstrap 区间的稳定性；单纯增加 bootstrap 重采样次数
不能补足缺失的采样。速率 PMF 的重采样若缺少支持或估计失败，会计为 rejected。只要存在失败 draw，主势垒/
速率误差和置信区间就不可用（`null`，曲线数组为 NaN）。`supported_draw_*` 结果只描述接受子集，条件是支持
完整且拟合有效，不能当作无条件置信区间。应检查请求数、实际尝试数、有效数、拒绝数及分块状态。
通用 FES 对每个 bin 作同样区分：只有所有请求的 draw 均支持该 bin，且相关性分块可用时，才报告主
`standard_error`，否则为 NaN。`supported_draw_standard_error` 与 `bootstrap_valid_per_bin` 保留条件误差
和支持次数。点估计 mask 与不确定度诊断回答不同的问题；有限误差条仍不等于物理收敛。

应按要报告的物理量制定误差目标。静态速率因子 $k_{\rm QTST}\propto\exp(-\beta\Delta F)$ 中，势垒误差大小
$|\delta F|$ 对应最多 $\exp(\beta|\delta F|)$ 倍的速率偏离。若将该因子限制为 1.10，需要
$|\delta F|\le k_BT\log(1.10)$，在 300 K 约为 **0.0568 kcal/mol**。这个例子把全部 10% 速率预算分给势垒；
传输系数、PES、时间步、珠子数以及 RPMD 近似的误差仍须分别评估。通用缺省值不能替所有体系满足该预算。

方法依据：[Kästner 和 Thiel (2006)](https://doi.org/10.1063/1.2206775)讨论高斯 UI 近似和相关误差传播；
[Shirts 和 Chodera (2008)](https://doi.org/10.1063/1.2978177)给出 MBAR；
[PyMBAR 时间序列指南](https://pymbar.readthedocs.io/en/latest/timeseries.html)说明平衡与相关性诊断。
[RPMDrate 官方输入示例](https://greengroup.mit.edu/rpmdrate-input/)是示范参数，不是普适精度保证。
[OpenMM RPMD 指南](https://docs.openmm.org/latest/userguide/library/09_rpmd_plugin.html)说明珠子数收敛，以及平衡
路径积分和近似实时 RPMD 的区别。

摘要含 `converged` 或 `physical_converged`，单次运行中保持 `false`；PCET 另报告 `numerical_convergence` 和
`physical_applicability_unverified`。理论、实现与方法来源见[科学约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md)、
[MBAR 速率 PMF](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md)和各任务专题。
