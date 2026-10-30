# 科学概念、单位与收敛

本章解释 PyRPMD 中各类计算结果的物理含义。先明确要计算的量，再选择任务、势能和采样方法，可以避免把自由能、采样时间、反应速率和扩散系数混为一谈。实际输入与命令见[工作流指南](workflows.md)、[周期材料与表面](periodic.md)和 [PCET](pcet.md)。

## 选择计算问题

| 目标 | 任务 | 主要结果与归一化 |
| --- | --- | --- |
| 两个气相反应物的热反应速率 | `rate` | PMF、传输系数和双分子速率，`cm^3 molecule^-1 s^-1` |
| 显式发射两个反应物并分类产物 | `drpmd` | 热速率或固定碰撞能截面，取决于初态分布 |
| 指定 CV 上的平衡自由能 | `fes` | 相对于 CV 体积元的概率密度及自由能 |
| 多入口、多 TS 的独立速率 | `network` | 每个入口与 TS 的独立速率矩阵 |
| 用历史偏置探索 1D/2D CV | `metadynamics` | 有限时间偏置自由能估计和 HILLS |
| 固定晶胞中的平衡核量子统计 | `pimd` | 物理能量估计器、环回转半径 |
| 热化后无恒温 RPMD 输运 | `diffusion` | 质心 MSD、VACF、扩散张量与诊断 |
| 一个材料位点或表面盆地的选定出口 | `site_rate` / `surface_rate` | 盆地归一化的一阶速率，`s^-1` |
| 局域电化学弱耦合质子–电子转移 | `pcet` | 还原/氧化模型速率与同位素效应，`s^-1` |

`pcet` 使用一维质子本征态和 golden-rule 速率表达式；它不执行 RPMD 传播。其余动力学任务使用同一套真实原子质量环聚合物约定。`network` 当前仅支持 `aggregation="separate"`；不同 TS 的通量没有自动相加为总速率。

## 原子单位与数组约定

内部计算采用原子单位。对低层 Python API，数值已经是原子单位；输入文件可使用经过量纲检查的字符串。不要依赖几何文件和内联数组拥有相同的缺省单位。

| 量 | 内部单位 / 形状 | 推荐输入方式 |
| --- | --- | --- |
| 位置、晶胞 | bohr；`q[P,N,3]`，`cell[3,3]` | `geometry_unit="angstrom"`、`cell_unit="angstrom"` |
| 能量 | hartree | `"0.7 eV"`、`"1 kcal/mol"` |
| 力 | hartree/bohr，物理势能的负梯度 | PES 输出 `[B,N,3]` |
| 质量 | 电子质量 | `masses_amu` 或带 `amu` 的同位素质量 |
| 时间 | 原子时间 | `"0.1 fs"`、`"2 ps"` |
| 温度 | kelvin 输入；内部使用 `beta=1/(k_B T)` | `temperature="300 K"` |
| 动量 | 原子单位，`p[P,N,3]` | 由采样器按质量和温度生成 |
| PES batch | 坐标 `[B,N,3]`，能量 `[B]` | B 是本次调用的帧数，不一定等于总珠子数 |
| D 维集体变量 | `value[D]`、`jacobian[D,P,N,3]` | 自定义 CV 必须返回完整环坐标 Jacobian |

所有原子索引从 0 开始。XYZ 坐标缺省为 angstrom；内联坐标缺省为 bohr。晶胞每一行是晶格矢量，满足 `r = s @ cell`，详细边界约定见[周期结构](periodic.md)。

势能接口返回每个珠子的**物理势能和物理力**。环弹簧由传播器加入；PES 不应加入弹簧、除以珠子数或重复加入 umbrella。接口的单位转换、错误处理及可选 virial 见[势能接口规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md)。

## 珠子数和环聚合物 Hamiltonian

对 P 个珠子和 N 个原子，PyRPMD 使用求和形式的 Hamiltonian：

$$
H_P=\sum_{j=0}^{P-1}\left[
\sum_i\frac{|p_{ji}|^2}{2m_i}+V(q_j)
+\sum_i\frac{m_i\omega_P^2}{2}|q_{ji}-q_{j-1,i}|^2
\right],\qquad \omega_P=\frac{P}{\beta}.
$$

原子单位中 $\hbar=1$，平衡分布是 $\exp[-(\beta/P)H_P]$。因此单个动量分量的 Maxwell 方差是 $m_iP/\beta$。P=1 时没有环弹簧，确定性传播回到经典分子动力学；增加 P 用于检查路径积分离散化，不等于增加独立实验样本。

质心定义为 $\bar q_i=P^{-1}\sum_j q_{ji}$。对于非线性函数 s，以下两种 CV 通常不同：

$$
s_{\mathrm{centroid}}=s(\bar q),\qquad
s_{\mathrm{bead\ average}}=\frac{1}{P}\sum_j s(q_j).
$$

输入中的 `estimator="centroid"` 与 `"bead_average"` 必须按要研究的物理量选择。一个 centroid CV 的 Jacobian 已包含 $1/P$，自定义插件不能再漏乘或重复乘这一因子。

若物理 umbrella 为 $W(s)=\tfrac12(s-c)^TK(s-c)$，求和 Hamiltonian 中加入的是 $P W$，完整环上的偏置力为 $-P J^T K(s-c)$。这一约定解释了为什么通用 FES 的 MBAR 约化偏置使用 $\beta W$，而不是再次乘以 P。混合长度、角度等不同 CV 时，中心和刚度矩阵各分量必须具有匹配量纲。

## 采样过程与真实时间传播

PIMD、umbrella 窗口和受约束母链的目的，是生成所需的平衡分布。其恒温动力学时间不直接解释成实验输运或反应时间。dRPMD 碰撞、RPMD 回穿子轨迹及 `diffusion` 的生产段则使用无恒温、无历史偏置的确定性传播。

| 确定性积分器 | 自由环处理 | 需要检查的误差 |
| --- | --- | --- |
| `cayley-bcb` | Cayley 自由环映射，缺省方法 | 物理力与分裂的有限步长误差 |
| `normal-mode-bab` | 精确正常模自由环转动 | 物理力与分裂误差；自由环精确不等于完整动力学精确 |
| `velocity-verlet` | 显式弹簧与物理力 | 环最高频率稳定界及更严格的物理势限制 |

所有方法都需要减小时间步检查目标结果。Verlet 的自由环必要稳定条件为 $|\Delta t|\max\omega_k<2$；通过这一检查仍不能保证强键势或刚性偏置稳定。Cayley 的自由环稳定性也不是任意大步长的精度保证。参见[积分器说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/integrators.md)。

普通反应和 FES 工作流采用与所选积分器匹配的 Andersen 采样。材料 `pimd` / `diffusion` 热化还支持 PILE-L：Cayley 对应 `bcocb`，正常模对应 `baoab`，Verlet 使用 Andersen。`native=true` 表示允许使用编译子步；`loop_backend="native"` 要求整个融合传播循环兼容，两者含义不同。周期/固定原子和 PILE-L 的限制见[周期任务指南](periodic.md)。

## 自由能的测度和速率归一化

通用 FES 输出的是所选 CV 体积元上的密度自由能：

$$
F(s)=-\beta^{-1}\log\rho(s)+C.
$$

直方图概率质量必须除以 bin 体积才是密度。坐标变换会改变密度测度，因此距离 FES 中包含的几何简并度不会被自动删除。二维自由能面上的一个势垒，也没有唯一的速率前因子。

气相 `rate` 使用包含通量质量度量的专用 PMF，并采用两个远区片段的双分子归一化。材料/表面的局部位点速率使用位置密度和显式盆地概率，输出一阶速率。两者的 PMF、分母和单位必须配套，不能只替换前因子。详见[气相速率](workflows.md)及[盆地速率](periodic.md)。

`pimd` 报告的物理能量也与环 Hamiltonian 不同。对全部原子可动的体系，势能估计是每珠子物理势能的平均，centroid-virial 动能为

$$
K_{\mathrm{cv}}=\frac{3N}{2\beta}
-\frac{1}{2P}\sum_{ji}(q_{ji}-\bar q_i)\cdot F_{ji}.
$$

物理内能估计为 $\langle V\rangle_P+K_{\mathrm{cv}}$。虚拟环动量的瞬时动能不直接用作量子物理动能；`ring_polymer_hamiltonian` 用于传播诊断。当前材料估计器要求全部原子可动。

## 如何判断结果可信

一次成功运行只说明指定输入完成了计算。物理结论还依赖势能适用性、反应定义、离散参数和有限采样。建议把验证记录分成以下层次，并保存每次改变后的完整输入与结果：

1. **模型与接口。** 核对元素、同位素、质量、电荷、自旋及 PES 定义，做力的有限差分检查；周期体系另检查晶胞、周期平移和部分 PBC。
2. **数值参数。** 分别改变珠子数、时间步和相关网格/bin 设置，比较所关心的能量、自由能、速率或 D，而不只看轨迹是否发散。
3. **平衡与覆盖。** 增加热化、独立副本与生产长度，检查 CV 慢模、窗口 overlap、有效样本数、盆地和分割面的支持。断连窗口之间的相对自由能不可识别。
4. **任务特有条件。** 气相检查 `rinf` / 初始分离、`bmax`、分类与最大碰撞时间；局部速率检查盆地和回穿平台；扩散检查长时线性区、参考系、尺寸与拟合区间；PCET 检查坐标域、网格、态数与弱耦合模型。
5. **统计误差。** 用独立副本或保留相关性的 block/parent bootstrap。重叠时间起点、同一母链发射的多条轨迹，以及一个环的多个珠子，都不能简单当作独立样本。

误差条描述已采样问题的统计不确定度，通常不包含势模型误差、未访问慢模、有限尺寸和珠子数误差。缺失误差应保持 `null`；未采样 bin 应保持 mask，不能把缺失值解释为零或自行填平后报告速率。dRPMD 未决轨迹的上下界也不等同于统计置信区间。

程序保留 `converged=false` 或 `physical_converged=false`，PCET 另报告 `numerical_convergence` 与 `physical_applicability_unverified`。这些状态帮助区分“完成计算”“通过特定数值检查”和“物理预测已经收敛”。理论与实现细节可继续阅读[科学约定](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/science.md)、[MBAR 速率 PMF](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mbar-rate-pmf.md)及各任务专题中的方法来源。
