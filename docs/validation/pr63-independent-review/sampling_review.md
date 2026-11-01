# PR #63：增强采样独立审阅与数值对照

审阅对象为 `3faac3aa42b5424b2711367608e0138c82f62e0a`。这里的实验在本轮独立执行；原 PR 中 Müller 势、PLUMED、稀有事件采样的旧结果没有被计为本轮重新执行的结果。本报告针对新增 CV、OPES、共享 MetaD、交换采样和偏置动力学时间分析。

**本轮发现并修复了静态 Hamiltonian 检查被复合势绕过的问题。公式级对照达到浮点精度；实际 RETIS 实验仍有一项接口条件概率超出点态 95% 区间，因此没有建立完整的动力学准确性或收敛结论。** 下面保留该偏差及全部种子数据。

## 1. 确认并修复的科学契约缺陷

**严重性：P1。含有动态成员的复合势能可以绕过静态 Hamiltonian 检查，导致平衡副本交换接受率不再对应固定的目标分布。**

原 `potentials/contraction.py::_component` 只检查部分 `metadata` 标记。`CommitteePotential`、`DeltaLearningPotential` 等包装后公布自己的执行元数据，成员的 `time_dependent`、`history_dependent`、`stateful` 声明可能被隐藏。即使构造时所有成员都静态，后续更新借用的成员也会绕过原 `replica_exchange.py::_static`；原检查只递归处理 `CompositeBias`。

本轮将构造边界和交换边界同时补齐：三个动态标记在对象属性、`metadata` 和 `capabilities` 中均被检查；交换检查递归访问已有的 committee、delta、difference、RPC、Python 和 step-cache 包装，并拒绝循环组合。先沿已知组合关系检查循环，再访问 cache 转发的元数据，因此 cache 自循环或 `cache → committee → cache` 都会在属性转发递归前被拒绝。静态组合仍按原有 Hamiltonian 运行，没有改动合法交换的接受公式。

`advance` 也在第一步 MD 前检查所有副本的势能与 bias。仅在块末交换时拒绝动态提供者仍可能调用它、改变外部历史；恢复位置、动量和 RNG 无法撤销这些副作用。新增回归确认第二个副本发生非法动态更新时，第一个副本和第二个副本均没有 PES 调用，时钟、传播统计及全部 RNG 都不变。

`sampling_composition_witness.py` 从审阅提交中取出旧的交换模块，在独立模块中执行。它没有替换运行中的生产文件。对于两个坐标完全未改变的副本，旧代码对同一动态 committee 分别得到 `0.0615` 和 `0.0855` 两个 log 接受率，并调用成员 PES 八次；修复后的代码在第一次 PES 调用前拒绝同一输入。原始结果保存在 `sampling_composition_witness.json`。

47 项新增组合回归在原提交上为 **46 失败、1 通过**，修复后全部通过；与已有交换/采样边界测试合并后的批次为 **73 通过**。另一次增强采样已有测试批次为 **143 通过、1 跳过**，其中真实 TorchScript CV 测试执行成功，跳过的是当前测试环境缺少 PLUMED Python wrapper 的直接比较。这些批次有重叠，不相加作为独立测试总数。全仓最终门槛另见总报告。

完整仓库测试结束后，又补充了上面的 cache 环路与 `advance` 前置检查，新增 13 项回归。最终该测试文件合计 **60 项：原提交 59 失败、1 通过，修复后 60 全部通过**；与已有采样边界测试合并的最终门槛为 **86 通过**。证据为 `sampling_final_guard_baseline_pytest.log` 与 `sampling_final_guard_pytest.log`，没有覆盖写掉早期 47 项回归的日志。

这些最后的修改只改变非法组合的拒绝时机与 cache 遍历，不改变合法静态势的传播方程。随后单独重跑 8 组解析交换对照，结果与之前逐项完全一致；`sampling_final_guard_equations.json` 同时记录前后源文件 hash。RETIS 的原始执行身份、链与结果均保留，没有用最后的 guard hash 重标旧数据。

## 2. 论文定义与当前实现边界

| 方法 | 本轮核对的定义 | 已验证的部分 | 仍不能据此声称的能力 |
| --- | --- | --- | --- |
| Path CV / RMSD / 几何 CV | Branduardi 等的路径进度与距离；最优刚体对齐后的距离 | 全 bead Jacobian、平移/转动不变性、二面角圆均值 | 尚未复现该文的蛋白或分子自由能面 |
| OPES / OPES-explore | 加权无偏 KDE 与不加权目标 KDE；两种不同的 bias 前因子 | 周期 Gaussian 密度、梯度、归一化、初始权重、宽度规则和恢复 | 自适应宽度采用已声明的 Welford/Scott 规则，不能当作论文/PLUMED 的扩散窗口自适应；无 kernel compression |
| Shared MetaD | 多 walker 共用 history；well-tempered 高度依赖既有 bias | 同一批次所有 walker 使用沉积前 history 的公式 | 没有把共享 history 的 walker 当独立统计重复；此实验不证明 FES 收敛 |
| Temperature / Hamiltonian exchange | 交换四个约化势能；正确重缩放动量 | 包含目标温度弹簧作用量的有限 bead 接受率、详细平衡、动量分布因子 | 没有新建持久化/MPI 的生产交换工作流 |
| 偏置运行的动力学时间 | 无扰动过渡区条件下的加速时钟和首达时间统计 | 时间积分、能量零点不变性、右删失暴露时间 | 时间数组代数不能证明过渡区未受扰动；这不是完整 OPES-flooding 偏置实现 |
| RETIS | 正接口路径系综的 shooting 与 path exchange | 实际双阱模型路径链及解析通量概率对照 | 没有 minus/flux 系综；不输出经本轮验证的绝对速率或分子级速率 |

OPES 的两种 bias 为

\[
V_\mathrm{METAD}(s)=\frac{1-1/\gamma}{\beta}\log(P(s)/Z+\epsilon),\qquad
V_\mathrm{explore}(s)=\frac{\gamma-1}{\beta}\log(p_\mathrm{WT}(s)/Z+\epsilon).
\]

前者的新 kernel 权重为 \(\exp[\beta V_\mathrm{old}(s_n)]\)，后者权重恒为 1。当前实现使用归一化 Gaussian；独立 oracle 对周期坐标直接叠加 33 个 Gaussian 镜像，不调用生产代码的 kernel、概率或归一化函数。自适应宽度的 oracle 直接对已展开的历史计算样本标准差，而非复用 Welford 递推。

原论文及官方实现同时强调：探索频繁、bias 稳定、重加权自由能收敛是不同判断。新的公式级对照不能把原 PR 中尚未收敛的 Müller 轨迹升级成已收敛结果。

## 3. 新执行的确定性对照

| 对照 | 参数范围 / 数量 | 最大绝对偏差 |
| --- | --- | ---: |
| 全环 CV Jacobian | 4 种 CV × 1/5 beads × centroid/bead average；四阶有限差分，步长 `1e-5` | `1.26294e-10` |
| CV 刚体不变性 | 与上项相同的 16 组配置；另含跨 ±π 的圆均值例子 | 全部低于 `1e-12` |
| OPES bias | METAD/explore × fixed/adaptive；二维、一维周期、6 次沉积、4 个查询点 | `1.66533e-15` hartree |
| OPES bias 梯度 | 同上 | `1.44329e-15` |
| OPES log Z | 独立批量 Gaussian 矩阵均值 | `8.88178e-16` |
| 自适应 kernel 宽度 | 独立批量标准差与权重 ESS | `2.22045e-16` |
| 交换 log 接受率 | 1/2/8/32 beads；温度交换及同时 Hamiltonian 交换 | `1.70530e-13` |
| 共享 MetaD 高度 | 4 walkers，2 个完整同步批次 | 低于 `2e-15` |
| 偏置时钟积分 | 4 个不等长时间区间 | `8.88178e-16` |
| 偏置能量零点不变性 | bias 和过渡区参考一起加 37 hartree | `1.77636e-15` |

交换 oracle 直接构造

\[
u_i(q)=\frac{\beta_i}{P}\left[
\frac{k_i}{2}\sum_{j,a,c}q_{jac}^2+
\frac12\sum_{j,a,c}m_a(P/\beta_i)^2(q_{jac}-q_{j+1,a,c})^2
\right],
\]

再计算 \(u_i(q_i)+u_j(q_j)-u_i(q_j)-u_j(q_i)\)。两个温度为 `beta=0.8, 2.3`，质量为 `1.2, 2.7`；Hamiltonian 对照使用不同的 harmonic 力常数。这里验证接受核的代数及详细平衡，不把八个配置点称为八次平衡模拟。

偏置动力学还对 **40,000 个独立合成首达时间**实施独立右删失：真实指数速率为 `0.7`，观测到 25,474 个事件、14,526 个删失样本。估计为 `0.7050826193`，渐近标准误 `0.0044176513`，距真值 `1.15` 个标准误。合成数据只验证删失似然和暴露时间处理，没有从分子轨迹推断速率。

## 4. 实际 RETIS 路径实验

使用可解析的三维可分模型

\[
U(x,y,z)=(x^2-1)^2+(y^2+z^2)/2,
\quad m=1,\quad\beta=1.3,\quad P=1.
\]

反应物/产物盆地为 `x <= -0.8` 与 `x >= 0.8`，三个正接口为 `(-0.8,-0.5,-0.2)`。横向 harmonic 分量从条件概率中抵消。连续时间、正向 canonical flux 的独立参考为

\[
P(\lambda_{i+1}\mid\lambda_i)
=\exp[-\beta(U(\lambda_{i+1})-U(\lambda_i))],
\]

其中最后一步取真正势垒顶 `x=0`，因为越过势垒后会到达 B。三项为 **0.5696290041、0.6269866691、0.9031017969**，乘积为 **0.3225426189**。

固定最终预算为两个步长 `0.04`、`0.02`，每个步长四个种子 `638361–638364`，每条链 300 次 burn-in、1,200 次生产 sweep。每次 sweep 对三个接口各尝试一次 shooting，再执行相邻交换；拒绝后的重复路径保留在统计中。共 9,600 次生产 sweep、28,800 个生产接口观测，不能把这些相关观测当作独立轨迹。初始路径由真实动力学产生，没有用坐标插值生成。

600 次生产 sweep 的早期运行只用于运行时间与流程检查，保存在 `sampling_pilot_*`；最终预算随后固定为 1,200，未依据显著性继续增加。两组最终结果使用相同四个种子，时间步比较按配对种子处理。

数值汇总在 `sampling_retis_summary.csv`，两组完整结果在 `sampling_results_dt0.04.json` 与 `sampling_results_dt0.02.json`。区间为四个种子结果的 Student-t 95% 区间，属于点估计的有限重复诊断；未进行多重比较校正，也不能排除慢混合。每个种子内部先求条件概率及其乘积，再计算种子之间的区间，从而不把三个相关接口误当独立误差源。

| 观测量 | 连续时间解析参考 | `dt=0.04`，均值 ± 95% 半宽 | `dt=0.02`，均值 ± 95% 半宽 |
| --- | ---: | ---: | ---: |
| `-0.8 → -0.5` | 0.5696290041 | 0.564375 ± 0.059196 | 0.567500 ± 0.049295 |
| `-0.5 → -0.2` | 0.6269866691 | 0.637917 ± 0.054571 | 0.630000 ± 0.043510 |
| `-0.2 → B` | 0.9031017969 | 0.917292 ± 0.023556 | **0.914792 ± 0.009211** |
| 三项乘积 | 0.3225426189 | 0.331018 ± 0.062803 | 0.327203 ± 0.039954 |

两个乘积的区间包含解析参考，但 `dt=0.02` 最后一项条件概率的区间约为 `[0.905581, 0.924002]`，**没有包含参考 0.903102**。这是保留的未解决验证结果。比较对象是连续时间参考，而实现对有限步长轨迹的离散配置检测接口；离散积分/接口偏差、有限 burn-in 和混合、四个种子的不确定性估计都可能影响结果。单项点态偏差且未校正多重比较，不能识别具体来源。

按相同种子配对，细步长减粗步长的乘积差为 `-0.003815 ± 0.070903`；如此宽的区间无法建立时间步收敛。两个步长的路径长度拒绝次数均为零，但这也不能替代独立的路径长度支持扫描。没有增选种子、延长到显著性消失，或丢弃该接口。

`sampling_retis_raw_dt*.npz` 保存每次生产 sweep 的 crossing 指示量、保留路径长度和 walker 标签，足以重新计算报告的概率与相关性诊断；它没有保存每个 shooting proposal 的所有 Cartesian 帧。生产脚本、种子和参数保留用于重生成路径。

本模型只有一个 bead；数值积分、离散接口检测、有限 burn-in、路径长度支持和统计误差仍是不同误差源。`max_path_steps=2500` 是每个完整路径的上限，没有做独立的路径长度上限扫描。本轮没有计算独立反应物通量，因此以上概率乘积不能直接标成物理速率。

## 5. 复现与证据

在项目根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python docs/validation/pr63-independent-review/sampling_experiments.py --retis-samples 1200 --retis-seeds 4 --dt 0.04
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python docs/validation/pr63-independent-review/sampling_experiments.py --retis-samples 1200 --retis-seeds 4 --dt 0.02
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python docs/validation/pr63-independent-review/sampling_composition_witness.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python docs/validation/pr63-independent-review/sampling_final_guard_equations.py
python docs/validation/pr63-independent-review/sampling_figures.py
```

生产 driver 在开始前保存所检查模块的 SHA-256，并在结束时确认这些文件未变化。结果记录实际执行源文件 hash；后续纯校验边界修补应保留这一执行身份，不能重标旧数据。时间、平台与 NumPy/SciPy 版本也在 JSON 中。图 `sampling_validation.png` / `.pdf` 完全由本轮 JSON 生成。

## 6. 一手资料

1. Branduardi, Gervasio & Parrinello, *From A to B in free energy space*, JCP **126**, 054103 (2007), [doi:10.1063/1.2432340](https://doi.org/10.1063/1.2432340)。
2. Invernizzi & Parrinello, *Exploration vs Convergence Speed in Adaptive-Bias Enhanced Sampling*, JCTC **18**, 3988–3996 (2022), [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9202311/)。
3. PLUMED 2.9 官方文档：[OPES_METAD](https://www.plumed.org/doc-v2.9/user-doc/html/_o_p_e_s__m_e_t_a_d.html)、[OPES_METAD_EXPLORE](https://www.plumed.org/doc-v2.9/user-doc/html/_o_p_e_s__m_e_t_a_d__e_x_p_l_o_r_e.html)。文档明确说明自适应宽度的扩散窗口、EXCLUDED_REGION、Explore 的 SIGMA 与 √γ 约定。
4. Raiteri et al., *Efficient Reconstruction of Complex Free Energy Landscapes by Multiple Walkers Metadynamics*, JPCB (2006), [doi:10.1021/jp054359r](https://doi.org/10.1021/jp054359r)；Barducci, Bussi & Parrinello, PRL **100**, 020603 (2008), [doi:10.1103/PhysRevLett.100.020603](https://doi.org/10.1103/PhysRevLett.100.020603)。
5. Sugita & Okamoto, *Replica-exchange molecular dynamics method for protein folding* (1999), [doi:10.1016/S0009-2614(99)01123-9](https://doi.org/10.1016/S0009-2614(99)01123-9)；Sugita, Kitao & Okamoto, *Multidimensional replica-exchange method for free-energy calculations*, [作者预印本](https://arxiv.org/abs/cond-mat/0009120)。
6. Salvalaglio, Tiwary & Parrinello, *Assessing the Reliability of the Dynamics Reconstructed from Metadynamics*, JCTC **10**, 1420–1425 (2014), [doi:10.1021/ct500040r](https://doi.org/10.1021/ct500040r)。
7. van Erp, *Reaction rate calculation by parallel path swapping*, PRL **98**, 268301 (2007), [doi:10.1103/PhysRevLett.98.268301](https://doi.org/10.1103/PhysRevLett.98.268301)；[PyRETIS 作者团队的方法说明](https://pyretis.org/dev/user/intro.html)给出正接口、minus 系综与通量的区别。
