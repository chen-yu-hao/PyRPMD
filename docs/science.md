# 科学约定、估计量和验证范围

PyRPMD 目前实现三个可执行流程：Bennett–Chandler RPMD 双分子速率、
直接碰撞 dRPMD，以及一维或二维 umbrella 自由能面。它们共享相同的
真实质量环聚合物动力学、批量 PES 接口和任务/珠子并行层。
`converged: false` 是有意保留的：一次任务正常完成、出现一个时间平台，
都不能代替珠子数、时间步长、统计长度和入口距离的收敛研究。

## 1. 单位和环聚合物 Hamiltonian

内部长度为 bohr，能量为 hartree，质量为电子质量，时间为原子时间。
温度输入的标准单位是 kelvin。带单位字符串会进行量纲检查；Python
低层 API 中的数值已经是原子单位。势能面返回物理势能和**负梯度力**，
不包含自由环弹簧力，也不自行乘除珠子数。

对 P 个珠子，程序使用求和形式

\[
H_P(q,p)=\sum_{j=0}^{P-1}\left[
  \sum_i\frac{|p_{ji}|^2}{2m_i}
  +V(q_j)
  +\sum_i\frac12m_i\omega_P^2|q_{ji}-q_{j-1,i}|^2
\right],\qquad \omega_P=P/\beta,
\]

并以 `exp[-(beta/P)*H_P]` 采样，使用真实物理质量 `m_i`。
因此每个笛卡尔动量分量的 Maxwell 方差为 `m_i*P/beta`。
数组轴为 `q[P,N,3]` 和 `p[P,N,3]`；一个 PES batch 的轴为 `[B,N,3]`。
这一约定必须与采用平均 Hamiltonian 的实现区分，否则温度、弹簧频率、
umbrella 力和热通量都会出现 P 因子错误。

生产轨迹采用 B–C–B 分裂：物理势半步、自由环的 Cayley 步、物理势半步。
Cayley 步在环正常模中实现，其 centroid 模是精确自由漂移。有限时间步下，
这一积分器仍然需要时间步收敛检验。普通工作流的平衡采样采用周期性
Andersen 动量重采样；速率 child 和直接碰撞生产段不加 thermostat。
低层 `Dynamics.sample_step` 另外提供 BCOCB/PILE-L，尚不是输入文件工作流
的可选采样后端。原生 C++ 与 NumPy 参考实现使用相同的 Cayley 映射。

## 2. CV 和 umbrella 的 P 因子

内置 CV 包括原子间距离、两个键长之差、夹角、笛卡尔分量、片段质心
距离和 RPMDrate 的 xi。一般几何 CV 支持两种明确不同的估计器：

\[
s_{\rm centroid}(q)=s(\bar q),\qquad
s_{\rm bead\ average}(q)=P^{-1}\sum_j s(q_j).
\]

对非线性 s，两者通常不相等。任何 CV 都返回 `value[D]` 和
`jacobian[D,P,N,3]`。Jacobian 是所选**完整估计器**对所有珠子坐标的导数，
因此 centroid CV 中已经有 `1/P`。Python CV factory 使用同一规范；支持
`module:factory` 和 `file.py:factory`，能够在独立进程/MPI worker 中重建。
当前扩展协议针对静态、可微 CV，不支持依赖模拟时间的驱动协议。

物理 umbrella 为

\[
W(s)=\tfrac12(s-c)^T K(s-c).
\]

在求和 Hamiltonian 中实际加入 `P*W`，力是
`-P * J.transpose() @ K @ (s-c)`。例如一个 centroid 距离的 umbrella，
每个珠子受到相同的 centroid 约束力，而不是再缩小 `1/P` 的力。
K 必须对称正定；标量、对角数组或完整矩阵均可。中心和弹簧常数的
量纲分别依赖各个 CV；混合长度/角度的非对角项也会检查量纲。
CV 的 `unit`/`units` 声明只改变数值型输入中心的解释，不改变内部 CV 数学。
数值型 stiffness 已是原子单位，推荐输入有明确单位的字符串。

## 3. 一维、二维自由能面

通用 FES 使用直接按 MBAR 方程实现的 SciPy 求解器。
它不是对 PyMBAR 源代码的复制，也没有将多个没有共同支持的窗口拼成
任意偏移的曲面。对具有相同温度、P、PES 和 Hamiltonian 的 umbrella，
共同的物理约化势能可以从所有状态和目标状态同时消去，只需构造

\[
u_k(s_n)=\beta W_k(s_n),\qquad
f_k=-\log\sum_n\frac{e^{-u_k(s_n)}}{
  \sum_l N_l e^{f_l-u_l(s_n)}}.
\]

先用固定点迭代初始化，再求解凸目标，并独立检查自洽方程残差。
返回目标态权重、每个窗口的无量纲自由能、overlap 矩阵和连接分量。
同一 umbrella 的不同 replica 是独立时间序列，各自保留随机数身份。
当前输入不支持混合不同温度或不同珠子数后直接套用上述消去关系。

输出是 CV 坐标的**概率密度**对应的自由能：

\[
F(s_1,s_2)=-\beta^{-1}\log\rho(s_1,s_2)+C.
\]

直方图概率质量除以每个 bin 的体积，因此非等宽 bin 也有正确测度。
这是对 `ds1 ds2` 的密度；距离坐标的几何简并度自然包含在采样分布里。
程序不会自动减去 `-2*kBT*log(r)`，也不会把二维 CV 面直接解释成某个
双分子速率的反应 PMF。

实际输出包含：

- `free_energy`、`standard_error`、`probability_mass`、`density` 和 bin 边界；
- 原始计数及 `mask`：无支持或低于最小计数的 bin 不外推；
- 窗口 overlap 矩阵和连接分量；断连时全局 FES 置为 masked/NaN，
  因为不同分量之间的相对自由能不可识别；
- 各时间序列的统计低效因子、有效样本数诊断和加权样本 ESS。

不确定度使用每条序列独立的 circular-block bootstrap，block 长度取
初始正序列自相关估计给出的低效因子向上取整。每次 bootstrap 重新
求解 MBAR，并以共同参考 bin 对齐自由能常数；失去连接的重复样本不用于
全局误差估计。短轨迹、隐藏慢模和未访问区域可能使误差偏乐观，不能仅
凭 bootstrap 小就宣称采样充分。

`reanalyze_fes` 从完成的窗口 checkpoint 重新构建曲面，可以更换 bins、
mask 阈值和 bootstrap 设置，不会重新调用 PES 或传播轨迹。

## 4. 标准 RPMD 双分子速率

### 4.1 反应坐标和适用范围

当前实现采用一个过渡态参考结构、两个入口片段和等长的 forming/breaking
键对列表。对每个键对，

\[
s_0=R_\infty-R_{AB},\qquad
s_1=(r_{f,TS}-r_f)-(r_{b,TS}-r_b),\qquad
\xi=\frac{s_0}{s_0-s_1}.
\]

所有几何量作用于 centroid。入口面是 xi=0，参考 TS 是 xi=1。
同一 TS 的多个键对使用最小 `s1` 分支。对显式原子置换生成的多个等价 TS，
严格沿用 RPMDrate `_surface.f90` 的顺序：每个成断键对先对 TS 取最大值，
再对键对取最小值，即 `s1 = min_bond(max_TS(values))`。这个顺序不能一般性地
换成 `max_TS(min_bond(values))`；多键回归测试特意区分二者。

`reaction.equivalent_paths` 保存从旧原子到新原子的完整置换；程序检查元素、
同位素质量和反应物片段划分，梯度与 Hessian 使用被选中的解析分支。
分支交界处是分段可微的，使用者必须检查重要采样区域中的分支切换。
当前支持一个参考 TS 的等价置换，不支持任意互不等价的多个 TS 几何。
H + H₂ 中 `[[0,2,1]]` 加上默认原始路径给出
`s1 = r12 - min(r01,r02)`，同时识别 H0H1 和 H0H2 两个交换出口。
这时 `symmetry_factor=1`，避免重复计数；额外路径/电子权重仍需用户明确给出。

### 4.2 速率 PMF 的测度

标准速率不是把通用一维 FES 的势垒直接代入一个随意的前因子。
按 RPMDrate 的实现，对 xi 定义

\[
Z_\xi=\sum_i\frac{|\nabla_i\xi|^2}{m_i},\qquad
f_\xi=\sqrt{Z_\xi/(2\pi\beta)}.
\]

速率 umbrella 的物理附加势为
`W_umbrella - log(f_xi)/beta`，求和 Hamiltonian 再乘 P。
对应的无 umbrella 目标测度包含 `delta(xi-x)*f_xi`。
程序使用解析 centroid Hessian 计算这个 metric 的力，并用有限差分
测试了其与能量的符号、梯度和 P 因子。

目前速率 PMF 的最终曲线使用 RPMDrate 的 Gaussian umbrella integration：
由每个窗口的均值、方差和已知 umbrella 力估计局部自由能导数，再按
窗口的局部高斯密度加权积分。同时独立计算 MBAR overlap；如果 overlap
断连，PMF 外推只能作为诊断，程序不会继续给出一个可靠速率。
Gaussian 窗口近似是否充分，需要用窗口分布、窗口间距和采样长度检验。

令 xi* 为指定的分割面，或已采样 PMF 上的最大值，则

\[
k_{QTST}=4\pi R_\infty^2
\frac{1}{\sqrt{2\pi\beta\mu}}
\exp\{-\beta[W_F(\xi^*)-W_F(0)]\},\qquad
\mu=M_A M_B/(M_A+M_B).
\]

另乘明确输入的电子权重和对称因子。速率单位由原子单位体积/时间
换算为 `cm^3 molecule^-1 s^-1`。入口在 Rinf 处是否足够解耦、势能面
的长程尾是否处理合适，都必须通过改变 Rinf 检验。

### 4.3 Constrained parent 和 paired children

Recrossing 使用没有 xi 分母奇点的线性分割函数

\[
S=(1-\xi^*)s_0+\xi^*s_1.
\]

Parent 以 S=0 的约束正则系综采样。新实现的 centroid Cayley-RATTLE
在位置步内沿旧法向求解约束乘子，在最终受约束坐标处计算新力，最后
沿新法向投影动量。自由环内部模保持 Cayley 传播，不把所有珠子强行
压到同一个几何面。Andersen 重采样后同样投影到切向动量分布。
这一约束正则系综的配置边缘包含对应的质量 metric 因子。

对每个 parent 配置，child **重新抽取未约束的 Maxwell 动量**，运行
`+p` 和 `-p` 一对没有 umbrella、没有 thermostat 的生产轨迹。若把
child 动量也投影到切向，将会错误地消除初始通量。

定义 `g = grad(S)`、`v_S = sum_i g_i dot pbar_i/m_i` 和
`f_S = sqrt(sum_i |g_i|^2/m_i/(2*pi*beta))`，计算

\[
\kappa(t)=\frac{\sum_{\rm children}(v_S/f_S)\,h[S(q_t)]}
 {\sum_{\rm children}(v_S/f_S)\,h(v_S)}.
\]

一对动量的分母为 `abs(v_S)/f_S`。程序保存完整 `kappa(t)`，不只保存
最后一点。PES 失败或约束失败会终止并报告；不会仿照某些参考实现的
回退分支，把失败计算替换成默认 kappa=0.5 或 1。

统计时，先把同一 parent 的所有 child pair 合并为一个 cluster，再在
parent 链内使用连续 block，并在多个独立 parent 链之间重采样。
PMF barrier 的窗口 block bootstrap 与 kappa 的 parent bootstrap 共同
形成速率误差估计。有限 parent 数目时这些误差仍只是初步诊断。

目前自动平台检查比较指定尾部的前后两段平均值，容许相对漂移阈值
或两倍 bootstrap 漂移标准误中的较大者；还要求正平台、至少十个
尾部时间点。`rate_cm3_s` 仅在这个检查通过、且 child 端点的缩放能量
变化 `|ΔH_P|/(P*kBT)` 不超过 0.1 时填写。同时保存每个 child pair 的最大
端点能量变化 `|ΔH_P|/P`；这两个量不依赖 PES 能量零点，适合比较具有
大负绝对能量的真实势能面。旧的 scaled drift 字段为兼容保留，其分母
含总能量，不能作为跨 PES 的精度指标，也不再决定是否发布速率估计。
旧 checkpoint 若没有新的能量诊断，结果会保留候选估计并明确说明限制，
而不会把缺失值当作零误差。
`rate_estimate_cm3_s` 保留可审查的有限时间
估计；状态为 `plateau_estimate` 或 `pilot_requires_convergence`，不叫
“validated rate”。端点漂移不是整条轨迹的最大能量误差，用户仍需做
时间步和轨迹长度测试。

## 5. 只给 TS 的准备流程

如果显式提供两个片段和成断键，使用这些定义。否则 `prepare --pilot`
或实际 `run` 会执行数值准备：

1. 检查 TS 处最大力，默认要求不超过 `1e-3 hartree/bohr`。
2. 以默认 `1e-4 bohr` 位移中心差分物理力，构造对称 Hessian，按真实
   原子质量加权，并寻找恰好一个明显负特征值。
3. 沿不稳定模式双向扰动，进行质量加权、能量下降的短轨迹。
4. 用有限集合的共价半径和默认 1.25 倍阈值建立端点连通图，识别两片段
   入口及对应成断键。只有元素标签和连接图同构时才视为等价方向。
5. 如入口方向、反应通道或负模数存在歧义，输出候选和诊断，要求显式
   反应定义；不自行生成一个看起来合理但不可审查的速率问题。

这些是准备工具，不是成熟量子化学程序的收敛 IRC。Hessian 步长、低频
模式、结构质量、半径图的化学适用性和下降轨迹终点都应检查。某些元素
没有自动半径，复杂重排也可能不满足成断键一一配对；这时应显式输入。

`examples/rate/toy_exchange.py` 是新写的双 Morse 加三体耦合教学势，具有
两个自由双分子渐近区和一个可识别的线性 saddle。H 标签用来演示质量
与连接图，**不是拟合的 H3 PES，也不是物理 H + H2 速率基准**。
`minimal.inp` 演示仅指定 TS 和 PES 的入口，`pilot.inp` 用短参数运行整个
PMF/parent/child 流程。自动准备和全流程均已实际运行。

## 6. 直接碰撞 dRPMD

dRPMD 先热化两个反应物的内部环聚合物自由度，然后对整个分子的全部
珠子施加一致的刚体转动和平移，构造相对碰撞运动。固定的是片段的
centroid COM，不是每个珠子的 COM；环内部模没有因此被删除。
撞击参数按面积均匀抽样 `b=bmax*sqrt(U)`，方向各向同性。

默认 `thermal_flux` 抽取

\[
E\sim\operatorname{Gamma}(2,1/\beta),\qquad
k_c=\pi b_{max}^2\sqrt{8/(\pi\beta\mu)}\,\langle I_c\rangle.
\]

这个能量分布已经包含碰撞通量的速度加权，不能再给事件额外乘一个
速度。另一个 `maxwell` 模式抽取 `Gamma(3/2,1/beta)`，对应事件估计量
才是 `pi*bmax^2*v*I_c`。`fixed_energy` 模式输出该能量下的截面，不将它
冒充热速率。

事件结果保留 product、nonreactive、unresolved 和 failed。未完成或失败
的事件不被当作无反应；有未知结果时报告观测下界和可能上界。
默认 `bmax='auto'` 的 R0/4 只是一项 pilot 起点，其速率点估计会被扣留。
显式输入 bmax 也不是收敛证明：必须增加 bmax、入口距离和最大传播时间，
检查截断贡献和 unresolved 比例。通道定义和持续性阈值也必须符合反应。

## 7. 并行、恢复和可以复现的内容

每个窗口/replica、parent 链和 child pair 有固定的逻辑 ID。种子由基础
seed 和这些 ID 派生，任务调度顺序不重分配随机序列。checkpoint 保存
`q,p,step,time`、随机数发生器完整状态以及已采集样本。
窗口和 parent 按 checkpoint 间隔保存；child 以完成的一对轨迹为恢复粒度，
被中断的 pair 从其保存的 parent 和固定随机种子重新传播。
恢复核对物理配置；改变 resume 标志、输出路径和并行布局不会改变同一
条随机序列。FES 的 `analysis.measure` 会改变采样 Hamiltonian，明确
属于物理指纹；bins 和 bootstrap 属于纯后处理。

进程和 MPI 以独立轨迹/窗口为外层任务；一个组内还可按珠子分配 PES
batch。一个积分步的力必须属于该次坐标请求，组内汇合后才继续积分。
MPI 工作流所有 rank 进入相同阶段，仅 coordinator 做汇总分析和总文件
写入。当前测试包含本机四 rank 的 G=2、B=2 布局；这验证 MPI 组和集合
操作，但不代表已经在真实跨物理节点网络或调度系统上验证性能。
不同 PES 的浮点归约顺序和外部实现可能影响最后几位；可重启不等于对
任意外部 GPU/MPI PES 承诺全平台逐位一致。

## 8. 已做的验证与仍需做的科学验证

| 范围 | 自动/集成检查 | 不能由此推出 |
|---|---|---|
| CV 与 bias | centroid/bead-average 全 Jacobian、xi Hessian、metric 力的有限差分；不同 P 的 umbrella 缩放 | 任意用户 CV 的正确性 |
| 一维/二维统计 | 已知二维高斯 umbrella 的解析 harmonic FES；非等宽 bin 密度；断连 mask；自相关诊断 | 真实复杂体系的遍历性 |
| 约束动力学 | 非线性距离面上的位置/速度约束与正反传播可逆性 | 任意大时间步都准确 |
| 标准速率 | 真实 paired child、完整 PMF/parent/child 管线、教学 TS 自动推断 | 与已发表化学反应速率相符 |
| 恢复 | 完成与恢复后的样本、最终状态和 kappa 一致；改变物理设置拒绝 | 间接 Python 依赖或远程 PES 永久不变 |
| 并行 | 串行/进程/MPI 流程与组内珠子分配；四 rank workflow/resume 集成脚本 | 真实多节点、所有 GPU/PES 的扩展效率 |

发表或生产计算仍需逐项验证 P、dt、Rinf/bmax、窗口数/间距/刚度、每窗口
采样长度、独立 replica、parent 相关性、child 平台时间、PES 和反应通道。
PES adapter 的软件 force-check 能发现单位、符号和导数错误，但不能证实
一个机器学习或经验势能面对目标化学反应的物理精度。

## 9. 参考与代码来源

- [RPMDrate](https://github.com/GreenGroup/RPMDrate)：MIT；本项目核对了用户仓库
  参考版本 `19ba2602bad5b2145b0d0ba0716d7ba3d90a1003` 的 `_surface.f90`、
  `_main.f90` 和 `main.py` 中坐标、flux metric、umbrella integration 和
  paired recrossing 估计量。具体保留的版权声明见 `THIRD_PARTY_NOTICES.md`。
- [Caracal](https://github.com/Trebonius91/Caracal)：MIT；核对版本
  `7f95ce9818b7f6d81d7cb3b634f360373c090820` 的 `src/umbrella.f90` 和
  `src/recross.f90`，包括其 parent/child 任务组织。没有移植失败时设置
  默认 transmission coefficient 的回退行为。
- Shirts and Chodera, “Statistically optimal analysis of samples from multiple
  equilibrium states,” *J. Chem. Phys.* **129**, 124105 (2008),
  [doi:10.1063/1.2978177](https://doi.org/10.1063/1.2978177)：MBAR 方程来源。

本实现参考并保留必要 MIT 归属，同时重写了 Python CV、二维统计、
约束 Cayley-RATTLE、任务并行和数据恢复接口；它不是对整个参考程序
直接改名，也不声称已与每个参考实现的全部科学输出逐项回归一致。
