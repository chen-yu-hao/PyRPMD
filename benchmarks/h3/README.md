# 真实体系测试：H + H₂ / BKMP2

这组计算使用 BKMP2 基态 H₃ 势能面，实际传播 16 珠子的环聚合物，检验
dRPMD、二维自由能面、伞抽样 PMF 和 recrossing。它不是裸 PES 的等高线图，
也不把少量碰撞和短时间平台称为已经收敛的反应速率。

势能论文、固定源码、SHA256、精度选项与作者独立数值参考见
[SOURCE.md](SOURCE.md)。默认统一温度 **1500 K**，氢核质量使用
`1.00782503223 u`；原子索引为 0、1、2，原反应物为 H0 + H1H2。

## 运行

需要从仓库根目录运行，安装 PyRPMD 原生内核、GNU Fortran 和图表依赖：

```bash
python -m pip install -e '.[test]' matplotlib
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output runs/h3/pes-check.json
python benchmarks/h3/numerics.py --output runs/h3/numerics.json

python benchmarks/h3/drpmd.py --output runs/h3/drpmd --workers 2
python benchmarks/h3/rpmd.py --stage fes2d --output runs/h3/fes2d --workers 2
python benchmarks/h3/rpmd.py --stage rate --output runs/h3/rate --workers 2

python benchmarks/h3/render.py runs/h3/drpmd
python benchmarks/h3/render.py runs/h3/fes2d
python benchmarks/h3/render.py runs/h3/rate
python benchmarks/h3/report.py runs/h3 --output runs/h3/VALIDATION.md
```

每个新任务使用新的输出目录。dRPMD runner 会自动再打开已完成任务，检查
所有轨迹记录与持续母轨迹时钟是否完全一致；中断续跑可以加 `--resume`。
FES/rate runner 同样支持 `--resume`。不要在运行中修改输入、分类器、势能
源码或二进制；科学指纹会阻止把不同定义混进同一结果。

`fes2d.inp` 和 `rate.inp` 也是可直接使用的用户输入，例如：

```bash
pyrpmd run benchmarks/h3/fes2d.inp --output runs/h3-fes-direct
pyrpmd run benchmarks/h3/rate.inp --output runs/h3-rate-direct
```

这些直接命令产生常规工作流数据；专用 Python runner 另外生成
`benchmark.json`，执行明确的物理/数值门槛，并在门槛失败时返回非零退出码。
`--workers` 只改变执行进程数；珠子数、随机数身份与采样参数在输入中固定。

## 四项计算

### 1. dRPMD

初始 H₂ 采用优化平衡键长。母轨迹热化内部环聚合物自由度，并在持续链上
间隔抽样；碰撞采用热通量能量分布、面积均匀 impact parameter 和各向同性
取向。默认入口距离 `R0=10 bohr`、`bmax=3 bohr`，生产碰撞段没有 thermostat。

分类器按原子身份区分 H0H1、H0H2 两个交换产物与原来的 H1H2。只有一个
centroid H–H 键短于 2.5 bohr、剩余原子离分子 COM 至少 10 bohr、向外运动，
且标签持续满足规定时间，才认为碰撞已经完成。三个游离原子等情况保留为
未决，不计入非反应；势能失败也独立记账。

报告包括反应数、未决/失败数、两个交换通道、每个母轨迹的热化步数、
轨迹能量漂移、碰撞能量和取向统计、最外 impact shell 的反应计数、
IID Wilson 区间与母轨迹 cluster bootstrap。少量独立母轨迹仍不足以
完整表征相关性；`bmax` 之外的贡献也不由一次有限样本自动排除。

### 2. 二维 RPMD 自由能面

两个 CV 为 centroid 距离 `r01` 和 `r12`，单位 bohr；实际二维 umbrella
采样经 MBAR 去偏。输出是相对于 `dr01 dr12` 的概率密度对应的自由能，
不自动减掉距离 Jacobian。未访问/计数不足的区域保持 mask。

检查窗口 overlap 连通性、有效样本数、过渡态邻域覆盖、非零环聚合物
回转半径和 bootstrap；还报告相同氢原子置换要求的
`F(r01,r12)=F(r12,r01)` 在有限采样中的偏离，不施加对称平均掩盖统计问题。

### 3. 伞抽样 PMF

采用 RPMDrate 的入口面/过渡面反应坐标和 flux metric 测度，窗口覆盖
`xi=0` 的入口至 `xi≈1` 的过渡态。PMF 使用 Gaussian umbrella integration，
同时通过 MBAR overlap 检查窗口连通性。保存原始窗口样本、均值、方差、
有效样本数以及 block-bootstrap 势垒误差。

该 runner 还对完整原始样本执行独立的 histogram-MBAR 重分析，使用
50、100、150 个 bin 和 40 次按时间块抽样。UI 与 MBAR 均采用同一个
flux metric 测度、`xi=0` 参考和实际 recrossing 使用的分割面；不因更换
PMF 估计器移动分割面。每次 bootstrap 同时计算两者势垒，保留差值的
协方差。结果写入 `pmf-diagnostics.json`，图中直接叠加曲线。

窗口明显偏离高斯分布时，两种估计器可能给出不同势垒。数值门槛通过
并不抹去这种差异；报告会保留两者和配对置信区间。次级 MBAR 速率沿用
相同的 κ，其置信区间**以 κ 固定为条件**，不包含 transmission 的误差，
不能当作独立物理验证或完整的最终速率不确定性。

两个等价交换出口显式使用：

```toml
[reaction]
fragments = [[0], [1, 2]]
forming_bonds = [[0, 1]]
breaking_bonds = [[1, 2]]
equivalent_paths = [[0, 2, 1]]
symmetry_factor = 1.0
```

因此 `s1=r12-min(r01,r02)`。不再额外乘 2；参考几何随原子标签一起置换。
这修正了只使用一个形成键时，另一交换出口不能被稳定识别的问题。

### 4. Recrossing

使用 Cayley–RATTLE 在分割面上采样母轨迹，并从两个等价面分支初始化。
每个母构型重新抽取未约束的 Maxwell 动量，运行 `+p/-p` 成对子轨迹。
子轨迹不加 umbrella 或 thermostat，记录完整 `kappa(t)`、parent-block
bootstrap 误差与端点能量变化。

检查正初始通量、实际的非平凡 transmission、约束残差、时间尾部平台和
不依赖 PES 能量零点的 `|ΔH_P|/(P*kBT)`。平台通过只说明选定时间区间的
诊断通过，不等于珠子数、长时间传播或统计误差已收敛。

## 独立数值检查

`pes.py --check` 复核作者给出的绝对能量和径向导数，优化 H₂ 与线性 TS，
检查恰一个 Hessian 负模、Cartesian 力的中心差分、平移/转动/原子置换
不变性，以及 batch/scalar 一致性和非法几何护栏。

`numerics.py` 在真实 BKMP2 势上比较 P=1、8、16、32 时的 C++ 与 NumPy
传播，并从相同初态比较 `dt=0.1/0.05/0.025 fs` 的 NVE 能量误差和末态误差。
这里不同 P 使用各自的数值初态，**不构成珠子数的物理收敛研究**。

## 输出与 GitHub Actions

每个阶段的 `benchmark.json` 保存参数、门槛、实测统计和局限性；dRPMD
原始数据在 `raw/`，FES/rate 在 `run/`。图表输出为 PNG 和 PDF；FES 同时
保存 HDF5、NPZ、CSV，PMF 和 recrossing 保存 NPZ、CSV 与 checkpoints。
最终 HDF5 样本与每条流的 checkpoint 必须逐值、等长度一致，才能通过
完整性门槛；仅有正确前缀的中断记录不能冒充完成的 checkpoint。

仓库的 **Real H3 BKMP2 validation** 工作流先运行势能参考值和 NVE 检查，
再并行执行 dRPMD、FES2D、PMF+recrossing 三个作业。最终 artifact
`h3-complete-validation` 包含完整输入、原始结果、图表与 `VALIDATION.md`；
单阶段 artifact 也独立保留，保存期 30 天。下载的第三方 PES 源码和其
二进制不进入这些结果包；其来源和哈希保存在验证报告里。

较长的 FES/rate 采样可选择 `--profile extended`；dRPMD 可显式设置
`--trajectories`、`--parents`、`--beads`、`--timestep-fs`、`--bmax-bohr`、
`--separation-bohr` 等。参数加大本身并不是收敛证明，应比较独立重复、
窗口分布、相关时间、PMF/κ(t) 及最终速率在变化前后的误差。

## 固定 96 窗与增强采样对照

`enhanced_sampling.py` 固定 T=1500 K、P=16、dt=0.1 fs、两个副本和
βK=400。固定窗在 ξ=-0.05…1.10 放置 96 个中心，自适应从 6 个中心开始。
`short` 使用 100 fs 平衡/200 fs 生产，`production` 使用 300 fs/1000 fs；
两者都是有限预算诊断，不运行 recrossing，也不输出已收敛速率。

```bash
python benchmarks/h3/enhanced_sampling.py --method fixed96 --output runs/h3/fixed96 --bootstrap 20
python benchmarks/h3/enhanced_sampling.py --method adaptive --output runs/h3/adaptive --bootstrap 20
python benchmarks/h3/enhanced_sampling.py --method metadynamics --output runs/h3/meta-diagnostic
```

可用 `--library` 指定已校验的 BKMP2 库，用 `--seed` 做独立重复。
每次使用新目录；`config.json`、原始 checkpoint 和 `comparison.json` 保存
完整预算及结果。`--production-steps` 可用于等总步数对照：计入所有 pilot
和平衡步数，根据实际冻结窗数分配剩余生产预算，不能把“每窗一样长”当成
“总成本一样”。输出包含同一固定分割面 ξ*=1.00925 的 MBAR 势垒、按时间块
bootstrap 的误差、逐流 ESS、总积分步数，以及共同支持区的 PMF。

Metadynamics 这一项是**无约束 ξ 的失败模式诊断**：它保持 `cv_density`
测度，不与 `rate_flux` PMF 作数值比较；输出范围不是墙，ξ 在反应区外可能
奇异。更多 hills 或更长时间不能替代正确的 CV/空间限制。实测结果与建议见
[增强采样检查报告](../../docs/validation/enhanced-sampling-2026-10-31/README.md)。
