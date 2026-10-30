# 周期 RPMD 核心复用与效率验证（2026-10-30）

基线为 `af90f20d05368fc910c493977cf29856300b5864`。本次修改覆盖固定晶胞几何、
材料传播中的势能复用、回穿统计和 PMF 重加权；完整回归输出见
[`pytest.txt`](pytest.txt)。这些检查验证实现与恢复契约，不代替真实材料的物理收敛研究。

最终完整测试：**755 passed、13 skipped、146 subtests passed**，用时 103.58 秒。
此前的针对性测试与该结果有重叠，不累加计数。

## 实现和可核对的收益

| 部分 | 修改 | 验证 |
| --- | --- | --- |
| 周期距离 | 值/梯度不再调用稠密 Hessian；旧导入路径保留为别名 | 1,000 原子、1 珠子的普通值/梯度调用，开发环境 `tracemalloc` 峰值约 72 MB 降至 74 KB；显式 Hessian 的有限差分检查通过 |
| 固定晶格 | 周期距离、键差、角度共用不可变 `FixedCellGeometry`，缓存正交性/QR | 所有 8 种 PBC 组合、正交及倾斜晶胞与大镜像偏移的 1,280 次基线对照逐位一致；另有独立枚举最近像测试 |
| 仿射 CV | surface height/lateral、projection/fractional 共享值/梯度/Hessian 基础实现 | 保留质量权重、origin、fractional 未归一化倒格矢和两种 bead estimator；覆盖导数和单位 |
| 材料 PES | 每步事务内复用相同坐标的物理能量、力和 virial | Python 确定性传播加保存帧观测量的 provider 调用由 5 次减至 2 次；PILE-L 对应步骤由 3 次减至 2 次；三种确定性积分器的 q/p/观测量逐位一致 |
| 材料导出 | 样本任务不再先导出一遍全轨迹，由分析统一导出 | 完成恢复、观测失败回滚、离线分析继续覆盖相同输出契约 |
| 扩散数组 | 验证全输入后只复制被测原子/轴；参考 COM 在选出的数据上扣除 | C 连续和 strided 数据对照相同；Fortran 布局的独立对照最大舍入差约 `1.6e-14`，相对约 `9.7e-15`；输入不被修改 |
| surface MBAR | 前置 FES 向 surface PMF 传递可验证的权重快照 | 初始点估计少解一次；6 次 bootstrap 的 solve 调用由 7 次减至 6 次，完整结果逐位一致；不同 bins/min_count 仍独立重建 |
| site MBAR | bootstrap 预计算一次 reduced-potential 矩阵，重抽样列 | 5 窗口 × 每窗口 1,500 样本、12 次 bootstrap 的矩阵构建由 12 次减至 1 次；12 个 QTST draws 逐位一致 |

上述内存与调用次数是指定路径的测量，不是整个材料模拟的统一加速倍数。
PES 缓存不跨事务；不改变 provider 的进程资源寿命、原生绑定或 checkpoint 内容。
另以编译 C ABI harmonic provider 强制三种 native 积分器对照，q/p/量子观测量一致，
`close()` 及关闭后拒绝访问的行为保持一致。

## 统计正确性

`analysis/recrossing.py` 共用母构型聚合、母链及链内相关块 bootstrap 和平台诊断，
保留工作流原有随机命名空间及 draw 顺序。两母链 × 三母构型 × 两对合法 `±p`
子轨迹的基线 κ draws、κ 标准误、联合速率误差和区间保持相同；基线 draws 已固化
为回归测试，而不是在测试中复制一份新实现。

纠正的边界情况包括：

- 单母构型不能因生成 100 个 bootstrap draws 而得到“零误差”或完整联合区间。
- 零通量母构型和母链不增加有效支持；零通量 bootstrap draw 被拒绝时，主误差留空。
- 单母链至少两个有效相关块才有条件误差，并标明
  `conditional_on_single_chain_mixing`。rate/surface 保留原有允许充分单链的政策；
  site 主误差继续要求至少两条独立非零通量母链。
- 所有新增链级诊断使用 JSON 稳定的字符串键，首次结果与保存/恢复结果一致。
- site PMF 逐 draw 记录 MBAR 不收敛、重叠断开或盆地支持不足，不只留下 draw 总数。

新增 PIMD `sampling_diagnostics` 报告统计低效因子、有效样本数、相关块和条件标准误。
至少四个完整块才提供该误差，已有副本间误差保持不变。扩散
`nve_energy_diagnostics` 使用 `β[H_P(t)-H_P(0)]/P`，只检查保存的无恒温生产帧；
超过 `0.1 kBT`/珠子时添加 warning，不改变 MSD 候选判定，不认定物理收敛。

## 接口与恢复

1. 周期 CV 的晶胞/PBC 快照不可原地修改；整体赋值 `cv.cell = new_cell` 或
   `cv.pbc = new_flags` 会重新验证并重建缓存。复制和 pickle 后仍保持只读。
2. 可复用的 MBAR 结果绑定完整有序 bias matrix、stream counts、overlap threshold。
   参数改变拒绝复用；数组在 deepcopy、pickle protocol 4/5 后仍不可写，诊断独立复制。
3. `surface_pmf.json` 和 summary 中嵌套 PMF 使用 `format_version=2`，逐样本
   `weights` 移至 `surface_pmf.npz`，位置由 `sample_weights_file` 和
   `sample_weights_key` 指明。低层数组 API 仍返回权重。
4. checkpoint schema、科学指纹、随机种子和逐步提交语义不变；完整测试覆盖周期
   surface/site/materials、气相和 network 的中断恢复。全套测试中发现并修复了新增
   链诊断整数键在 JSON 恢复后变字符串的跨工作流问题。

## 验证范围

运行环境为 Python 3.12、NumPy 2.3.5、SciPy 1.17，已编译本地 C++/OpenMP，
安装 ASE。完整测试使用 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
可选真实 BKMP2、指定 Ni–H EAM 模型、PyTorch 的跳过情况列于原始日志；跳过不算通过。
本机没有 MPI launcher/mpi4py，因此没有运行实际多 rank 流程；已检查进程序列化，
仓库 CI 中现有的 MPI materials/surface smoke tests 仍负责该执行路径。

本轮没有迁移全历史 NPZ checkpoint 到分块存储，也没有合并 site/surface 的完整
母子轨迹执行器。它们涉及恢复格式或约束传播的更大边界，仍是后续可独立推进的优化。
