# 常见问题与诊断

出现问题时，保留完整错误、输入、`pyrpmd info` 的输出、代码 commit 和运行目录。按以下层次排查：输入 → 模型 →
积分与采样 → 统计分析 → 续跑。

## 安装与命令

| 现象 | 处理 |
| --- | --- |
| 找不到 `pyrpmd` | 激活安装时使用的环境；用 `python -m pip show PyRPMD` 核对 |
| C++/OpenMP 编译失败 | 安装带 OpenMP 的 C++17 编译器，或在干净环境中用 `PYRPMD_NO_OPENMP=1` 或 `PYRPMD_NO_NATIVE=1` 构建 |
| `pyrpmd info` 中缺少某个积分器 | 更新源码后重新安装，避免加载旧的 `.so` |
| pytest 出现跳过项 | 缺少可选依赖（ASE、BKMP2、EAM 模型、PyMBAR、Torch、matplotlib 等）；安装后即可运行这些测试 |
| `native="required"` 报错 | 扩展缺少所需积分器；重新构建，或在新任务中使用缺省的 `native=true` |
| `loop_backend="native"` 被拒绝 | PES、CV、动态偏置或周期/固定原子需要 Python 循环；使用 `auto` |

## 输入与单位

| 现象 | 处理 |
| --- | --- |
| `Output directory is not empty` | 新计算使用新的 `--output`；继续已有计算用 `pyrpmd resume DIR` 或 `run INPUT --resume` |
| `... overrides; beads is not applicable` 等 | 该任务不接受这个覆盖；PCET 只接受 `--temperature` 和 `--output` |
| 未知字段或任务组合无效 | 查看 `--help` 和[输入](input.md)中的 section 表 |
| 找不到 XYZ、factory 或模型 | 输入中的路径相对于输入文件；命令行路径相对于当前目录 |
| 时间不是步长的整数倍 | 改用步数，或选用能整除该时间的步长 |
| 原子相距过远或重叠 | 核对内联 bohr 与 XYZ angstrom，以及 `geometry_unit` / `cell_unit` |
| 原子索引越界 | 索引从 0 开始；坐标、质量、分类器和模型须使用同一原子顺序 |
| 周期 provider 被拒绝 | 使用支持晶胞/PBC 的 provider（ASE、周期 Python 插件、i-PI） |
| `pimd` / `diffusion` / `site_rate` 拒绝固定原子 | 这些任务要求全部原子可动；固定层使用表面 `fes` 或 `surface_rate` |

对外部程序使用 `--check-pes` 前，先启动 i-PI driver 或准备好 calculator 的模型文件。

## 势能与动力学

| 现象 | 处理 |
| --- | --- |
| 力的有限差分检查失败 | 检查力的符号、Å/bohr 与 eV/hartree、原子顺序以及模型的光滑性 |
| `NaN`/`Inf` 或 provider 返回非零状态 | 检查失败构型是否在模型的适用范围内 |
| 能量漂移或环振动不稳定 | 缩短时间步并检查力、约束与偏置；velocity-Verlet 受自由环最高频率限制 |
| 换积分器后结果改变 | 每种积分器有自己的时间步误差；分别对时间步收敛 |
| 固定原子在初始检查点中移动 | 核对输入体系与检查点属于同一个 slab |

恒温 PIMD 的能量在采样中会涨落。无恒温 RPMD 的稳定性、PIMD 能量估计量和环 Hamiltonian 分别检查。

## 自由能、速率与扩散

| 现象 | 下一步 |
| --- | --- |
| FES `disconnected_overlap` | 增加桥接窗口、调整刚度或延长采样 |
| 大量 mask 或空格子 | 降低网格分辨率或增加采样 |
| 没有 bootstrap 区间 | 阅读 bootstrap 诊断中的逐次失败和样本量信息 |
| 回穿没有平台 | 增加独立母链、延长子轨迹并检查分割面 |
| dRPMD 不报告热速率 | 设置显式的 `bmax` 和分类规则，并检查未决/失败轨迹 |
| 扩散系数为 `null` | 阅读 `analysis_status`；把生产段延长到扩散区 |
| PCET 速率显示 0 | 读取 log 速率；速率发生了下溢 |
| Metadynamics 拒绝 bootstrap | metadynamics 重分析使用 `--bootstrap 0` |

## 续跑与文件

`run` 不覆盖非空目录。继续运行用 `resume DIR`，新的科学设置用新目录。科学指纹不匹配表示体系、方法或依赖发生了变化；
容器校验失败表示保存的状态已损坏。

普通几何 CV（`distance`、`angle`、`bond_difference`）作用于展开几何；最小镜像需要显式的 `periodic_*` 类型。早期材料分支中
普通 CV 使用最小镜像，这类运行不能按当前约定续跑，请改用相应的显式 CV 重新开始。

重分析会覆盖派生结果，比较分箱或拟合方案前先保存原摘要和结果表。在并行文件系统上，检查点安全依赖原子 rename 和 fsync；
保留完整的运行目录与环境记录。

## 报告问题

提供最小输入、可共享的势能或模型替代、确切命令、traceback、`pyrpmd info` 的输出、CPU/GPU/MPI 布局、出错阶段和 commit。
图像应包含 mask 区域和诊断。源码结构与测试见[开发者指南](development.md)。
