# 下一版本（未发布）

本说明汇总[原 0.2.0 发布记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/releases/0.2.0.md)
之后、截至 2026-11-03 的 `main@ceee59c` 已合入改动，以及下述文档整理。
安装包版本仍为 0.2.0；本说明没有指定新版本号、标签或发布日期。

## 找到正确入口

中英文 README 聚焦安装、可运行的首次计算、能力表与文档链接。详细限制和证据移入对应的
[能力章节](../manual/capabilities.md)。[手册索引](../index.md)覆盖新合入的精度控制、张量 PIMD、
实验性 f-QCMD 与批量 ML 势；[参考文献目录](../manual/references.md)区分已实现方法与相关研究，
并记录原始论文。

精度控制指南继续纳入 Sphinx，现已加入正式目录。文档 CI 检查 README 大小与章节对应关系、
中英文章节集合、本地文件及锚点、外部链接。保留原有指南文件名，公共入口不再显示内部规划标签。
下列未提供中文全文的技术指南均明确标为英文。

## 动力学、势能与量子统计

- **原生传播：** free、harmonic、Morse-pair、double-well 以及兼容的固定晶胞 periodic-cosine
  势具有原生绑定。共享 CV/偏置/度规内核与解析 Hessian–向量积去除重复实现和稠密度规计算。
  见[内置势（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/builtin-native-potentials.md)与
  [传播栈（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/propagation-stack.md)。
- **张量工作流：** `sampling.engine="torch"` 以单个 float64 设备批次运行固定晶胞 `pimd` 和
  `diffusion`，原子检查点保存状态与 RNG。不支持的 NPT、约束、交换或并行 worker 组合显式报错。
  见[工作流契约（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/torch-materials-workflows.md)。
- **势能 provider：** 批量 MACE、NequIP、fairchem、DeePMD、metatomic 支持批次和张量接口；
  OpenMM 有持久 Context 与 RPMD copy-bank 评估路线；LAMMPS 有珠子亲和的持久实例与可选私有 MPI
  通信器。框架版本、限制和实测见[批量 ML（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/batched-ml-potentials.md)
  与[力引擎性能（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/t19-performance.md)。
- **交换统计：** 费米可观测量通过玻色采样的符号重加权获得，并检查平均符号。周期交换可选高斯镜像求和
  `full_winding`，或显式近似 `minimum_image`；各向同性 NPT 使用独立的全珠子 BZP 路线。
  尚无超流估计量，也没有解决符号问题。见[路径积分作用量（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-path-integral.md)。
- **GLE/PIGLET：** 导入带单位与来源信息的 gle4md 风格参数，或生成已支持的谐振目标参数族；拟合目标与残差进入
  身份记录。[凝聚相量子统计（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/condensed-phase-nqe.md)
  说明估计量、恒压路线和独立对照。

## 速率、采样与光谱

- **精度驱动速率：** `pyrpmd precision plan`、`run`、`status` 以 PES 帧预算执行冻结的热速率设计，
  保存可验证的执行回执，处理恢复与重放。统计精度、数值等效性和物理有效性分别判断。首版执行器支持本地串行固定伞窗口；
  见[精度控制（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/rate_precision.md)。
- **瞬子：** Hessian provider、Bofill 优化和 `pyrpmd instanton` 支持速率与劈裂收敛扫描。
  正常区 golden-rule 速率使用两个束缚/受约束非绝热面及常数弱耦合；预因子仍为稠密实现，最多 4096 个活动坐标。
  见[瞬子（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/instanton.md)、
  [归一化（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/low-temperature.md)和
  [相位投影（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tunnelling-phase-projection.md)。
- **路径采样：** RETIS 增加 minus 系综和内部通量估计；REPPTIS 与直接 FFS 提供 Python API。
  通用 TOML 工作流和 infinity-RETIS 不在本次交付中，见[路径采样（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/transition-interface-sampling.md)。
- **增强采样：** ABF/eABF、steered 功与自由能差、有限温度 string、学习型 CV 导出和预先声明的 PLUMED 文件重启，
  分别见[增强采样（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/enhanced-sampling.md)、
  [string（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/finite-temperature-string.md)、
  [mlcolvar（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/mlcolvar.md)和
  [PLUMED（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/plumed.md)。
- **实验性 f-QCMD：** 弯曲三原子分子的准质心、条件采样、IBI 修正及单珠子生产接入现有光谱流程。
  选定的实现与模型边界见[f-QCMD（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/qcmd.md)。
  成对物理光谱验证和 h-CMD 仍待完成，本说明不宣称已解决曲率问题。
- **态选择 QCT：** 显式 `j0_energy_policy="preserve_actions"` 策略在去除角动量时保留量子化作用量；
  历史能量策略仍可使用。见[态选择准备（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/state-selected.md)。

## 非绝热动力学与模型开发

经典 **IESH** 与已实现的**多态 MASH** 提供单珠子工作流、检查点和独立小体系对照，
不表示已实现对应环聚合物版本。见[IESH 与多态 MASH（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/nonadiabatic-iesh-multistate.md)。

**PySCF SA-CASSCF** 桥提供能量、梯度、NAC 和重叠；**spin–boson HEOM** 提供需要单独收敛的参考方法。
[电子结构指南（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-nonadiabatic.md)说明其边界与可选依赖。
可恢复的**主动学习循环**连接采样、参考标注与训练，另有可选 CPU MACE 参考训练器；
见[模型生命周期（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/tier2-ml-lifecycle.md)。

## 兼容性与验证

既有默认引擎和科学身份保持不变。显式引擎选择有各自的状态与重启契约，legacy、native、Torch 检查点不能互换。
legacy/native 引擎收敛任务合并后，才能关闭这一轮最终文档整合。

原生发布、认证归档路径和 Windows JSON 替换重试的可移植性修复已经合入。对历史上未记录盘符路径插件或模型哈希的
Windows 旧运行，在续算前拒绝执行；不能用今天的源文件冒充缺失的历史身份。见
[可移植持久化（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/portable-persistence.md)和
[Windows 兼容策略（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/development.md)。部分 Windows 跳过登记仍待单独清理。

集成还修正了方差抵消的接受判断、原生 double-well 的 FMA 抵消、PLUMED HILLS 格式和进程测试的时序假设。
[集成路线记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/roadmap.md)保留精确源码版本和测试结果。
本轮文档整理不改变数值内核或检查点格式。

## 仍需补充的证据

已发布 CPU 耗时仅适用于记录的硬件、模型与批次尺寸。本轮性能验证**尚无 CUDA 实测**；
[报告（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/t19-performance.md)提供可运行驱动并明确这项限制。
LAMMPS 持久实例并非在所有配置下都更快。

实时 QCMD、h-CMD、infinity-RETIS、VES、环聚合物 IESH/多态 MASH 与 TDDFT NAC 仍属独立工作。
单元测试或小型解析示例通过，不等于分子速率、光谱或材料性质已物理收敛。
