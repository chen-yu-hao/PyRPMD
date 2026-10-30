# PCET、材料与表面分支整合验证（2026-10-30）

本记录验证三个功能分支共同工作的程序契约，不替代各分支的势能来源、
物理适用性与采样收敛研究。原分支的历史数值记录保持原样。

## 整合来源

基线主分支：`4115580fcd6f54aa5f74613772dc052f32e5481a`。
依次以保留父历史的 merge 合入：

| PR | 分支 | 原始提交 |
| --- | --- | --- |
| #5 | feature/electrochemical-pcet | beb38a19302d49c0739e6b8bb1b1c4261d54ef0a |
| #6 | feat/materials-diffusion | 9c27602ed2d4c5ad442172386dc13ecc29f4b382 |
| #7 | feat/surface-catalysis | b4b5dc489cb537fbbee068bade23ca959e2a3986 |

## 需要实质解决的冲突

- 合并 PCET、PIMD、diffusion、site_rate、surface_rate 的输入字段、CLI 和 runner 分派。
- 同时支持显式 Python 周期 PES 两种契约：`periodic=True` 配合 provider-unit
  `evaluate(q, *, cell, pbc)`；`supports_pbc=True` 配合 canonical-AU `bind_system(system)`。
  保留 ASE 的 D/T 元素映射与物理 virial，以及 i-PI 和周期解析模型。
- 共用有界 QR 最近晶格像算法，保留正交晶胞快速路径；拒绝超出可靠整数范围的位移。
- `cell_unit` 缺省继承几何单位（inline bohr / XYZ angstrom），有 cell 默认全周期；
  生产输入应显式写 `cell_unit` 和 `pbc`。System.pbc 统一为不可变 tuple。
- 普通 `distance`/`bond_difference`/`angle` 使用展开几何。最小镜像需明确
  `periodic_distance`/`periodic_bond_difference`/`periodic_angle`。
- 对合并前含有歧义普通 CV 的周期配置，科学指纹纳入 `explicit-periodic-v1`。
  旧歧义记录拒绝恢复/再分析，不能在相同哈希下更换数学定义；分子与明确周期 CV
  的记录保持原有身份。
- 旧非周期 cell 的两种序列化形式（显式 false PBC / 省略 PBC）保留各自字面身份，
  不宽松忽略科学哈希。真实 FES 完成恢复测试禁止新传播并逐数组核对 checkpoint。
- PIMD、diffusion、site_rate 的估计器仍假设所有原子可动，显式拒绝 fixed_atoms；
  固定层用于 FES/surface_rate，避免将固定自由度错误加入物理动能或速率质量度量。

## 本机实测

Python 3.12；源码安装编译 C++17/OpenMP。三种确定性积分器均可用，见
[capabilities.json](capabilities.json)。

| 检查 | 结果 | 原始输出 |
| --- | --- | --- |
| 整合后完整 pytest | 575 passed, 13 skipped, 146 subtests passed | [pytest-full.txt](pytest-full.txt) |
| 最终科学指纹补丁与新增交互回归，以及相邻 checkpoint/PCET/CV 功能 | 126 passed, 24 subtests passed | [pytest-final-targeted.txt](pytest-final-targeted.txt) |
| 实际 4-rank MPI 材料流程 | PIMD、diffusion、site_rate、精确恢复、离线分析 PASS | [mpi-materials.txt](mpi-materials.txt) |
| 实际 4-rank MPI 表面流程 | 串行与 G=2/B=2 的 27 个数组/checkpoint 文件精确一致；固定层与完成恢复 PASS | [mpi-surface.txt](mpi-surface.txt) |
| 文本/语法 | git diff --check、Python compileall 通过 | 提交前运行 |

完整 pytest 的 13 项跳过：10 项未构建的可选真实 BKMP2、1 项未显式配置的
Ni–H EAM 模型、2 项未安装的 PyTorch。跳过不计为通过。最终 targeted 结果与
完整测试有重叠，不能相加解释为一次更大的完整测试。最后补丁只改变特定周期 CV
的科学指纹，其相邻功能和新回归在 targeted 检查中重测。

本机 MPI 5.0.2 使用如下执行环境；默认 CMA 需要当前容器未实现的
`process_vm_readv`，因此按 MPICH 配置关闭 CMA，并使用 TCP/self。未改变仓库
科学参数、未声称跨节点或 GPU 验证。

```bash
export MPIR_CVAR_CH4_CMA_ENABLE=0
export UCX_TLS=tcp,self
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mpiexec -n 4 python tests/mpi_materials_smoke.py /fresh/output/materials
mpiexec -n 4 python tests/mpi_surface_smoke.py /fresh/output/surface
```

常规 GitHub CI 继续使用所安装的 OpenMPI，并同时保留材料和表面 MPI 测试。
上述短流程验证接口组合与恢复行为，不表明任何真实材料速率、扩散系数或电极
PCET 模型已经达到物理收敛。

## 手册与最终整合树验收

在所有科学修复与手册合入同一工作树后，重新执行完整测试：
**584 passed、13 skipped、146 subtests passed**（81.98 秒），见
[最终测试原始输出](pytest-final-tree.txt)与[实际导入路径/原生能力](final-import.txt)。
13 项跳过仍为未构建真实 BKMP2 的 10 项、未配置 Ni–H 模型的 1 项和缺少 Torch 的 2 项。
本节取代上面的初轮计数作为最终整合测试结论，不把两轮计数相加。

新增综合手册有 1 个首页与 14 个章节，覆盖全部 10 类工作流：

- Sphinx HTML 和 singlehtml 均以 `-W --keep-going` 构建成功。
- 手册的仓库文件链接和生成 HTML 的本地链接均无缺失。
- 入门输入实际 CLI 运行得到 3 个窗口、6 条流、120 个样本；完成后恢复仍为 120 个样本。
- 离线重分析 `bins=8, bootstrap=10` 成功；教学结果保持 `converged=false`。
- 完整低层 API 示例运行 100 步并打印环 Hamiltonian 变化和后端统计。

[手册验收数据](manual-validation.json)同时记录最终科学源码和测试目录的 Git tree 哈希。
页面组织参考 [PySCF 用户指南](https://pyscf.org/user/index.html)和
[Quickstart](https://pyscf.org/quickstart.html)，内容依据本项目源码，不复制参考手册文字。

本地 sdist 与 CPython 3.12 Linux wheel 构建成功，47 个 Python 源模块均与整合源码逐字一致。
sdist 包含 `docs/conf.py`、手册首页、14 篇章节、CSS 和两个入门示例，排除 `docs/_build`。
[包内容核对](final-package-inspection.json)保存本地验证归档的哈希；这些归档先于本节
新增验收记录生成，所验证的科学模块和手册正文保持一致。

独立新环境安装该 wheel，并在源码树外使用 sdist 解包示例执行
`tests/package_smoke.py`：三积分器 native/Python 对照、5 个工作流示例（含全部 18 条件 PCET）、
两种积分器下 dRPMD 实际母/子传播及全部完成后恢复均通过。
额外确认材料、site-rate 和 surface 模块可从已安装 wheel 导入。
具体数据见[已安装包验证](installed-wheel-smoke.json)。
