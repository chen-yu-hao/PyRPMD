# 开发、验证与维护

## 源码结构

| 路径 | 职责 |
| --- | --- |
| `src/pyrpmd/input.py`、`units.py`、`system.py` | 输入检查、单位、分子与周期身份 |
| `src/pyrpmd/runner.py`、`cli.py` | Python 与命令行入口、工作流分派 |
| `src/pyrpmd/dynamics.py`、`native/` | 环积分器、恒温器、约束、C++17/OpenMP |
| `src/pyrpmd/potentials/` | provider 生命周期、单位与批处理约定 |
| `src/pyrpmd/cv.py`、`bias.py` | CV 与 Jacobian、偏置与测度 |
| `src/pyrpmd/workflows/` | rate、FES、材料、表面、PCET 等工作流 |
| `src/pyrpmd/analysis/` | 自由能、扩散与统计分析 |
| `src/pyrpmd/pcet.py`、`pcet_input.py` | PCET 内核与模型输入 |
| `src/pyrpmd/nonadiabatic.py`、`potentials/diabatic.py` | NRPMD mapping 变量、矩阵势与 MInt 传播 |
| `src/pyrpmd/checkpoint.py`、`io.py` | 原子提交、校验与科学身份 |
| `src/pyrpmd/parallel/` | 任务进程、珠子池、MPI 与 GPU 放置 |
| `examples/`、`tests/`、`benchmarks/` | 可运行示例、回归测试、固定来源的基准 |

### 公共工具约定

标量整数、有限性以及正负号和上下界检查统一使用 `validation.py`。整数检查拒绝布尔值和小数；
调用方可以保留本领域的异常类型与错误消息。单位换算和任务专属约束仍由各自的解析器处理。

`io.jsonable` 和 `io.write_json` 默认采用严格的元数据转换；`io.report_jsonable` 和
`io.write_report_json` 把不可用的数值估计表示为 JSON `null`。可调用对象、dataclass 和
`to_dict` 的转换策略用于保留已有表示。通用 NPZ 检查点容器通过 `checkpoint.atomic_savez`
完成原子压缩写入和完整性封装；`save_metadata_npz` 只负责组装 dRPMD 元数据，再委派给该写入器。

运行指纹共用编码器，同时保留已有模式：`io.fingerprint(config, system)` 保留包含分析设置的
manifest 身份；提供 `extras` 时使用采样流身份，排除纯分析与执行设置，同时保留采样测度、
通量轴和 CV 几何约定。FES 与 OPES 使用采样流形式。dRPMD 在 `drpmd_persistence.py` 中组装
带版本的物理参数、父链身份策略和本地依赖摘要，再调用 `io.fingerprint_payload`。修改编码或
模式时，必须明确如何兼容续跑，并提供回归验证。

### 按职责拆分的实现模块

以下路径相对于 `src/pyrpmd/`：

| 模块组 | 职责 |
| --- | --- |
| `input.py`、`input_*.py` | `JobSpec` 加载、覆盖和解析顺序；字段模式、科学选项、路径、运行参数、系统/provider 和任务约束分别实现。 |
| `workflows/drpmd.py`、`drpmd_*.py` | 分派、采样、轨迹传播、出射通道判定、分析和续跑约定。 |
| `workflows/rate.py`、`rate_*.py` | 流程协调、约束几何、选项、PMF 准备、父/子轨迹和统计汇总。 |
| `analysis/fes.py` | FES 重建与既有分析导入入口。 |
| `analysis/mbar.py`、`umbrella.py`、`fes_statistics.py`、`rate_pmf.py` | 分别负责 MBAR/可复用权重，伞形势能/直方图/UI，相关性/bootstrap 统计，以及速率势垒估计。 |

兼容入口通过显式重新导出或小型适配函数，保留既有导入与已序列化的全局对象引用。进程/MPI
工作函数保持可导入；dRPMD 将工厂和持久化回调显式传入工作内核。FES 适配器在调用时解析
既有诊断使用的求解器、重采样和 UI 回调。新模块从实际负责该功能的模块导入依赖，避免反向
依赖兼容入口。

修改公共接口时，在同一提交中更新文档、示例以及检查数值与状态行为的测试。修改采样测度、力、约束、随机流或续跑身份时，
附上科学依据和回归测试。

## 运行测试

```bash
python -m pip install -e '.[test,ase]'
python -m pytest -q
git diff --check
```

原生测试需要编译扩展，socket 测试需要 loopback 权限。被跳过的测试没有运行；安装对应的可选依赖即可覆盖。有 MPI 时另运行：

```bash
mpiexec -n 4 python tests/mpi_integration.py
mpiexec -n 4 python tests/mpi_workflow_smoke.py runs/mpi-workflow-check
```

材料与表面的 MPI 脚本和真实模型基准见各自的验证记录。这些脚本在单机上运行；多节点和 CUDA 运行需要单独检查。

## 打包

```bash
python -m pip install build
python -m build
```

sdist 包含源码、原生头文件、手册源文件、示例与来源说明；wheel 包含编译扩展和公开 ABI 头文件。把 wheel 安装到独立环境
并在源码目录之外测试，避免 editable 安装掩盖缺失的模块。wheel 文件名取决于 Python 与平台，安装实际生成的文件。

## 手册

手册有两个版本，各为一个 Sphinx 项目：

| 版本 | 入口与章节 | 配置 |
| --- | --- | --- |
| 英文（缺省） | `docs/index.md`、`docs/manual/*.md` | `docs/conf.py` |
| 中文 | `docs/zh/index.md`、`docs/zh/manual/*.md` | `docs/zh/conf.py`（复用英文配置，设置语言与标题） |

两者共用 `docs/_static/manual.css` 与 `docs/_templates/language-switcher.html`，使用 [Sphinx](https://www.sphinx-doc.org/)、
[MyST](https://myst-parser.readthedocs.io/) 和 [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/) 构建。
章节结构参照 [PySCF](https://pyscf.org/user/index.html) 的 Quickstart / User Guide / API / Contributor Guide。

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
python -m sphinx -W --keep-going -b singlehtml docs/zh docs/_build/singlehtml/zh
```

HTML 版本提供导航、搜索和分章节页面；singlehtml 把全部内容放在一页。把中文版构建到英文输出下的 `zh/`，语言切换才能
找到每页的对应页面。构建输出被 Git 忽略。GitHub 的 **Documentation** 工作流严格构建两个版本、运行入门示例，并上传
`pyrpmd-manual-html` 产物。

两个版本保持相同的核心章节文件、标题、命令和表格，并在同一提交中更新。扩展指南和验证报告可先提供英文版，并在中文 API
章节中添加简要入口；对应译文发布前，这些指南的语言按钮打开中文手册目录。两版之间的链接使用 GitHub 完整 URL，因为每个
Sphinx 项目只包含自己语言的文件。

## 写作约定

每个教程给出要解决的问题、可运行的输入、所需环境、输出路径与单位以及关键参数。用 `literalinclude` 引入真实代码，
用 autodoc 生成 API 签名，使其随源码更新。

直接、以任务为中心地写作：说明命令做什么、产生什么。每项限制只在读者需要据此行动的地方说明一次，并写成对行为的平实描述。

公开示例须通过输入解析，入门示例须能运行、读取结果并续跑。手册内部链接使用相对 `.md` 路径；验证报告、示例源码和规范
使用 GitHub 完整 URL。示例中的占位符须明确标出。

周期几何、固定原子支持或模型身份改变时，检查旧检查点如何被解释：能验证旧身份的状态予以迁移，其余显式拒绝。

## 来源与许可

项目许可证和第三方声明见 [LICENSE](https://github.com/chen-yu-hao/PyRPMD/blob/main/LICENSE) 和
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md)。改编自 RPMDrate 等来源的
代码保留其来源 commit、作者与许可证。外部电子结构程序、机器学习模型和 EAM/PES 数据各有自己的许可证。

提交说明中报告运行的测试、跳过项、硬件和已知限制。
