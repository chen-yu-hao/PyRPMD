# 开发、验证与维护手册

## 源码结构

| 路径 | 职责 |
| --- | --- |
| `src/pyrpmd/input.py`, `units.py`, `system.py` | 严格输入、单位、分子/周期身份 |
| `src/pyrpmd/runner.py`, `cli.py` | Python 与 CLI 公共入口、工作流分派 |
| `src/pyrpmd/dynamics.py`, `native/` | 环积分器、热化、约束、C++17/OpenMP |
| `src/pyrpmd/potentials/` | provider 生命周期、单位与 batch 契约 |
| `src/pyrpmd/cv.py`, `bias.py` | CV/Jacobian、偏置和测度 |
| `src/pyrpmd/workflows/` | rate、FES、材料、表面、PCET 等工作流 |
| `src/pyrpmd/analysis/` | 自由能、扩散和统计分析 |
| `src/pyrpmd/pcet.py`, `pcet_input.py` | 局域 PCET 数值核与模型输入 |
| `src/pyrpmd/nonadiabatic.py`, `potentials/diabatic.py` | 实验 NRPMD 映射变量、矩阵势与 MInt 传播 |
| `src/pyrpmd/checkpoint.py`, `io.py` | 原子提交、校验与科学身份 |
| `src/pyrpmd/parallel/` | 任务进程、珠子池、MPI 与 GPU 放置 |
| `examples/`, `tests/`, `benchmarks/` | 可运行使用例、回归、来源固定的体系验证 |

修改公共接口时同时更新文档、相关示例与真正检验数值/状态行为的测试。
采样测度、力、约束、随机流或恢复身份的更改必须有相应科学依据与回归。

## 本地回归

```bash
python -m pip install -e '.[test,ase]'
python -m pytest -q
git diff --check
```

原生测试需要构建扩展；没有可选依赖而跳过的测试不算通过。
Socket 测试需要本地 loopback 权限。MPI 环境中进一步执行：

```bash
mpiexec -n 4 python tests/mpi_integration.py
mpiexec -n 4 python tests/mpi_workflow_smoke.py runs/mpi-workflow-check
```

材料和表面专用 MPI 脚本、真实模型基准及其环境设置见各自验证记录。
测试在单机多 rank 上通过不自动证明跨节点性能或真实 CUDA 路径。

## 包构建

```bash
python -m pip install build
python -m build
```

检查 sdist 包含源码、原生头文件、手册源文件、示例和来源说明；wheel 包含当前平台
编译扩展及公开 ABI 头文件。安装构建出的 wheel 到独立环境并离开源码目录验证，
可避免 editable install 掩盖漏打包模块。wheel 文件名与 Python/平台有关，安装时选择实际生成的文件。

## 手册的结构与构建

手册有中英文两个版本，各自是一个 Sphinx 项目：

| 版本 | 入口与章节 | 配置 |
| --- | --- | --- |
| 中文 | `docs/index.md`、`docs/manual/*.md` | `docs/conf.py` |
| English | `docs/en/index.md`、`docs/en/manual/*.md` | `docs/en/conf.py`（复用中文配置，仅改语言与标题） |

两者共享 `docs/_static/manual.css` 与 `docs/_templates/language-switcher.html`。
采用 [Sphinx](https://www.sphinx-doc.org/)、[MyST](https://myst-parser.readthedocs.io/) 和
[PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)。
组织方式参考 [PySCF](https://pyscf.org/user/index.html) 的 Quickstart / User Guide / API / Contributor Guide，
内容依据本项目实现，保留其自身命令和对象约定。

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/en docs/_build/html/en
python -m sphinx -W --keep-going -b singlehtml docs docs/_build/singlehtml
python -m sphinx -W --keep-going -b singlehtml docs/en docs/_build/singlehtml/en
```

普通 HTML 提供导航、搜索和独立章节；singlehtml 将正文集中到一页。英文版必须输出到
中文版目录下的 `en/`，导航栏语言切换才能找到对应页面。
生成文件在忽略目录，不提交到源码。GitHub 的 **Documentation** 工作流执行两种语言的严格构建与
入门示例 smoke，并上传 `pyrpmd-manual-html` 构建产物。

两种语言保持同一章节文件名、标题层级、示例命令与表格内容。修改一章时在同一提交中更新
另一语言的对应章节；两版之间的跨语言链接使用 GitHub 完整路径，因为每个 Sphinx 项目
只包含自身语言的文件。

专题规范和历史验证仍留在原 `docs/` 路径，通过仓库链接引用。
Sphinx 只把当前综合手册纳入导航，避免将针对旧提交的实验记录混作当前教程。
源码分发保留全部这些原始证据。

## 写作与验证约定

每个教程应明确：所解决的问题、可直接运行的输入、必要环境、输出路径和单位、
关键参数，以及该示例不能证明的物理结论。真实代码用 `literalinclude` 包含；
新增 API 签名用 autodoc 从源码生成，避免手写签名随版本漂移。

公共例子至少能通过当前输入解析。关键入门例应执行完整 run、读取结果和完成后 resume。
文档链接分两类：当前手册内部使用相对 `.md`；历史报告、示例源码和既有规范使用
GitHub 完整路径。示例中的占位符必须明确标记。

改变周期几何、固定原子支持或模型身份时，先比较旧 checkpoint 的语义；
可以迁移的状态应先验证旧身份，无法无歧义解释的状态应明确拒绝。
不能仅因为数组形状相同就接受物理意义变化后的恢复。

## 来源、许可与科学证据

项目许可证和第三方通知分别见
[LICENSE](https://github.com/chen-yu-hao/PyRPMD/blob/main/LICENSE)及
[THIRD_PARTY_NOTICES.md](https://github.com/chen-yu-hao/PyRPMD/blob/main/THIRD_PARTY_NOTICES.md)。
改写或移植 RPMDrate 等来源时保留来源 commit、作者和许可。
外部电子结构程序、机器学习模型和 EAM/PES 数据有各自的许可与适用域。

在提交说明中报告实际测试结果、跳过项、硬件和已知限制。
代码回归、独立数值对照、有限样本体系验证与最终物理预测是不同层次的证据。
不要把历史分支的统计结果换成新提交的结果，也不要将模型 pilot 写成收敛材料常数。

