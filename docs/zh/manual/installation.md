# 安装与环境

## 从源码安装

从 Git 仓库把 PyRPMD 安装到新的虚拟环境中，然后检查编译得到的能力：

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pyrpmd --version
pyrpmd info
```

PyRPMD 需要 **Python 3.11 或更高版本**、NumPy ≥ 1.26、SciPy ≥ 1.11 与 h5py ≥ 3.10，完整依赖见
[`pyproject.toml`](https://github.com/chen-yu-hao/PyRPMD/blob/main/pyproject.toml)。安装时会编译 C++17
扩展，Linux 上启用 OpenMP。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

上面使用 POSIX shell 语法。Windows 用户请换用 PowerShell 激活命令，并为原生构建配置 C++17 编译器；
仓库的打包 CI 当前覆盖 Linux 与 macOS。使用下面的 NumPy 参考实现时不需要编译器。
若 shell 找不到 `pyrpmd`，可把命令替换为 `python -m pyrpmd`，也能确保它与 `python -m pip` 使用同一 Python 环境。

`pyrpmd info` 输出 JSON，关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `native.available`、`native.openmp` | C++ 扩展与 OpenMP 已加载 |
| `native.integrators` | 已编译的积分器：`cayley-bcb`、`normal-mode-bab`、`velocity-verlet` |
| `workflows` | 11 类可用 `task`，包括 `opes` |
| `input_schema_version` | 当前输入版本（1） |

ASE、MPI、Torch 和外部势能库等可选功能需要各自的依赖，见[可选依赖](#可选依赖)。

## 编译器与参考实现

构建使用系统中找到的 C/C++ 编译器，或 `CC`、`CXX` 指定的编译器：

```bash
CC=gcc CXX=g++ python -m pip install -e .
```

没有 C++ 编译器时，安装 NumPy 参考实现：

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

PowerShell 中先执行 `$env:PYRPMD_NO_NATIVE="1"`，再运行 `python -m pip install -e .`，最后用
`Remove-Item Env:PYRPMD_NO_NATIVE` 移除这个构建设置。

保留 C++ 内核、不使用 OpenMP：

```bash
PYRPMD_NO_OPENMP=1 python -m pip install -e .
```

这两个变量在构建时生效。更新源码（尤其是原生接口）后重新安装，使 Python 始终加载与源码一致的扩展；
`pyrpmd info` 显示实际加载的内容。

运行时由两个输入设置控制编译代码：`dynamics.native="required"` 要求编译的积分内核；
`dynamics.loop_backend="native"` 要求整个传播循环走融合原生路径。默认工作流引擎下，自定义 Python 势或周期/固定原子体系
在 Python 外层循环中使用编译的积分子步骤。单独选择的原生 PIMD 引擎有自己的势能与执行要求，见[执行与并行](execution.md)。

## 可选依赖

| 安装选项 | 启用功能 | 另需 |
| --- | --- | --- |
| `python -m pip install -e '.[ase]'` | ASE calculator 势能 | calculator 的模型或程序 |
| `python -m pip install -e '.[mpi]'` | MPI 轨迹组与珠子分块 | 与 mpi4py 匹配的 MPI 库和启动器 |
| `python -m pip install -e '.[gpu]'` | Torch 势能 | 与节点驱动匹配的 PyTorch/CUDA 及模型 |
| `python -m pip install -e '.[plots]'` | Matplotlib 出图与 `pyrpmd report` 的 HTML/JSON 分析报告 | 保存的 PyRPMD 结果；出报告不需要外部 PES |
| `python -m pip install -e '.[analysis]'` | PyMBAR 对照及绘图依赖 | 常规 FES 使用内置 MBAR |
| `python -m pip install -e '.[test]'` | pytest 测试 | 缺少可选接口时相应测试跳过 |
| `python -m pip install -e '.[docs]'` | 构建本手册 | Sphinx、MyST、PyData 主题 |

MPI 启动器与 MPI 库应来自同一环境；在集群上先加载站点 module，再安装 mpi4py。按驱动选择 PyTorch 的
CPU 或 CUDA wheel。BKMP2 基准还需要 GNU Fortran，其构建脚本会下载第三方势能源码并核对校验值。

## 验证安装

先用内置谐振子入门例子检查完整用户流程。安装后可在任意目录运行，无需可选模型、绘图包或测试包：

```bash
pyrpmd init installation-check
pyrpmd validate installation-check/quickstart.inp --check-pes --output installation-check/run
pyrpmd run installation-check/quickstart.inp --output installation-check/run
pyrpmd analyze installation-check/run
pyrpmd resume installation-check/run
```

保存的摘要应显示 `task="fes"`、3 个窗口、6 条采样流与 120 个样本；已完成任务在 `resume` 后仍为完成状态。
[快速入门](quickstart.md)解释诊断字段，以及为什么短 pilot 不能作为收敛检验。重复检查时请选择另一个入门目录。

开发者可安装 `.[test]`，再从仓库根目录运行 `python -m pytest -q`。
缺少依赖时，可选接口与外部模型的测试可能跳过，请阅读具体原因。编译器、socket、MPI 与真实模型的验证要求见
[开发者指南](development.md)。

## 构建本手册

手册有英文与中文两个版本，章节一一对应。把中文版构建到英文版输出目录下的 `zh/`，语言按钮才能互相链接：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

打开 `docs/_build/html/index.html`（英文）或 `docs/_build/html/zh/index.html`（中文）。构建包含从当前源码
生成的 API 参考，`-W` 把警告视为错误，无需外部势能或长轨迹。发布与开发说明见[开发者指南](development.md)。
