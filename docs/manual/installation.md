# 安装与环境

## 从源码安装

PyRPMD 的可复现起点是 Git 仓库。下面的命令在一个新的虚拟环境中安装当前检出的源码，
并检查实际加载的计算能力。不要假设同名 PyPI 项目对应这一份源码。

```bash
git clone https://github.com/chen-yu-hao/PyRPMD.git
cd PyRPMD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pyrpmd --version
pyrpmd info
```

需要 **Python 3.11 或更高版本**、NumPy ≥ 1.26、SciPy ≥ 1.11、h5py ≥ 3.10。
默认构建 C++17 扩展；Linux 构建默认启用 OpenMP。版本与依赖以仓库
[`pyproject.toml`](https://github.com/chen-yu-hao/PyRPMD/blob/main/pyproject.toml)为准。
上述 shell 激活命令用于 Linux/macOS；Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。

`pyrpmd info` 会报告加载到的原生能力。完成安装不表示所有外部势能库、MPI、CUDA
和可选接口都已经就绪；按实际使用的功能安装对应依赖。

## 编译器与参考实现

构建器会寻找可用的 C/C++ 编译器，也接受显式的 `CC`、`CXX`。
Linux 上例如：

```bash
CC=gcc CXX=g++ python -m pip install -e .
```

没有 C++ 编译器时，可在干净环境中安装 NumPy 参考路径：

```bash
PYRPMD_NO_NATIVE=1 python -m pip install -e .
```

如果只有 OpenMP 不可用，可保留 C++ 内核而禁用 OpenMP：

```bash
PYRPMD_NO_OPENMP=1 python -m pip install -e .
```

这两个环境变量控制**构建**。已有目录中的旧 `.so` 不会因设置变量自动消失；
需要检查 `pyrpmd info`，或使用新检出与新环境。更新源码尤其是原生接口后应重新安装编译，
避免 Python 与旧扩展混用。

输入中的 `dynamics.native="required"` 要求编译积分内核；
`dynamics.loop_backend="native"` 要求整段传播满足融合原生路径。
它们不是同一个开关。自定义 Python 势或周期/固定原子体系可以使用编译积分子步骤，
同时由 Python 外层循环驱动，详见[执行与并行](execution.md)。

## 可选依赖

| 安装选项 | 用途 | 还需准备 |
| --- | --- | --- |
| `python -m pip install -e '.[ase]'` | ASE calculator 势能接口 | 目标 calculator 自己的模型/程序 |
| `python -m pip install -e '.[mpi]'` | MPI 轨迹组和珠子分块 | 与 mpi4py 匹配的 MPI 库及启动器 |
| `python -m pip install -e '.[gpu]'` | Torch 势能接口 | 与节点驱动匹配的 PyTorch/CUDA 安装和模型 |
| `python -m pip install -e '.[analysis]'` | 独立 PyMBAR 对照工具 | 常规 FES 使用包内 MBAR 实现，不强制依赖 PyMBAR |
| `python -m pip install -e '.[test]'` | pytest 回归 | 可选接口缺失时相关测试会跳过 |
| `python -m pip install -e '.[docs]'` | 构建本手册 | Sphinx、MyST、PyData 主题 |

MPI 启动器与 Python 加载的 MPI 应来自同一环境。集群上先遵循站点的 module/作业规范，
再安装 mpi4py。PyTorch 的 CPU/CUDA wheel 应按实际驱动选择；安装 extra 不自动证明
有可用 GPU。BKMP2 基准另需 GNU Fortran，并通过仓库脚本下载和校验第三方势能源码。

## 验证安装

在源码仓库根目录执行：

```bash
python -m pytest -q
pyrpmd validate examples/manual/quickstart.inp --check-pes
python examples/manual/low_level.py
```

记录通过、跳过和失败三类结果。跳过真实 BKMP2、Torch 或某个 MPI 测试，表示该环境
未验证对应路径。运行完整的[快速入门](quickstart.md)，还可以检查输出、分析与续跑。

## 构建手册

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

用浏览器打开 `docs/_build/html/index.html`。构建包括目录、章节交叉引用和从当前源码导入的
公共 API；`-W` 使文档警告导致失败。无需在生成手册时连接外部势能程序或运行长轨迹。
开发与发布说明见[开发者指南](development.md)。

