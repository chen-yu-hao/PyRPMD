# 安装与环境

## 从源码安装

从 Git 仓库把 PyRPMD 安装到新的虚拟环境中，然后检查编译得到的能力：

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

PyRPMD 需要 **Python 3.11 或更高版本**、NumPy ≥ 1.26、SciPy ≥ 1.11 与 h5py ≥ 3.10，完整依赖见
[`pyproject.toml`](https://github.com/chen-yu-hao/PyRPMD/blob/main/pyproject.toml)。安装时会编译 C++17
扩展，Linux 上启用 OpenMP。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。

`pyrpmd info` 输出 JSON，关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `native.available`、`native.openmp` | C++ 扩展与 OpenMP 已加载 |
| `native.integrators` | 已编译的积分器：`cayley-bcb`、`normal-mode-bab`、`velocity-verlet` |
| `workflows` | 11 类可用 `task`（`rate`、`drpmd`、`fes`、`network`、`metadynamics`、`opes`、`pcet`、`pimd`、`diffusion`、`site_rate`、`surface_rate`） |
| `enhanced_sampling`、`initial_ensembles` | 增强采样方法与选态初始制备（`thermal_parent`、`ne_rpmd`、`qct`） |
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

保留 C++ 内核、不使用 OpenMP：

```bash
PYRPMD_NO_OPENMP=1 python -m pip install -e .
```

这两个变量在构建时生效。`PYRPMD_NO_NATIVE=1` 同时禁用 C++ PIMD 驱动（`sampling.engine="native"`、`pyrpmd-native`），此时它们会明确报错而不回退。更新源码（尤其是原生接口）后重新安装，使 Python 始终加载与源码一致的扩展；
`pyrpmd info` 显示实际加载的内容。

运行时由两个输入设置控制编译代码：`dynamics.native="required"` 要求编译的积分内核；
`dynamics.loop_backend="native"` 要求整个传播循环走融合原生路径。自定义 Python 势或周期/固定原子体系
在 Python 外层循环中使用编译的积分子步骤，见[执行与并行](execution.md)。

## 可选依赖

| 安装选项 | 启用功能 | 另需 |
| --- | --- | --- |
| `python -m pip install -e '.[ase]'` | ASE calculator 势能 | calculator 的模型或程序 |
| `python -m pip install -e '.[mpi]'` | MPI 轨迹组与珠子分块 | 与 mpi4py 匹配的 MPI 库和启动器 |
| `python -m pip install -e '.[gpu]'` | Torch 势能 | 与节点驱动匹配的 PyTorch/CUDA 及模型 |
| `python -m pip install -e '.[analysis]'` | PyMBAR 对照与 matplotlib | 无；常规 FES 使用内置 MBAR |
| `python -m pip install -e '.[plots]'` | `pyrpmd report` 与 `--report` 出图 | matplotlib |
| `python -m pip install -e '.[mace]'`、`'.[nequip]'`、`'.[fairchem]'`、`'.[deepmd]'` | 现成的 ML 势能 provider | 模型检查点 |
| `python -m pip install -e '.[tblite]'`（别名 `xtb`）、`'.[pyscf]'` | 即时 xTB 与 HF/DFT provider | 无 |
| `python -m pip install -e '.[pesgen]'` | [势能生成](pes-generation.md)中的 PySCF 参考数据 | 可选原生 Caracal 工具；JaxPIP 有单独固定的依赖栈，见 [JaxPIP](jaxpip-generation.md) |
| `python -m pip install -e '.[test]'` | pytest 测试 | 缺少可选接口时相应测试跳过 |
| `python -m pip install -e '.[docs]'` | 构建本手册 | Sphinx、MyST、PyData 主题 |

MPI 启动器与 MPI 库应来自同一环境；在集群上先加载站点 module，再安装 mpi4py。按驱动选择 PyTorch 的
CPU 或 CUDA wheel。BKMP2 基准还需要 GNU Fortran，其构建脚本会下载第三方势能源码并核对校验值。

## 验证安装

在仓库根目录执行：

```bash
python -m pytest -q
pyrpmd validate examples/manual/quickstart.inp --check-pes
python examples/manual/low_level.py
```

只安装 `.[test]` 时，依赖 ASE、BKMP2、EAM 模型、PyMBAR、Torch 或 matplotlib 的测试会跳过，其余测试应全部通过。
随后运行[快速入门](quickstart.md)，检查输出、分析与续跑。

## 构建本手册

手册有英文与中文两个版本，章节一一对应。把中文版构建到英文版输出目录下的 `zh/`，语言按钮才能互相链接：

```bash
python -m pip install -e '.[docs]'
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b html docs/zh docs/_build/html/zh
```

打开 `docs/_build/html/index.html`（英文）或 `docs/_build/html/zh/index.html`（中文）。构建包含从当前源码
生成的 API 参考，`-W` 把警告视为错误，无需外部势能或长轨迹。发布与开发说明见[开发者指南](development.md)。
