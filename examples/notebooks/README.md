# PyRPMD Jupyter Notebook 交互式案例

本目录提供四个涵盖 **势能面（PES）**、**化学反应速率（Rate）**、**固体与周期性材料计算（Solid-State Materials）** 以及 **GPU 加速与 PyTorch 张量动力学（GPU Acceleration）** 的可执行 Jupyter Notebook 案例。

所有 Notebook 均采用解析模型或仓库自带示例输入，所有运行产物均写入 `tempfile.TemporaryDirectory()` 临时目录，不会修改仓库中的被追踪文件；每个 Notebook 在单核 CPU 上均可在数秒内完整运行。

## 案例索引

| Notebook | 核心模块 | 涵盖内容 |
| --- | --- | --- |
| [`01_pes_potentials.ipynb`](01_pes_potentials.ipynb) | **PES 势能面** | 内置解析势与 C++ 原生绑定、`ToyExchange` 反应势与有限差分力检验（`check_potential`）、C ABI 共享库编译与调用（`CABIPotential`）、环聚合物收缩（RPC `SplitPotential` / `DifferencePotential`） |
| [`02_reaction_rate.ipynb`](02_reaction_rate.ipynb) | **Rate 反应速率** | 过渡态（TS）鞍点与通道检验（`prepare`）、伞形抽样 PMF（UI 与 MBAR 估计器对比）、分割面约束母链与回穿透射系数 $\kappa(t)$、全自动速率配置（`rate.auto = true`）与低温环聚合物瞬子（Instanton）隧穿速率 |
| [`03_solid_state_materials.ipynb`](03_solid_state_materials.ipynb) | **固体与周期性材料** | 周期晶胞平衡 PIMD（原初与质心维里动能估计量、核量子离域回转半径 $R_g^2$）、固定基底表面板层模型（`fixed_atoms` Slab）、各向同性恒压系综（BZP NPT）、周期固体质心扩散诊断（`diffusion`）与晶格位点跃迁速率（`site_rate`） |
| [`04_gpu_acceleration.ipynb`](04_gpu_acceleration.ipynb) | **GPU 加速与张量后端** | 自动设备选择（`cuda` / `cpu` `float64`）、设备驻留张量动力学（`TorchPES`、`TorchState`、`TorchDynamics` 与 `cayley-bcocb` 采样）、可微周期性 PyTorch 势能面、张量材料工作流（`sampling.engine = "torch"`）与原子级检查点续算、跨轨迹批量并行吞吐量测试 |

## 运行方式

### 1. 交互式运行

```bash
pip install -e ".[test,plots,gpu]"
jupyter lab examples/notebooks/
```

### 2. 自动化测试与验证

仓库测试套件包含针对全部 Notebook 的结构与端到端执行测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_notebooks.py
```

> **科学边界说明**：Notebook 中的短时运行参数仅用于演示软件接口、工作流阶段与物理诊断门禁，不代表物理收敛的速率或材料性质预测。实际研究计算请遵循 [`docs/convergence.md`](../../docs/convergence.md) 进行珠子数、时间步长、窗口重叠与采样时长的系统收敛检验。
