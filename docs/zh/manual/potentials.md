# 连接势能与扩展插件

## 标量 PES 接口

每个执行进程构造自己的势能对象。每次调用接收一批结构（通常是该进程负责的珠子），返回每个结构的
**物理能量和力**。环弹簧、偏置、约束和珠子缩放由 PyRPMD 加入。

实验性 NRPMD 使用单独的矩阵接口，返回完整 diabatic 矩阵及其正梯度，见[非绝热教程](nonadiabatic.md)。

| 数据 | 形状 | 单位 / 含义 |
| --- | --- | --- |
| 坐标 `q` | `float64[B,N,3]` | bohr，C 连续 |
| 能量 `energy` | `[B]` | 每个结构的总势能，hartree |
| 力 `forces` | `[B,N,3]` | 能量的负梯度，hartree/bohr |
| 状态 `status` | `[B]` | 0 表示成功，非零表示该结构失败 |
| 信息 `messages` | B 个字符串 | 可选的失败说明 |
| Virial | 可选 `[B,3,3]` | hartree，由 provider 实现时提供 |

`B` 是一次调用中的结构数。每个结构独立求值。

## 一个完整的 Python provider

把这个谐振子势保存为 `my_pes.py`：

```python
import numpy as np
from pyrpmd.potentials import PESResult


class Harmonic:
    metadata = {"execution_device": "cpu", "thread_safety": "context_serial"}

    def __init__(self, k=0.002):
        self.k = float(k)

    def evaluate(self, q):
        energy = 0.5 * self.k * np.sum(q * q, axis=(1, 2))
        return PESResult(energy, -self.k * q)

    def close(self):
        pass


def build_pes(k=0.002):
    return Harmonic(k)
```

并在输入中指向它：

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
params = {k = 0.002}
```

把 factory 放在独立的可导入模块中。每个工作进程调用 factory 并持有自己的上下文（GPU 句柄、socket 或 Fortran COMMON
状态等），在 `close()` 中释放。

## 单位与梯度检查

Python、f2py 和 Torch 适配器可以声明模型的单位：

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
units = {positions = "angstrom", energy = "eV"}
derivative = "force"
```

此时模型以 Å 和 eV 工作，由适配器转换。返回正梯度的 Python provider 声明 `derivative="gradient"`。旧 f2py 接口总是
返回正梯度，由适配器取负；Torch 输出总是解释为力，或作为能量由 autograd 得到力。C ABI v1 使用原子单位。

在几个有代表性的构型上对新模型做有限差分，并检查原子顺序、能量零点、电子态和适用范围：

```python
import numpy as np
from pyrpmd.potentials import check_potential
from my_pes import build_pes

provider = build_pes()
try:
    q = np.array([[[0.1, -0.2, 0.3]]])  # 该 provider 的单位为 bohr
    print(check_potential(provider, q, epsilon=1e-5))
finally:
    provider.close()
```

## 适配器

| Provider | 用途 |
| --- | --- |
| `builtin` / `free` / `harmonic` 等 | 解析模型势 |
| `python` | `module:factory` 或 `file.py:factory`；包装自己的模型 |
| `rpmdrate` / `f2py` | 旧 Fortran 包装，`(3,N,B)` 布局与正梯度 |
| `cabi` | 实现版本化 PyRPMD PES v1 ABI 的共享库 |
| `ase` | factory 返回 ASE Calculator；处理 eV/Å、晶胞/PBC 与 stress |
| `torch` | Torch 模型给出批量能量或能量与力；用于分子体系 |
| `ipi` | 与外部 driver 通信的同步 i-PI 力协议 |
| `diabatic_python` | 实验性 NRPMD 的实对称或复 Hermitian 矩阵势及其梯度 |

电子计算中 D 和 T 按氢传递，同时保留各自的核质量。

Torch 适配器在其设备上求值模型，每次调用传输坐标和力；每个轨迹进程加载自己的模型副本。

## 周期模型

周期任务需要声明支持晶胞/PBC 的 provider。Python provider 使用以下两种协议之一：

| 协议 | 声明与调用 | 单位 |
| --- | --- | --- |
| 每次调用传入晶胞 | `metadata["periodic"]=True`，`evaluate(q, *, cell, pbc)` | q 与晶胞使用 provider 声明的长度单位 |
| 绑定体系 | `metadata["supports_pbc"]=True`，先 `bind_system(system)` 再 `evaluate(q)` | 绑定的 `system` 为原子单位；q 使用 provider 单位 |

显式实现其中一种协议。材料示例还提供 `periodic_cosine` 模型。i-PI 的 POSDATA 传递固定晶胞及其逆矩阵，但不含 PBC 标志，
因此部分周期体系以及带晶胞的非周期体系需设置 `potential.params.init.pbc`，driver 从 INIT 中读取。周期任务拒绝 C ABI v1、
f2py 和 Torch provider。

固定层属于动力学约束；势能始终接收全部原子。见[材料与表面](periodic.md)。

## 原生共享库与 Fortran

实现 ABI 的 `.so` 用法如下：

```bash
pyrpmd pes inspect ./libpes.so --geometry ts.xyz
pyrpmd pes check ./libpes.so --geometry ts.xyz --step 1e-5 --tolerance 1e-5
pyrpmd run job.inp --pes ./libpes.so --output runs/native-pes
```

`pes check` 的步长以 bohr 计，容差以 hartree/bohr 计。目标文件（`.o`）通过 `pyrpmd pes build` 和一个写明 wrapper、
编译器、PIC 选项与链接依赖的 manifest 链接成共享库。f2py 扩展使用 `rpmdrate` / `f2py` provider，并须与 Python 和 NumPy
的 ABI 匹配。协议与生命周期见 [PES v1 ABI](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/ABI.md)。

## 自定义 CV 与分类器

自定义 CV 的 factory 返回一个对象，其 metadata 声明 `dimension`、规范 `units` 和可选的 `periods`；
`value_and_jacobian(state)` 返回 `value[D]` 和 `jacobian[D,P,N,3]`，其中已包含该估计量的全部珠子因子。该协议适用于静态、可微的坐标。

`outcomes.factory` 为 dRPMD 提供产物分类器。同组成交换和异构化通道需显式编码。完整示例与返回值见
[插件规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md)和
[二维自定义 CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/custom_cv.py)。
