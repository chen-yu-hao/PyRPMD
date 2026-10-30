# 连接势能与扩展插件

## 先确认物理接口

高层工作流每个执行进程分别构造势能对象。一次 PES 调用接收一个结构批次，
通常是该进程负责的若干珠子。**只返回物理势能和力**：不要把环弹簧、umbrella、
约束、温度或 `1/P` 缩放混入模型。

| 数据 | 形状 | 规范单位/含义 |
| --- | --- | --- |
| 坐标 `q` | `float64[B,N,3]` | bohr，C-contiguous |
| 能量 `energy` | `[B]` | 每个结构的总势能，hartree |
| 力 `forces` | `[B,N,3]` | 负能量梯度，hartree/bohr |
| 状态 `status` | `[B]` | 0 为成功；失败不能用零力掩盖 |
| 信息 `messages` | B 个字符串 | 可选失败说明 |
| 物理 virial | 可选 `[B,3,3]` | hartree，provider 真正实现时才提供 |

`B` 是一次批处理大小，不保证等于整个作业的珠子数 P。
势能模型应对不同结构独立求值，不能在 batch 维度混合能量。

## 一个完整 Python provider

下面的非周期谐势可以保存为 `my_pes.py`：

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

在同目录输入中替换势能 section：

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
params = {k = 0.002}
```

factory 应位于独立且可导入的模块。不要把不可序列化的 GPU context、socket 或
Fortran COMMON 状态从父进程传给所有 worker；每个 worker 应创建和关闭自己的上下文。

## 单位与梯度检查

Python/f2py/Torch 适配层可声明本身的坐标和能量单位，例如：

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
units = {positions = "angstrom", energy = "eV"}
derivative = "force"
```

此时 `my_pes.py` 的参数和实现必须真正使用 Å 与 eV。Python provider 若返回正梯度，声明
`derivative="gradient"`；适配层负责符号和单位转换。旧 f2py 接口固定接收正梯度并取负；
Torch 的直接输出固定解释为力，或由能量 autograd 计算负梯度，不使用这一 derivative 开关。
C ABI v1 始终使用原子单位。

对于任何新接入模型，在几个具有代表性的构型上先做能量/力有限差分，并检查原子顺序、
势能零点、电子态和模型定义域：

```python
import numpy as np
from pyrpmd.potentials import check_potential
from my_pes import build_pes

provider = build_pes()
try:
    q = np.array([[[0.1, -0.2, 0.3]]])  # bohr for this provider
    print(check_potential(provider, q, epsilon=1e-5))
finally:
    provider.close()
```

检查力差是程序接入的诊断，不能证明模型在所有反应路径、温度和周期结构上准确。

## 选择已有适配器

| Provider | 入口与使用条件 |
| --- | --- |
| `builtin` / `free` / `harmonic` 等 | 教学解析模型；不当作真实化学势能面 |
| `python` | `module:factory` 或 `file.py:factory`；适合用户模型包装 |
| `rpmdrate` / `f2py` | 旧 Fortran wrapper，适配 `(3,N,B)` 布局及正梯度 |
| `cabi` | 实现版本化 PyRPMD PES v1 ABI 的共享库 |
| `ase` | factory 返回 ASE Calculator，适配 eV/Å、cell/PBC 与可用 stress |
| `torch` | factory/model 提供批次能量或能量与力；当前周期体系不支持 |
| `ipi` | 同步 i-PI 力协议；需要外部 driver 与对应连接 |

ASE 接口能连接 calculator，并不保证某个具体模型的元素、周期边界、自旋、下载许可
和设备支持。周期 D/T 核在电子计算中按 H 元素处理，核质量仍保留。

PyRPMD 的 Torch 接口包含 CPU/设备传输；它不等于整个积分器在 GPU 上执行。
多个轨迹进程各自加载模型，显存需求也相应增加。

## 周期模型的显式契约

周期任务必须使用声明支持 cell/PBC 的 provider。整合版保留两种已存在的 Python 协议：

| 协议 | 声明与调用 | 单位 |
| --- | --- | --- |
| 按调用传递周期参数 | `metadata["periodic"]=True`，`evaluate(q, *, cell, pbc)` | q、cell 使用声明的 provider 长度单位 |
| 绑定完整体系 | `metadata["supports_pbc"]=True`，`bind_system(system)` 后 `evaluate(q)` | 绑定的 `system` 使用原子单位；q 仍遵守 provider 单位声明 |

模型应明确实现其中一个协议，不依靠捕获 `TypeError` 猜测是否支持周期。
绑定不会让一个非周期势自动成为周期势。材料示例还提供 `periodic_cosine` 教学模型；
i-PI 的 POSDATA 传递真实固定晶胞及其逆矩阵，但协议中没有 PBC 标志；
部分周期或带 cell 的非周期体系还需显式配置 `potential.params.init.pbc`，并由 driver
遵守这一 INIT 契约。外部 driver 必须正确实现该周期问题。
当前 C ABI v1、旧 f2py、Torch 周期组合会被拒绝，不能忽略晶胞继续算。

固定层由动力学约束处理，势能仍应看到全部原子的结构。更多边界见[材料与表面](periodic.md)。

## 原生共享库与旧 Fortran

实现 ABI 的 `.so` 可使用：

```bash
pyrpmd pes inspect ./libpes.so --geometry ts.xyz
pyrpmd pes check ./libpes.so --geometry ts.xyz --step 1e-5 --tolerance 1e-5
pyrpmd run job.inp --pes ./libpes.so --output runs/native-pes
```

`pes check` 的步长为 bohr，容差为 hartree/bohr。普通 `.o` 不是可直接加载的共享库；
需要 manifest 明确 wrapper、编译器、PIC 与链接依赖，再通过 `pyrpmd pes build` 构建。
任意 `.so` 也不等于 PyRPMD ABI 库：f2py 扩展要选择相应 provider，且与 Python/NumPy ABI 匹配。
完整协议与生命周期见 [PES v1 ABI](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/ABI.md)。

## 自定义 CV 与分类器

自定义 CV factory 返回含 metadata 的对象，声明 `dimension`、规范 `units` 和可选 `periods`；
`value_and_jacobian(state)` 返回 `value[D]` 与 `jacobian[D,P,N,3]`。
Jacobian 必须已经包含 estimator 的全部珠子因子。当前协议是静态可微坐标，
不把隐式有限差分或依赖外部时钟的坐标视作同一能力。

dRPMD 的 `outcomes.factory` 用于定义有物理依据的产物分类器。连通图推断只能用于 pilot；
同组成交换或复杂异构化应显式编码并验证。完整例子与分类返回值见
[插件规范](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/potentials.md)及
[二维自定义 CV](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/fes/custom_cv.py)。

