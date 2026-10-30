# 势能面与 Python 插件

## 统一物理接口

所有 provider 接收 C-contiguous `float64[B,N,3]`，B 是一次调用中的结构数（通常是分配给该 provider 的珠子）。接口返回：

```python
from pyrpmd.potentials import PESResult

# q: bohr; energy: hartree; forces: hartree/bohr
return PESResult(energy, forces, status, messages)
```

能量形状 `[B]`，力 `[B,N,3]`，status `[B]`（0 成功），messages 可选。也可返回 `(energy, forces)`，适配器补成功状态。力必须是 **负梯度**。只返回物理 PES，不加入环弹簧、umbrella、约束或 `1/P` 因子。失败状态由调用方显式处理，不能用零力掩盖失败。

可选 `PESResult(..., virial=virial)` 接受 `[B,3,3]` hartree 物理构型 virial。
它不含动量/环弹簧，不能直接作为量子压力。周期 Python provider 须声明
`metadata['periodic']=True`，并通过 `evaluate(q, *, cell, pbc)` 或匹配系统的
已绑定 cell/PBC 接收固定晶胞。也兼容 `supports_pbc=True` 与 canonical-AU `bind_system(system)` 回调契约。
完整约定见[材料计算](materials.md)与[表面计算](surface-catalysis.md)。

Python、f2py 与 Torch 接口可声明 `units={positions="angstrom", energy="eV"}`；入口/出口自动转换到原子单位。C ABI 始终使用规范 AU。需要梯度而非力的 Python PES 可声明 `derivative="gradient"`。单位字符串的 provider 支持范围见 `potentials/python.py`；复杂单位变换可在自定义 wrapper 中明确实现。

## Python PES

```python
# my_pes.py
import numpy as np

class Harmonic:
    metadata = {"execution_device": "cpu", "thread_safety": "context_serial"}

    def __init__(self, k):
        self.k = k

    def evaluate(self, q):
        return 0.5*self.k*np.sum(q*q, axis=(1, 2)), -self.k*q

    def close(self):
        pass

def build_pes(k=0.002):
    return Harmonic(k)
```

```toml
[potential]
provider = "python"
factory = "my_pes.py:build_pes"
params = {k = 0.002}
```

factory 在每个实际势能进程中创建自己的对象。带有全局 Fortran COMMON、GPU context 或 socket 的对象不跨进程传输。普通 Python 输入中的配置仍使用可导入 factory，避免让 lambda/本地闭包承担重启和 MPI 序列化职责。低层 `Dynamics` 可直接接受实现 `evaluate` 的对象。

## 周期表面 PES

周期 System 的 cell/PBC 会传入 ASE `Atoms`。自定义 Python provider 须声明
`metadata["supports_pbc"]=True` 并实现 `bind_system(system)`，接收原子单位的
完整系统；所有串行／进程／珠子 worker 均执行该绑定。当前 C ABI v1、原 f2py、
Torch 和 i-PI socket 工厂不能承载该固定晶胞契约，周期请求会明确拒绝。
参见 [表面 PES 契约](surface-catalysis.md#周期-pes-契约和性能) 与
`examples/surface/toy_surface.py`、`examples/surface/ase_emt.py`。

## 既有 RPMDrate/f2py 势能

```toml
[potential]
provider = "rpmdrate"
module = "./potential.cpython-312-x86_64-linux-gnu.so"
function = "get_potential"
initializer = "initialize_potential"
with_dimensions = false
```

也可使用 Python 模块名。扩展模块文件名须保留它真实的 Python 初始化模块名。适配器把 `[B,N,3]` 转为 Fortran-contiguous `(3,N,B)`，读取 `(V[B], dVdq[3,N,B], info)`，把正梯度变成力。没有 info 的二元组也支持。`with_dimensions=true` 用于额外接收 `natoms,nbeads` 的 wrapper；其他旧签名需要用户提供薄 Python/Fortran wrapper。没有初始化函数时，Python 输入可设置 `initializer=None`；TOML 可用空字符串。

现有编译模块必须与运行 Python 和其 NumPy/f2py ABI 兼容。PyRPMD 不把任意 `.so` 都视为 f2py，也不猜测 Fortran 参数列表。当前测试验证了 legacy 协议的形状、初始化和梯度符号；真实的特定 Fortran PES 需在用户编译环境再验证。

## C ABI `.so` 与 `.o`

现成 `.so` 如果实现 [PES v1 ABI](../native/ABI.md)，可直接使用：

```bash
pyrpmd run job.inp --pes ./libpes.so
pyrpmd pes inspect ./libpes.so --geometry ts.xyz
pyrpmd pes check ./libpes.so --geometry ts.xyz
```

`pes check` 对实际能量做中心差分并检查力，检查步长单位 bohr。C ABI metadata 同时检查原子顺序、质量以外的电子态元数据、数据布局和单位。Fortran 可使用 `ISO_C_BINDING` + `BIND(C)` 实现同一个 ABI。

普通 `.o` 不能直接动态加载。manifest 负责链接，例如：

```toml
# pes.toml; 本文件中的路径相对于 manifest。
compiler = "gfortran"
objects = ["legacy_pes.o"]
wrapper_sources = ["pyrpmd_wrapper.f90"]
include_dirs = ["include"]
compile_flags = ["-O3", "-fPIC"]
library_dirs = ["lib"]
link_libraries = ["lapack", "blas"]
```

```bash
pyrpmd pes build legacy_pes.o --manifest pes.toml --output libpes.so
# 或运行时自动链接到内容寻址缓存：
pyrpmd run job.inp --pes legacy_pes.o --pes-manifest pes.toml
```

既有对象若没有 PIC 或缺少链接依赖，会给出链接错误，不重猜编译配置。缓存包括 manifest、对象/wrapper 内容和编译器版本。`PYRPMD_PES_CACHE` 可指定共享/本地缓存位置。C 示例与可运行构建命令见 [examples/pes](../examples/pes/README.md)。

## ASE

```toml
[potential]
provider = "ase"
factory = "calculators.py:build_calculator"
```

factory 返回 ASE Calculator。适配器逐结构更新 Atoms，从 ASE eV/angstrom 转换为 AU。因此能通过 ASE 对接已有 Calculator，但实际模型下载、许可证、适用元素、电子态、设备与 batch 能力由模型自身决定。已实际验证 ASE Lennard-Jones；没有声明所有 MACE/DeepMD/xTB 模型都已跑过。

ASE 收到真实晶胞/PBC，D/T 映射为电子势能上的 H 元素；核质量由 Dynamics 保留。
若 calculator 提供 stress，adapter 转换为 `virial=-volume*stress`。
真实 Ni–H EAM 的验证脚本与模型来源见[材料 pilot](../benchmarks/materials/README.md)。

## Torch 与多 GPU

```toml
[potential]
provider = "torch"
factory = "torch_model.py:build"
device = "cuda:0"
dtype = "float64"

[parallel]
backend = "process"
mode = "gpu"
devices = [0, 1]
trajectories_per_gpu = 2
```

Torch factory 接收 `device`，可返回 `torch.nn.Module`。`forward(q)` 返回 `[B]` 或 `[B,1]` 总能量时，由 autograd 求力；也可返回 `(energy, forces)` 或相应字典。每个 batch item 必须是独立结构的能量，不能在 batch 维度耦合结构。输出不得 detach 能量而又不提供力；参考接口要求 float64。已有 TorchScript 文件可用 `model="model.pt"`。

factory 也可返回有 `evaluate` 的封装 provider，并明确 metadata 中的实际 `execution_device`。请求 CUDA 但设备不可用，或返回 CPU provider 时会报错。每卡多轨迹是独立进程/模型副本；珠子在一次 PES 调用中 batch。当前动力学状态和接口数组在 CPU，Torch 路径包含主机/设备传输，并非整个 RPMD 积分器驻留 GPU。

## i-PI socket

```toml
[potential]
provider = "ipi"
params = {host="127.0.0.1", port=31415, timeout=60.0, init={periodic=false}}
```

PyRPMD 充当同步 force server，外部 driver 连接并按 i-PI 协议提供力。每个 context 对应一个 driver。并行执行器为不同轨迹/珠子 context 分配不同端口或 Unix socket 后缀；需自行启动相应数量的 driver 并连接正确地址。多节点 hostname/防火墙/driver 批处理由站点配置。固定晶胞任务传入真实 cell 与逆矩阵；部分 PBC 另需匹配的 `params.init.pbc` 和服务端支持，因为 POSDATA 不传边界布尔标志。非周期分子保留原协议行为。

周期能力当前不扩展到分子 Torch、C ABI v1 或 f2py；这些入口会拒绝周期系统。

## 自定义 CV

factory 返回有 `metadata` 和 `value_and_jacobian(state)` 的对象。metadata 定义 `dimension`、`units`（规范 `bohr` / `hartree` / `radian` / `dimensionless`）及可选 `periods`。只读 `state.q` 形状 `[P,N,3]`，返回 `value[D]`、`jacobian[D,P,N,3]`，导数必须包含 estimator 的所有 bead 因子。`centroid`、`bead_average` 或 `full_ring` 是插件声明的估计量语义，不会对已经完整的导数再次除 P。

这是静态坐标协议；不支持依赖外部时钟的 CV 或隐式有限差分梯度。二维自定义完整例子见 [custom_cv.py](../examples/fes/custom_cv.py)。

## 自定义产物分类器

`outcomes.factory` 指向接收 `system` 与 `params` 的工厂，返回可调用对象。`classifier(state)` 返回以下之一：

- `None`：目前不能判定；
- `{"status":"nonreactive", "channel":None}`：已确认回到反应物；
- `{"status":"product", "channel":"channel_name"}`：已确认指定产品。

字串 `"reactants"`/`"nonreactive"` 或其他通道名也是简写。返回的标签仍需保持 `persistence_steps`；自定义分类器负责自身的远区、身份、键连或其他物理终态判据。可实现 `validate_reactants(q)` 检查母轨迹仍属于入口态。
