# PySCF 数据集、JaxPIP 拟合与原生动力学

这个可选分子流程连接显式划分的参考数据、JaxPIP 线性 PIP 或 PIP 神经网络、
仓库内的 ONNX 导出器，以及现有 C ABI 势能接口。PySCF 离线计算参考能量和梯度；
生产动力学通过 ONNX Runtime C++ 插件计算力。

结果是给定分子构型域内的拟合模型。拟合和导出记录始终保留
`physical_validation=false`；误差阈值或导出检查通过，不代表 RPMD 物理量已收敛，
也不代表得到了全局反应势能面。另一个[Caracal 流程](pes-generation.md)
使用相同参考数据约定生成 QMDFF/dE-EVB/TREQ。

## 安装离线工具

在仓库根目录中，为拟合与导出建立独立环境：

```bash
python -m venv .venv-jaxpip-offline
. .venv-jaxpip-offline/bin/activate
python -m pip install -e . PySCF==2.14.0
python -m pip install -r tools/requirements-jaxpip-export.txt
```

[固定依赖文件](https://github.com/chen-yu-hao/PyRPMD/blob/main/tools/requirements-jaxpip-export.txt)
包含 JaxPIP、JAX、Equinox 与 ONNX 导出工具。这些仅属于**离线拟合/导出环境**，
不是普通 PyRPMD 动力学的依赖。数据集目录锁当前使用 POSIX `fcntl`。
原生执行另需 ONNX Runtime C/C++ SDK 和已编译插件，仅安装 Python wheel 不够；
构建步骤见[原生 ONNX 说明](https://github.com/chen-yu-hao/PyRPMD/blob/main/native/onnx/README.md)。

下面的 H₂ pilot 会自行生成键长扫描、数据划分和基组：

```bash
python examples/pesgen/jaxpip/h2_pipeline.py --kind linear --output runs/h2-linear
python examples/pesgen/jaxpip/h2_pipeline.py --kind nn --output runs/h2-nn
```

RHF/STO-3G 扫描用于检查接口，不证明模型覆盖一般 H₂ 构型、样本彼此独立，
或参考计算达到研究所需精度。

## 收集显式划分的参考数据

准备原子顺序一致的多帧 XYZ，建立 `data.toml`：

```toml
schema = "pyrpmd.jaxpip-data/1"

[geometries]
train = "train.xyz"
validation = "validation.xyz"
test = "test.xyz"

[qm]
method = "RHF"
basis = "sto-3g"
charge = 0
spin = 0
conv_tol = 1e-12
```

```bash
python -m pyrpmd.pesgen jaxpip-data data.toml --output runs/jaxpip-data
```

XYZ 坐标为 **Å**；存储的参考量使用 bohr、hartree 和 hartree/bohr 的**正能量梯度**，
拟合时取负号转换为力。此流程只请求能量和梯度。参考引擎也支持 UHF/RKS/UKS，
DFT 必须显式设置 `xc`。`spin=N_alpha-N_beta`，模型多重度为 `spin+1`。
各帧的原子顺序、电子态、PySCF 版本与设置必须一致。SCF 收敛是必要条件，
但不能单独证明电子态连续性或波函数稳定性。

| 划分 | 当前实现中的用途 |
| --- | --- |
| `train` | 必须提供至少两点；决定拟合参数、NN 缩放与优化停止条件。 |
| `validation` | 可选诊断报告；不用于提前停止或选择 checkpoint。 |
| `test` | 可选诊断报告；原生交接的显式误差门槛需要非空测试集。 |

拟合前应按独立轨迹或反应路径组划分数据。收集器依据元素对距离签名拒绝跨组重复，
包括刚体变换和相同元素置换。这个保守检查也可能拒绝同距离谱的不同结构；
它不能发现全部近重复或证明统计独立。选择模型或误差阈值时，应保留最终测试集。

也可以用 `[references] train=["reference-a.json", "reference-b.json"]`
及可选的 `validation`/`test` 列表，导入已校验的 `ReferencePoint` JSON；
此时不使用 `[geometries]` 或 `[qm]`。

## 拟合线性 PIP 或 PIP-NN

提供适合指定原子顺序和组成的 JaxPIP `.json`/`.json.gz` 基组。基组是描述符，
不是已经拟合的势能。上游模型为 `PolynomialLinearModel` 与
`PolynomialNeuralNetwork`，保留其 Equinox 检查点格式。
以下 H,H 示例需要对应的双原子基组；用于其他分子时，
同时更换基组和 `basis_symbols`。保存为 `fit.toml`：

```toml
schema = "pyrpmd.jaxpip-fit/1"
dataset = "runs/jaxpip-data"
basis = "basis.json"
basis_symbols = ["H", "H"]

[fit]
kind = "linear"
position_unit = "bohr"
energy_unit = "hartree"
force_weight = 1.0
ridge = 1e-10

[acceptance]
energy_rmse_max = 3e-5
force_rmse_max = 3e-4
```

```bash
python -m pyrpmd.pesgen jaxpip-fit fit.toml --output runs/jaxpip-fit
```

两个 TOML 文件中的路径均相对于输入文件所在目录。PIP-NN 使用 `kind="nn"`，
可配置 `hidden_layers=[16,16]`、`activation="tanh"` 或 `"isru"`、`seed=0`、`max_steps=500`。
描述符的 `alpha` 使用声明的模型长度约定。基组第一项必须是常数；检查指数范围、
原子数和确定性训练探针上的同元素置换不变性。探针不能证明完整代数对称性，
基组生成和选择也不属于这个流程。

线性模型使用能量/力联合加权的增广 SVD 最小二乘，可选 ridge 正则化，常数项不惩罚。
NN 使用 L-BFGS-B 与 JAX 能量/力目标导数，只优化网络层，冻结描述符和仅由训练集确定的
缩放参数。力标签通过真实能量导数参与拟合，不另建力网络。两种损失都是按声明尺度
归一化的逐分量均值。线性模型把训练能量偏移恢复到常数项，不对测试能量另行平移零点。
验证集和测试集不参与训练、缩放、停止或检查点选择。

模型长度单位可选 `bohr`/`angstrom`，能量单位可选 `hartree`/`eV`。
无论模型使用哪组单位，`energy_scale`、`force_scale`、报告误差和 `[acceptance]`
门槛始终使用 **hartree 与 hartree/bohr**。示例阈值不适用于所有体系，应预先按研究目标确定。

检查 `metrics.json` 中每组的 RMSE、最大误差和优化诊断。
`within_requested_test_bounds=true` 要求测试集非空、显式提供两个正阈值，
且两种测试 RMSE 均达标。输出还包括 `model.eqx`、匹配的基组、`probe.npy`
和带散列的 `model.json`。导出探针来自训练构型，只检查数值转换，不能替代独立精度测试。

耗尽 NN 迭代预算不会报告为优化收敛。密集拟合的 `max_design_elements` 默认上限为
2000 万，更大的问题需要其他拟合实现。反复依据测试误差手动调参仍然属于数据泄漏；
这样选模后，应重新保留独立测试集。

## 导出并交给原生势能接口

使用当前仓库内的导出器：

```bash
python -m pyrpmd.pesgen jaxpip-export runs/jaxpip-fit \
  --exporter tools/export_jaxpip_onnx.py --output runs/jaxpip-native
python -m pyrpmd.pesgen jaxpip-run runs/jaxpip-native existing-job.inp \
  --library /absolute/path/libpyrpmd_onnx.so --device cpu \
  --output runs/jaxpip-dynamics
```

导出子进程检查动态 float64 能量/力输出、批量大小 1/3/32 的 JAX/ONNX 数值一致性、
独立帧行为和力的有限差分。外层还核对文件散列、单位、原子顺序及电子态。
成功目录包含 `model.onnx`、`export-report.json`、拟合文件、日志及 `export.json`。

`jaxpip-run` 替换一个现有、有效、非周期分子输入的势能部分，不替用户定义反应坐标
或速率归一化。输入原子顺序、电荷和多重度须与模型一致。
在 `[dynamics]` 设置 `loop_backend="native"`、`block_steps=256` 可明确要求受支持的
C++ 传播路径；不兼容的 Python CV/偏置组合会报错。CUDA 还需匹配 GPU SDK 和设备依赖，
CPU 检查通过不能作为 GPU 验证。

缺少测试门槛或未达标时，原生交接默认拒绝；只有显式添加 `--allow-unvalidated`
才能执行 **pilot**。门槛通过也仍是 `physical_validation=false`：外推、长程渐近、
相关通道、轨迹能量守恒及数值/物理收敛需要另外验证。
对应 Python API 位于 `pyrpmd.pesgen` 的 `jaxpip_*` 模块中：`collect_dataset`、
`fit_jaxpip`、`export_jaxpip`、`native_potential_spec`、`run_jaxpip_job`。
模型交接与运行路径不导入 JAX 或 PySCF。

## 续跑与来源记录

`jaxpip-data --resume` 仅在输入和设置相同时复用中断的参考计算。
`jaxpip-fit --resume` 复用完整且匹配的拟合，不是恢复中断的优化器状态。
数据、设置、依赖身份变化或完成文件损坏会被拒绝。不完整的拟合、导出需要新目录；
导出不会覆盖已有模型包。请将数据集、拟合/导出清单与日志和后续动力学结果一并保留。

[本地验证记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/jaxpip-generation-2026-10-31/README.md)
保留作者当时的环境和数值证据。H₂ 示例包含 16/4/4 个训练/验证/测试点；指定
`--exporter` 与 `--library` 后，还会比较八珠子 100 步原生/参考传播，并检查原生块内
零 Python PES 回调。这些检查验证接口，不验证物理速率收敛。

[整合源码审查记录](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/jaxpip-generation-2026-10-31/REVIEWED-INTEGRATION.md)
记录了全部 44 项测试通过，以及使用当前仓库导出器完成两类真实模型导出和原生传播的结果。

上游源码与方法：[CQPES/JaxPIP](https://github.com/CQPES/JaxPIP)，Li 等，*JCTC* 2026，
DOI `10.1021/acs.jctc.6c01659`。使用时应同时引用上游方法和参考数据的电子结构计算层级。
