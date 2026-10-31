# PySCF 到 JaxPIP 势能生成

这条可选离线流程复用 [Caracal 势能生成](pes-generation.md) 的 PySCF 参考点格式，
训练真实 JaxPIP 线性 PIP 或 PIP-NN，保存上游 Equinox 检查点，再使用已有原生 ONNX
接口。它不把 QMDFF 转换为 PIP，也不新增一套动力学积分器。

本实现依赖势能生成 PR #35；推理及导出复用 PR #34。集成测试固定接口提交
`5c7c55aa7622dbadc37c47a8ed65c6b666eade02` 和 JaxPIP 提交
`a18c8cca1e795202194d4082919ebd71722df125`。本路线不需要安装 Caracal。

## 环境与命令

新参考点需要 PySCF；训练需要 JAX、JaxPIP、Equinox 和 SciPy；导出环境按照原生接口的
`tools/requirements-jaxpip-export.txt` 安装。编译原生接口后，将动态库路径显式传入。
动力学生产阶段不需要 JAX 或 PySCF，也不逐步回调 Python 势能。

```bash
python -m pyrpmd.pesgen jaxpip-data data.toml --output runs/pip-data
python -m pyrpmd.pesgen jaxpip-fit fit.toml --output runs/pip-fit
python -m pyrpmd.pesgen jaxpip-export runs/pip-fit \
  --exporter external/onnx-bridge/tools/export_jaxpip_onnx.py --output runs/pip-native
python -m pyrpmd.pesgen jaxpip-run runs/pip-native job.inp \
  --library /absolute/path/libpyrpmd_onnx.so --output runs/pip-dynamics
```

## 数据输入

```toml
schema = "pyrpmd.jaxpip-data/1"
[geometries]
train = "train.xyz"
validation = "validation.xyz"
test = "test.xyz"
[qm]
method = "RHF"
basis = "sto-3g"
conv_tol = 1e-12
```

XYZ 使用 angstrom。参考点内部统一为 bohr、hartree 和正能量梯度；训练力取负梯度。
只计算能量与梯度，不请求不必要的 Hessian 或布居分析。也可用 `[references]` 中各集合
的 JSON 路径列表导入已有参考点，此时不再指定 `[qm]`。

原子顺序、电子态、PySCF 版本和全部计算设置必须一致。至少两个训练点。训练、验证、
测试集合显式指定；跨集合的刚体变换或置换重复构型会被保守拒绝。这不是统计独立性的
证明，时间相关或同一路径数据仍应按轨迹/路径组划分，而不是随机拆分相邻帧。

## 训练输入

```toml
schema = "pyrpmd.jaxpip-fit/1"
dataset = "runs/pip-data"
basis = "basis.json.gz"
basis_symbols = ["H", "H"]
[fit]
kind = "linear" # 或 nn
alpha = 1.0
position_unit = "bohr" # 或 angstrom
energy_unit = "hartree" # 或 eV
energy_scale = 0.01 # 总是 hartree
force_scale = 0.01 # 总是 hartree/bohr
force_weight = 1.0
ridge = 1e-10 # 仅线性模型，常数项不正则化
hidden_layers = [16, 16]
max_steps = 500
[acceptance]
energy_rmse_max = 3e-5
force_rmse_max = 3e-4
```

基组由用户提供，要求第一项为常数，原子顺序须显式声明；检查指数范围、原子数和采样
探针上的同元素置换不变性。探针检查不能替代基组完整代数对称性的证明。

线性模型采用能量/力联合加权 SVD 拟合；PIP-NN 采用 L-BFGS-B，只优化网络层，冻结
描述符和仅由训练集确定的缩放参数。验证和测试标签不参与拟合、缩放、停止或模型选择。
耗尽迭代预算不会报告为优化收敛。报告分别列出三个集合的原子单位误差，且不对测试能量
重新平移零点。密集拟合有内存规模检查；完整拟合支持校验续跑，中断优化不恢复优化器状态。

## 导出与边界

显式指定的导出脚本属于可信可执行代码，在独立进程中运行。复用接口的 JAX/ONNX
能量力一致性、有限差分、1/3/32 批量大小检查，核对哈希和原子/单位/电子态元数据后才发布
完整模型包。失败保留日志，不覆盖已有结果。原生作业只注入普通 C ABI 势能配置，复用现有
JobSpec 和 runner；没有另写 Python 逐步积分循环。

没有显式测试集及误差阈值通过时，试算需 `--allow-unvalidated`。阈值通过只说明给定测试
域上的拟合精度，所有报告的 `physical_validation` 仍为 false。CPU 是本次验证目标；不声称
GPU 加速、周期应力、非绝热耦合或跨轨迹自动批处理。

`examples/pesgen/jaxpip/h2_pipeline.py` 提供真实 H2/RHF-STO-3G 的 16/4/4 点试算。
指定导出脚本和动态库可检查八珠子 100 步原生传播与参考路径一致，以及原生块内零 Python
势能回调。这不是全局反应势能或收敛的物理速率验证。
