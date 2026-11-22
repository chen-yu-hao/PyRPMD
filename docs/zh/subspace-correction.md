# RPMD 速率的量子子空间修正

**中文** · [English](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-correction.md)

> **我应该选择哪套量子修正工作流？**
> - **热反应速率常数乘性修正 $k \approx k_{\mathrm{RPMD}} \times C_S$（本页）：** 在 1–3 维反应子空间 $S$ 中计算量子与 RPMD 速率之比 $C_S = k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$，支持在 `.inp` 中配置 `[rate.subspace_correction]`、独立命令行 `pyrpmd subspace-correct correct.toml` 或 `pyrpmd.analysis` Python API。
> - **时间关联函数、光谱响应与在线 Ehrenfest 轨迹：** 若需频域响应缺陷传递（`run_quantum_correction`）、在线环聚合物 + 一维波函数耦合演化（`OnlineQuantumRPMD`）或含时通量–布居曲线修正（`hybrid_flux_side_correction`），请前往 **[量子修正英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-correction.md)**。
> - **经过多个耦合电子态的反应（锥形交叉、系间窜越、电荷转移交叉）：** 见 **[多电子态耦合子空间](multistate-subspace.md)**（research，Python API），其中给出多态子空间精确速率，以及与 $C_S$ 相乘的非绝热因子 $\eta_\mathrm{NA}$。

RPMD 能很好地处理反应环境效应（零点能、绝热跟随、回穿），但在沿反应坐标的深穿隧、共振与量子相干方面存在方法学误差。子空间修正保留全维 RPMD 速率，仅通过 1 至 3 个反应关键坐标上的精确量子计算修正上述内禀缺陷：

$$
k \approx k_{\mathrm{RPMD}}(\text{full system}) \times C_S, \qquad
C_S = \frac{k_{\mathrm{QM}}(H_S)}{k_{\mathrm{RPMD}}(H_S)}
$$

分子与分母使用同一个子空间哈密顿量 $H_S$ 和相同的反应物归一化，因此 $C_S$ 仅分离出子空间 $S$ 内部的 RPMD 方法误差。修正采用乘性传递：虽然 $H_S$ 的绝对速率可能与全维体系相差若干数量级，但二者共有的前因子在比值中精确抵消。当 $S$ 为全维体系时 $C_S$ 精确成立；当 $S$ 为低维子空间时属于可传递缺陷近似。

---

## 1. 三种使用场景速选

根据你是要发起一次全新的 `rate` 计算、对已保存的 `rate` 运行目录做后处理，还是不跑伞抽样直接由势能面在 Python 中计算 $C_S$，从下表选择对应入口：

| 使用场景 | 推荐入口 | 是否需要全维伞抽样？ | 核心命令 / API |
| --- | --- | --- | --- |
| **A. 随 `rate` 一步运行子空间修正** | `.inp` 中配置 `[rate.subspace_correction]` | 需要（本地 `umbrella` PMF；解析输入时自动开启 `output.centroid_path = true`） | `pyrpmd run calculation.inp` |
| **B. 对已有 `rate` 运行做后处理或重调参数** | `.inp` 加 `--stage subspace`，或独立运行 `pyrpmd subspace-correct correct.toml` | 需要已保存的 `rate` 目录（使用质心路径或 `--stage subspace` 时须含 `output.centroid_path = true`；`path.source = "mep"` 也可直接由保存的鞍点与 PES 构建 IRC） | `pyrpmd run calculation.inp --stage subspace` 或 `pyrpmd subspace-correct correct.toml` |
| **C. 纯 Python API 无伞抽样计算 $C_S$（或用于 `surface_rate` / 自定义模型）** | `pyrpmd.analysis` Python API | 不需要（直接由 $V$、$\nabla V$ 与鞍点/路径构造 $H_S$） | `project_linear_subspace`、`project_path_subspace`、`subspace_correction`、`correct_rate_run` |

(optional-native-rate-stage)=
### 场景 A：原生 `rate` 内置子空间阶段（单份 `.inp` 一步运行）

直接使用常规 `rate` 输入文件与运行器，无需单独编写第二份修正配置文件：

```toml
# 常规的 system、potential、reaction、umbrella 与 recrossing 等表保留在同一文件中
# （完整教学输入见 examples/rate/subspace.inp）。
task = "rate"

[rate.subspace_correction]
enabled = true
mode = "apply"             # 默认；或设为 "factor_only" 仅算因子
dimension = "auto"         # 或显式指定 1、2、[1, 2]
environment = "adiabatic"  # 也可选 "free_energy" 或 "frozen"
model_2d = "rph"           # 默认 MEP 反应路径哈密顿量；或 "plane"
anchor = "none"            # 或 "skeleton"（低温瞬子骨架锚定）
threshold = 0.5
reuse = true

[rate.subspace_correction.path]
source = "mep"             # 默认从细化鞍点出发的 IRC；也可选 "centroid"、"instanton" 等
step = 10.0                # 裸原子单位：bohr * sqrt(m_e)

[rate.subspace_correction.quantum]
refinement = 3
truncation_kt = 30.0
# 低温或含刚性横模时务必收敛 truncation_min（单位 Eh，约 6 倍最高横模量子）。
# 2D 截断后保留的网格点数取决于温度、质量与 PES；求解前会在 solver_plans 中给出所需的 dense_limit。
truncation_min = 0.0
dense_limit = 8000

[rate.subspace_correction.rpmd]
walkers = 32
sweeps = 800
trajectories = 1024
seed = 17

# 有效性门禁（下方显示默认值）。设为 "allow" 会显式覆写并记录到报告中。
[rate.subspace_correction.validity]
unconverged_path = "reject"   # 反应物与产物两侧路径延拓均须到达可接受终点
unstable_bath = "reject"      # 1D 路径模型的法平面横向 Hessian 必须稳定
outside_grid = "reject"       # RPMD 估计量采样必须留在 V_S 网格内部
max_outside_fraction = 0.01
submerged_surface = "reject"  # RPMD 分割面必须是质心自由能瓶颈（不可深陷于入口络合物下方）
max_submerged_kt = 1.0
folded_ribbon = "reject"      # RPH 2D 路径曲率半径不得在横向基态波函数宽度内发生坐标折叠
max_curvature_width = 0.5
on_invalid = "stop"           # 或 "diagnose"：门禁拒绝时仍求解分量速率供诊断，但绝不应用该因子
```

```bash
pyrpmd run calculation.inp
# 仅修改子空间参数后，在同一输入/输出目录上单独重跑子空间阶段：
pyrpmd run calculation.inp --stage subspace
# 常规检查点续跑也会在主流程结束后自动执行已启用的子空间后处理：
pyrpmd run calculation.inp --resume
```

完整可运行的教学输入见 [`examples/rate/subspace.inp`](../../examples/rate/subspace.inp)。
`enabled` 默认为 `false`；设为 `true` 时，输入解析器会在伞抽样开始**之前**自动开启
`output.centroid_path = true`。PMF 与回穿阶段继续使用原有的原生速率执行器与后端；随后的子空间求解在
root rank 上运行一次，调用 SciPy/NumPy 精确量子求解器与子空间 RPMD 求解器。

`mode = "apply"` 计算修正因子 $C_S$ 并乘上已保存的全维 RPMD 速率估计；
`mode = "factor_only"` 计算相同的量子/RPMD 比值，但不要求全维速率已发布，也不写出修正后速率文件（它仍需要已保存的体系/PES 与完整窗口质心矩，并非免采样的纯过渡态计算；在完整 `run` 中不会跳过常规速率采样阶段，若已有采样数据请加 `--stage subspace`）。

温度、珠子数、原子质量、PES 与输出目录全部从主 `rate` 任务自动继承。`[rate.subspace_correction.path]`、
`.quantum` 与 `.rpmd` 中的有量纲数值一律采用**裸原子单位**，不支持带单位字符串。原生 `.rpmd.windows`
为整数窗口数（至少为 4）。子空间表拒绝未知字段，且不接受 `source`、`output` 或独立的 `nbeads`。路径环境可选
`frozen`、`adiabatic` 与 `free_energy`（`relaxed` 仅在 Python API 线性子空间投影中提供）。

原生集成支持使用**本地伞抽样 PMF** 的气相 `task = "rate"`（含独立的 `[rate.pmf]` 伞抽样阶段），拒绝外部 PMF、
metadynamics 和 OPES（因其不含窗口质心统计矩）。`surface_rate` 与 `extend` 扩展代次不支持该原生阶段，请使用场景 C（Python API）。

#### 输出文件、续跑与异常处理

所有后处理产物均保存在 `RUN/subspace/` 目录下：
- `path.npy`：拼接并延拓后的笛卡尔反应路径（单位 $\mathrm{bohr}$）。
- `C_S_1d.json` 和/或 `C_S_2d.json`：序列化的 `SubspaceCorrection` 记录，含精算量子速率、子空间 RPMD 速率、诊断信息与有效性门禁结果。
- `subspace_pipeline.json`：完整流水线报告（路径终点状态、采样与静态曲率 $\times$ 宽度、求解器计划 `solver_plans`、各维模型详情 `models`、门禁汇总 `validity`、嵌套灵敏度 `nested_sensitivity`、瞬子锚定 `instanton_anchor`）。
- `request.json` 与 `status.json`：解析后的子空间请求参数、缓存校验键 `reuse_key`、执行资源预算及最终阶段状态（`"complete"`、`"invalid_model"`、`"solver_capacity_exceeded"` 或 `"failed"`）。
- `rate_quantum_corrected.json`（`mode = "apply"` 时生成）：修正后的速率点估计、可发布速率（当源速率可发布时）、合成标准误差及源数据文件的 SHA-256 校验值。

完整运行返回值与 `results/summary.json` 会附加 `subspace_correction` 状态记录，不会覆盖原始速率字段；根目录下的
`summary.json`、PMF、回穿文件、manifest 与已保存输入均不会被改写。`--stage subspace` 仅向 `RUN/subspace` 写入，若此前完整运行留下了旧状态，以最新的 `RUN/subspace/status.json` 为准。

修改子空间网格、维度或子空间采样控制不会改变主任务的 RPMD 物理指纹（`physics_fingerprint`）。只要解析后的物理配置一致且 PES/模型文件哈希未变，即可直接编辑 `.inp` 并运行 `--stage subspace`。`--stage subspace` 会在调用 PES 前校验保存的输入与 manifest 身份，拒绝修改过温度、珠子数、反应定义、主采样预算或体系的输入，且与 `--resume` 及热启动选项互斥。若缺失窗口质心矩或窗口未采样完，必须重新执行伞抽样；在检查点之后才打开开关无法补全早期窗口统计量。

原生阶段的 `reuse = true` 会同时校验源数据文件哈希、数值参数、实现代码哈希、珠子数、温度与已保存因子校验和；任何参数变更或记录损坏都会触发重算，绝不仅凭温度相同就复用旧结果。复用因子时仍会重建/检查路径并将其应用到当前保存的速率估计上。

在完整 `rate` 运行中，若子空间阶段抛出异常，会将状态记为 `status = "failed"` 并给出警告，未修正的全维 RPMD 速率保持完好；而在显式调用 `--stage subspace` 时，失败会记录诊断并以非零状态码退出 CLI。请始终检查最新的 `status.json`（重试失败或切换为 `factor_only` 时，早期生成的修正文件不会作为有效结果保留；若无任何模型通过门禁，旧的 `rate_quantum_corrected.json` 会被自动清理）。

(one-command-from-a-rate-run)=
### 场景 B：对已保存的速率运行做独立后处理（`pyrpmd subspace-correct`）

```bash
pyrpmd subspace-correct correct.toml
```

`correct.toml` 指定一个已完成的速率运行目录（`[source] rate_run = "..."`）以及可选的
`[path]`、`[subspace]`、`[quantum]`、`[rpmd]`、`[validity]`、`[execution]` 和 `[output]`
表（完整示例见 `benchmarks/clhbr_subspace/correct.toml`）：

```toml
[source]
rate_run = "../../runs/clhbr-300K-production"

[path]
source = "mep"           # 默认从细化鞍点出发的 IRC；也可选 "centroid"、"instanton" 等
step = 10.0              # 质量加权延拓步长 (bohr * sqrt(m_e))
spread_factor = 2.0      # 取向锁定窗口：离散度 <= 2 倍过渡态离散度
smoothing = 1.5          # xbar(xi) 局部二次拟合带宽（以窗口间距为单位）

[subspace]
dimension = "auto"       # 始终算 1D；当 曲率 x 宽度 > threshold 时自动追加 2D
threshold = 0.5
environment = "adiabatic"
model_2d = "rph"         # "rph"（默认）或 "plane"
anchor = "none"          # "none" 或 "skeleton"

[quantum]
refinement = 3           # RPMD 样条网格比 DVR 网格细 3 倍（共享同一 V_S）
truncation_kt = 30.0     # 能量截断 DVR：保留 V <= floor + max(30 kT, truncation_min)
truncation_min = 0.0     # 低温下设为约 6 倍最高横模量子（单位 Eh）
threshold_margin = 1.0   # 热通量算符限制在 E >= 通道阈值 - 1 kT
reference_kt = 0.3       # 反应物参考切片：势能与渐近端之差 <= 0.3 kT 的平坦自由通道
product_drop_kt = 15.0   # 路径落至最高点以下 15 kT 时开始吸收产物
dense_limit = 8000

[rpmd]
windows = 48
walkers = 32
sweeps = 800
trajectories = 1024
seed = 11

[output]
directory = "../../runs/clhbr-300K-production/subspace"
reuse = false
```

当需要使用伞抽样窗口的质心统计矩（如选用 `"mep"` 以外的 `path.source`，或计算采样曲率 $\times$ 宽度诊断）时，原速率运行必须在输入中启用：

```toml
[output]
centroid_path = true
```

开启后，速率 PMF 阶段的每个伞窗口会对每个对齐到过渡态几何的环聚合物（先分配最近的等价反应通道，再做质量加权 Kabsch 对齐）累加记录：平均质心、质心协方差、珠子绕质心的协方差 $\Sigma(\xi)$ 以及平均原子间距离。这些统计矩具有可加性，多副本与检查点续跑能够精确合并，最终写入 `centroid_path.npz` 并由 `mean_path_from_rate_run(run_dir)` 读取。该开关仅用于分析（不计入物理指纹，也不改变动力学轨迹），但会停用一条融合原生快速路径，因此在未启用 `[rate.subspace_correction]` 时默认关闭。若反应路径在越过过渡态后发生弯曲，请将伞抽样窗口延伸到分割面之后（例如 `stop = 1.45`）。

(workflow)=
### 场景 C：纯 Python API 无伞抽样计算 $C_S$（`pyrpmd.analysis`）

当你希望直接从势能面函数/梯度和过渡态几何计算 $C_S$ 而无需运行全维伞抽样（或者需要将修正因子应用到 `surface_rate` 或外部速率结果）时，可在 Python 中按五步直接构造并求解 $H_S$：

1. **构建子空间哈密顿量 `H_S`**（`pyrpmd.analysis.subspace_model`）。坐标采用电子质量加权（$\mathrm{bohr}\cdot\sqrt{m_e}$），第 0 轴为反应坐标，分割面为第 0 轴上的位置：
   - `model_from_function(V, axes, masses=..., dividing_surface=...)` — 哈密顿量本身已是 1–3 维模型。
   - `project_linear_subspace(V, grad, reference, masses, basis, axes, environment=...)` — 高维体系的线性子空间投影。正交补空间可选 `"frozen"`（冻结）、`"relaxed"`（静态能量极小化）、`"adiabatic"`（弛豫能量加简谐零点能，默认）或 `"free_energy"`（弛豫能量加温度 $\beta$ 下的简谐量子自由能）。
   - `project_path_subspace(V, grad, path, masses, axis, ...)` — 弯曲的一维反应路径（如 IRC/MEP，或由 `ring_polymer_mean_path` 得到的环聚合物平均路径），配合法平面正交截断热浴与弧长坐标。`zero_point = "harmonic"` 保持历史投影 Hessian 频率；`zero_point = "rph"` 在线性法平面截口上使用二次反应路径势能 Hessian `N.T @ H @ N`；`zero_point = "dvr"` 沿局部法向模式求解独立的一维 sinc-DVR 截口（当 `environment = "free_energy"` 时采用截口热自由能）。`dvr_points`、`dvr_half_width` 与 `dvr_spacing` 会记录到模型溯源信息中。
2. **精确量子速率** `exact_quantum_rate(model, beta)`（`pyrpmd.analysis.subspace_quantum`）。在 sinc-DVR 网格上带吸收势的热通量–通量公式：
   - `method = "sylvester"`（底层 API 默认至 `dense_limit = 1600`，流水线默认 `8000`）一次性求解 $H^\dagger X - X H = i F$；无时间截断，对长寿命量子共振精确成立。
   - `method = "flux_eigenstates"`（不含稠密专用选项的更大 2D/3D 网格）通过 Chebyshev Boltzmann 算符与分块子空间迭代求出显著的热通量本征态，再用分裂算符传播。`max_time` 时仍陷俘在网格内的波包模方按后半程吸收分支比外推分配，并在 `diagnostics["extrapolation"]` 中报告 $T/2$、$3T/4$ 与 $T$ 处的估计值。
3. **同一 `H_S` 的 RPMD 速率** `ring_polymer_rate(model, beta, nbeads)`（`pyrpmd.analysis.subspace_rpmd`）：质心受约束热力学积分（精确平均力，采用精确自由环传播与 Metropolis 判据的正常模 HMC）加无约束回穿轨迹。报告误差综合了 walker 统计误差、TI 求积误差与传输系数 bootstrap 误差。
4. **合成修正因子** `subspace_correction(model, beta, nbeads)` 返回 `SubspaceCorrection` 对象（含 `factor`、`log_standard_error`、`valid`、`validity` 及两个子结果），可用 `.write(path)` 保存为 JSON。请使用与全维 RPMD 运行一致的珠子数 `nbeads`。
5. **应用修正因子。** 对内存中的速率数值调用 `apply_subspace_correction(rate, correction, standard_error=..., beta=..., bead_factor=..., anchor_log_shift=...)`，或对已保存的 PyRPMD `rate`/`surface_rate` 目录调用 `correct_rate_run(run_dir, correction)`。后者会核对 `input/resolved.json` 中的 `beta`，绝不修改源数据文件，记录源文件的 SHA-256 并写出 `rate_quantum_corrected.json`（若源运行本身未达到可发布门禁，`corrected_rate_*` 保持为空，仅报告修正后的点估计）。

```python
import numpy as np
from pyrpmd.analysis import (
    project_linear_subspace,
    subspace_correction,
    harmonic_bead_factor,
    apply_subspace_correction,
    correct_rate_run,
)

# 1. 围绕过渡态 x_ts 投影 1D 或 2D 子空间（axes 单位为 bohr * sqrt(m_e)）
sub = project_linear_subspace(
    V, grad, x_ts, masses, basis_mw,
    axes=[np.linspace(-150.0, 150.0, 301)],
    environment="adiabatic",
    dividing_surface=0.0,
    rigid_body=True,
)

# 2. 在匹配的 beta 与珠子数 P 下计算 C_S = k_QM(H_S) / k_RPMD(H_S)
corr = subspace_correction(sub, beta, nbeads=32)
corr.write("C_S_300K.json")

# 3a. 应用到已保存的 rate 或 surface_rate 运行目录：
correct_rate_run("runs/rate-300K", corr)

# 3b. 或应用到内存中的速率数值（可选传入正交补空间有限珠因子 B_P）：
bp = harmonic_bead_factor(omega_ts_complement, omega_reactant_complement, beta, nbeads=32)
res = apply_subspace_correction(k_rpmd, corr, standard_error=se_rpmd, beta=beta, bead_factor=bp["factor"])
```

---

## 2. 完整参数与单位速查表

> **子空间输入的四条核心单位规则（`input_subspace.py` & `subspace_pipeline.py`）**
> 1. **禁止使用带单位字符串：** `[rate.subspace_correction.*]` 与 `correct.toml` 中的所有数值参数必须填写裸数字（`allow_strings = False`）。写入 `"10.0 bohr"` 或 `"300 K"` 会直接触发 `InputError`。
> 2. **质量加权坐标与网格长度以电子质量 $m_e = 1$ 计（非 amu）：** `step`、`mep_displacement`、`spacing_1d`、`spacing_2d`、`padding`、`dvr_half_width`、`dvr_spacing` 的物理单位均为 **$\mathrm{bohr}\cdot\sqrt{m_e}$**。由于 $1\,\mathrm{amu} \approx 1822.89\,m_e$，氢原子移动 $0.23\,\mathrm{bohr}$ 对应的质量加权弧长约为 $0.23 \times \sqrt{1836} \approx 10\,\mathrm{bohr}\cdot\sqrt{m_e}$。
> 3. **能量与频率单位为 $\mathrm{E_h}$（Hartree，$\hbar = 1$）：** `truncation_min`、`asymptote_tolerance` 以及软模频率阈值 `soft_mode_frequency`（$2.3\times 10^{-4}\,\mathrm{E_h} \approx 50\,\mathrm{cm}^{-1}$）均以 $\mathrm{E_h}$ 为单位。
> 4. **笛卡尔差分步长单位为 $\mathrm{bohr}$，交越温度单位为 $\mathrm{K}$：** `hessian_displacement`（默认 $5\times 10^{-3}\,\mathrm{bohr}$）是最轻原子的物理笛卡尔位移；`crossover_temperature` 单位为开尔文（$\mathrm{K}$）。

### 2.1 顶层控制、子空间模型构造、源目录与输出

| `.inp` 键名 (`[rate.subspace_correction]`) | `correct.toml` 键名 | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| `enabled` | *(不使用)* | `bool` | `false` | `true`, `false` | —（启用原生子空间阶段，并在伞抽样前自动设 `output.centroid_path = true`） |
| `mode` | *(API `apply=`)* | `str` | `"apply"` | `"apply"`, `"factor_only"` | —（`"apply"` 写出 `rate_quantum_corrected.json`；`"factor_only"` 仅计算并保存 $C_S$） |
| `reuse` | `[output] reuse` | `bool` | `true` (`.inp`) / `false` (`correct.toml`) | `true`, `false` | —（未启用 `[execution]` 表时，是否复用通过内容与哈希校验的已保存 $C_S$） |
| *(自动继承主运行)* | `[source] rate_run` | `str` | `None`（`correct.toml` 必填） | 已保存的 `rate` 目录路径 | —（`.inp` 中禁止出现 `source`） |
| *(固定为 `<run>/subspace`)* | `[output] directory` | `str` | `None` (`<rate_run>/subspace`) | 输出目录路径 | —（`.inp` 中禁止出现 `output`） |
| `dimension` | `[subspace] dimension` | `str`、`int` 或 `list[int]` | `"auto"` | `"auto"`, `1`, `2`, `[1]`, `[2]`, `[1, 2]` | 无量纲（`"auto"` 始终算 1D，当 $\kappa\rho > \text{threshold}$ 或 1D `bath` 门禁失稳时自动加 2D） |
| `threshold` | `[subspace] threshold` | `float` | `0.5` | $> 0$ | 无量纲 — `"auto"` 模式下触发 2D 子空间的曲率 $\times$ 宽度阈值（$\kappa\rho$） |
| `environment` | `[subspace] environment` | `str` | `"adiabatic"` | `"frozen"`, `"adiabatic"`, `"free_energy"` | —（正交环境模处理方式；`"relaxed"` 仅在 Python `project_linear_subspace` 中可用） |
| `model_2d` | `[subspace] model_2d` | `str` | `"rph"` | `"rph"`, `"plane"` | —（`"rph"`：展开 MEP 上的反应路径哈密顿量；`"plane"`：穿过过渡态的固定平面） |
| `anchor` | `[subspace] anchor` | `str` | `"none"` | `"none"`, `"skeleton"` | —（`"skeleton"` 在 $T < T_c$ 时加入全维相对裸势骨架 $S^0$ 的瞬子最速下降修正 $A_H - A_{S^0}$） |
| `curvature` | `[subspace] curvature` | `str` | `"none"` | `"none"`, `"small"` | —（1D 路径有效质量：`"none"` 为单位质量，`"small"` 为小曲率穿隧 SCT 有效质量 $\mu_{\mathrm{eff}}(\sigma)$） |
| `zero_point` | `[subspace] zero_point` | `str` | `"harmonic"` | `"harmonic"`, `"rph"`, `"dvr"` | —（1D 路径法平面横向修正：历史投影 Hessian、RPH 法平面 Hessian 或一维 sinc-DVR 截口） |
| `dvr_points` | `[subspace] dvr_points` | `int` | `81` | $\ge 8$（偶数自动加 1 变奇数） | 点数 — `zero_point = "dvr"` 时每个一维法向模式截口的 DVR 网格点数 |
| `dvr_half_width` | `[subspace] dvr_half_width` | `float` | `None`（自动取 $8/\sqrt{\omega}$） | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — `zero_point = "dvr"` 时每个一维法向截口 DVR 盒子的半宽 |
| `dvr_spacing` | `[subspace] dvr_spacing` | `float` | `None`（使用 `dvr_points`） | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — `zero_point = "dvr"` 时一维法向截口 DVR 的网格间距（给定后覆盖 `dvr_points`，最少 8 点） |
| `hessian_displacement` | `[subspace] hessian_displacement` | `float` | `5e-3` | $> 0$ | $\mathrm{bohr}$ — 有限差分法平面 Hessian 中最轻原子的笛卡尔位移步长（$h_{\mathrm{mw}} = d\sqrt{\min m_i}$） |

### 2.2 反应路径构建与渐近延拓参数 (`path`)

| `.inp` 键名 (`[rate.subspace_correction.path]`) | `correct.toml` 键名 (`[path]`) | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| `source` | `source` | `str` | `"mep"` | `"mep"`, `"pmf_minimum"`, `"centroid"`, `"centroid_all"`, `"bead_weighted"`, `"least_action"`, `"instanton"` | —（反应路径核心段来源；默认 `"mep"` 从细化一阶鞍点出发双向追迹 IRC） |
| `step` | `step` | `float` | `10.0` | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 最速下降渐近延拓的质量加权弧长步长 |
| `max_steps` | `max_steps` | `int` | `None`（无步数上限） | $\ge 1$（若显式设置） | 步数 — 可选的延拓步数硬上限；耗尽该步数将导致 `path` 门禁失败（`max_steps_exhausted`） |
| `spread_factor` | `spread_factor` | `float` | `2.0` | $> 0$ | 无量纲 — 筛选取向锁定伞窗口的阈值（Kabsch 对齐离散度 $\le \text{spread\_factor} \times$ TS 离散度） |
| `asymptote_tolerance` | `asymptote_tolerance` | `float` | `1e-6` | $> 0$ | $\mathrm{E_h}$ — 末尾 $10 \times \text{step}$ 弧长窗口内的最大能量波动容差，满足即判定 `asymptote_converged` |
| `asymptote_patience` | `asymptote_patience` | `int` | `50` | $\ge 1$ | 窗口数 — 末尾能量范围连续多少个窗口未下降 10% 即终止并报 `asymptote_not_approached` |
| `smoothing` | `smoothing` | `float` | `1.5` | $\ge 0$ | 无量纲（$2\,\mathrm{median}(\Delta\xi)$ 的倍数）— 质心路径对 $\xi$ 做局部二次平滑的带宽（`0` 为关闭） |
| `reactant_xi` | `reactant_xi` | `float` | `0.0` | $< 1.0$ | 无量纲 $\xi$ — 反应物侧延拓爬出前驱反应络合物势阱时的目标反应坐标（$\xi = 0$ 对应分离参考 $R_\infty$） |
| `mep_displacement` | `mep_displacement` | `float` | `None` (`0.1 * step`) | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — `source = "mep"` 时从细化鞍点沿虚频模式向两侧跨出的初始位移 |
| `ts_gradient_tolerance` | `ts_gradient_tolerance` | `float` | `1e-5` | $> 0$ | $\mathrm{E_h}/\mathrm{bohr}$ — P-RFO 鞍点细化的最大笛卡尔振动梯度收敛阈值（用于 `"mep"` 与 `"skeleton"` 锚定） |
| `pmf_constraint_tolerance` | `pmf_constraint_tolerance` | `float` | `1e-8` | $> 0$ | 无量纲 $\xi$ — `source = "pmf_minimum"` 在各窗口超曲面极小化时的最大 $\lvert\xi(x) - \xi_{\mathrm{window}}\rvert$ 约束残差 |
| `least_action_points` | `least_action_points` | `int` | `9` | $\ge 2$ | 点数 — `source = "least_action"` 在路径与端点弦之间扫描插值参数 $\alpha \in [0, 1]$ 的网格点数 |
| `instanton_beads` | `instanton_beads` | `int` | `None`（`max(8, nbeads)` 或锚定时 `max(nbeads, 64)`） | $\ge 1$（内部取 $\ge 8$ 的偶数） | 珠子数 — `source = "instanton"` 与 `anchor = "skeleton"` 优化热瞬子时使用的全环珠子数 |
| `instanton_max_iterations` | `instanton_max_iterations` | `int` | `200`（锚定时 `300`） | $\ge 1$ | 迭代步数 — 热瞬子优化的最大迭代次数 |
| `crossover_temperature` | `crossover_temperature` | `float` | `None`（由 TS 虚频自动计算） | $> 0$ | $\mathrm{K}$ — `source = "instanton"` 在自动评估 TS Hessian 失败时使用的备用交越温度 $T_c$ |

### 2.3 精确量子求解器与 DVR 网格参数 (`quantum`)

| `.inp` 键名 (`[rate.subspace_correction.quantum]`) | `correct.toml` 键名 (`[quantum]`) | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| `spacing_1d` | `spacing_1d` | `float` | `None`（按热 de Broglie 波长 / `refinement` 自动定） | $> 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 1D 路径 sinc-DVR 沿弧长的网格间距 |
| `spacing_2d` | `spacing_2d` | `float` 或 `[float, float]` | `None`（按 $\sigma_\perp/3$ 与 $\lambda_{\mathrm{dB}}$ 自动定） | $> 0$（标量或长度为 2 的列表） | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 2D 粗 DVR 网格间距（细 RPMD 样条网格间距为 `spacing_2d / refinement`） |
| `refinement` | `refinement` | `int` | `3` | $\ge 1$ | 无量纲倍数 — 2D 势能投影与 RPMD 样条细网格相对于量子 DVR 粗网格的加密倍数 |
| `absorber_fraction` | `absorber_fraction` | `float` | `0.2` | $(0, 0.5)$（严格 $0 < x < 0.5$） | 无量纲比例 — 1D 路径两端吸收势默认占总路径长度的比例 |
| `threshold_margin` | `threshold_margin` | `float` | `1.0` | $> 0$ | $k_B T$ — 热通量算符能量下限裕度：$E_{\mathrm{floor}} = \max(E_{\mathrm{thresh},R}, E_{\mathrm{thresh},P}) - \text{threshold\_margin}\cdot k_B T$ |
| `truncation_kt` | `truncation_kt` | `float` | `30.0` | $> 0$ | $k_B T$ — 2D 能量截断 DVR 相对于 $E_{\mathrm{floor}}$ 的热能量截断高度 |
| `truncation_min` | `truncation_min` | `float` | `0.0` | $\ge 0$ | $\mathrm{E_h}$ — 与温度无关的最小能量截断裕度：有效截断偏移为 $\max(\text{truncation\_kt}\cdot k_B T,\ \text{truncation\_min})$ |
| `padding` | `padding` | `float` | `None` (`40.0`) | $\ge 0$ | $\mathrm{bohr}\cdot\sqrt{m_e}$ — 2D 模型围绕反应路径的横向边界扩展宽度（`rph` 取 $\max(\text{padding}, 6\sigma_{\perp,\max})$） |
| `dense_limit` | `dense_limit` | `int` | `8000` (流水线) / `1600` (底层 API) | $\ge 1$ | 网格点数 — 稠密 Sylvester 路线允许保留的最大 DVR 网格点数（内存约 $192 N^2$ 字节） |
| `reference_kt` | `reference_kt` | `float` | `0.3` | $> 0$ | $k_B T$ — 判定反应物自由渐近通道（及高温产物渐近区）时允许偏离端点势能的最大幅度 |
| `reference_density_tolerance` | `reference_density_tolerance` | `float` | `1e-2` | $> 0$ | 无量纲 — 反应物参考切片处允许的最大 Wigner–Kirkwood 量子/质心密度失配 $\beta^2\lvert V''\rvert/12 + \beta^3(V')^2/24$ |
| `product_drop_kt` | `product_drop_kt` | `float` | `15.0` | $> 0$ | $k_B T$ — 产物势能低于势垒顶点多少 $k_B T$ 时视为产物已不可逆生成（用于产物阱终止与吸收势起点） |

### 2.4 子空间 RPMD 求解器参数 (`rpmd`)

| `.inp` 键名 (`[rate.subspace_correction.rpmd]`) | `correct.toml` 键名 (`[rpmd]`) | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| *(自动继承顶层 `beads`)* | `nbeads` | `int` | `None`（继承 `run.nbeads`） | $\ge 1$ | 珠子数 — 环聚合物珠子数 $P$（`.inp` 中禁止单独指定，强制继承主运行 `beads`） |
| `windows` | `windows` | `int` | `48` | $\ge 4$ | 窗口数 — 从 `reactant_reference` 到 `dividing_surface` 的受约束质心热力学积分（TI）窗口数 |
| `walkers` | `walkers` | `int` | `32` | $\ge 2$ | 链数 — 每个 TI 窗口并行演化的独立 HMC 环聚合物 walker 数 |
| `burn_in` | `burn_in` | `int` | `200` | $\ge 0$ | 扫描步数 — 每个 walker 在统计平均力之前的 HMC 热化平衡扫描步数 |
| `sweeps` | `sweeps` | `int` | `400` | $\ge 2$ | 扫描步数 — 每个 walker 的生产 HMC 扫描步数 |
| `leapfrog_steps` | `leapfrog_steps` | `int` | `8` | $\ge 1$ | 步数 — 每次 HMC 提议轨迹的平均蛙跳步数（每次在 $[0.5, 1.5] \times \text{leapfrog\_steps}$ 内均匀随机） |
| `hmc_dt` | `hmc_dt` | `float` | `None` ($0.3/\omega_{\mathrm{th}}$) | $> 0$ | $\hbar/\mathrm{E_h}$（原子时间单位，$\approx 0.024189\,\mathrm{fs}$）— 受约束质心 HMC 提议积分步长 |
| `trajectories` | `trajectories` | `int` | `512` | $\ge 2$ | 轨迹对数 — 从分割面窗口抽取并发射成对 $\pm p$ 回穿子轨迹的母构型数（共 $2\times\text{trajectories}$ 条子轨迹） |
| `md_dt` | `md_dt` | `float` | `None` ($0.1/\omega_{\mathrm{th}}$) | $> 0$ | $\hbar/\mathrm{E_h}$（原子时间单位）— 无约束 RPMD 回穿轨迹的积分步长 |
| `md_time` | `md_time` | `float` | `None` ($0.3 L_0 \sqrt{\beta m_0}$) | $> 0$ | $\hbar/\mathrm{E_h}$（原子时间单位）— 回穿子轨迹的总传播时长 |
| `plateau_fraction` | `plateau_fraction` | `float` | `0.3` | $(0, 1]$ ($0 < x \le 1$) | 无量纲比例 — 在 `[0, md_time]` 末端取多大比例的时间窗口平均得到传输系数 $\kappa$ |
| `bootstrap` | `bootstrap` | `int` | `400` | $\ge 2$ | 重采样次数 — 计算传输系数标准误差的成对轨迹 bootstrap 次数 |
| `seed` | `seed` | `int` | `0` | $\ge 0$ | 无量纲整数 — 子空间 HMC 与回穿采样的随机数发生器种子 |
| *(自动)* | `backend` | `str` | `"auto"` | `"auto"`, `"native"`, `"python"` | —（在 `correct.toml` 或 API 中选择 C++17 B-样条原生扩展或纯 Python 参考实现） |

### 2.5 有效性门禁与容差策略 (`validity`)

| `.inp` 键名 (`[rate.subspace_correction.validity]`) | `correct.toml` 键名 (`[validity]`) | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| `unconverged_path` | `unconverged_path` | `str` | `"reject"` | `"reject"`, `"allow"` | —（`path` 门禁策略：反应物/产物延拓均须到达可接受终点，且 1D 反应物参考切片须位于自由通道） |
| `unstable_bath` | `unstable_bath` | `str` | `"reject"` | `"reject"`, `"allow"` | —（`bath` 门禁策略：1D 路径法平面 Hessian 不得出现高于 `soft_mode_frequency` 的虚频） |
| `outside_grid` | `outside_grid` | `str` | `"reject"` | `"reject"`, `"allow"` | —（`grid` 门禁策略：RPMD 估计量越出 $V_S$ 网格边界的比例须 $\le \text{max\_outside\_fraction}$） |
| `max_outside_fraction` | `max_outside_fraction` | `float` | `0.01` | $[0, 1]$ | 无量纲比例 — `grid_exit.estimator_outside_fraction` 的最大允许上限 |
| `submerged_surface` | `submerged_surface` | `str` | `"reject"` | `"reject"`, `"allow"` | —（`surface` 门禁策略：RPMD 分割面不得低于上游质心自由能最高点超过 `max_submerged_kt`） |
| `max_submerged_kt` | `max_submerged_kt` | `float` | `1.0` | $\ge 0$ | $k_B T$ — RPMD 分割面相对于反应物参考切片与分割面之间质心自由能瓶颈的最大允许下沉深度 |
| `folded_ribbon` | `folded_ribbon` | `str` | `"reject"` | `"reject"`, `"allow"` | —（2D RPH 模型 `ribbon` 门禁策略：$\lvert\kappa\rvert\sigma_\perp \le \text{max\_curvature\_width}$） |
| `max_curvature_width` | `max_curvature_width` | `float` | `0.5` | $> 0$ | 无量纲 — 2D RPH 模型从参考切片到产物吸收势之间 $\max \lvert\kappa\rvert/\sqrt{2\omega_n}$ 的上限 |
| `tight_truncation` | `tight_truncation` | `str` | `"reject"` | `"reject"`, `"allow"` | —（2D 能量截断 DVR 的 `truncation` 门禁策略：截断裕度须至少容纳 `min_truncation_quanta` 倍横向振动量子） |
| `min_truncation_quanta` | `min_truncation_quanta` | `float` | `2.5` | $> 0$ | 无量纲（$\hbar\omega_\perp$ 倍数）— 参考切片与分割面谷底上方的最小截断能量安全裕度 |
| `soft_mode_frequency` | `soft_mode_frequency` | `float` | `2.3e-4` | $\ge 0$ | $\mathrm{E_h}$（$\approx 50\,\mathrm{cm}^{-1}$）— 软模虚频豁免阈值（低于该值的法平面虚频视为碎片自由转动等软模，不触发 `bath` 失败） |
| `on_invalid` | `on_invalid` | `str` | `"stop"` | `"stop"`, `"diagnose"` | —（`"stop"` 跳过未通过预检门禁的模型求解；`"diagnose"` 仍求解其量子/RPMD 速率并记入 `diagnostic_factor`，但绝不用于修正速率） |

### 2.6 多核执行资源与并行调度 (`execution`)

| `.inp` 键名 (`[rate.subspace_correction.execution]`) | `correct.toml` 键名 (`[execution]`) | 类型 | 默认值 | 允许取值 / 范围 | 精确物理单位与含义 |
| --- | --- | --- | --- | --- | --- |
| `cpus` | `cpus` | `"auto"` 或 `int` | `"auto"` (`.inp`) / `1` (`correct.toml`) | `"auto"` 或 $\ge 1$ | CPU 核心数 — 总 CPU 预算（`.inp` 中 `"auto"` 由 `[parallel]` 推导，`correct.toml` 与 MPI 模式下自动检测进程/节点 CPU 亲和性），受操作系统 CPU 亲和性上限约束 |
| `workers` | `workers` | `int` | 省略（$\min(\text{未完成任务数}, \text{cpus})$） | $\ge 1$ | 进程数 — `"process"` 后端的最大并发任务进程数；本地调度器依据任务工作量与扩展上限从空闲核心预算中动态分配 OpenMP 线程并重叠执行轻量与重度 2D 任务 |
| `blas_threads` | `blas_threads` | `int` | `8`（$\min(8, \text{cpus})$） | $\ge 1$ | 线程数 — 单个任务内稠密本征求解、Schur 分解与 Sylvester 方程的最大 BLAS 线程数 |
| `memory_limit_mb` | `memory_limit_mb` | `float` | 物理内存/cgroup 的 `75%` (`.inp`) / `4096.0` (`correct.toml`) | $> 0$ | $\mathrm{MiB}$（$2^{20}$ 字节）— 工作内存上限（MPI 模式下为每节点预算），任务准入前与求解器计划均会检查 |
| *(由 `[parallel]` 自动推导)* | `backend` | `str` | `"process"` | `"serial"`, `"process"`, `"mpi"` | —（仅 `correct.toml` / Python API 可配；`.inp` 中由全局 `[parallel]` 自动决定） |
| *(始终开启)* | `resume` | `bool` | `true` | `true`, `false` | —（仅 `correct.toml` / Python API 可配；是否复用 `checkpoint_dir` 中已完成的单任务原子检查点） |
| *(固定为 `<subspace>/tasks`)* | `checkpoint_dir` | `str` | `<output.directory>/tasks` | 目录路径 | —（仅 `correct.toml` / Python API 可配；存放 `<fingerprint>.json` 任务检查点的目录） |

---

(validity-gates)=
## 3. 有效性门禁与故障排查表

### 七项有效性门禁

得出一个正的比值 $k_{\mathrm{QM}}(H_S) / k_{\mathrm{RPMD}}(H_S)$ 并不意味着它就是合法的速率修正因子：只有当 $H_S$ 本身是物理自洽的有效子空间模型时，该比值才有意义。流水线会评估六项物理门禁（`path`、`bath`、`truncation`、`grid`、`ribbon`、`surface`）和一项求解前容量门禁（`solver`）；任何未通过门禁的模型都会保留完整诊断记录，但**绝不会被应用到最终速率**。

| 门禁名称 | 控制策略 / 阈值键名 | 通过条件 | `subspace_pipeline.json` 中的诊断证据位置 |
| --- | --- | --- | --- |
| `path` | `unconverged_path` (`"reject"`) | 反应物与产物两侧延拓均到达可接受终点——渐近收敛 `asymptote_converged`、爬出反应物前驱络合物到达 $\xi \le \text{reactant\_xi}$、落入低于势垒顶点至少 `product_drop_kt` 的产物势阱，或落入能量已收敛至 `reference_kt` 内的渐近浅范德华出口阱（`product_asymptote`）——且 1D 反应物参考切片位于平坦自由通道内 | `path.reactant_termination`、`path.product_termination`（含 `status`、`accepted`、`reason`、`steps`、`arc_length`、`tail_energy_range`、`end_energy`、`drop_below_top`、`minima`、`exit_coordinate`）；`models.1.model_construction.reactant_reference` |
| `bath` | `unstable_bath` (`"reject"`)、`soft_mode_frequency` (`2.3e-4` $\mathrm{E_h}$) | 1D / 2D RPH 路径沿途或 2D `plane` 模型有效反应区域内的补空间横向 Hessian 不存在曲率低于 $-\text{soft\_mode\_frequency}^2$ 的失稳模式 | `models.D.model_construction.significant_unstable`（弧长或网格坐标位置、虚频大小、失稳性质分类） |
| `truncation` | `tight_truncation` (`"reject"`)、`min_truncation_quanta` (`2.5`) | （能量截断 2D/3D DVR）反应物参考切片与分割面处的截断安全裕度至少为 `min_truncation_quanta` $\hbar\omega_\perp$ | `solver_plans.2.truncation_confinement`（含 `truncation_quanta`、`required_truncation_min`、`recommended_truncation_min`、`recommended_dense_limit`） |
| `grid` | `outside_grid` (`"reject"`)、`max_outside_fraction` (`0.01`) | 环聚合物速率估计量采样到 $V_S$ 网格外部（常数延拓区）的份额不超过 `max_outside_fraction` | `corrections.D.rpmd.diagnostics.grid_exit` |
| `ribbon` | `folded_ribbon` (`"reject"`)、`max_curvature_width` (`0.5`) | （仅 2D RPH 模型）从反应物参考切片到产物吸收势起点之间，路径曲率与横向模式宽度之积 $\lvert\kappa\rvert / \sqrt{2 \omega_n}$ 不超过 `max_curvature_width` | `models.2.model_construction.ribbon_fold` |
| `surface` | `submerged_surface` (`"reject"`)、`max_submerged_kt` (`1.0`) | RPMD 分割面低于反应物参考切片至分割面之间质心自由能最高点的幅度不超过 `max_submerged_kt` | `corrections.D.rpmd.diagnostics.free_energy_bottleneck` |
| `solver` | `quantum.dense_limit`、`execution.memory_limit_mb` *(不可覆写)* | 规划的量子求解路线支持该模型的选项与网格规模，且量子与 RPMD 任务均不超过内存预算 | `solver_plans.D`（以及 `models.D.solver_plan`） |

### 常见故障速查表

| 报错现象 / 门禁拒绝状态 | 物理与数值成因 | 精确修复方案 |
| --- | --- | --- |
| `solver_plans.2.status = "dense_limit_exceeded"`（阶段状态 `"solver_capacity_exceeded"`） | 2D 网格在能量截断后保留的 DVR 点数 $N_{\mathrm{retained}}$ 超过 `quantum.dense_limit`（常见于高温或调大 `truncation_min` 后） | 查看 `subspace_pipeline.json` 或 `status.json` 中 `solver_plans.2` 的 `required_dense_limit` 与 `dense_memory_mb`。若物理内存充足（约 $192 N^2$ 字节），调大 `quantum.dense_limit`；否则适度增大 `quantum.spacing_2d` 或减小 `quantum.padding` 并验证网格收敛 |
| `solver_plans.D.status = "memory_limit_exceeded"` | 稠密量子或子空间 RPMD 任务的预估峰值内存超出 `execution.memory_limit_mb` | 调大 `execution.memory_limit_mb`（或减小 `execution.workers` 使单节点同时运行的任务数减少），或按 `recommendations` 缩小 2D 网格 |
| `path` 门禁失败：`asymptote_not_approached` 或 `step_collapsed` | 最速下降延拓在长程渐近尾部徘徊、跑出拟合势能面定义域，或步长在急转弯处折半至崩溃 | 检查势能面在远区的平滑性；调整 `path.step`、`path.asymptote_tolerance` 或 `path.asymptote_patience`；若原用了含噪伞质心路径，改用默认的 `path.source = "mep"` |
| `path` 门禁失败：`reactant reference slice ... rule = "not_free"` | 反应物端不存在长度至少 4 个网格间距、势能变化 $\le \text{reference\_kt}$（$0.3\,k_B T$）且 Wigner–Kirkwood 密度偏差 $\le \text{reference\_density\_tolerance}$（$10^{-2}$）的平坦自由通道 | 将反应物路径继续向渐近自由区延伸（减小 `path.asymptote_tolerance`，或在含反应物前驱络合物时将 `path.reactant_xi` 设为更小的负值） |
| 1D 模型 `bath` 门禁失败（`tangent_misaligned` 或 `ridge`） | 沿 1D 路径存在虚频高于 `soft_mode_frequency`（$50\,\mathrm{cm}^{-1}$）的法平面横向振动模式 | 保持 `dimension = "auto"`（或设为 `2` / `[1, 2]`）：流水线会自动触发维度升级（`dimension_escalation`）把失稳横向坐标纳入 2D 子空间 $S$。若由单精度拟合势在 H 原子上的差分噪声引起，可适当增大 `hessian_displacement`（如 `1e-2` bohr） |
| 2D RPH 模型 `ribbon` 门禁失败（`kappa * sigma > 0.5`） | 反应路径在鞍点附近急剧转弯，曲率半径 $1/\kappa$ 进入横向基态波函数宽度 $\sigma_\perp = 1/\sqrt{2\omega_n}$ 内部，导致 RPH 坐标折叠 | 改用固定过渡态平面模型：设置 `model_2d = "plane"` |
| `surface` 门禁失败（`surface_below_maximum_kt > 1.0`） | 淹没势垒（submerged barrier）位于入口离子–偶极络合物后方，其质心自由能比入口通道自由能瓶颈低 $1\,k_B T$ 以上，短时回穿轨迹陷在络合物内 | 将 RPMD 分割面移至门禁报告的自由能瓶颈位置 `free_energy_bottleneck.maximum_position`（纯 Python 模型可使用 `capture_surface()`） |
| `grid` 门禁失败且 `dominant: ti` 或 `dominant: surface` | 受约束 TI 或分割面上的环聚合物珠子越出了 2D 网格的横向边界 | 增大 `quantum.padding`（默认 `40.0` $\mathrm{bohr}\cdot\sqrt{m_e}$）以拓宽垂直于反应通道方向的盒子宽度 |
| `grid` 门禁失败且 `dominant: recrossing` | 回穿轨迹在势能仍向外上升（斜率 $> 0.1\,k_B T / \text{热波长}$）处离开网格，或离开网格后又折返穿过分割面 | 沿反应通道方向延伸盒子或调整吸收势起点（调大 `quantum.product_drop_kt` 或调小 `quantum.reference_kt`） |
| 报 `InputError: ... requires locally sampled umbrella windows` 或 `centroid_path.npz` 不完整 | 在检查点续跑时才中途开启 `[rate.subspace_correction]`，或主任务使用了 MetaD/OPES/外部 PMF | 从头运行带 `rate.subspace_correction.enabled = true`（或 `output.centroid_path = true`）的伞抽样；或者改用 `path.source = "mep"` 的 `pyrpmd subspace-correct` / Python API |

### 各门禁的物理机制详解

**`ribbon`（RPH 条带坐标折叠门禁）。** 2D RPH 坐标 $x(s, q) = w(s) + q n(s)$ 在 $q = 1 / \kappa$ 处发生雅可比奇异与坐标折叠。当路径曲率半径 $1/\kappa$ 与横向基态波函数宽度 $\sigma_\perp = 1 / \sqrt{2 \omega_n}$ 相当时，坐标折叠线落在波函数内部，反应路径哈密顿量不再适用（进入大曲率区）。已验证的弯曲势谷模型满足 $\kappa \sigma_\perp \in [0.21, 0.25]$；而在具有深络合物且鞍点急转弯的 8D 模型上 $\kappa \sigma_\perp = 0.56$，RPH 量子速率偏高 66%。该门禁在 $|\kappa|\sigma_\perp > \text{max\_curvature\_width}$（`0.5`）时拒绝 RPH 模型并提示改用 `model_2d = "plane"`。

**`surface`（淹没分割面门禁）。** 只有当回穿传输系数 $\kappa(t)$ 收敛到长时间极限时，Bennett–Chandler 速率才与分割面位置无关。对于位于入口离子–偶极络合物之后、能量低于反应物渐近线的淹没势垒（submerged barrier），中心鞍点并非自由能瓶颈：该处的质心自由能低于入口通道，导致穿过中心鞍点的 QTST 通量比入口通道瓶颈允许的最大捕获通量高出至多 $\exp(\beta \cdot \text{depth})$ 倍，且落入络合物势阱的轨迹在任何有限回穿时长内都会反复穿回。在 6D 淹没势垒基准模型上，穿过中心势垒的 RPMD 速率高达自由粒子通量的 2300 倍（而此时 $\kappa$ 仍停留在 0.79）。`surface` 门禁在分割面低于上游质心自由能峰值超过 `max_submerged_kt`（`1.0` $k_B T$）时拒绝该因子，并指出自由能最大值所在位置；此时应将 RPMD 分割面设在入口自由能瓶颈处（而带吸收势的精确量子速率不依赖于分割面位置）。

**`path`（反应路径与参考切片门禁）。** 路径延拓默认不设步数上限：渐近区有多远完全由势能面决定（例如离子–偶极长程尾比中性反应需要多得多的步数），靠步数截断停下的路径并未真正到达渐近区。当沿路径最近 `10 * step` 弧长窗口内的能量变化小于 `asymptote_tolerance` 时，延拓以 `asymptote_converged` 正常结束（使用弧长窗口而非固定 10 个点，是为了防止急转弯处步长减半伪造出“平坦尾部”）。若连续 `asymptote_patience`（50）个窗口内尾部能量极差未能下降 10%（如束缚坐标、跑出拟合势定义域或路径游走），则以 `asymptote_not_approached` 失败终止；最小步长下仍无法解析切向时记为 `step_collapsed`；势能求值失败或非有限时记为 `potential_failure`。`path.max_steps` 为可选项，一旦设置并触及上限，即记为 `max_steps_exhausted` 并令门禁失败。

当下降延拓遇到局部能量极小点（前驱或后继反应络合物，常见于带淹没势垒的离子–分子反应）时，会以 `minimum_reached` 停在该极小点（issue #191）：最速下降无法离开极小点，若强行沿入射切向继续推进只会爬上势壁。极小点是否可接受按反应物/产物两侧分别判定：
- **产物侧：** 当极小点低于路径最高能量至少 `quantum.product_drop_kt`（$15\,k_B T$）时，产物已不可逆生成，门禁将其接受为 `accepted: "product_well"`，且 1D 模型仅在势阱内已满足下沉阈值的区域放置产物吸收势（`models.1.product_absorber`）。
- **反应物侧：** 极小点不能作为终点，因为量子与 RPMD 速率都必须在络合物外侧的**自由反应物通道**切片上归一化。反应物延拓会记录所经极小点（`minima`），并沿反应坐标 $\xi$ 递减方向（$-\nabla\xi$，质量加权并投影掉刚体模式）逐步爬出势阱：每一步在向 $-\nabla\xi$ 推进一个步长的超平面上极小化 $V$。由于 $\xi$ 单调递减，路径绝不会折返落回阱内。当到达 $\xi \le \text{path.reactant\_xi}$（默认为 $\xi = 0$，即反应物碎片相距 $R_\infty$ 的速率参考点）时以 `coordinate_target_reached`（`accepted: "reactant_reference"`）结束；若在此之前能量已平坦则以 `asymptote_converged` 提前结束。若通过底层库 API 调用且未提供反应坐标函数，停在反应物络合物中的延拓仍会因 `bound pre-reaction complex` 被门禁拒绝。

1D 模型默认将反应物参考切片放在距反应物端 `absorber_fraction + 0.05` 倍路径总长处（位于反应物吸收势内侧）。与 2D 模型相同，该参考切片必须位于**自由反应物通道**内：切片处的裸势能（不用含软模噪声的 $V_S$）与反应物端点之差不超过 `quantum.reference_kt`（$0.3\,k_B T$），且 Wigner–Kirkwood 密度偏差不超过 `quantum.reference_density_tolerance`（$10^{-2}$）。若默认位置不满足自由通道条件，参考切片会自动移至从反应物端算起平坦通道的最后一个点，并在其前方自动收缩反应物吸收势宽度（`models.1.model_construction.reactant_reference.rule = "free_channel"`）；若不存在至少 4 个网格间距长的平坦通道（`"not_free"`），则 `path` 门禁失败。

**`bath`（横向热浴稳定性门禁）。** 振动绝热模型要求垂直于路径的法平面内所有正交模式均为束缚态。负的法平面曲率没有实零点能；直接丢弃该项会人为压低模型恰好失效处的 $V_S$。虚频低于 `soft_mode_frequency`（$2.3\times 10^{-4}\,\mathrm{E_h} \approx 50\,\mathrm{cm}^{-1}$）的模式属于分离碎片的自由相对转动等软模，程序会统计其点数（`soft_complement_points`）并给出所省略零点能的上界（`dropped_zero_point_bound`）而不判失败。超过该阈值的虚频将令 `bath` 门禁失败，并由全振动空间（切向 + 法平面）Hessian 诊断其物理性质：`tangent_misaligned`（仅有 1 个不稳定振动模，即反应模本身因路径切向偏离鞍点不稳定模 `tangent_to_lowest_mode_degrees` 而泄漏到法平面）或 `ridge`（存在 2 个及以上不稳定振动模，表明某个横向坐标处于分水岭脊线而非热浴模）。无论哪种情况，该横向坐标都必须纳入子空间 $S$，因此在 `dimension = "auto"` 下会自动触发向 2D 模型的升级（`dimension_escalation`）。法平面有限差分 Hessian 将最轻原子移动 `hessian_displacement`（$5\times 10^{-3}\,\mathrm{bohr}$），避免了早先固定质量加权步长在氢原子上仅移动 $2\times 10^{-5}\,\mathrm{bohr}$ 从而被单精度拟合势噪声淹没的问题。

**`grid`（有界网格越界门禁）。** 在有界网格外部，样条插值的 $V_S$ 被延拓为常数（力为零）。`spline_outside_fraction` 统计了所有梯度求值（包括被拒绝的 HMC 提议、平衡段、以及已经穿过平坦产物通道末端不可逆离开的回穿轨迹），无法反映速率估计量本身受污染的程度。`grid_exit` 对此做了精确分解：按阶段统计（`stages`）、真正进入热力学积分与分割面统计的已接受构型珠子越界比例（`ti_sample_outside_fraction`、`surface_sample_outside_fraction`），以及回穿轨迹中离开网格后又折返穿过分割面、或在 $V_S$ 仍向外爬升（斜率 $> 0.1\,k_B T / \text{热长度}$）处越界的受污染通量份额（`recrossing_contaminated_flux_fraction`）。门禁以三者中的最大值 `estimator_outside_fraction` 与 `max_outside_fraction`（`0.01`）比较，并在原因中标明主导项 `dominant_component`：若 `ti` 或 `surface` 主导，需增大横向 `padding`；若 `recrossing` 主导，则需沿通道延伸盒子或吸收势。

**`solver`（求解器容量与内存门禁）。** 在启动任何速率求解之前，每个维度的模型都会先生成求解器计划（`solver_plans`）：记录完整网格形状、能量截断后保留的点数 `retained_points`、量子求解器将采用的路线（`sylvester` 或 `flux_eigenstates`）、稠密与实际路线内存估算，以及在启用 `[execution]` 表时子空间 RPMD 任务的内存估算。由于 2D 模型的能量截断（`truncation_energy`）、能量下限（`energy_floor`）和参考切片线密度归一化（`normalization = "slice"`）仅在稠密 Sylvester 路线上实现，当保留点数超过 `dense_limit` 时无法求解，门禁会在求解前直接拒绝该模型（`status = "dense_limit_exceeded"`），并给出 `required_dense_limit`、所需内存与修改建议 `recommendations`。该门禁不可覆写，且在 `on_invalid = "diagnose"` 下同样生效。

各维度独立求解与门禁判定。当某维度的门禁失败时，`C_S_Dd.json` 记录 `valid: false`、`factor: null`、诊断比值 `diagnostic_factor`、门禁详情 `validity` 及量子/RPMD 两个分量结果；`apply_subspace_correction` 拒绝使用 `valid = false` 的对象。若没有任何维度通过门禁，不写出修正速率文件（并自动删除旧运行遗留的 `rate_quantum_corrected.json`），`pyrpmd subspace-correct`（及显式 `--stage subspace`）以非零状态码退出。当曲率 $\times$ 宽度推荐 2D 但仅 1D 通过门禁时，报告会在 `warnings` 中明确指出最终应用的是 1D 因子。

---

(cpus-for-the-stage)=
## 4. 多核性能与内存规划指南

### 原生 `rate` 阶段的 CPU 自动分配

原生子空间阶段直接复用同一份 `.inp` 为 `rate` 主任务配置的 CPU 资源，无需额外设置：

| 全局 `[parallel]` 配置 | 子空间阶段使用的 CPU 核心数 |
| --- | --- |
| 省略 / 默认（1 worker, 1 thread） | `1`：保持不变的串行执行路线 |
| `backend = "serial"` 或 `"process"` | `trajectory_workers × bead_workers × threads_per_rank` |
| `backend = "mpi"` | Rank 0（唯一执行该阶段的进程）的 `threads_per_rank` |

实际核心数受当前进程的操作系统 CPU 亲和性上限约束（例如 `taskset` 或 Slurm/cgroup CPU 集合）。当可用核心数 $> 1$ 时，各维度的量子（QM）与 RPMD 任务会在独立进程中并发执行（MPI 模式下在 Rank 0 内部使用线程调度），每个任务内部使用 OpenMP 并行传播环聚合物，并将 BLAS 线程数限制在至多 8 线程。内存预算默认取物理内存与 cgroup v2 `memory.max` 较小值的 75%。派生工作进程会自动设置 `OMP_WAIT_POLICY=PASSIVE`（除非环境变量已显式指定）。如需在 `.inp` 中单独覆盖：

```toml
[rate.subspace_correction.execution]
cpus = 16                 # 默认 "auto"：按上表从 [parallel] 推导；设为 1 则强制串行
# workers = 4             # 最大并发任务进程数；默认等于未完成任务数
# blas_threads = 8        # 默认 min(8, cpus)
# memory_limit_mb = 32000 # 当作业内存上限不由 cgroup v2 控制时建议显式指定
```

这些执行资源参数绝不会改变修正因子数值或因子缓存身份（`reuse_key`）：原生 RPMD 结果与线程数严格无关，随后在不同核心数下运行 `--stage subspace` 可直接复用 `subspace/tasks/` 中已完成的单任务检查点。`subspace/status.json` 会记录实际使用的执行资源预算。

### 内存估算公式与 `dense_limit` 规划

在分配任何大型数组之前，`quantum_solver_plan()` 与任务调度器会按下式估算每个任务的峰值工作内存：

- **稠密 Sylvester 量子求解路线（`method = "sylvester"`）：**
  $$
  \mathrm{Memory}_{\mathrm{dense}} \approx 64\,\mathrm{MiB} + \frac{192\,N_{\mathrm{retained}}^2 + 64\,N_{\mathrm{grid}} + 32\,N_\perp^2}{2^{20}}\,\mathrm{MiB}
  $$
  其中 $N_{\mathrm{retained}}$ 是满足 $V_S \le E_{\mathrm{trunc}}$ 的保留 DVR 网格点数（复 Schur 分解与分块上三角 Sylvester 求解期间同时驻留 4 个 $N_{\mathrm{retained}} \times N_{\mathrm{retained}}$ 复双精度矩阵，约合 $192\,N_{\mathrm{retained}}^2$ 字节）：
  - $N_{\mathrm{retained}} = 4000 \implies \sim 3.0\,\mathrm{GiB}$
  - $N_{\mathrm{retained}} = 8000 \text{（默认 } \texttt{dense\_limit}\text{）} \implies \sim 11.8\,\mathrm{GiB}$
  - $N_{\mathrm{retained}} = 11500 \text{（200 K H + H}_2\text{ 基准算例）} \implies \sim 24.3\,\mathrm{GiB}$
- **迭代热通量本征态路线（`method = "flux_eigenstates"`，不含稠密专用选项的 1D/2D/3D 网格）：**
  $$
  \mathrm{Memory}_{\mathrm{iter}} \approx 64\,\mathrm{MiB} + \frac{256\,N_{\mathrm{grid}}\,N_{\mathrm{states}} + 64\sum_a n_a^2}{2^{20}}\,\mathrm{MiB}
  $$
- **子空间 RPMD 任务（`ring_polymer_rate`）：**
  $$
  \mathrm{Memory}_{\mathrm{RPMD}} \approx 64\,\mathrm{MiB} + \frac{192\max(N_{\mathrm{win}} N_{\mathrm{walk}},\, 2 N_{\mathrm{traj}})\,P\,D + 48 N_{\mathrm{traj}}(N_{\mathrm{steps}} + 1) + 64 N_{\mathrm{grid}}}{2^{20}}\,\mathrm{MiB}
  $$

请将 `execution.memory_limit_mb` 设得略低于作业申请的物理内存上限，为 Python 解释器、动态库及主控进程留出余量。单节点运行多个并发 `workers` 时，只有当“活动任务预估内存之和 + 新任务预估内存 $\le \text{memory\_limit\_mb}$”时，调度器才会准入新任务。

(hybrid-cpu-and-mpi-execution)=
### 混合 CPU / OpenMP 与多节点 MPI 执行

有界网格上的子空间 RPMD 求解器默认优先调用 C++17 原生扩展（`backend = "auto"`）。它复制 SciPy 三次 B-样条的节点与系数，在 C++ 中直接计算完全一致的势能与解析梯度；HMC 提议轨迹与成对回穿轨迹全程释放 GIL，无 Python 回调开销。OpenMP 将独立的环聚合物分发到各线程，SIMD 加速正常模更新，自由环演化使用 FFT（非 2 的幂次珠子数使用直接 Fourier 变换）。

Python 端仍掌管 NumPy 随机数流、随机化蛙跳步数、Metropolis 接受判据、burn-in 步长自适应、walker 统计、热力学积分及成对轨迹 bootstrap。改变线程数不会改变随机数序列或归约顺序。在 Python 中单独调用 `ring_polymer_rate` 时，可传入 `threads=32`，并用 `backend="python"`（参考实现）或 `backend="native"`（强制要求编译内核）切换后端；若模型带有未网格化的 Python 势能回调，`backend="auto"` 会自动回退并记录原因。

量子求解器保留 SciPy 本征求解与复 Schur 分解；稠密 DVR 组装避免了完整的 Kronecker 临时张量，Sylvester 方程使用 LAPACK `ztrsyl` 分块三角求解与 BLAS 矩阵乘法更新（遇到病态缩放或近简并时自动回退至完整 LAPACK 求解）。

在独立 `correct.toml` 中添加 `[execution]` 表，即可并发重叠执行量子与 RPMD 任务（同时计算 1D 和 2D 时共 4 个子任务）：

```toml
[execution]
backend = "process"
cpus = "auto"        # 或显式正整数，例如 32
# workers = 4        # 可选：限制最大并发任务进程数
blas_threads = 8
memory_limit_mb = 8192
resume = true
# checkpoint_dir = "subspace-tasks"  # 默认：输出目录 / tasks

[rpmd]
backend = "native"
```

本地 `process` 调度器依据各任务的工作量估算与扩展上限动态分配空闲 OpenMP 线程，并将轻量阶段与重度 2D 任务重叠执行，避免重度任务被静态限制在 $\lfloor\text{cpus}/\text{workers}\rfloor$ 线程上。在 Python 脚本中可同样调用任务调度器：

```python
from pyrpmd.analysis import subspace_correction, subspace_corrections

# 在脚本中使用多进程后端时，请放在 if __name__ == "__main__": 保护块内
corr = subspace_correction(model, beta, 32,
    execution={"backend": "process", "cpus": 32, "workers": 2})
corrections = subspace_corrections({
    "1d": {"model": model1, "beta": beta, "nbeads": 32},
    "2d": {"model": model2, "beta": beta, "nbeads": 32},
}, execution={"cpus": 32, "workers": 4, "memory_limit_mb": 8192})
```

对于多节点集群，安装 `mpi` 附加依赖并设置 `execution.backend = "mpi"`，在所有 MPI rank 上启动相同命令（例如 2 个节点、每节点 2 个 rank、每 rank 绑定 16 核）：

```bash
mpiexec -n 4 --map-by ppr:2:node:PE=16 pyrpmd subspace-correct correct.toml
```

在 MPI 模式下，`cpus` 与 `memory_limit_mb` 均为**每个共享内存节点（per-node）**的总预算，由同一节点上的本地 rank 平均分配（`workers` 仅对本地 `process` 后端生效）。Rank 0 负责构建模型、写出流水线报告、动态分发任务并同时参与计算；rank 之间仅传递已网格化的模型和量子网格数组，不序列化活动 PES 对象或闭包。

每个完成的量子或 RPMD 任务都会以内容指纹命名（`<fingerprint>.json`）原子写入检查点目录（含完整的 RPMD 时间序列）。指纹涵盖网格、质量、温度、珠子数、势能元数据、吸收势/截断数组、数值与采样参数、随机种子、求解器实现代码哈希及 NumPy/SciPy 版本。改变线程数或 MPI rank 布局可直接复用已完成任务的检查点；任何物理或采样参数改变则会生成新指纹。

---

## 5. 模型原理与收敛判据

### 5.1 六步流水线物理原理

无论通过 `[rate.subspace_correction]` 还是 `pyrpmd subspace-correct` 调用，流水线均按以下六步执行，并将所有中间物理量写入 `subspace_pipeline.json`：

1. **反应路径核心段来源（`path.source`）。** 默认 `source = "mep"` 为下方详述的最小能量路径（IRC）；`"centroid"` 为历史默认的取向锁定窗口平均质心路径（0.2.x 之前的默认值）。`"bead_weighted"` 沿珠子绕质心协方差 $\Sigma(\xi)$ 的横向主本征矢将每个锁定窗口质心向弯道内侧平移 $\sqrt{\lambda_{\max}}$（符号由局部曲率方向决定）。`"instanton"` 在交越温度 $T_c$ 以下使用 `pyrpmd.instanton.optimize_instanton` 优化得到的折叠半环瞬子路径（在 $T \ge T_c$ 或优化未收敛时回退为质心路径并记录原因）。`"least_action"` 在平滑核心路径与端点连线弦之间做一维参数扫描，选取一维欧氏 WKB 作用量最小的路径。`"centroid_all"` 使用全部伞窗口的平均质心。

   质心是热力学平均而非真实几何构型：天平动（libration）或开环坐标的平均质心键长短于环聚合物实际访问的任何键长，且渐近区自由旋转的反应物碎片不存在有物理意义的平均几何。因此程序提供两种直接建立在势能面上的无收缩路径核心（`pyrpmd.analysis.subspace_paths`）：
   - `"mep"`（默认）：内禀反应坐标（IRC）。先在剔除刚体模式的振动子空间内用分区有理函数优化（P-RFO，有限差分 Hessian，梯度收敛判据 `ts_gradient_tolerance` 默认 $10^{-5}\,\mathrm{E_h}/\mathrm{bohr}$）将速率运行的过渡态几何细化到严格一阶鞍点，再沿唯一虚频模式向两侧跨出 `mep_displacement`（默认 `0.1 * step`），随后向两侧执行最速下降延拓。它在全局均为最小能量路径，严格满足法平面振动绝热模型的物理假设（若鞍点细化后不稳定模个数不为 1 则直接拒绝）。
   - `"pmf_minimum"`：在每个伞窗口的 $\xi = \mathrm{const}$ 超曲面上极小化势能 $V$，从最靠近过渡态的窗口开始向两侧逐窗推进以保持同一势盆连通。它用真实的物理键长覆盖整个 PMF 采样区间，且在 $\xi = 1$ 处穿过鞍点（属于特征坐标路径，$\nabla V \parallel \nabla\xi$）。
2. **平均路径与向两侧渐近区的延拓。** 对于质心类路径，程序先筛选出对齐质心保持紧凑的取向锁定窗口（离散度不超过过渡态附近的 `spread_factor = 2.0` 倍），并对 $\bar x(\xi)$ 关于采样坐标 $\xi$ 做局部二次回归平滑（对线性参考构型则由轴向平均键长重建，避免绕轴 Kabsch 旋转不确定导致的投影缩短）。随后，任何选定的核心路径都通过最速下降法向两侧延拓（每步在前方半球的球面上极小化势能并剔除刚体模式；爬坡段采用超平面极小化，急转弯处自动折半步长），直至末尾 10 个步长弧长内的能量变化低于 `asymptote_tolerance`。产物侧允许停在深后继络合物势阱中，反应物侧则沿 $-\nabla\xi$ 爬出前驱络合物直至 $\xi = \text{reactant\_xi}$（`0.0`）。
3. **曲率 $\times$ 宽度（$\kappa\rho$）诊断。** 同时在采样窗口上利用珠子协方差 $\Sigma(\xi)$ 计算 $\rho = \sqrt{2\,n^{\mathsf T}\Sigma n}$，并在整条路径上利用曲率法向模式的简谐珠子宽度计算静态 $\kappa\rho$。当最大值超过 `threshold`（默认 `0.5`）时，在 `dimension = "auto"` 下自动启用 2D 子空间。
4. **1D 反应路径模型。** 沿弧长构造法平面截断的振动绝热（`"adiabatic"`）、热自由能（`"free_energy"`）或冻结（`"frozen"`）模型并求解 $C_S(\mathrm{1D})$。当路径曲率很大时，法平面在弯道内侧相交使绝热势出现尖点（cusp），此时 $C_S(\mathrm{1D})$ 不可信，必须升级至 2D 模型。
5. **2D 子空间模型（`model_2d`）。**
   - `"rph"`（默认）：MEP 上的反应路径哈密顿量（`pyrpmd.analysis.subspace_rph`）。第 0 轴为弧长 $s$，第 1 轴为沿曲率法向 $n(s)$ 的横向位移 $q$，其余正交模式保持绝热。在 $x(s, q) = w(s) + q n(s)$ 下，动能算符为 $p_s^2 / [2(1 - \kappa q)^2] + p_q^2 / 2$（$\kappa = t'\cdot n$），弯道内侧路径更短，自然刻画切角穿隧（corner cutting）。法向 $n(s)$ 采用无扭转的 Bishop 标架沿路径平行输运（选取承载最大 Boltzmann 加权曲率的方向，报告为 `captured_curvature`）。具有该度规的二维条带与平面上的弯曲条带等距同构（isometric）：程序将其沿具有相同有符号曲率的平面曲线展开（develop）到平坦二维平面上，使动能恢复为标准平面拉普拉斯算符 $-(\partial_X^2 + \partial_Y^2)/2 + V_S$，从而直接复用高精度 2D 求解器。该模型在 2D、3D 与旋转 8D 弯曲势谷基准上与全维精确修正的误差在 3.3% 以内（而 1D 模型偏低 4–14%，见 [docs/validation/2026-11-13-subspace-rph-2d](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-13-subspace-rph-2d/README.md)）。
   - `"plane"`：穿过过渡态、由速率分割面法向（质量加权 $\nabla\xi$，故 2D 分割线严格对应 $\xi = 1$）与补空间内路径主位移方向张成的固定二维平面（对共线三原子转移反应即为精确的共线反应平面）。吸收势绑定在路径弧长上，使发夹型（hairpin）弯折通道在同一侧离开网格时也能被正确吸收。量子与 RPMD 共享同一份在 `refinement`（默认 3）倍细网格上投影的 $V_S$：量子 DVR 每隔 3 个节点取精确值，RPMD 在细网格上做三次 B-样条插值（避免在窄势谷上直接对粗 DVR 节点插值产生 Runge 振铃）。产物吸收势起点取“低于势垒最高点 `product_drop_kt`（$15\,k_B T$）”与“进入产物渐近线 `reference_kt`（$0.3\,k_B T$）以内”二者中较早达到者，保证阈值 $\max(V_{\mathrm{top}} - \text{product\_drop\_kt}\cdot k_B T,\ V_{\mathrm{asymptote}} + \text{reference\_kt}\cdot k_B T)$ 随温度连续变化。

   **反应物参考切片与网格节点对齐。** 所有模型均在反应物参考切片上按单位长度归一化量子速率与 RPMD 速率。由于在斜坡上量子密度与质心密度存在 $\mathcal{O}(\hbar^2)$ 的 Wigner–Kirkwood 偏差（$\rho_{\mathrm{Q}}/\rho_{\mathrm{cl}} = 1 - \beta^2 V''/12 + \beta^3 (V')^2/24$，而质心密度修正为 $-\beta^2 V''/24$），该偏差在 $C_S$ 中不会抵消。因此参考切片必须同时满足势能偏离端点 $\le \text{reference\_kt}$（$0.3\,k_B T$）**且** $\beta^2|V''|/12 + \beta^3(V')^2/24 \le \text{reference\_density\_tolerance}$（$10^{-2}$）。此外，在两种 2D 模型中，反应物参考切片都会对齐到量子 DVR 网格的第 0 轴节点上，确保量子线密度与 RPMD 热力学积分起点严格重合在同一切片上。
6. **嵌套灵敏度判定** $\ln(C_{\mathrm{2D}}/C_{\mathrm{1D}})$ 自动选定最终应用的维度，并写出 `rate_quantum_corrected.json`。

**放热或吸热反应的能量下限截断（`energy_floor`）。** 对于强放热或强吸热反应，计算盒子中包含低能量通道的深束缚本征态：它们的净反应通量为零，但 Boltzmann 权重极其巨大（例如 300 K 下的 Cl + HBr 高达 $\exp(25)$），吸收势的微小反射就会彻底淹没真实的反应通量。因此 1D 与 2D 模型均将热通量算符限制在高于较高通道阈值减去 `threshold_margin`（$1\,k_B T$）的本征态上（`exact_quantum_rate(..., energy_floor=...)`）。由于累积反应几率在较高通道阈值以下恒为零，该截断在连续极限下严格精确（Cl + HBr 300 K 完整验证见 [docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md)）。

(choosing-the-subspace)=
### 5.2 子空间选择与低温计算三大核心要点

子空间 $S$ 必须包含**产生 RPMD 方法误差的自由度**，而不仅仅是耦合很强的模式。静态耦合强度并非正确判据：一个沿光滑路径绝热跟随的高频模式本身就能被 RPMD 准确处理，将其加入 $S$ 对 $C_S$ 的改变往往只有百分之几。决定子空间维度的两项核心判据为：
- `curvature_width_criterion(curvature, width)` — 质量加权路径曲率与横向量子宽度（$\sqrt{2}\times\text{珠子 RMS}$）之积 $\kappa\rho$。当 $\kappa\rho \ll 1$ 时 1D 路径足以承载耦合；当 $\kappa\rho \sim 1$ 时横向振动跨越弯道（如共线 $\mathrm{H + H_2}$ 为 0.80–1.01），必须将该横向坐标纳入 $S$（默认阈值 `0.5`）。
- `nested_subspace_sensitivity([C_1D, C_2D, ...])` — 增加一个维度时 $\ln C_S$ 的变化量 $\ln(C_{d+1}/C_d)$，这是检验子空间收敛的最终判据。

完整理论推导见 [Subspace transfer theory（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-transfer-theory.md) 与 [The instanton anchor（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-instanton-anchor.md)。

#### 低温计算必读要点（避开三个经典收敛坑）

1. **能量截断 DVR 的硬壁效应（`truncation_min`）与 `dense_limit` 内存耦合：**
   在 2D 模型中，势能高于 $E_{\mathrm{floor}} + \max(\text{truncation\_kt}\cdot k_B T,\ \text{truncation\_min})$ 的网格点会被剔除以控制稠密矩阵规模。**被剔除的 DVR 点等效于无限高硬壁**，会从两侧挤压横向振动波函数。只有当截断阈值高出局部势谷底（包含势垒脊线处）**至少约 6 倍最刚性横向振动模式的量子（$6\hbar\omega_\perp$，对应硬壁位于约 5 倍基态宽度之外）**时，2D 量子速率才会收敛：
   - 在低温下，默认的 $30\,k_B T$ 远小于高频伸缩横模的 $6\hbar\omega_\perp$（例如在 $200\,\mathrm{K}$ 下 $30\,k_B T \approx 0.019\,\mathrm{E_h}$，而 $\omega_\perp = 0.010\,\mathrm{E_h}$ 的横向模式需要 $6\hbar\omega_\perp \approx 0.060\,\mathrm{E_h}$）。在 $200\,\mathrm{K}$ 可分离 2D Eckart 模型（$\omega_\perp = 0.010\,\mathrm{E_h}$）上，截断裕度为 $0.016\,\mathrm{E_h}$ 时量子速率偏低 12%，$0.03\,\mathrm{E_h}$ 时偏低 1%，达到 $0.07\,\mathrm{E_h}$ 时精确到 $1 - 4\times 10^{-6}$。
   - **操作规范：** 低温计算务必检查 `C_S_2d.json` 中的 `diagnostics.truncation_margin`，并将 `quantum.truncation_min`（单位 $\mathrm{E_h}$）显式设为约 $6\hbar\omega_\perp$。
   - **与 `dense_limit` 和内存的关系：** 调大 `truncation_min`（或在固定 `truncation_kt` 下升高温度）会显著增加保留的 2D 网格点数 $N_{\mathrm{retained}}$。一旦 $N_{\mathrm{retained}} > \text{dense\_limit}$，`solver` 门禁会在求解前拒绝该模型（`status = "dense_limit_exceeded"`），并在 `solver_plans.2` 中给出所需的 `required_dense_limit` 与稠密内存估算（$\approx 192\,N_{\mathrm{retained}}^2$ 字节）。请据此同步上调 `quantum.dense_limit` 与 `execution.memory_limit_mb`。
2. **正交环境模的有限珠数因子 `harmonic_bead_factor` ($B_P$)：1D 与 2D 模型的本质物理差异：**
   `"adiabatic"` 与 `"free_energy"` 投影将正交补空间（环境模 $E = S^\perp$）已收敛到连续极限（$P \to \infty$）的量子零点能或自由能放入了 $V_S$；但全维 $P$ 珠 RPMD 速率在环境模上仍携带有限珠数配分函数误差：
   $$
   B_P = \frac{\prod_{k \in E^\ddagger} Z_P(u_k^\ddagger) / Z_\infty(u_k^\ddagger)}{\prod_{k \in E_r} Z_P(u_{k,r}) / Z_\infty(u_{k,r})}
   \approx \exp\!\left[-\frac{\sum_{k \in E_r} u_{k,r}^3 - \sum_{k \in E^\ddagger} (u_k^\ddagger)^3}{48 P^2}\right],
   \qquad u_k = \beta\hbar\omega_k
   $$
   要使 $|\ln B_P| < \delta$，所需珠子数 $P \gtrsim \sqrt{(\sum_r u_r^3 - \sum_\ddagger u_\ddagger^3)/(48\delta)}$ 随无量纲频率按 $u^{3/2}$ 增长（例如 $200\,\mathrm{K}$ 下的 $\mathrm{H + H_2}$，$u_r = 31.7$、$u_\ddagger = 14.8$，将环境珠数误差压到 5% 以内需要 $P \approx 100$）。当全维计算的 $P$ 未完全收敛时，可先将全维 RPMD 速率除以 $B_P$，再乘上 $C_S$（调用 `harmonic_bead_factor` 与 `apply_subspace_correction(..., bead_factor=B_P)`）。
   - **1D vs 2D 下使用 $B_P$ 的关键区别：** 在子空间 $S$ 内部，分子 $k_{\mathrm{QM}}(H_S)$ 与分母 $k_{\mathrm{RPMD},P}(H_S)$ 已经在同一个 $H_S$ 和相同珠子数 $P$ 下显式求解，因此**凡是已经包含在子空间 $S$ 内部的自由度，其有限珠数误差已经被 $C_S$ 本身精确修正**。计算 $B_P$ 时**只能包含正交于 $S$ 的剩余环境模式 $E = S^\perp$，绝不能把已进入 $S$ 的模式重复计入 $B_P$**：
     - 在 **1D 路径模型**中，过渡态处全部 $3N - 7$（或 $3N - 6$）个横向简正模与反应物全部振动模都属于正交环境 $E = S^\perp$（包括面内对称伸缩模與反应物键伸缩模），均需计入 $B_P(\mathrm{1D})$。
     - 在 **2D 子空间模型（`"rph"` 或 `"plane"`）**中，第二个子空间坐标（共线转移平面内的曲率法向/对称伸缩模及反应物伸缩振动）**已经显式进入 2D 子空间 $S$**，其有限 $P$ 误差已由 $C_S(\mathrm{2D})$ 承担。因此 $B_P(\mathrm{2D})$ **只能对剩余的 $3N - 8$ 个面外正交补空间模式（如三原子共线反应的简并弯曲模）求积**。若把包含反应物高频伸缩模的 1D $B_P$ 错误地套用在 $C_S(\mathrm{2D})$ 上，就会对该伸缩模的有限珠数误差造成双重修正。
3. **热激发环境模（`environment = "free_energy"`）与深穿隧瞬子骨架锚定（`anchor = "skeleton"`）：**
   - 当正交环境模式满足 $\beta\hbar\omega \lesssim 4$（存在显著热激发布居）时，应设置 `environment = "free_energy"`，使环境耦合的热平均在一阶微扰下严格精确（`"adiabatic"` 仅使用基态零点能，在 $\beta\hbar\omega = 2$ 时一阶耦合系数存在 31% 偏差）。
   - 在交越温度 $T_c$ 以下的深穿隧区，慢横向模式（$\omega_\perp \lesssim \lambda^\ddagger$）或 1D 降维可能改变瞬子最速下降缺陷 $A = \ln[2\pi / (\beta\hbar\eta_1)]$。设置 `anchor = "skeleton"` 会在 $\ln C_S$ 上叠加上全维体系相对于**裸势骨架模型 $S^0$** 的瞬子缺陷差 $A_H - A_{S^0}$（$\ln C_{\mathrm{anch}}^{\mathrm{skel}} = \ln C_S + A_H - A_{S^0}$），既消除了降维骨架的穿隧质量/交越温度偏差，又避免了直接减去 $A_S$ 对横向零点能软化的双重计数。

(numerical-parameters-to-converge)=
### 5.3 数值收敛检查清单

| 待收敛参数 | 配置位置 | 取值过小或不当的典型症状 |
| --- | --- | --- |
| 盒子长度与吸收势宽度/强度 | `exact_quantum_rate(cap_width=, cap_strength=)`，流水线 `absorber_fraction`、`product_drop_kt`、`reference_kt` | 吸收势反射导致量子速率振荡或偏高（200 K 下 16 bohr Eckart 盒子偏高 0.7%，32 bohr 降至 0.06%） |
| 网格间距 | 模型 `axes`，流水线 `spacing_1d`、`spacing_2d`、`refinement` | 加密网格时量子速率或 RPMD 自由能发生漂移 |
| 稠密路线容量上限 | 流水线 `quantum.dense_limit` | 并非物理收敛参数：当 2D 截断后保留点数超过该值时，模型在求解前被 `solver` 门禁拒绝（`solver_plans.2.status = "dense_limit_exceeded"`），并在计划中给出 `required_dense_limit` 与内存需求（约 $192\,N^2$ 字节） |
| 能量截断阈值（能量截断 DVR） | `exact_quantum_rate(truncation_energy=)`，流水线 `truncation_kt` 与 `truncation_min` ($\mathrm{E_h}$) | 被剔除的网格点相当于无限高硬壁。只有当局部势谷底以上的截断裕度超过约 6 倍最刚性横模量子（$6\hbar\omega_\perp$）时速率才收敛；低温下务必检查 `diagnostics.truncation_margin` 并设置 `truncation_min` |
| 传播总时长与时间步长 | `max_time`、`dt`（迭代量子路线） | `relative_extrapolation_spread` 偏大 |
| 热力学积分窗口数 | `ring_polymer_rate(windows=)`，流水线 `rpmd.windows` | `ti_quadrature_log_error` 与统计标准误差相当 |
| 环聚合物珠子数 | `nbeads`、`harmonic_bead_factor` ($B_P$) | $C_S$ 须修正与全维运行相同珠子数 $P$ 的估计量；对正交环境模 $E = S^\perp$ 除以 $B_P$ 或直接收敛 $P$ |
| 子空间 RPMD 采样预算 | `walkers`、`sweeps`、`trajectories` | `log_standard_error` 过大，导致 `nested_subspace_sensitivity` 无法分辨维度差异（`undetermined`） |
| RPMD 采样网格边界 | `quantum.padding`、`grid_exit.estimator_outside_fraction` | 越出网格的环聚合物会采样到常数延拓区；`grid_exit.estimator_outside_fraction` 必须保持在 `max_outside_fraction`（`0.01`）以下 |

(validation)=
### 5.4 验证记录与相关量子修正工具对照

- **2D Eckart + 势垒软化热浴（`examples/subspace_correction/eckart_bath_2d.py`）：** 对称 Eckart 势垒（$V_0 = 0.425\,\mathrm{eV}$，$a = 0.66\,\mathrm{bohr}$，$m = 1060$）耦合在势垒处软化 30% 并偏移 $0.2\,\mathrm{bohr}$ 的横向振动：
  [docs/validation/2026-11-07-subspace-correction](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-07-subspace-correction/README.md)。
- **Cl + HBr 重–轻–重共线氢转移（300 K）：**
  [docs/validation/2026-11-08-clhbr-subspace](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-08-clhbr-subspace/README.md)。
- **子空间传递理论基准测试（600、300、200 K）：**
  [docs/validation/2026-11-09-subspace-transfer](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-09-subspace-transfer/README.md) 与 [Subspace transfer theory（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-transfer-theory.md)。
- **2D、3D 与旋转 8D 弯曲势谷上的 2D RPH 与平面模型对比：**
  [docs/validation/2026-11-13-subspace-rph-2d](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-13-subspace-rph-2d/README.md)。
- **2D 势谷（M1–M4）与 6D–10D 复杂势能面上的瞬子骨架锚定：**
  [docs/validation/2026-11-14-subspace-instanton-anchor](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/validation/2026-11-14-subspace-instanton-anchor/README.md) 与 [The instanton anchor（英文）](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/subspace-instanton-anchor.md)。
- **单元测试（`tests/test_subspace_*.py`、`tests/test_rate_subspace.py`）：** 检验精确量子求解器与解析 Eckart 透射系数的一致性、迭代路线与稠密路线的一致性、可分离 2D 速率与 1D 速率的一致性、单珠 RPMD 与经典 TST 的一致性、各类子空间投影与解析绝热势的一致性、全部六项有效性门禁以及 `correct_rate_run` 的溯源与温度校验。

[量子修正英文指南](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/quantum-correction.md)涵盖频域响应（光谱）修正、在线 Ehrenfest 传播器与含时通量–布居曲线传递。对于热反应速率常数，请使用本页介绍的子空间速率修正模块。`hybrid_flux_side_correction(..., mode="ratio")` 与 `adaptive_quantum.matched_rate_defect(..., mode="ratio")` 可对外部计算好的曲线或平台值执行同样的乘性比值传递（其默认的加性模式对子空间模型的绝对速率不具有不变性）。
