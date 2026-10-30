# H + BrCl：多通道二维质心 PMF 回归

本目录使用公开的 **BrHCl_LEPS1 历史半经验反应势**，计算两个不同化学产物通道：

$$
\mathrm{H+BrCl\rightarrow HCl+Br},\qquad
\mathrm{H+BrCl\rightarrow HBr+Cl}.
$$

这是有明确来源的反应模型上的数值回归，用于检验二维 umbrella/MBAR、隐藏自由度采样、
珠子数对照和有支持网格上的通道连接。两种产物不是同种原子的等价标签交换。
反应本身的两个产物通道有原始研究支持[1]，但本基准不把 LEPS1 的计算解释为现代精度的
真实散射分支预测。**默认采样含取向 Metropolis 更新，只产生平衡构型样本；其时间序列
不能用于散射动力学、反应次数或速率。**

本文给出模型定义与复现方法，不预先给出任何运行的收敛结论。具体运行的证据以各自保存的
计划、模型校验、样本和分析报告为准。

## 模型、原子顺序与来源

源码固定为 ChemPotPy 提交
[`4716d22e87ae01770cad9ec995c4a02a8e479296`](https://github.com/shuyinan/chempotpy/tree/4716d22e87ae01770cad9ec995c4a02a8e479296)
中的
[`chempotpy/BrHCl/BrHCl_LEPS1.f`](https://github.com/shuyinan/chempotpy/blob/4716d22e87ae01770cad9ec995c4a02a8e479296/chempotpy/BrHCl/BrHCl_LEPS1.f)。
构建时检查字节数与 SHA256，下载的第三方源码保留在忽略提交的 `build/` 目录。
参数、许可来源、原始转换常数、能量零点和适配器细节见 [SOURCE.md](SOURCE.md)；
机器可读的固定来源在 [source.json](source.json)。

固定原子顺序为 **Br, H, Cl**，原子序数为 **35, 1, 17**。
扫描使用同位素 **⁷⁹Br、¹H、³⁵Cl**，质量记录在 `plan.json`。
模型为一个中性 doublet 电子面。它包含 HBr、HCl、BrCl 三种二原子渐近构型；
源码引用的原参数研究主要针对 Br + HCl 氢转移[2,3]。
将同一个解析面用于 H + BrCl 可以检验多产物计算，但不构成对这个入口的独立电子结构验证。
它也不解析卤素精细结构和非绝热耦合。

PyRPMD 适配器输入 bohr，输出 hartree 和 **力 `−dE/dq`**；固定原子顺序不会自动重排。
适配器直接调用原 POTLIB 的原子单位接口，保留原参数与舍入常数。
`BrHClPotential()` 默认不带墙；扫描显式加上下面定义的共同约束。

模型明确拒绝任意原子对距离小于 **0.2 bohr** 的构型。参考积分也限制在同一有效域中，
没有估算这个被排除 core 的物理概率。其他 PES 失败，包括 LEPS 奇异导数，不会被静默
改成零能量、普通拒绝或缺失 bin。

## 安装、构建与接口检验

以下命令均从仓库根目录运行。需要 Python 3.11+、GNU Fortran、C++17 编译器及原生
PyRPMD 内核；本扫描要求 native propagation，不使用纯 Python 回退。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

# 下载并编译经过校验的固定源码；构建会记录编译器、命令及文件哈希。
python benchmarks/brhcl/build_pes.py

# 检查解析公式、渐近能、力、刚性变换、批量接口和共同墙。
python -m benchmarks.brhcl.pes --check \
  --output runs/brhcl-validation/pes-verification.json

# 包括真实 PES、几何分类、P1 积分、取向 MC 和扫描完整性回归。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest tests/test_brhcl_*.py -q
```

仓库的 [BrHCl multichannel PMF regression](../../.github/workflows/real-brhcl.yml)
工作流会先构建固定来源的真实 PES，再运行这些回归和 1D/2D 精度测试，
使需要该势能面的测试在 CI 中实际执行。

离线构建使用 `python benchmarks/brhcl/build_pes.py --source /path/to/BrHCl_LEPS1.f`，
仍须满足同一来源校验。默认二进制是 `benchmarks/brhcl/build/libbrhcl.so`；
扫描可用 `--library /path/to/libbrhcl.so` 指定其他构建，但必须具有配套的构建来源文件。
修改适配器或二进制后应重新构建并另开运行目录。

## 扫描设置与复现命令

默认 `pilot` 的参数由 [scan.py](scan.py) 的 `plan()` 生成并写入 `plan.json`：

| 参数 | 默认值 |
| --- | --- |
| 温度 / 珠子数 / MD 步长 | 2000 K / `P=1` / 0.1 fs |
| 窗口中心 | `r_HBr` 为 1.15–4.95 Å；`r_HCl` 为 1.00–4.80 Å；各 11 点 |
| 每窗起始族 | `compact`、`open`，共 121 × 2 = 242 条流 |
| 二维 umbrella 刚度 | 对角阵，每轴 80 kcal mol⁻¹ Å⁻²；能量含标准 `1/2` 因子 |
| 每流热化 / 生产 | 500 fs / 1000 fs 的 MD 段 |
| 保存 / Andersen 刷新 / checkpoint | 每 1 fs / 10 fs / 100 fs |
| 取向 MC | 每次 Andersen 刷新前 32 个 Haar 候选 |
| 共同墙 | 半径 6 Å；刚度 20 kcal mol⁻¹ Å⁻² |
| 分析网格 | 32 × 32；范围 0.95–5.25 Å、0.85–5.10 Å |
| 基本支持门槛 | 每 bin 原始样本数 ≥20、重要性权重 ESS ≥20 |
| 默认 bootstrap | 12 次；用于初步误差诊断，不预设区间已稳定 |

`compact` 的第三距离接近 BrCl 平衡长度 2.136 Å，`open` 接近 `0.9*(r_HBr+r_HCl)`；
初始化会夹到严格满足三角不等式的范围，再作有偏几何松弛。这是两个横向起始族，
不是预先指定两个最终产物。不同起始族的一致性须从保存数据检验。

```bash
# 只写计划，不启动采样或声称完成计算。
python benchmarks/brhcl/scan.py plan --profile pilot --beads 1 \
  --output runs/brhcl-plan

# 先作短 smoke，验证构建、native loop、checkpoint 和分析连接。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile smoke --beads 1 \
  --workers 4 --output runs/brhcl-smoke
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 0 \
  --output runs/brhcl-smoke

# P1：有限约束经典 PMF，并与独立数值积分比较。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 1 \
  --dt-fs 0.1 --seed 20261030 --orientation-mc-trials 32 \
  --workers 4 --output runs/brhcl-p1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --reference --bootstrap 12 \
  --output runs/brhcl-p1

# P8：另开目录；经典积分不作为 P>1 质心 PMF 的真值。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 8 \
  --dt-fs 0.1 --seed 20261031 --orientation-mc-trials 32 \
  --workers 4 --output runs/brhcl-p8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 12 \
  --output runs/brhcl-p8
```

`run` 负责采样，`analyze` 负责重建；应分别执行。`--reference` 仅在 `analyze` 且 `P=1`
时使用。重复同一 `run` 命令可从 checkpoint 恢复；恢复保留 q/p、RNG、MC 计数和采样时钟。
相同输出目录不接受改变科学参数的计划。步长、珠子数、seed、墙半径、生产长度或 MC 次数的
对照必须另开目录。`--equilibration-fs`、`--production-fs` 可以设置新的采样长度；
`--save-interval-fs` 可以改变保存间隔，但必须与步长、阶段长度和 10 fs Andersen 周期对齐；
`--orientation-mc-trials 0` 可生成不含取向更新的对照，不应假定这个对照已充分混合。

计划也报告资源规模。默认 pilot 共保存 242,000 个构型；仅一个未合并的 float64
`[streams,total_samples]` reduced-potential 矩阵就需 468,512,000 字节，实际分析峰值更高。
`--workers` 控制采样进程，不能消除 MBAR 分析的内存分配。每个进程使用一个 native
线程，环境变量限制底层数学库线程，避免多进程过度争用。

### 较长采样、珠子数与步长对照

下面命令复现本次较长采样和敏感性检查的参数。四组每流热化均为 **2 ps**，每次
Andersen 刷新前使用 **64 个取向候选**，但生产长度和保存间隔分为两组：

| 运行 | 珠子数 | MD 步长 / fs | 生产 / ps | 保存间隔 / fs |
| --- | ---: | ---: | ---: | ---: |
| `p1-final` | 1 | 0.10 | 10 | 10 |
| `p8-final` | 8 | 0.10 | 10 | 10 |
| `p16-final` | 16 | 0.10 | 5 | 5 |
| `p8-halfdt-final` | 8 | 0.05 | 5 | 5 |

这些运行保持同一固定 seed，方便复现，不替代独立 seed 重复。每流均保存 1000 个生产
构型，但两个较短对照的采样时长只有主运行的一半，保存构型数相同不表示有效样本数相同。
`P16` 和半步长结果是**敏感性检查，不足以单独证明珠子数或步长收敛**；二者与主运行的
差异也包含采样噪声及采样长度差异。具体证据以运行报告为准。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 1 \
  --dt-fs 0.1 --seed 20261030 --equilibration-fs 2000 --production-fs 10000 \
  --orientation-mc-trials 64 --save-interval-fs 10 \
  --workers 4 --output runs/brhcl-validation-20261030/p1-final
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --reference --bootstrap 20 \
  --output runs/brhcl-validation-20261030/p1-final

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 8 \
  --dt-fs 0.1 --seed 20261030 --equilibration-fs 2000 --production-fs 10000 \
  --orientation-mc-trials 64 --save-interval-fs 10 \
  --workers 4 --output runs/brhcl-validation-20261030/p8-final
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 20 \
  --output runs/brhcl-validation-20261030/p8-final

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 16 \
  --dt-fs 0.1 --seed 20261030 --equilibration-fs 2000 --production-fs 5000 \
  --orientation-mc-trials 64 --save-interval-fs 5 \
  --workers 4 --output runs/brhcl-validation-20261030/p16-final
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 0 \
  --output runs/brhcl-validation-20261030/p16-final

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py run --profile pilot --beads 8 \
  --dt-fs 0.05 --seed 20261030 --equilibration-fs 2000 --production-fs 5000 \
  --orientation-mc-trials 64 --save-interval-fs 5 \
  --workers 4 --output runs/brhcl-validation-20261030/p8-halfdt-final
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 0 \
  --output runs/brhcl-validation-20261030/p8-halfdt-final
```

### 复用已校验的 P1 积分缓存

可从保存的 P1 计划和模型身份独立重建参考，不需要重新采样：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/reference.py \
  --run runs/brhcl-validation-20261030/p1-final \
  --output runs/brhcl-validation-20261030/reference32.npz
```

输出若已存在会拒绝覆盖；重算时选一个新的文件名。默认容差为 0.01 kBT，
可用 `--tolerance-kbt` 收紧。

已有 `reference32.npz` 及其同目录同名 `reference32.json` 元数据时，可以在重分析中复用
它们，避免重复确定性的三距积分。缓存用于独立参考，不复用任何轨迹的 MBAR 点估计或
bootstrap draw：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/brhcl/scan.py analyze --bootstrap 20 \
  --output runs/brhcl-validation-20261030/p1-final \
  --reference-cache runs/brhcl-validation-20261030/reference32.npz
```

`--reference-cache` 已包含启用参考比较的含义，无需同时传 `--reference`。读取时检查
归档完整性、源码 pin、PES 二进制哈希、`P=1`、温度/β、共同墙、core 有效域、原子与 CV
顺序，以及每个实际 bohr bin 边界。缓存和 `.json` 必须成对保留；科学目标不一致会拒绝，
不能复用到 P8/P16、不同墙或不同网格。没有匹配缓存时，使用前述 `analyze --reference`
重新积分；不要绕过身份检查。

## 坐标、测度与共同墙

令每种原子的 ring-polymer 质心为 $\bar q_a=P^{-1}\sum_b q_{a,b}$，取

$$
x=|\bar q_{\mathrm{Br}}-\bar q_{\mathrm H}|,\qquad
y=|\bar q_{\mathrm H}-\bar q_{\mathrm{Cl}}|.
$$

这里是**质心之间的距离**，不是逐珠距离的平均。保存的自由能是相对于 `dx dy` 的
质心概率密度 PMF，保留该坐标的构型熵，不额外移除径向 Jacobian。
第三个质心距离 $z=|\bar q_{\mathrm{Br}}-\bar q_{\mathrm{Cl}}|$ 不受 umbrella 直接控制，
但保存在几何中并用于通道判断和慢模诊断。

给定有限的 x、y 时，z 由三角不等式约束；整个解离体系的无约束平衡仍存在体积归一化问题。
扫描因此在**每个珠子的所有三对原子**上使用同一个平底软墙：

$$
W(q_b)=\frac12 k_\mathrm{wall}
\sum_{a<c}\max(0,|q_{a,b}-q_{c,b}|-R_\mathrm{wall})^2.
$$

墙在所有窗口一致，位于物理势适配器内，在移除 umbrella 后仍保留。因此结果是
**有限共同约束下、单电子面上的平衡 PMF**。它没有自动的无限体积或标准态解释。
比较不同珠子数时保持墙定义一致；改变墙半径则是在检查这个额外约束的影响。

## P1 独立积分基准

[reference.py](reference.py) 对同一个 PES 和共同墙进行数值积分，不使用轨迹或 MBAR 权重。
当 `P=1` 时，去掉整体平移并积分整体转动后，三距构型测度为

$$
\mathrm d\mu\propto x y z\,\mathrm dx\,\mathrm dy\,\mathrm dz.
$$

因此未归一化的二维经典密度为

$$
\widetilde\rho(x,y)=x y
\int_{\max(|x-y|,r_\min)}^{x+y}
z\exp[-\beta(V(x,y,z)+W(x,y,z))]\,\mathrm dz,
\qquad x,y\ge r_\min,
$$

其中 $r_\min=0.2$ bohr 是声明的模型域下界，域外密度为零；$\beta=(k_BT)^{-1}$。
省略的整体常数不影响自由能差。参考概率归一化到给定有限 x/y 分析矩形；被排除 core 的
概率没有通过这个计算得到物理上界。

与 histogram PMF 比较时，进一步对每个完整 bin 的面积平均密度：

$$
\rho_{ij}=\frac1{\Delta x_i\Delta y_j}
\int_{x_i}^{x_{i+1}}\!\int_{y_j}^{y_{j+1}}\rho(x,y)\,\mathrm dy\,\mathrm dx,
\qquad F_{ij}=-k_BT\ln\rho_{ij}+C.
$$

这一步包含隐藏距离的熵和有限 bin 效应，不能换成 `min_z V`、最低能量路径或 bin 中心的
势能切片。程序逐次提高 Gauss–Legendre 阶数，保存每个 bin 的阶数、误差估计和收敛标记。
默认相邻阶数的 log-density 差容差为 0.01，即 0.01 $k_BT$ 的自由能差；这只是数值
收敛诊断，不是严格误差界。比较仅使用同时有采样支持、积分收敛的 bin，并先对齐自由能常数。
这个基准**仅适用于 P1**，不能验证 P8 的量子质心分布。

## 为什么加入 Haar 取向 MC

在固定 x、y 时，BrCl 距离与分子取向仍可有很慢的松弛。二维窗口 overlap 连通，不保证
这些隐藏自由度已经混合[5,6]。本基准在 Andersen 动量刷新之前，加入保持目标构型测度的
取向更新；`compact/open` 分别分析及第三距离相关性仍用于检查其效果。

对整个 Cl ring 使用同一个独立 Haar $SO(3)$ 旋转，绕 H 质心作

$$
q'_{\mathrm{Cl},b}=\bar q_H+R(q_{\mathrm{Cl},b}-\bar q_H),
\qquad R\sim\mathrm{Haar}(SO(3)).
$$

Br/H 的所有珠子不变。这保留两个 umbrella CV，也保留 Cl 相邻珠子差分的范数，因此
全部 ring spring 和 umbrella 项不变。坐标变换 Jacobian 为 1，且 Haar 逆旋转同分布，
所以无需额外的角度 Jacobian。接受率为

$$
\alpha=\min\left\{1,
\exp\left[-\frac\beta P\sum_b
\big((V+W)(q'_b)-(V+W)(q_b)\big)\right]\right\}.
$$

[sampling.py](sampling.py) 从同一旋转轨道批量生成独立候选，按顺序相对于**最新已接受状态**
作 Metropolis 判断。使用的是逐珠物理势加共同墙的能量之和，而非质心构型能量。
已声明 core 内的候选被拒绝；其他 PES 异常照常中止。随后独立刷新全部动量，再传播下一段 MD。

取向 MC 子核精确保留所定义的构型测度；有限时间步 MD 本身仍可能造成离散偏差，因此仍需
步长对照和 P1 积分检验。MC 次数不是独立样本数，接受率也不证明混合。命令中的 fs 是 MD
段的工作量；加入 MC 后，该序列不是具有可解释反应时间的碰撞轨迹。

## 通道判据与二维连接

[channels.py](channels.py) 根据固定 Br,H,Cl 顺序的**全部三条质心距离**分类：

| 标签 | 必须形成的键 | 另外两对 |
| --- | --- | --- |
| `H+BrCl` | Br–Cl | 均明确断开 |
| `HCl+Br` | H–Cl | 均明确断开 |
| `HBr+Cl` | Br–H | 均明确断开 |
| `intermediate/ambiguous` | 灰区、额外成键、全部解离或重叠 | 保留不确定性 |

| 原子对 | 模型平衡距离 / Å | formed 上界 / Å | broken 下界 / Å |
| --- | ---: | ---: | ---: |
| HBr | 1.414 | 1.80 | 2.35 |
| HCl | 1.275 | 1.65 | 2.15 |
| BrCl | 2.136 | 2.65 | 3.35 |

边界定义为 `r <= formed` 成键、`r >= broken` 断开，中间保持灰区。formed 默认界比模型
平衡键长高约 25–30%，属于可配置的诊断选择，不是经动力学标定的产物判据。
任意质心距离小于 0.40 Å 另标为重叠；这个几何诊断界与 PES 的 0.2 bohr 有效域界不同。
质心分类不表示每个珠子具有同样的连接图，也不表示产物已经动力学承诺。

先用 MBAR 权重计算各 bin 的条件通道占比，分母包含 ambiguous 构型。一个通道在至少
80% 占比的受支持格子中选择最低自由能格作为盆地种子。三个标签对应不同化学组成，
不合并，也不把混合格子重新归一化成纯通道。缺少这样的种子，不能据此证明通道不存在。

连接使用共享的四邻接 minimax 实现：在受支持网格上寻找最高自由能最小的路径。
它不跨越 NaN、Inf 或 masked bin，也不沿对角穿过支持缺口。全局 overlap 不连通时
全局 PMF 被 mask；原始数量和权重 ESS 门槛还会进一步限制支持范围。

**所得连接只是二维投影上的支持路径。** 不同高维构型可以投影到相邻甚至同一个 bin；
它不是全维势能面的最小能量路径（MEP）。路径最高点不能自动称为真实过渡态，
也不能用于散射分支比或速率。路径本身尚无 bootstrap
置信区间。默认 ±5% 阈值对照检查标签与盆地选择，不等于路径屏障已对阈值收敛。

## 分析输出与结论边界

主要输出包括：

- `plan.json`、`model.json`：科学参数、同位素、模型与二进制身份。
- `streams/*.npz`：CV、质心构型、q/p、RNG、时钟、完整性校验与传播/MC 诊断。
- `sampling.json`：运行工作量与耗时说明；恢复运行的计时不等于全部样本的新采样耗时。
- `fes.npz`：有支持的 PMF、SE、mask、样本数量、权重 ESS、通道比例、overlap，
  以及两起始族的独立 PMF；启用 `--reference` 或 `--reference-cache` 时还含 P1 积分及误差数组。
- `summary.json`：投影盆地和连接、逐流通道/第三距离诊断、两起始族比较、阈值敏感性及限制。
- `reference-summary.json`：使用参考积分或其缓存时的积分阶数历史和与 P1 采样的比较。

两起始族、独立积分及不同珠子数/步长的 PMF 比较，均先在有限共同支持格子上减去
自由能差的算术平均，再计算以 $k_BT$ 为单位的 RMS。格子等权，不按平衡概率加权；
它是所比较投影区域的形状差异，不能解释为整体体系的概率误差。不同对照的共同支持
可能不同，比较 RMS 时应同时查看使用的格子数和 mask。

相关块 bootstrap 对每条流同时估计两个 umbrella CV、第三距离和夹角余弦的统计低效因子，
使用

$$
g=\max(g_x,g_y,g_z,g_{\cos\theta}),\qquad
\cos\theta=
\frac{(\bar q_{\mathrm{Br}}-\bar q_H)\cdot(\bar q_{\mathrm{Cl}}-\bar q_H)}{xy}.
$$

第三距离 z 与 `cos(Br–H–Cl)` 通过逐帧对齐的 `correlation_observables` 传入核心分析，
所得 g 不小于只依据两个 CV 的值。这些额外观测量仅影响相关性诊断和 bootstrap 块计划，
**不改变 MBAR 的原始样本数、点估计权重或
目标测度**，也不把它们变成新的有偏 CV。请求块长为 `ceil(5*g)`，实际长度仍受
`max(1,floor(N/2))` 上限约束，并保存截断、块数和有效性状态。
这将已显式监测的横向相关性纳入误差诊断，但不能保证未观测自由度已经混合，特别是 `P>1` 的内部珠子模。
权重 ESS 仍未作时间相关修正。
必须同时查看第三距/CV 的相关性比值、compact/open 的差异、支持范围、失败 draw，以及
P1 的独立积分误差，而非只看重叠图或误差条。

默认只有 12 次 bootstrap 重复，即使全部成功，也只提供初步有限样本 SE 诊断，不足以
宣称置信区间尾部或屏障差的误差已经稳定。`limited_blocks` 表示至少一条流少于 20 个
块或块长被截断，此时有限 SE 仍不可当作可靠区间；`unavailable` 时 SE 全部保留为 NaN。
还须检查每个 bin 的有效 draw 数，不能把某些 bin 的有限 SE 推广为整张曲面的精度保证。
目前通道条件占比、盆地选择、投影路径与不同运行之间的 RMS 没有联合 bootstrap 区间。

`P=1` 是经典基准，`P=8` 只是量子珠子数对照。获得物理上收敛的结论还需要适当的珠子数、
步长、独立随机种子、采样长度、取向更新强度、墙半径和网格对照。本例始终保留
`physical_converged=False`，不把进程正常结束、求解器收敛或某个单项检验当成全部误差已受控。

## 原始文献

1. J. C. Polanyi and W. J. Skrlac. “Energy distribution among reaction products. X.
   H + ICl → HCl + I, H + ClBr → HCl + Br, HBr + Cl.” *Chemical Physics* **23**,
   167–194 (1977). [DOI: 10.1016/0301-0104(77)89001-0](https://doi.org/10.1016/0301-0104(77)89001-0).
2. D. J. Douglas, J. C. Polanyi and J. J. Sloan. “Effect of changing reagent energy on
   reaction dynamics.” *Chemical Physics* **13**, 15–31 (1976).
   [DOI: 10.1016/0301-0104(76)80121-8](https://doi.org/10.1016/0301-0104(76)80121-8).
3. J. A. Kaye and A. Kuppermann. “Collinear quantum mechanical probabilities and rate
   constants for the Br + HCl(ν = 2, 3, 4) reaction using hyperspherical coordinates.”
   *Chemical Physics Letters* **92**, 574–580 (1982).
   [DOI: 10.1016/0009-2614(82)83653-1](https://doi.org/10.1016/0009-2614(82)83653-1).
4. P. J. Kuntz, E. M. Nemeth, J. C. Polanyi, S. D. Rosner and C. E. Young. “Energy
   Distribution Among Products of Exothermic Reactions. II. Repulsive, Mixed, and
   Attractive Energy Release.” *Journal of Chemical Physics* **44**, 1168–1184 (1966).
   [DOI: 10.1063/1.1726803](https://doi.org/10.1063/1.1726803). 源码所引扩展 LEPS 形式。
5. M. R. Shirts and J. D. Chodera. “Statistically optimal analysis of samples from
   multiple equilibrium states.” *Journal of Chemical Physics* **129**, 124105 (2008).
   [DOI: 10.1063/1.2978177](https://doi.org/10.1063/1.2978177).
6. F. Zhu and G. Hummer. “Convergence and error estimation in free energy calculations
   using the weighted histogram analysis method.” *Journal of Computational Chemistry*
   **33**, 453–465 (2012). [DOI: 10.1002/jcc.21989](https://doi.org/10.1002/jcc.21989).
