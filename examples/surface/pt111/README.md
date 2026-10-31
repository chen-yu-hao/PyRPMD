# 真实 CO/O/Pt(111)：三维 MetaD 与完整原子动力学

本例使用 39 原子、4 层 3×3 Pt(111)，固定底部 18 个 Pt，另外 21 个原子
（63 个笛卡尔自由度）参与动力学。PES 来自 Li 等人的公开 EANN；没有把
金属表面替换成一维势。PyRPMD 对所有活动坐标传播，三个 CV 只定义偏置。

论文：C. Li et al., *First principles reactive flux theory for surface reactions
revealing multiple channels and recrossing dynamics*, Nature Communications
**17**, 10859 (2026), [DOI](https://doi.org/10.1038/s41467-026-77226-4)。
模型：[Zenodo 21401971](https://doi.org/10.5281/zenodo.21401971)，CC-BY-4.0，
归属原作者。`system.json` 是论文 Supplementary Data 的 `data.lmp` 原子坐标、
质量、晶胞的 JSON 转录；下方给出模型 SHA256，下载脚本同时验证原压缩包。

## 运行短算

在仓库根目录运行。Torch 是可选依赖，CPU 环境可按 PyTorch 的 CPU 安装说明配置。

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,mpi,analysis]'
python examples/surface/pt111/fetch_model.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pyrpmd validate examples/surface/pt111/pilot.py --check-pes
pyrpmd run examples/surface/pt111/pilot.py --output runs/pt111-pilot
pyrpmd resume runs/pt111-pilot
pyrpmd analyze runs/pt111-pilot --recompute
pyrpmd project runs/pt111-pilot --axes 2 --output runs/pt111-height
```

短算使用 100 步热化、800 步生产、1 个副本；应输出 3 维 FES、40 个样本、
2 个 hills，以及一个含 NHC 状态的 checkpoint。冻结原子在所有珠子上的位置
保持初值，动量恒为零。此长度只检查完整工作流，不足以采样反应势垒。
因此三维网格及高度投影会有缺失区域；投影可能全部被 mask，这是正常的
支持检查结果。`converged=false`，没有从有偏轨迹越垒时间计算速率。

模型文件：`REANN_PES_DOUBLE.pt`，SHA256：
`64a2a91636d11c091585aaf1bc72af6397e9672a0a3bd02602f95ec1772fb6f4`。
`potential.params.model` 是显式路径，运行 manifest 会记录其内容哈希，
文件改变后不能沿用原 checkpoint。`eann.py` 将 bohr/hartree 转换为模型的
Å/eV，PBC 由模型在私有坐标上处理，不折回 PyRPMD 的动力学坐标。

`python examples/surface/pt111/check_dynamics.py` 在论文 SP1 构型上进一步检查
1／4 珠子的真实键长约束、固定基底和无偏子轨迹；`transition_state.json` 是该
公开构型的 Å 坐标转录。输出能量漂移及约束残差，不把短轨迹当作 κ 平台。

## 较长的论文参数输入

`production.py` 声明 600 K、P=1、0.2 fs velocity-Verlet、20 fs NHC 阻尼、
3 个独立副本、各 8 ps 热化和 40 ns 生产。两个键长使用 `periodic_distance`，对应原 PLUMED 的 WHOLEMOLECULES 成对最近像重建；动力学坐标仍保持展开。三个偏置 CV 为 r(C,O₁)、r(C,O₂)、
h(C)−mean(z(top9 Pt))；σ=0.2 Å，W₀=0.020 eV，每 400 步沉积一次，γ=39。
附加的 O、Pt 高度墙和 CV 墙与论文 PLUMED 文件数值一致，κ=150 eV/Å²，
W=κΔ²。热化时无墙和 hills；生产开始时重新初始化热浴，保持 q/p。
周期晶胞 z 长度为 66.9262008667 Å，XY 为论文倾斜晶胞。

```bash
pyrpmd validate examples/surface/pt111/production.py
# 明确选择长算时运行；每个副本有 200,000,000 个生产步。
pyrpmd run examples/surface/pt111/production.py --output runs/pt111-600K-production
```

这份输入已具备可执行功能，但不是逐轨迹复刻 LAMMPS/PLUMED 的声明：

- 采用公开双精度 TorchScript 模型；原 LAMMPS 示例使用 float 模型。旧 Fortran
  回穿程序与公开 TorchScript PES 的版本一致性仍未确认。
- NHC 采用全部 63 个活动自由度，不减去 COM 自由度。原 LAMMPS 输入没有
  显式指定 `compute_modify ... extra/dof`，通常默认扣 3；随机数和初速度算法也不同。
  PyRPMD 不把固定基底上并不守恒的活动组 COM 额外约束成常数。
- 本实现直接求 Gaussian 和；论文 PLUMED 使用网格插值。分析网格间距 0.04 Å
  对应 215×215×185 个 bins，但该网格不加速动力学。计算与分析成本随 hill 数增长，
  先用较短分段测量资源消耗，再安排 40 ns；本次开发验证没有运行这些长算。

参数相同或接近不等于统计结果已收敛。需检查不同种子、分段 FES、CV 覆盖、
珠子数和时间步，以及原文其他温度下各自明确的偏置设置。

## 通道投影和速率

自由能投影应计算 `−log(∫exp(−βF) ds_other)/β`。`pyrpmd project` 和
`pyrpmd.analysis.marginalize_fes` 实现该积分，并使用实际 bin 宽度。
选择第三轴得到高度 PMF；保留第二轴可得到 r(C,O₂) PMF。
要排除 CO₂ 区域，可创建三维布尔选择域：

```python
import numpy as np
from pyrpmd.constants import ANGSTROM_TO_AU

with np.load('runs/pt111-pilot/fes.npz', allow_pickle=False) as f:
    r1 = (f['edges_0'][1:] + f['edges_0'][:-1]) / 2
    r2 = (f['edges_1'][1:] + f['edges_1'][:-1]) / 2
    shape = f['free_energy'].shape
channel = np.broadcast_to(
    np.maximum(r1[:, None, None], r2[None, :, None]) >= 1.96 * ANGSTROM_TO_AU,
    shape,
).copy()
np.save('co_channel.npy', channel)
```

```bash
pyrpmd project runs/pt111-pilot --axes 2 --selection co_channel.npy --output runs/pt111-co-height
```

这是条件密度的投影；原文 SI 对 CO 脱附采用删除部分 Gaussian hills 的方法，
两者不等价。不能用删除 hills 或把空 bin 填成零的方式声称得到完整条件系综。
当前投影对选定域内任何缺失的积分切片保留 mask，不给出伪造 PMF。

独立速率使用 `task='surface_rate'` 的伞抽样 → PMF → 曲面约束母轨迹 → ±p
无偏子轨迹。已支持 `distance`、满足光滑性条件的 `periodic_distance` 键长和 `surface_height`：论文参考分割面分别为
1.98 Å（氧化，product_side='lower'）和 3.04 Å（高度，'higher'）。盆地必须由
实际采样定义并被支持，不能只填入论文分割面就报告完整催化速率。
`surface.free_energy` 将这个过程接通：读取三维源 FES、做条件投影、在相同域内
接受约束母构型，再运行配对子轨迹。源静态墙保留在母／子哈密顿量中，MetaD
偏置不保留；这与原文移除采样墙后的 Fortran 子轨迹不是逐步相同的算法。

```bash
PYRPMD_PT111_FES=runs/pt111-pilot pyrpmd run examples/surface/pt111/rate_from_fes.py --output runs/pt111-oxidation-rate
PYRPMD_PT111_FES=runs/pt111-pilot PYRPMD_PT111_CHANNEL=desorption PYRPMD_PT111_SELECTION=co_channel.npy pyrpmd run examples/surface/pt111/rate_from_fes.py --output runs/pt111-desorption-rate
```

短算的源 FES 不覆盖分割面时，速率任务应返回具体的支持不足原因，**不启动子轨迹**。
长算支持充足时才计算 QTST、κ 和候选速率；示例的母／子数量及有限盆地只是试算设置，
需根据收敛调整。源 PMF 的统计误差未被校准，正式速率保持空值。
多通道的定义、相同域与墙、误差范围见[完整使用说明](../../../docs/surface-catalysis.md)。

本次功能测试证明输入、力、传播、恢复和分析接口可用，不等于已复现论文
Fig. 4 的速率。先前静态势垒对照与部分原文回穿轨迹不一致的记录仍有效。
