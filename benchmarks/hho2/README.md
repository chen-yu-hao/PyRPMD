# H + HO₂：二维质心 PMF 与多通道投影分析

**当前状态：扫描程序和几何/拓扑测试已实现；没有随附经过验证的全维反应 PES，尚未产生真实 H + HO₂ PMF。**

本目录不会把教学谐振子或 H₂O₂ 局部振转势当成 H + HO₂ 的全局反应势。
`plan` 不需要势能；`check-pes`、`run` 必须显式指定模型清单、模型文件及 SHA256。
测试中的 harmonic mock 仅检验程序连接，不能作为化学结果。

## 物理定义

原子顺序固定为 **H0（入射）、H1（原 HO₂ 中的 H）、O0、O1**。
二维坐标为 `r_HH` 和 `r_OO`，均是**原子质心间距**；不是珠子距离的平均值。
保存的 PMF 是相对于 `dr_HH dr_OO` 的概率密度自由能，保留这个坐标选择产生的构型熵。

这两个坐标不能约束所有解离方向：在 H₂ + O₂ 区域，它们只约束两个分子的内部键长，
两个分子仍可任意远离。因此对每个珠子的所有六对原子施加相同的平底距离墙：

\[
V_\mathrm{wall}(q)=\tfrac12 k_\mathrm{wall}
\sum_{i<j}\max(0,r_{ij}-R_\mathrm{wall})^2.
\]

默认 `R_wall=6 Å`、`k_wall=20 kcal mol⁻¹ Å⁻²`。墙在所有窗口相同，
**保留在重建的目标 Hamiltonian 中**，所以输出属于有限约束下的 PMF。
这些初始参数尚未对 H + HO₂ 验证；必须改变墙的位置检查体积熵和连接结论的敏感性。

所有窗口从五种几何族分别启动：反应物、H 标签交换、H₂+O₂、OH+OH、H₂O+O。
内置结构仅是未优化的初始化模板；每窗先作几何 steering 和有偏势能松弛，再热化。
多个起始族可以暴露隐变量滞后，但不能保证各族已经混合。每个窗口保存质心几何、
完整 checkpoint、物理能、墙能和 ring spread，允许检查全部六个键长和隐藏自由度。

默认 `1500 K`、`P=1` 用于经典 pilot。`P=1` 的结果不会被称为量子 RPMD PMF。
必须另做 `P=8/16/32`、步长减半、独立 seed、采样延长和墙半径变化的对照。
`extended` 只是更长的预设，运行成功不代表收敛。

## 接入外部反应势

JSON 清单示例（占位值必须替换，路径相对于清单）：

```json
{
  "reference": "实际使用的全维反应势名称、DOI 和版本",
  "atomic_numbers": [1, 1, 8, 8],
  "multiplicity": 3,
  "potential": {
    "provider": "python",
    "factory": "pes_adapter.py:build",
    "params": {"model_file": "weights.dat"}
  },
  "artifacts": [
    {"path": "pes_adapter.py", "sha256": "替换为真实SHA256"},
    {"path": "weights.dat", "sha256": "替换为真实SHA256"}
  ]
}
```

也可以使用 PyRPMD `cabi` 或 `rpmdrate` provider，但必须保持指定的 HHOO 原子顺序、
明确电子态和单位，且记录所有实际模型文件。Python factory 内部使用的其他依赖和
动态加载数据也需要列入清单，清单本身不能自动发现任意用户代码的全部依赖。
适配器返回原子单位能量和**负梯度力**，接口见项目 `docs/potentials.md`。
清单会检查实际 provider 入口是否包含在已校验的 artifacts 中；内置教学势不被接受。
保存的 `physical_energies` 是裸 PES 能量，`wall_energies` 单独记录共同约束能。

`check-pes` 只验证五个初始化构型的能量/力有限性与有限差分一致性。
它不能替代原作者参考能量、渐近产物能、置换对称性、势能域和梯度的独立验证。
一个物理上错误但自洽的势也可能通过力检查。

```bash
# 无 PES 时先检查计划、窗口数与工作量。
python benchmarks/hho2/scan.py plan --profile pilot --output runs/hho2-plan

# 接入来源与数值均已验证的真实反应势后。
python benchmarks/hho2/scan.py check-pes --pes-config /path/to/model.json \
  --output runs/hho2-pes-check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/hho2/scan.py run --profile pilot \
  --pes-config /path/to/model.json --workers 4 --output runs/hho2-pilot

# 相同命令、相同科学参数和输出目录可从 checkpoint 续跑。
python benchmarks/hho2/scan.py analyze --output runs/hho2-pilot

# 另开输出目录，做更高珠子数/不同步长/独立 seed 的比较。
python benchmarks/hho2/scan.py plan --profile extended --beads 16 \
  --dt-fs 0.05 --seed 20261031 --output runs/hho2-p16-plan
```

`pilot` 的默认计划是 110 个中心 × 5 个起始族 = 550 条独立流；每流热化 200 fs、
生产 1000 fs、每 0.5 fs 保存。它是初始探索配置，窗口宽度和重叠仍需根据实际 PES 调整。
PES 壁垒可能阻止直角网格上的一些窗口稳定采样，此时应保留失败和支持缺口，不能补出通路。

当前 MBAR 实现在合并相同 Hamiltonian 之前会构造稠密的 `[streams, total_samples]` 矩阵。
`plan` 提供样本数和**单个 float64 矩阵的内存下界**：

| 预设 | 独立流 | 总保存样本 | 单矩阵内存下界 |
| --- | ---: | ---: | ---: |
| `pilot` | 550 | 1,100,000 | 4.84 GB（4.51 GiB） |
| `extended` | 1,710 | 17,100,000 | 233.93 GB（217.86 GiB） |

实际峰值还包括其他 MBAR 数组、bootstrap、几何和文件加载，明显高于这个下界。
`--workers` 不会消除分析阶段的分配；`extended` 不能直接当作普通工作站的默认生产配置。
本次没有实现分块 MBAR，也没有自动抽稀。确定 PES 和可用资源后，应先以 `smoke` 验证接入，
再规划采样保存间隔、窗口规模或专门的内存优化；不能为节省内存而合并起始族并丢失滞后诊断。

分析只接受计划中完整的流集合；每个流必须通过 checksum、模型/参数 fingerprint、采样时钟、
数组形状和 CV–质心几何一致性检查。不同计划不能覆盖现有 `plan.json`，改变科学参数需另开目录。

## 通道标签与连接的含义

`channels.py` 根据全部六个键长的成键图分类，保留 H 原子身份：

| 标签 | 几何意义 |
| --- | --- |
| `reactant_H0+H1O2` | 原入射 H0 游离，H1 仍在 HO₂ |
| `exchanged_H1+H0O2` | H1 游离，H0 进入 HO₂ |
| `H2+O2` | 独立 H–H 与 O–O 片段 |
| `OH+OH` | 两个独立 O–H 片段 |
| `H2O+O` | 一个 O 连两个 H，另一个 O 游离 |
| `intermediate/ambiguous` | 键长灰区、非匹配连接图或反应复合物 |

默认 formed/broken 阈值是 HH 1.00/1.40 Å、OH 1.25/1.75 Å、OO 1.65/2.10 Å。
这些是示例阈值；必须对阈值敏感性进行检查，不能把灰区强行归入某产物。
原子质心上的标签不等于每个珠子具有相同的成键图。

MBAR 会检查窗口 overlap。全局 overlap 不连通时，全局 FES 被 mask。
连接分析另外要求原始 bin 数量与重要性权重 ESS 支持，不能跨越 NaN、Inf 或 masked bin。
采用**四邻接** minimax 路径，禁止沿对角穿过缺口；只选通道重加权条件占比至少 80% 的受支持格子作盆地种子。
这给出二维投影上的路径和自由能最高点，**不证明高维真实反应路径相通**，也不给出速率或分支比。
当前路径尚无 bootstrap 置信区间，需在实际样本产生后结合独立重复验证。

H 原子标签交换不改变 `(r_HH, r_OO)`。对等质量 H 和置换对称的 PES / 墙，充分混合后两种身份入口
在每个 bin 的条件概率相等，各自至多为 50%，因此不能分别要求它们达到 80%。
程序仅在盆地 / 路径层把两种入口的条件占比相加，使用合并标签 `H+HO2`；原始身份标签比例保留，
另外保存 `basin_channel_names/fractions` 和分组映射。这组二维坐标无法分辨 H 身份交换路径。
如果其他通道找不到条件占比 ≥80% 的盆地格子，也不能据此判定该通道不存在或不连通。
同样，基于这两个 CV 的相关时间和较高的权重 ESS 无法排除正交自由度中的缓慢通道交换；
必须比较不同起始族的完整几何和通道标签分布。`summary.json` 中逐流的标签计数和相邻标签变化
可帮助发现滞后，但包含 ambiguous 标签变化，既不是已解析的动力学反应次数，也不是混合证明。

主要输出：

- `plan.json`、`model.json`：科学参数与模型来源。
- `streams/*.npz`：独立流的样本、质心构型、最终 q/p、RNG 和完整性校验。
- `fes.npz`：PMF、支持 mask、误差、窗口 overlap、各 bin 通道比例。
- `summary.json`：支持情况、通道盆地、投影连接和限制。
- `pmf.png`：PMF / SE / 权重 ESS 三联图，只显示受支持格子。

## 模型获取记录与原始文献

2026-10-30 的本次检查未取得可编译的适用全局模型；这是**模型输入缺失**，不是已完成的化学扫描。
核对了 PyRPMD 当前仓库、ChemPotPy 的 H₂O₂ 条目、POTLIB 官方 H₂O₂ 搜索、CQPES Model-Zoo、
PIPPy 示例和原论文附件入口。ChemPotPy/POTLIB 的 Koput–Carter–Handy 1998 条目是 H₂O₂ 振转模型，
不能据其分子式相同就用于多通道反应。部分出版商附件入口返回 403；未取得源码不等于证明作者没有提供模型。

- Li, J.; Li, J. *J. Phys. Chem. A* **2021**, 125, 1540–1552.
  [10.1021/acs.jpca.0c11213](https://doi.org/10.1021/acs.jpca.0c11213)：ground-triplet H+HO₂ 多通道 PES。
- Li, J.; Dawes, R.; Guo, H. *PCCP* **2016**, 18, 29825–29835.
  [10.1039/C6CP06232F](https://doi.org/10.1039/C6CP06232F)：全局最低 triplet H₂O₂ PES。
- Jasper et al. *Front. Phys.* **2022**, 10, 1003010.
  [10.3389/fphy.2022.1003010](https://doi.org/10.3389/fphy.2022.1003010)：singlet/triplet PES 和非统计动力学。
- [ChemPotPy H₂O₂ 模型说明](https://github.com/shuyinan/chempotpy/blob/4716d22e87ae01770cad9ec995c4a02a8e479296/chempotpy/H2O2/__init__.py)
- [CQPES Model-Zoo](https://github.com/CQPES/Model-Zoo/tree/40bfcc3c4a72d763dc77f4081e1de60980a44af7)

单一 triplet PMF 不能自动合并 singlet 通道。即使同一面上的 PMF 已收敛，也不能从平衡盆地占比推出
H + HO₂ 碰撞产物分支比；后者仍需要相应的动力学计算。
