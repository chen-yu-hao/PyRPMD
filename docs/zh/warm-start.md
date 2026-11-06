# 热启动：从旧运行的构型开始一个新运行

`resume` 继续的是**同一个**计算，因此会拒绝科学设置的改变。若想先用低精度跑，再以其结果作为高精度计算
（更多珠子、更小时间步、不同温度或势能面）的起点，请新建一个运行并从旧运行初始化：

```bash
pyrpmd run high.inp --output runs/high --init-from runs/low
pyrpmd run high.inp --output runs/high --init-from runs/low/checkpoints/metadynamics_0000.npz
pyrpmd run high.inp --output runs/high --init-from runs/low --init-bias   # 仅元动力学
```

Python 接口：`run(job, output=..., init_from=..., init_index=None, init_bias=False)`。

| 项目 | 规则 |
| --- | --- |
| 身份 | 新运行有自己的输入、输出目录和科学指纹。`manifest.json` 的 `initialized_from` 记录父运行的 `physics_fingerprint`、所用检查点、索引、珠子数和父运行路径 |
| 导入内容 | 只导入一个环聚合物构型 q。插值后的构型及可选的初始偏置保存在 `input/resolved.json` 的 `warm_start` 中，因此属于新运行的指纹；新运行续跑时直接使用这份状态，不再读取父运行 |
| 珠子数 | 珠子数不同时用环聚合物正规模式（Fourier）插值：质心严格保持，增加珠子时补零高频模式，减少珠子时截断高频模式 |
| 动量与随机数 | 按新温度和新珠子数从 Maxwell 分布重新抽取动量，使用新输入的种子；不复用父运行的 RNG、热浴或样本 |
| 平衡 | 必须重新平衡：新输入须设置 `sampling.equilibration_steps >= 1`，完整平衡阶段跑完后才进入产出阶段。`results/summary.json` 中的 `warm_start` 条目标明这是热启动 |
| 选择 | 运行目录中只能有一个含环构型的检查点，否则请直接给出检查点文件；检查点含多个构型时用 `--init-index` 选择 |
| 体系 | 原子及顺序相同，质量可以不同；固定原子位置不能变；周期体系须晶胞与周期性相同，父运行用了恒压器则拒绝（不导入其最终晶胞） |

支持的工作流：伞形抽样、元动力学、OPES、ABF/eABF、拉伸 MD、VES 等 FES（含固定窗口副本交换和共享/分布式
walkers）、PLUMED、带 `[path_integral]` 的 `pimd`，以及使用 legacy 材料引擎的 `pimd`/`diffusion`。
速率、dRPMD、表面/位点速率、反应网络、非绝热、PCET、光谱、路径抽样、tempering、自适应伞形抽样以及
native/torch 材料引擎自行准备初态，会拒绝 `--init-from`。

## 导入元动力学偏置

只有指定 `--init-bias` 时才导入偏置，且仅限独立、直接求和（非网格、非 walkers）的元动力学运行之间。要求：

* 元动力学 CV 完全一致（定义、原子、估计器和单位），Gaussian 宽度完全一致；
* 温度改变时，两次运行都必须是 well-tempered，否则拒绝。

两份偏置被视为同一 F(s) 的估计：V_old ≈ -f_old F，V_new ≈ -f_new F，其中 well-tempered 的
f = (γ-1)/γ，标准元动力学 f = 1。导入的 hill 高度乘以 f_new/f_old。这一换算假设新设置下 F(s) 变化不大，
需要自行检查。

导入的 hills 是静态的**初始偏置**：参与能量、力、well-tempered 高度因子和最终偏置估计，但不属于新运行自己
沉积的历史。`HILLS.csv`、检查点中的 `hills` 和 `hills_per_replica` 只包含新运行沉积的 hills，摘要另以
`initial_bias_hills` 报告导入数量。
