# 自动分析报告与出图

用一条命令把保存的 PyRPMD 结果整理为报告：

```bash
python -m pip install -e '.[analysis]'
pyrpmd report runs/my-run --output reports/my-run
```

可选 `analysis` 依赖包含绘图所需的 Matplotlib。完成后用浏览器打开
`reports/my-run/index.html`。报告读取 PyRPMD 原生结果文件，不调用势能、不重新传播轨迹，
也不执行保存的 Python 输入文件。

## 单个计算、多个计算与自动出图入口

在源码仓库根目录运行下面的谐振子示例，可以检查计算与出图的完整连接：

```bash
pyrpmd run examples/fes/harmonic2d.inp --output runs/report-harmonic
pyrpmd report runs/report-harmonic --output reports/harmonic \
  --formats png pdf svg --dpi 600 --energy-unit kcal/mol --length-unit angstrom
```

这是解析谐振子模型的短采样示例；完成运行不等于完成真实反应的收敛计算。
二维面中没有采样支持的格子仍保留掩膜。

已有计算也可以从 `analyze` 入口出图：

```bash
pyrpmd analyze runs/my-run --report --report-output reports/my-run
```

对新计算，`pyrpmd run INPUT --report` 在运行完成后生成默认报告。单个计算未指定报告输出路径时，
默认位置是 `RUN/analysis-report`。比较多个计算时必须显式给出 `--output`：

```bash
pyrpmd report runs/rate-300 runs/rate-500 runs/rate-700 \
  --output reports/temperature-series \
  --labels '300 K' '500 K' '700 K' --series RPMD RPMD RPMD
```

如果计算完成后生成报告失败，命令会返回错误，并明确给出已保存数值结果的位置及独立的
`pyrpmd report` 重试命令。修复绘图依赖或输出目录后，只需重试保存数据的报告；已完成的结果仍然可用。

`--labels` 是每个计算的展示名称；`--series` 是温度曲线、KIE 和参数比较使用的系列名。
使用这些选项时，每个输入目录各提供一个值。不指定系列时，程序根据元数据保守分组。
例如，不同温度采用不同珠子数的计算，可能需要在确认珠数收敛后显式归入同一个系列。

显式指定同一个系列表示你认为这些计算适合一起比较。仍需检查反应、势能、同位素、CV、
归一化和采样方案是否一致；单位相同本身不足以证明可比性。
气相速率分组读取 `preparation.json` 中已经就绪的实际反应定义及保存的 TS 参考几何；
推断通道缺失时不进行比较。模型/插件文件的已保存校验值和 metadynamics 静态墙也参与集合身份检查，
显式指定同一系列不能绕过这些条件。

## 输出内容与文件保护

| 输出 | 用途 |
| --- | --- |
| `index.html` | 本地可打开的结果总览、图片、表格和诊断 |
| `report.md` | 可阅读和进一步编辑的文字报告 |
| `report.json` | 结构化结果、来源、诊断及输出文件索引 |
| `summary.csv` | 每个计算的汇总数据 |
| `figures/` | 默认 PNG/PDF，可指定 SVG |
| `data/` | 每张图对应的 CSV，包含实际绘图数据及单位 |

通过 `report.json` 的输出索引定位具体图片和数据表。报告记录实际读取文件的 SHA-256 和软件版本。
导出的 CSV 中缺失值留空；二维绘图表保留网格中心坐标和掩膜列。
仍应保留原始计算目录；绘图表格不能替代输入、轨迹和采样档案。

已有输出目录默认不会被替换。需要重新生成时使用 `--overwrite`，且目标目录必须具有有效的
PyRPMD 报告标记；此选项不能用于覆盖任意目录或输入计算目录。

```bash
pyrpmd report runs/my-run --output reports/my-run --overwrite
```

## 根据现有数据自动选择图片

| 原始结果 | 可生成的分析与图片类别 |
| --- | --- |
| 一维 PMF/FES | 曲线、有效点、已有不确定度、极值与元数据 |
| 二维 FES | 自由能分布及已有采样覆盖、掩膜信息 |
| 伞抽样 | 保存的窗口分布、重叠以及约束和采样诊断 |
| 气相、位点或表面速率 | PMF、传输系数与原工作流的速率诊断 |
| 多个可比计算 | PMF/recrossing 叠图、速率温度依赖 |
| Metadynamics | 保存的 bias/FES 及已有历史信息 |
| dRPMD | 产物与其他结局计数、受支持的通道统计，保留未决和失败项 |
| 材料与扩散 | 已保存的观测量、MSD/VACF 及原拟合诊断 |
| PCET | 已有模型、通道贡献和扫描结果 |

只有数据确实存在时才生成对应图片；缺失的数据会记录在报告中，不用假定值补齐。
本版本解析 PyRPMD 原生输出，不包含 Post-RPMDrate 的旧 RPMDrate `.dat` 格式或执行 `input.py` 的逻辑。
材料报告同时支持 NPZ 和默认的带校验分块轨迹，只读取导出的已提交前缀，不采纳未提交的孤立块。
电子摩擦中的能量变化标为与电子热浴的能量交换；无恒温器扩散仍保留 NVE 诊断。

## 单位、二维坐标与不确定度

`--energy-unit` 和 `--length-unit` 只改变可以转换的展示单位，默认是 `kcal/mol` 和 `angstrom`。
它们不改变计算，也不会把角度或无量纲 CV 变成长度。双分子速率的单位是
`cm3 molecule^-1 s^-1`，一阶速率是 `s^-1`；不同维度的速率不会混成一个系列。
长度单位选项作用于 PMF/FES 坐标轴。轨迹原子对距离和环尺寸明确使用 angstrom，
MSD/VACF 使用 angstrom 与 fs，固定碰撞能截面保留标明的 `bohr²` 单位。

二维自由能面与二维势能切面不同。自由能面的横纵轴必须是实际采样的 CV，其数值对应这些坐标下的
概率密度测度。只有采样坐标本来就是 `r(H1,H2)` 和 `r(H2,H3)`，才能把图标成这两个键长。
不能通过改轴名称转换坐标。没有采样支持的区域保持掩膜，不插值补出反应路径；掩膜本身也不能
解释为高势垒。
测度与 CV 单位来自 FES 数组档案；如果汇总文件明确给出了冲突声明，对应 FES 不出图；
缺少测度时不叠加比较曲线。

保存的标准误和区间保持各自含义，没有可用误差时不生成误差带。标准误不等于 95% 置信区间，
改变珠子数或时间步得到的差异属于参数敏感性。窗口重叠不足、有效样本少、轨迹失败、平台检查
未通过等诊断，在画出图片之后仍然有效。
热速率的 percentile 95% 区间、结构化不确定度说明和势垒区间保留在报告指标中，不转换为 1 SE。
固定碰撞能散射的独立初态 Wilson 区间、母链 bootstrap 95% 区间与未决结局界限分别保留。

报告不会认证物理收敛。一条平滑 PMF、看似平坦的 recrossing 尾部或成功运行的短测试，仍需要
[科学约定](science.md)中的收敛研究来支持正式预测。

## 正式结果与诊断估计

默认使用原工作流已发布的速率字段。若原工作流因检查未通过而保留空的最终速率，报告不会把
有限时间估计提升为最终结果。需要查看这些诊断估计时可显式启用：

```bash
pyrpmd report runs/rate-pilot --output reports/rate-diagnostics --include-estimates
```

估计值会与已发布速率分开展示并标明诊断用途，绝不纳入 Arrhenius 拟合。
原结果中存在正式速率字段，也不意味着珠数、时间步或全部物理收敛问题均已解决。

### 可选 Arrhenius 拟合

```bash
pyrpmd report runs/rate-300 runs/rate-500 runs/rate-700 \
  --output reports/arrhenius --series RPMD RPMD RPMD --fit-arrhenius
```

拟合默认关闭。启用后，需要同一可比系列中至少三个不同温度、正的已接受速率。
程序对 `ln(k)` 与 `1/T` 做无权重描述性回归，报告活化能和拟合优度。
这不为原速率创造误差，也不能证明整个温区只有一个活化能；跨越隧穿区时尤其应检查曲率。

### 动力学同位素效应 KIE

```bash
pyrpmd report runs/H-300 runs/H-500 runs/D-300 runs/D-500 \
  --output reports/kie --series H H D D --kie H D
```

参数顺序是分子、分母，上例得到 `k_H/k_D`。温度必须匹配、速率单位必须相容，反应与归一化也应可比。
程序不插值补温度点。仅当两个速率都保存了标准误时，才在独立估计假设下给出一阶传播标准误：

$$
\operatorname{SE}(R) = |R|\sqrt{
  \left(\frac{\operatorname{SE}(k_H)}{k_H}\right)^2 +
  \left(\frac{\operatorname{SE}(k_D)}{k_D}\right)^2},
\qquad R = k_H/k_D.
$$

配对计算可能存在协方差，此表达式不估计该协方差。缺少任意一侧的标准误时，不假造比值误差。

### 参数收敛对比

```bash
pyrpmd report runs/P8 runs/P16 runs/P32 --output reports/beads \
  --series bead-check bead-check bead-check --convergence beads
```

可选变量是 `beads`、`dt_fs`、`rinf_bohr`、`bmax_bohr` 和 `samples`，只能绘制元数据中已有的值。
把差异归因于一个参数之前，应控制其他科学设置。指定 `--convergence` 只生成比较，
不替代足够参数范围、独立采样和误差分析。

## 导入带来源的参考数据

通过 `--reference reference.csv` 添加比较数据。列格式为：

| 列名 | 含义 |
| --- | --- |
| `series` | 参考系列名称 |
| `temperature_K` | 开尔文温度 |
| `rate` | 所声明单位下的速率 |
| `rate_unit` | `cm3 molecule^-1 s^-1` 或 `s^-1` |
| `source` | 文献、DOI、数据集标识或明确的合成数据说明 |
| `standard_error` | 可选，与速率同单位的标准误 |

图中会区分参考数据角色并保留来源。不要把 95% 区间宽度、含义不明的实验误差、
不同理论方法间的离散程度填进 `standard_error`。

仓库中的
[reference-rates.csv](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/analysis/reference-rates.csv)
仅包含**合成示例数据**，其中误差值同样是合成的。它只演示格式，不是实验值，也不能用来验证势能面：

```bash
pyrpmd report runs/rate-pilot --output reports/reference-format \
  --reference examples/analysis/reference-rates.csv
```

## Python 接口

```python
from pyrpmd.analysis.reporting import generate_report

report = generate_report(
    ["runs/rate-300", "runs/rate-500", "runs/rate-700"],
    output="reports/rates",
    formats=("png", "pdf", "svg"),
    dpi=600,
    energy_unit="kcal/mol",
    length_unit="angstrom",
    labels=["300 K", "500 K", "700 K"],
    series=["RPMD", "RPMD", "RPMD"],
    arrhenius_fit=True,
)
```

此外支持 `reference`、`include_estimates`、`kie`、`convergence` 和 `overwrite`，
含义与命令行一致。KIE 示例为 `kie=("H", "D")`。
`arrhenius_fit=False` 和 `include_estimates=False` 是默认设置。

## 文献参考与绘图格式

PMF、recrossing、同位素和速率比较参考
[Gui 等，JCTC 2022](https://doi.org/10.1021/acs.jctc.2c00522)中的图型；
多通道比较参考 [H + MgH 研究](https://arxiv.org/abs/2007.07059)，采样检查也参考
[NO/Pd(111) 研究](https://doi.org/10.1039/D2SC06559B)。
[Post-RPMDrate](https://github.com/chen-yu-hao/Post-RPMDrate)提供了将采样诊断与最终结果一起展示的功能参考；
本模块独立实现，不导入或复制其源代码。

紧凑图宽度为 3.33 in，较大图为 7 in，字体至少 8 pt。默认输出 300 dpi 彩色 PNG 与矢量 PDF，
可显式增加 SVG 或调整 DPI。
[JCTC 当前绘图规范](https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce)
分别规定彩色、灰度和黑白线图的最低分辨率为 300、600 和 1200 dpi。
投稿前仍应按最终版面大小检查导出图片。
