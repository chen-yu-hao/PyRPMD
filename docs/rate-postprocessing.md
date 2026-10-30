# 二维自由能速率与通道分支后处理

`pyrpmd rate-postprocess` 读取已保存的采样结果，计算静态通量、通道速率和分支比，
不重新调用 PES 或传播轨迹。输出另存为 JSON，保留源文件 SHA256、单位、测度和状态，
不会覆盖原来的 PMF、FES 或 `summary.json`。
命令行输出简短 JSON 摘要；完整时间曲线、父构型信息和 bootstrap 数组保存在结果文件中。

## 支持的三条路线

| 数据 | 计算 | 需要的额外信息 |
| --- | --- | --- |
| 已有气相 `rate` 工作流 | 复算入口前因子、PMF 静态因子、signed 再穿越与单通道速率 | 原运行目录中的输入和摘要 |
| 二维 `cv_density` FES | 沿折线积分正向通量 | 法向热速度、完整分界曲线、游离反应物体积或已知参考面通量 |
| 二维 `rate_flux` FES | 沿辅助轴积分，比较两个固定 `s0` 截面 | 明确 `flux_axis=0`、已知入口通量、两个截面的完整支持 |

后两条路线都可附加来自同一物理体系、同一分界面的多通道有符号通量记录。
普通产品盆地的 Boltzmann 权重不是动力学分支比。已有 `network` 输出的独立 TS
速率也不会在这里自动相加：不同 TS 可能重复计数同一物理通量。

## 已有一维气相速率

```bash
pyrpmd rate-postprocess runs/h3-rate --bootstrap 200 --seed 7
```

默认写入 `runs/h3-rate/rate_postprocess.json`。也可通过 Python 调用：

```python
from pyrpmd.analysis.rate_io import postprocess_rate

result = postprocess_rate("runs/h3-rate", bootstrap=200, seed=7)
print(result["static"]["k_tst_cm3_s"])
print(result["rate_estimate_cm3_s"])
print(result["rate_cm3_s"])
```

适配器使用 `input/resolved.json`、`rate_pmf.json`、`rate_pmf.npz`、
`recrossing.npz` 和 `summary.json`。它独立重算

\[
k_{\rm TST}^{(2)}=\frac{4\pi R_\infty^2}{\sqrt{2\pi\beta\mu}}
e^{-\beta\Delta W}\,g_{\rm electronic}\,g_{\rm symmetry}
\]

并核对已保存的分界面、PMF 方法、静态前因子和父构型通量总和。
原计算若缺少平台或能量检查未通过，后处理仍保留 `rate_cm3_s=null`；
有限时间候选值保留在 `rate_estimate_cm3_s`。它不会把旧 pilot 升级为收敛结果。

## 普通二维概率密度面

对与 `ds0 ds1` 对应的自由能面，静态正通量为

\[
I_\Sigma=\int_\Sigma e^{-\beta F(s)}\langle v_n\Theta(v_n)\mid s\rangle\,d\ell.
\]

支持两种归一化：

\[
k_\Sigma^{(2),+}=V\frac{I_\Sigma}{\int_\mathcal R e^{-\beta F(s)}d^2s},
\qquad
k_\Sigma^{(2),+}=k_{\rm ref}^{(2),+}\frac{I_\Sigma}{I_{\rm ref}}.
\]

第一种的 `V` 必须是游离反应物的可及相对平移体积；接触复合物盆地不能代替
自由双分子反应物。第二种的参考曲线必须是已知完整入射通量的截面，不能随意选一个
局部像素，再把它称为渐近入口。

Python API 示例（所有数值使用运行的原子单位/CV 坐标尺度）：

```python
import numpy as np
from pyrpmd.analysis import surface_rate_2d

with np.load("run/fes.npz", allow_pickle=False) as data:
    result = surface_rate_2d(
        data["free_energy"], [data["edges_0"], data["edges_1"]],
        beta=beta, mask=data["mask"], measure="cv_density",
        curve=target_curve,
        normal_speed=positive_normal_speed,
        reference_curve=entrance_curve,
        reference_normal_speed=entrance_positive_speed,
        reference_rate_au=known_entrance_rate,
    )
```

`normal_speed` 可为正标量、每个折线顶点的数值，或
`normal_speed(points, normals)` 函数。`points` 和 `normals` 都是 `[N,2]`。
函数返回每个点的正向热速度平均。也可用常量 `inverse_mass`，此时使用
`sqrt(inverse_mass / (2*pi*beta))`。两者互斥。

这里需要的是 `E[v_n theta(v_n)]`，不是 `E[v_n | v_n>0]`，两者相差因子二。
一般质量度量需要 `E[sqrt(n^T G n)|s]`，不能随意替换成 `sqrt(E[n^T G n|s])`。
两个 CV 的坐标尺度、速度和质量度量必须匹配。`F` 已包含的 Jacobian 不能再次乘入。

分界曲线应完整覆盖所定义的通量且不重复经过相同物理面片。积分器将折线按每个
插值单元分段，以避免稀疏顶点跳过缺失支持。插值需要的 bin 被 mask 时直接报错；
不会外推越过未采样区。自由能在 bin 中心之间双线性插值，在最外半个 bin 内采用
常值延展。绝对积分依赖自由能零点，归一化速率不依赖该零点。

对应 JSON spec 使用 `method="surface_2d"`，字段与上述关键字一致；还必须提供
`reference_definition`，或使用 `reactant_mask`/`reactant_volume_au` 时提供
`reactant_definition`。这些文字记录归一化的物理意义，不替代实际采样验证。

## 已包含法向通量权重的二维面

二维 `rate_flux` 是可选采样测度，不等于普通概率密度。第一 CV 必须是
`rpmdrate_xi`，并在输入中显式设置：

```toml
[analysis]
measure = "rate_flux"
flux_axis = 0
```

采样在第一个 CV 的法向通量测度下进行。辅助 CV 可以是键长等变量。此时

\[
A(x)=\int e^{-\beta F_{\rm flux}(x,y)}dy,\qquad
k^{(2),+}(x^*)=k_{\rm ref}^{(2),+}\frac{A(x^*)}{A(x_{\rm ref})}.
\]

不要再乘法向热速度。此面只适用于固定 `s0` 截面，不能旋转后用于另一方向的通量。
横向积分使用各 bin 的真实宽度；在 `s0` 非 bin 中心处，只沿 `s0` 插值自由能。
这通常不等同于先边际化再插值，因此解析一致性测试使用精确 bin 中心。

spec 示例中的量需由实际采样及入口物理确定：

```json
{
  "schema": "pyrpmd.rate_postprocess",
  "schema_version": 1,
  "method": "flux_slice_2d",
  "surface": 1.0,
  "reference": 0.0,
  "reference_rate_au": 1.0,
  "reference_definition": "Replace this example value with the physical entrance flux",
  "flux_axis": 0
}
```

上述 `1.0` 是格式示例，不能当成 H+H2 的入口通量。真实 benchmark 会自动计算并
保存其正确数值。温度默认从 `input/resolved.json` 读取；导入无该文件的 FES 时必须
在 spec 中显式提供 `beta`。`summary.json` 的测度、能量单位、CV 单位和连通性不可省略。

```bash
pyrpmd rate-postprocess run --spec rate-spec.json --output rate-result.json
```

## 多通道有符号通量与联合误差

`channel_rates` 接受：

| 数组 | 形状 | 含义 |
| --- | --- | --- |
| `time_au` | `[T]` | 严格递增的原子时间 |
| `parent_numerator` | `[P,T,C]` | 每个父构型已汇总的通道有符号分子 |
| `parent_denominator` | `[P]` | 所有通道共享的正向通量分母 |
| `parent_ids` | `[P,2]` | 唯一的 `(chain,parent)` 非负整数身份 |
| `channel_names` | `[C]` | 互斥产物通道名 |

每个子轨迹对的通道贡献包含正、反动量传播的差：

\[
\kappa_i(t)=\frac{\sum_p N_{pi}(t)}{\sum_p D_p},\quad
k_i=k_\Sigma^+\kappa_i,\quad b_i=\frac{\kappa_i}{\sum_j\kappa_j}.
\]

父构型内的多个子轨迹不能当作独立样本。实现对所有通道共同重采样相同的父链和
相关块，以保留通道之间及总通量的协方差。每条通道和总通量都要检查平台；总通量
已经平坦而分支继续变化时，不报告正式分支。负净通量保留为诊断，不裁剪到零。

误差是**固定静态 PMF 因子时的通道再穿越条件误差**，不包含静态 FES 误差。
旧一维运行中已存在的 PMF+recrossing 联合误差会保留在 `source_rate_uncertainty`。
单条父链的误差也明确标注其依赖链内混合假设。

CLI 可以通过 spec 的 `channel_flux` 指向 NPZ。该文件除上述数组外需包含标量字段：

```python
schema = "pyrpmd.channel_flux"
schema_version = 1
estimator = "signed_flux_side"
time_unit = "atomic"
surface_identity = "an identity covering physics, CV and the dividing surface"
```

spec 必须提供完全相同的 `surface_identity`。这是生产者与后处理之间的显式身份
合同；它应覆盖温度、珠数、势能、CV、分界面及统计权重，不能随意复用标签。
导入通量的几何产物承诺、能量守恒和离散化收敛仍需独立验证，故顶层正式
`rate_cm3_s` 保留为空，候选值和通道平台放在明确标注的字段中。

## 真实 H + H2 / BKMP2 回归

```bash
python benchmarks/h3/build_pes.py
python benchmarks/h3/pes.py --check --output validation/pes-check.json
python benchmarks/h3/rate_postprocess.py \
  --profile quick --workers 2 --output validation/h3-rate-postprocess
pyrpmd rate-postprocess validation/h3-rate-postprocess/run \
  --spec validation/h3-rate-postprocess/rate-postprocess-spec.json \
  --output validation/h3-rate-postprocess/cli-result.json
```

`--profile pilot` 使用更多珠子和更长采样。benchmark 采样带入口参考的
`(xi,r12)` 二维通量自由能，检查非均匀横向积分与同样本直接 MBAR 加权的一维边际
是否一致，并运行真实成对 RPMD 子轨迹，分别记录两个带原子标签的交换出口。
两个出口对应同一化学反应的不同原子排列，不能额外乘二，也不强制结果为 50:50。

该测试使用较粗的辅助坐标分箱以验证速率边际化，不是高分辨率二维反应路径图。
报告保留横向有限域、域外权重、父构型数、步长、珠数和能量诊断；`converged=false`。
同样本一维/二维一致性是实现验收，不是两次独立物理采样的收敛证明。
