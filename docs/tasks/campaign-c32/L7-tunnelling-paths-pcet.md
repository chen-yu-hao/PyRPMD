---
orphan: true
---

# L7 隧穿、路径采样与 PCET

| 字段 | 内容 |
| --- | --- |
| CPU | 30–31（2 核） |
| 预算 | 92 核时（计划 70） |
| 工作流 | `pyrpmd instanton`（热速率、基态劈裂、golden-rule）、`pyrpmd vtst`（TST/CVT/ZCT/SCT）、低温交叉检查、`path_sampling`（RETIS、REPPTIS、FFS、∞RETIS）、committor、`pcet` |
| 外部模型 | 无（解析模型）；pyPCET 可选 |
| 建议分支 | `agent/c32-l7-tunnelling` |
| 记录 | `docs/validation/c32-l7-tunnelling-<日期>/README.md` |
| 复核 | L7 复核 L3；L3 复核 L7 |
| 启动 | `$C32_LAUNCH run --plan "$C32_PLAN" --root "$C32_DATA" --lane L7 ...` |

## 背景

这条车道的模型都便宜，核心价值是**同一模型上的多条独立路线**：精确量子、instanton、VTST/SCT、
RPMD Bennett–Chandler、路径采样和直接 MFPT。它们彼此独立，任何一条的归一化错误都会在比较中暴露。
本机实测：`examples/sampling/transition_interface.py` 约 7 s，`pyrpmd vtst` 在 `examples/rate/` 的
`minimal`、`pilot`、`research`、`from_endpoints` 四个输入上都能完成（第四轮 T31 退回时这四个输入都报错）。

## 子测试

### L7-a Eckart 势垒：速率随温度的五路比较（约 25 核时）

模型：`examples/instanton/eckart.py` 的一维 Eckart 势垒（参数不变）。温度取交叉温度 T_c 的
0.5、0.75、1.0、1.5、2.0 倍（T_c 由势垒曲率算出，写进预注册）。

| 路线 | 入口 | 适用范围 |
| --- | --- | --- |
| 精确量子 k(T) | 示例中的量子交叉检查（或独立的透射系数热平均） | 全部温度，作为参考 |
| ring-polymer instanton | `pyrpmd instanton`（P 加倍直到作用量与预因子变化小于预注册阈值） | T < T_c |
| VTST / SCT | 对应的 `rate` 输入上运行 `pyrpmd vtst --temperature ...` | 全部温度 |
| RPMD 速率 | `rate` 或 `site_rate`（P 由 P 与 2P 比较确定） | T ≳ 0.75 T_c |
| 经典 TST | 解析 | 只作背景 |

判据：instanton 的珠子收敛记录；每条路线与精确量子的比值和区间。RPMD 在深隧穿区与精确量子的偏离是已知的
物理近似，报告而不设阈值；但 T ≥ T_c 时 RPMD 与精确量子的比值区间应在预注册范围内（建议 [0.8, 1.25]）。

### L7-b 对称双阱基态劈裂（约 5 核时）

`examples/instanton/low_temperature.py` 与 `pyrpmd instanton` 的基态劈裂模式，对独立 DVR 精确劈裂：
P 与 β 的收敛序列；判据为外推后与 DVR 的相对差在预注册阈值内，未外推的值单独报告。

### L7-c 路径采样四种方法对同一速率（约 30 核时）

模型：`examples/sampling/path_sampling_quartic.py` 的四次双阱（或路径采样示例使用的同一模型）。
`examples/sampling/path_sampling_retis.inp`、`path_sampling_repptis.inp`、`path_sampling_ffs.inp`、
`path_sampling_infinite_retis.inp` 各放大采样、8 次独立运行（不同种子、独立目录）。

- 参考：同一模型的解析或数值精确速率（[路径采样指南](../../transition-interface-sampling.md) 中的四次势速率），
  以及 L7-d 的直接 MFPT。
- 判据：每种方法 8 次运行的区间包含参考值；方法之间按区间比较效率（单位力评估的方差）。
  FFS 噪声大是已知的，报告而不删除。
- 中途中断一次 RETIS 运行并续算，与不中断的同种子运行逐位比较。

### L7-d 同一模型的直接 MFPT 与 committor（约 15 核时）

1. 在 L7-c 的模型上用长时间无偏轨迹数跃迁（与路径采样相同的界面定义），得到第三个独立速率。
2. `examples/sampling/committor.py --shots 1024 --beads 4`：沿反应坐标取 7 个点，p_B(ξ) 与解析/数值参考比较，
   报告 A/B/未判定的计数和保守区间；0.5 交点的位置与 TS 的关系。

### L7-e PCET 扫描（约 5 核时）

`examples/pcet/harmonic_scan.inp` 放大到 5 个温度 × 5 个电位 × H/D：检查正逆速率比满足细致平衡
（与 ΔG 的关系在数值精度内）、H/D KIE 随温度的趋势、Tafel 斜率；pyPCET 可用时按
[PCET 记录](../../validation/pcet-model-2026-10-30/README.md) 的方法对照（记录的一致度为 0.062%）。

## 不做

- 不修改 instanton/VTST 的收敛参数默认值；收敛失败（例如 WKB 求积不收敛）如实记录为结果。

## 交付物

- `benchmarks/campaign_c32/L7/`：输入、精确参考脚本（含独立测试）、预注册；验证记录与判据汇总表。

## 派发提示词

```text
你是 PyRPMD（chen-yu-hao/PyRPMD）C32 验证战役的车道 L7 agent，负责"隧穿、路径采样与 PCET"。
这是测试与验证任务：不修改 src/、native/ 和已登记的基准工具。

开始前完整阅读：docs/tasks/README.md 的 G3、G5、G9–G14；docs/tasks/campaign-c32/README.md；
docs/tasks/campaign-c32/L7-tunnelling-paths-pcet.md；docs/instanton.md；docs/low-temperature.md；docs/vtst.md；
docs/transition-interface-sampling.md；docs/pcet.md。

要求：
- 你的 CPU 是 30–31（2 核），预算 92 核时；所有计算用 $C32_LAUNCH run --lane L7 启动。
- 每个模型上有多条独立路线（精确量子、instanton、VTST/SCT、RPMD、路径采样、MFPT）；
  先写好精确参考并单独测试，再比较。温度、珠子数阶梯和比值区间在生产前提交。
- 收敛失败、未判定 shot 和被拒绝的输入都是结果，如实记录。
- T+24 h、T+40 h 更新 $C32_DATA/L7/PROGRESS.md；T+50 h 停止生产。
- 在分支 agent/c32-l7-tunnelling 提交记录并开 PR（G11 + 判据汇总表）；Phase 2 复核 L3。
```
