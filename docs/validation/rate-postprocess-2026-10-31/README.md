# 二维自由能速率后处理与真实 H + H2 验证

本记录验证二维自由能通量积分、双分子入口归一化、通道分辨有符号再穿越及离线文件接口。
基线为 `main@3dc82c399a7701d19d636ee78b43f2e8ba7659f9`。
新实现与科学适用范围见 [使用说明](../../rate-postprocessing.md)；完整采样配方见
[H3 benchmark](../../../benchmarks/h3/RATE_POSTPROCESS.md)。

## 实际计算结果

势能为固定源 SHA256 的 BKMP2；在 1500 K、步长 0.1 fs、`Rinf=10 bohr` 下采样。
两个 CV 为 `(xi,r12)`，使用明确的 `rate_flux` 测度及 `flux_axis=0`。
`xi=0` 作为已知自由入射球面参考，`xi=1` 为固定测试分界面。

| 项目 | quick | pilot |
| --- | ---: | ---: |
| 珠数 | 8 | 16 |
| umbrella 采样流 | 96 | 144 |
| 保存 CV 样本 | 19,200 | 86,400 |
| 独立受约束父链 | 2 | 2 |
| 父构型 | 32 | 64 |
| 子轨迹，包含正反动量两方向 | 128 | 256 |
| TST 速率，cm³ molecule⁻¹ s⁻¹ | 4.818519203916103e-12 | 9.385762579084539e-12 |
| 总传输系数 | 0.9531094930005963 | 0.8831643907033088 |
| 候选总速率，cm³ molecule⁻¹ s⁻¹ | 4.592576395458114e-12 | 8.289171289443113e-12 |
| 标号出口 H0H1+H2 | 0.5364298995 | 0.5064608363 |
| 标号出口 H0H2+H1 | 0.4635701005 | 0.4935391637 |
| 各分支条件标准误差 | 0.35715 | 0.34757 |
| 最大终点能量误差 `abs(delta H)/(P kBT)` | 0.0142821 | 0.0277941 |
| 二维积分/同样本直接一维边际相对误差 | 1.11e-16 | 1.11e-16 |

所有这些都是初步测试估计。两次计算同时改变珠数和采样量，候选总速率仍相差约 1.8 倍，
尚未获得物理收敛。分支约 50:50 只与置换对称性定性相容，约 0.35 的条件标准误差
不支持高精度分支结论。两个出口是同一化学反应的原子标号排列，不能再额外乘二。

## 已完成的验收

最终完整本地测试：**1478 passed、13 skipped、153 subtests passed**，零失败。
跳过项均为未安装的可选运行时或外部参考数据。统计与命令见 [tests.json](tests.json)。

- 二维通量积分与直接使用同一批 MBAR 样本权重构造的一维边际，达到浮点精度一致。
  这是独立代数实现的核对，不是两次独立物理采样的对照。
- 实际运行正、反动量子轨迹。两个通道的 signed numerator 之和逐点复现已有单通道估计器。
- 所有通道联合重采样相同的父链和相关块，保留分支与总通量之间的协方差。
- 所有产品侧终点均通过额外的键长、分离距离及向外运动诊断：quick 57 个，pilot 123 个。
  这仅是终点诊断，没有声称持续承诺或长期稳定产物分类已经完成。
- Python 文件适配器和公开 CLI 都从保存文件重新计算，数值一致，并保持正式
  `rate_cm3_s=null`、`converged=false`。
- 真实网格发现并修复了 `1.0` 与相应 bin 中心相差一个 ULP 时，极小伪插值权重误触 masked
  邻格的问题。仅作受格距限制的 ULP 吸附，真实缺支持仍拒绝。
- 审查新增了 FES/PMF 元数据矛盾、PMF 曲线与 JSON 势垒不符以及覆盖源表格的拒绝测试。

采样图为 23×2 的非均匀直方图，辅助轴只用于验证横向积分，不是高分辨率二维路径图。
有限辅助轴范围是 `[0,4] bohr`；未观察到域外样本不等于已经证明尾部截断误差为零。
本轮未计算静态 PMF bootstrap，动态误差以静态前因子固定为条件。

## 无需势能和新动力学的离线重放

本目录保存真实运行的精简可重放包，包括后处理所需的 FES 数组、父构型通道通量、
解析输入和元数据，可以直接复算：

```bash
pyrpmd rate-postprocess docs/validation/rate-postprocess-2026-10-31/quick/run \
  --spec docs/validation/rate-postprocess-2026-10-31/quick/rate-postprocess-spec.json \
  --output /tmp/h3-quick-rate-replay.json

pyrpmd rate-postprocess docs/validation/rate-postprocess-2026-10-31/pilot/run \
  --spec docs/validation/rate-postprocess-2026-10-31/pilot/rate-postprocess-spec.json \
  --output /tmp/h3-pilot-rate-replay.json
```

精简 FES 只删除后处理不使用的 `weights`、`overlap` 和 `_pyrpmd_checkpoint` 三个数组，
其余数组的形状、类型和字节内容均与实际采样文件一致。本次本地运行的完整源 FES 保留在
`runs/h3-rate-postprocess-{quick,pilot}/run/fes.npz`，未纳入版本控制；每个 profile 的
`replay-subset.json` 记录完整源文件和精简文件的 SHA256，以及删除数组的形状、类型和
SHA256。`h3-summary.json` 中的 `source_artifact_sha256` 保留原始来源摘要，
`replay_artifact_sha256` 则标识本目录中的可重放文件。

`surface-identity.json` 包含完整反应坐标、等价路径、
体系、温度、珠数、积分器、实际 PES 库 SHA256 和固定源身份。`metadata-upgrade.json`
记录身份字段加固：只升级 provenance，前后物理通量数组摘要不变。
GitHub Actions 工作流已配置将完整 umbrella 原始流、检查点和势能检查保存到
`h3-rate-postprocess` artifact；工作流成功执行并上传后可获取。

精确机器可读候选值、运行参数和限制见 [h3-summary.json](h3-summary.json)。
