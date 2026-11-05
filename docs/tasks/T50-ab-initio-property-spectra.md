---
orphan: true
---

# T50 从头算偶极驱动的光谱（socket `extras` 消费端）

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 2（第五轮） |
| 前置 | T28（含第五轮增补：`PESResult.extras`） |
| 建议分支 | `agent/t50-ab-initio-spectra` |

## 背景

i-PI 协议的 `GETFORCE` 回复在力和维里之后附带一个 `extras` 字符串。i-PI 3 的驱动以 JSON 形式
在其中返回偶极矩、Born 有效电荷、电子摩擦张量等附加量，i-PI 用这些量计算从头算 IR 光谱和
电子摩擦动力学。具体哪些外部程序发送哪些键，需要逐一查证（例如 ASE `SocketClient` 不发送偶极，见下文）。

PyRPMD 的 i-PI 服务端读取 `extras` 后直接丢弃（`src/pyrpmd/potentials/socket.py:90`），
T28 的第五轮增补负责解析它，并通过 `PESResult.extras` 交给调用方。

光谱工作流目前只能在保存的轨迹帧上**事后**调用性质模型（`properties.py` 的
`evaluate_property_trajectory`，支持 Python 和 TorchScript 模型）。对于从头算体系，
事后在每一帧上重新做电子结构计算的代价与整条轨迹相当。正确的做法是在动力学过程中，
直接记录同一次力计算得到的偶极。

## 目标

让 `task="spectroscopy"`（TRPMD、RPMD、PA-CMD）能直接使用外部电子结构程序在力计算时返回的
每个珠子的偶极，不增加额外的电子结构调用，并给出可复现的从头算 IR 光谱示例。

## 范围

**要做：**

1. **性质来源 `pes_extras`。** 在 `properties.py` 中新增一种性质来源：从 `PESResult.extras`
   中按键取出偶极（必要时取 Born 有效电荷），转换为 PyRPMD 的内部单位，并做形状与有限值校验。
   每个客户端的键名和单位约定（CP2K、FHI-aims、QE、ASE `SocketClient`）在指南中逐一写明，
   未实机验证的标注为“未实测”（G10）。
2. **在线记录。** 光谱工作流在采样帧上记录每个珠子的性质，与轨迹一起原子提交到 checkpoint；
   续算后不丢帧、不重复。只在采样帧要求 `extras` 中有性质；缺失时报错并保留已提交前缀。
3. **身份。** 性质来源（`pes_extras`、键名、单位）进入 run identity；同一运行目录不能在续算时更换性质来源。
4. **示例。** 新建 `examples/ab_initio/ir_water/`：对单个水分子或小团簇做短 TRPMD，生成相对 IR 光谱。
   示例只验证数据流，不声称物理收敛。客户端的选择：
   - **不要使用 ASE `SocketClient`**：已核实（ASE 当前版本 `socketio.py` 的 `sendforce`），它的 `extras`
     只是一个占位字节，不发送偶极。
   - 推荐使用 i-PI 自带的 Python 驱动 `i-pi-py_driver`（`pip install ipi`）：其驱动类的
     `compute_structure` 可以返回 JSON 格式的 `extras` 字符串（参见 `ipi/pes/dummy.py`）。
     编写一个包装 tblite/xtb 的驱动类，返回能量、力和偶极。`ipi` 作为可选测试依赖，缺失时跳过。
   - CP2K、FHI-aims、QE 是否以及如何发送偶极，需要逐一查证官方文档或源码后再写入指南；
     未查证或未实机运行的一律标注“未实测”。

**不做：**

- 不实现基于 `extras` 的电子摩擦张量（EF-RPMD）；在指南中列为后续工作。
- 不计算绝对 IR 强度或光学前因子（与已有光谱工作流的边界一致）。

## 文件所有权

- **独占：** `src/pyrpmd/properties.py`（新增来源）、`workflows/spectroscopy.py`（在线记录）、
  `spectroscopy_input.py`、新建 `examples/ab_initio/ir_water/`、新建 `docs/ab-initio-spectroscopy.md`（`orphan: true`）、相关测试。
- **共享注册点：** `docs/conf.py`、T47 合并后的注册表。
- **只读：** `potentials/socket.py` 与 `potentials/base.py`（由 T28 修改）。

## 交付物

- `pes_extras` 性质来源与在线记录。
- 示例与指南。
- 验证记录 `docs/validation/ab-initio-spectra-<日期>/README.md`。

## 验收标准

- 用可控的假 i-PI 客户端（返回已知的偶极函数）运行光谱工作流：
  - 记录的每个珠子的偶极与假客户端的解析值一致；
  - 电子结构调用次数与不记录性质时相同（计数断言）；
  - 得到的光谱与把同一偶极函数包装为 `CallablePropertyModel` 事后计算的光谱逐位一致。
- 中断续算后的性质序列与不中断时逐位一致。
- 在实机程序上：socket 返回的偶极与同一程序独立单点计算的偶极在 SCF 容差内一致；
  偶极对坐标的有限差分与 Born 有效电荷一致（程序提供时）。
- 采样帧缺少 `extras` 性质时给出明确错误，已提交的前缀保持完整。
- 端到端测试走真实的输入解析与工作流分派；mock 只替代外部程序（G19）。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "spectroscopy or properties or extras"
python -m pytest -q -m "not campaign"
```
