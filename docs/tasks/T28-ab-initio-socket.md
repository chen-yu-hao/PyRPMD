---
orphan: true
---

# T28 从头算 PIMD/RPMD：i-PI 服务端配方与稳健性

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 后续依赖本卡 | T46 |
| 建议分支 | `agent/t28-ab-initio-socket` |

## 背景

`src/pyrpmd/potentials/socket.py` 已经实现了持久的 i-PI 力服务端，支持多个客户端并发、
动态分配、超时和按输入顺序返回结果；`tests/test_ipi_clients.py` 覆盖了协议行为。
因此 CP2K、Quantum ESPRESSO（`pw.x --ipi`）、FHI-aims、DFTB+、LAMMPS（`fix ipi`）和
ASE `SocketClient` 理论上都可以为 PyRPMD 提供从头算力。

但仓库中没有任何经过实机验证的从头算配方或示例，`examples/` 和 `docs/manual/` 都没有提到
这些程序。而这正是 i-PI 最主要的使用场景。

## 目标

让“PyRPMD 驱动 + 外部电子结构程序提供力”成为一条有文档、有测试、经过实机验证的一等路线，
覆盖分子 RPMD 速率和周期 PIMD 两类用法。

## 范围

**要做：**

1. **实机验证至少两个外部程序。** 优先选择能在 Linux CI 中安装的程序：
   - Quantum ESPRESSO（发行版软件包，或 conda-forge 的 `qe`）；
   - DFTB+（conda-forge）或 xtb/tblite，经 ASE `SocketClient` 接入。
   每个程序都要完成：分子 PIMD、周期 PIMD（含维里，用于 NPT），以及一个小型 RPMD 速率试点
   （只验证流程，不声称收敛）。
2. **其余程序写配方，标注为“未实测”。** CP2K、FHI-aims、VASP（经 ASE）、LAMMPS `fix ipi`
   都给出输入模板，并写清单位、晶胞约定和 PBC 责任方。
3. **稳健性：**
   - 客户端崩溃或断开后，同一几何重新分派；
   - 超时策略可以配置；
   - 支持 UNIX socket 与 TCP；
   - 多客户端时按珠子批次负载均衡；
   - 外部程序的 SCF 未收敛状态能映射到 `PESResult.status`；
   - 外部程序的版本、输入文件哈希和赝势或基组文件哈希进入 run identity。
4. **续算语义：** 外部程序的波函数外推等内部状态不进入 PyRPMD checkpoint。
   文档写清续算后第一步可能产生的 SCF 差异，并在测试中验证续算后的物理结果在 SCF 容差之内。

**不做：**

- 不实现 RPC/MTS 工作流接入（见 T46）。
- 不打包或分发任何电子结构程序。

## 文件所有权

- **独占：** `src/pyrpmd/potentials/socket.py`、新建 `examples/ab_initio/`（每个程序一个子目录，
  附 README）、新建 `docs/ab-initio.md`（`orphan: true`）、`tests/test_ipi_*.py`、
  新增 CI job 文件 `.github/workflows/ab-initio.yml`。
- **共享注册点：** `potentials/__init__.py`、`pyproject.toml`（可选依赖）。

## 交付物

- 实机验证过的示例和未实测的配方模板。
- 稳健性改进及其测试，使用可以模拟崩溃、慢速、错误状态的假客户端。
- 验证记录 `docs/validation/ab-initio-socket-<日期>/README.md`。

## 验收标准

- 对每个实机验证的程序：
  - PyRPMD 经 socket 得到的能量和力，与同一程序独立运行的单点结果一致（误差在 SCF 收敛阈值内）；
  - 力与有限差分对照；
  - 周期体系的维里与有限应变能量差分对照；
  - NVE 短轨迹的能量漂移在报告中给出，并与时间步长、SCF 阈值对应。
- 客户端在批次中途被杀掉后，计算继续，结果与不杀时一致（在 SCF 容差之内）；
  所有客户端都失败时给出明确的错误，并且不写入半个步长。
- CI job 在程序不可用时跳过，不使失败；可用时运行一个不超过 10 分钟的小型端到端测试。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "ipi or socket"
python -m pytest -q -m "not campaign"
```

## 第五轮增补（2026-11-04）：解析 `extras`

本节扩充本卡范围，优先于上文的“不做”与“文件所有权”中与之冲突的内容。消费端见 [T50](T50-ab-initio-property-spectra.md)。

**背景。** `src/pyrpmd/potentials/socket.py:90` 在读取 `GETFORCE` 回复时执行
`self.recv(extra, deadline)` 并丢弃内容。i-PI 3 的驱动在 `extras` 中以 JSON 返回偶极、
Born 有效电荷、电子摩擦张量等附加量（参见 i-PI 的 `ipi/pes/dummy.py`）。丢弃它意味着从头算 IR 光谱
和基于从头算摩擦张量的 EF-RPMD 都无法经 socket 实现。注意 ASE `SocketClient` 只发送一个占位字节，不发送偶极。

**追加要做：**

1. 解析 `extras`：空串或占位字节视为“无附加量”；合法 JSON 对象按键保存；非 JSON 的非空内容按原始字节保存，
   并记录一次警告，不报错。大小上限沿用现有的 16 MiB 检查。
2. 在 `potentials/base.py` 的 `PESResult` 中追加可选字段 `extras`（长度为 B 的元组，元素为字典或 `None`），
   默认 `None`。`validate_result` 只检查长度，不解释内容；已有字段、校验和所有现有提供者的行为保持不变。
3. socket 服务端按输入顺序把每个珠子的 `extras` 放进 `PESResult.extras`；客户端重新分派（见上文稳健性要求）时，
   采用最终成功那次计算的 `extras`。
4. 不解释任何物理键名与单位，这些由消费端（T50）负责。

**追加验收：**

- 用可控的假客户端发送 JSON、空串、占位字节、非 JSON 内容和超限大小，分别得到预期结果或明确错误。
- `pip install ipi` 可用时，用 `i-pi-py_driver` 返回 JSON `extras` 的驱动端到端测试；不可用时跳过。
- 所有现有提供者返回的 `PESResult.extras` 为 `None`，已有测试与 run identity 不变。

**追加共享注册点：** `potentials/base.py`（只追加 `PESResult.extras` 可选字段）。
