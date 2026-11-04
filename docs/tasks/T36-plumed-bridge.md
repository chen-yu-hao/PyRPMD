---
orphan: true
---

# T36 PLUMED 桥升级：多 walker、珠子 CV、MPI

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t36-plumed-bridge` |

## 背景

`src/pyrpmd/plumed.py` 是可选的 PLUMED 质心偏置桥：PLUMED 计算 W(q_centroid)，
环聚合物哈密顿量接收 P·W；每个完整的 MD 步之后调用一次 `update`。

当前限制（见 `docs/plumed.md`、`docs/tier2-methods.md`）：

- 只能运行一条串行轨迹；
- 只支持质心偏置；
- 只有在初次运行前设置 `[plumed] restart="files"` 时才能续算。

PLUMED 有数百种 CV 和多种偏置方法（VES、PBMetaD、OPES 系列），以及原生的 multiple-walker 机制
（`WALKERS_N`/`WALKERS_DIR` 与 multi-sim）。打通这些能力，比在 PyRPMD 中重新实现更划算。
原路线图的 T24（VES）可以通过这条路线满足。

## 目标

让 PLUMED 路线支持多个副本和 multiple walkers、珠子分辨的 CV 和 MPI 执行，
并给出明确的续算语义，使 PyRPMD 可以直接使用 PLUMED 生态中的偏置方法。

## 范围

**要做：**

1. **多副本。** `sampling.replicas>1` 时，每个副本拥有独立的 PLUMED 实例、工作目录和输出前缀。
2. **multiple walkers。** 通过 PLUMED 自身的 `WALKERS_*` 文件共享机制或 multi-sim 通信器，
   让多个 walker 共享同一份偏置历史；明确更新时钟与提交顺序。
3. **珠子分辨模式。** 新增 `mode="beads"`：把 P×N 个珠子坐标作为 PLUMED 原子传入
   （原子序号映射写进输出），偏置力直接作用到各个珠子。
   质心模式保持为默认。文档写明两种模式对应不同的哈密顿量。
4. **MPI。** 每个 walker 或副本对应一个 MPI 组，复用 `src/pyrpmd/parallel/` 中的组划分；
   根进程负责汇总输出。
5. **续算。** 在所有新模式下，`restart="files"` 都能保存 PLUMED 外部状态的哈希与偏移量。
   恢复时检查文件一致性；不一致时拒绝续算，并给出明确提示。
6. **VES 示例。** 用 PLUMED 的 VES 模块（PLUMED 编译时启用了 VES）运行一个二维模型示例，
   与 PyRPMD 原生的 MetaD/OPES 结果对照。

**不做：**

- 不在 PyRPMD 中原生实现 VES。
- 不改变现有单轨迹质心模式的行为与 checkpoint 格式。

## 文件所有权

- **独占：** `src/pyrpmd/plumed.py`、`workflows/plumed.py`、`examples/plumed/`、`docs/plumed.md`、相关测试。
- **共享注册点：** `input_tasks.py`（`[plumed]` 新字段的校验）。

## 交付物

- 实现、更新后的英文指南和 VES 示例。
- 验证记录 `docs/validation/plumed-bridge-<日期>/README.md`。验证需要使用真实的 PLUMED 库，
  并写明版本与编译选项。

## 验收标准

- 已有 PLUMED 测试（包括 mock kernel 与真实 PLUMED）全部通过，单轨迹行为不变。
- 在二维双势阱上，4 个 PLUMED multiple walker 的 MetaD 得到的 FES 与 PyRPMD 原生
  shared-walker MetaD 的结果在统计误差内一致。
- 在 `mode="beads"` 下，PLUMED 给出的珠子偏置力与有限差分对照；对每个珠子施加相同的
  单珠势时，结果与解析值一致。
- 中途中断后用 `restart="files"` 续算，所有新模式的偏置历史和轨迹与不中断时一致
  （或在 PLUMED 文件精度之内）；外部文件被篡改时拒绝续算。
- 4 进程 MPI 测试通过。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "plumed"
mpiexec -n 4 python -m pyrpmd run examples/plumed/multiple_walkers.inp --output runs/t36-mw
```
