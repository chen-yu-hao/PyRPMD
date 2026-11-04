---
orphan: true
---

# T38 多进程 multiple-walker MetaD/OPES

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t38-multi-walkers` |

## 背景

`src/pyrpmd/shared_metadynamics.py` 的 `SharedMetadynamicsBias` 已经实现了同步的 multiple walkers：
所有 walker 共享同一份历史，在沉积屏障处整批提交，整个系综写入一个原子 checkpoint。
但它只能在单进程中串行运行（见 `docs/tier2-methods.md`：“no asynchronous multi-process file sharing”）。
OPES 也没有多 walker 共享核的模式。

对于昂贵的 PES，walker 必须分布到多个进程或节点上，才能获得实际的加速。

## 目标

让 MetaD 和 OPES 的 multiple walkers 分布到多个进程或节点。提供两种模式：

- **同步模式：** 可复现，结果与现有串行实现逐位一致；
- **异步模式：** 追求吞吐量，结果在统计上等价。

两种模式都保证共享历史在任何进程崩溃后不会损坏。

## 范围

**要做：**

1. **同步 MPI 模式。** 在沉积步做 allgather，各进程用相同的提交前偏置计算新的高斯核；
   结果与单进程的 `SharedMetadynamicsBias` 逐位一致。
2. **异步文件模式。** 每个 walker 向自己的只追加日志写入核，写入采用原子方式并带校验和；
   定期读取其他 walker 的日志并合并（参考 PLUMED `WALKERS_DIR` 的思路，独立实现）。
   已合并的位置记录在 checkpoint 中。
3. **OPES。** 为 `opes.py` 增加共享核的 multiple-walker 模式，支持上述两种模式；
   压缩与归一化规则在指南中写明。
4. **容错。** 单个 walker 崩溃后，其他 walker 继续运行；日志的半条记录会被识别并忽略；
   续算时从已提交的位置恢复。
5. **入口。** 在 `[metadynamics]` 和 `[opes]` 中新增 `walkers_mode = "serial"|"mpi-sync"|"async-files"`；
   默认值保持现有行为。

**不做：**

- 不修改 PLUMED 路线（归 T36）。
- 不改变现有单进程 shared-walker 的 checkpoint 格式。

## 文件所有权

- **独占：** `src/pyrpmd/shared_metadynamics.py`、`workflows/shared_metadynamics.py`、`opes.py`、
  `workflows/opes.py`、`bias.py`（只限 multiple-walker 相关的追加接口）、相关测试。
- **共享注册点：** `input_tasks.py`、`input_fes.py`（`walkers_mode` 字段的校验）。

## 交付物

- 两种模式的实现、OPES multiple walkers、英文指南（追加到 `docs/enhanced-sampling.md`）。
- 验证记录 `docs/validation/multi-walkers-<日期>/README.md`。

## 验收标准

- `mpi-sync` 模式在 2 进程和 4 进程下，与单进程 shared-walker 的偏置历史和轨迹逐位一致。
- `async-files` 模式在二维双势阱上得到的 FES，与同步模式在统计误差内一致；
  并给出吞吐量对比（使用一个人为放慢的 PES）。
- 故障注入测试：杀掉一个 walker、截断一条日志记录、重复启动同一 walker，
  共享历史都保持一致，续算正确。
- OPES multiple walkers 的 FES 与单 walker OPES 在统计误差内一致。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "metadynamics or opes or walker"
mpiexec -n 4 python -m pyrpmd run examples/metadynamics/multiple_walkers.inp --output runs/t38
```
