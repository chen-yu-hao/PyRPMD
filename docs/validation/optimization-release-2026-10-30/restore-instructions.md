# PyRPMD 原性能证据包：离线恢复补充

此补充包修复原证据包恢复说明中的 Git 元数据缺口。原归档、原补丁、原
benchmark harness、两份 native 二进制及五轮性能数据均保持原样。

## 实际发现与修复

原 `RESTORE.txt` 要求解开 `baseline-source.tar`、`candidate-source.tar` 并复制
各自的 `.so`。这样可以得到源码和二进制，但源码归档不包含 `.git`，而原
`audit_optimization.py` 在启动时会调用 `git rev-parse HEAD`，并要求基线提交
严格等于 `6fde29d2039c01921f00393bf17830b939af131b`。隔离实证中原命令在
控制器第 579 行因 `not a git repository` 失败，还未创建测量输出目录。

这里提供一个仅含原基线及其 41 个祖先提交的 Git bundle。恢复脚本先从该
bundle 建立两个真实的 Git checkout，再对候选应用原 `candidate.patch`，
最后复制原二进制。这样既能满足原 harness 的检查，又保留候选在原测量中
“HEAD 仍为基线、变更已暂存”的准确状态。无需网络访问或修改原 harness。

`baseline.bundle` 只公布这一条 ref：

```text
6fde29d2039c01921f00393bf17830b939af131b refs/heads/measured-baseline
```

bundle 大小为 2,380,058 字节，含完整的 42 提交历史；没有其他分支、标签、
新候选提交、工作区文件或本地 Git 配置。

## 一条经过验证的恢复命令

把本补充包、原证据包及两者的 `.sha256` 文件放在同一目录。使用能导入原
扩展的 Python 环境执行以下命令，其中 `restored-pyrpmd` 必须尚不存在：

```bash
sha256sum -c perf-audit-optimization-evidence.tar.gz.sha256
sha256sum -c perf-audit-optimization-restore-supplement.tar.gz.sha256
tar -xzf perf-audit-optimization-restore-supplement.tar.gz
python perf-audit-optimization-restore-supplement/restore.py \
  --archive perf-audit-optimization-evidence.tar.gz \
  --destination restored-pyrpmd \
  --smoke
```

脚本本身只依赖 Python 标准库和 Git；导入和 smoke 需要下述运行环境。
它固定核对原 archive 与 bundle 的 SHA256，拒绝覆盖任何已有目标目录。
随后会核对原始清单、完整源码、Git tree、测量时的每个源码指纹、native
和 PES 指纹，并测试实际导入位置。最后调用原 harness 执行全部八个缩小
功能 case；`--smoke` 不采集时间，不产生加速比。省略 `--smoke` 会只执行
恢复、校验和导入测试，并在结果 JSON 中给出可单独执行的 smoke 命令。

脚本产生：

```text
restored-pyrpmd/baseline/
restored-pyrpmd/candidate/
restored-pyrpmd/evidence/perf-audit-optimization/
restored-pyrpmd/evidence/perf-audit-optimization-evidence/
restored-pyrpmd/restore-validation.json
restored-pyrpmd/smoke-output/                 # 使用 --smoke 时
```

恢复后的候选仍是原测量的冻结 tree，而不是之后可能出现的分支最新版本：

| 身份 | 精确值 |
| --- | --- |
| 两个 checkout 的 HEAD | `6fde29d2039c01921f00393bf17830b939af131b` |
| baseline index tree | `80e21b261a1b3b66084af982ffd696907843f4cc` |
| candidate index tree | `899cc0618ba92db03e891c236845dac671704c00` |
| 原 harness SHA256 | `8b05c7c9e78a5fe9b0d00616f6204d5e7669fc4b668903d9cd1e6973177c7be1` |
| 原证据包 SHA256 | `00daad7cf6e48ecbc17625500dcbf0a2a18eb30d732f22f58d5be123a7493cae` |
| baseline bundle SHA256 | `ab064c3b1aadcaba5212a0ca6d4b8cb8450f40914616a1ce706ac4782a7867a6` |

## 运行环境与路径说明

原扩展文件名为 `_native.cpython-312-x86_64-linux-gnu.so`。本次恢复测试使用
CPython 3.12.14、Linux x86_64、glibc 2.39，以及原记录的 NumPy 2.5.4、
SciPy 1.18.1、h5py 3.16.0、threadpoolctl 3.6.0。OpenMP 扩展依赖的系统
运行库也必须可解析。`restore.py` 用启动它的同一 Python 解释器执行导入
及 smoke，不安装软件，不编译或替换 `.so`。

若换平台或重新编译，二进制将成为新的测量对象，需要重新记录其指纹；
不要把新二进制写回原证据。本补充测试的结论是原保存二进制在匹配环境中
可以恢复和导入。源码构建、wheel 安装验证属于独立检查。

原报告中的 checkout、输出目录和构建 provenance 中的绝对路径是历史
记录，并非恢复后的目标地址。恢复脚本生成新的绝对路径传给原 harness，
保留历史报告与 provenance 的字节不变。BKMP2 的同一个库和相邻 JSON
实际使用路径是：

```text
restored-pyrpmd/evidence/perf-audit-optimization-evidence/pes/libbkmp2.so
restored-pyrpmd/evidence/perf-audit-optimization-evidence/pes/libbkmp2.build.json
```

原外层 archive 解包后包含两个同级目录。manifest 路径的基准分别是：

| 文件 | 其键名相对的目录 |
| --- | --- |
| `perf-audit-optimization-evidence/raw-files-sha256.json` | `perf-audit-optimization/` |
| `perf-audit-optimization-evidence/bundle-files-sha256.json` | `perf-audit-optimization-evidence/` |
| 本补充包 `files-sha256.json` | `perf-audit-optimization-restore-supplement/` |

原总报告位于 `perf-audit-optimization/report.json`。各 manifest 不包含
自身；外层 archive 的独立 SHA256 覆盖整个包，包括 manifest 本身。

## 已完成的隔离验证与证据

2026-10-30 的恢复验证已完成，无正式性能重测：

- 原外层 archive：40,880,804 字节，SHA256 与原 sidecar 一致。
- 原 raw manifest：1,041 个文件、42,300,624 字节全部匹配。
- 原 bundle manifest：12 个被记录文件全部匹配。
- 完整源码：baseline 357 文件、candidate 366 文件，内容与可执行位均匹配。
- 测量指纹：baseline 77 项、candidate 78 项全部匹配；两个完整 Git tree
  与上述值相同。
- 原补丁通过 `git apply --check --index`，实际应用后得到精确的候选 tree。
- 两个原生扩展均从恢复后的各自 checkout 导入；baseline 无 `sample_free`，
  candidate 有 `sample_free`，符合测量对象；两者 native/PES 哈希均匹配。
- 原 harness 的八个 case 全部通过 `--smoke`；报告模式为
  `smoke_functional_only`，所有 `seconds_raw` 为空，未产生 speedup。

证据索引：

| 路径 | 内容 |
| --- | --- |
| `verification/initial-integrity.json` | 原恢复失败、全部 manifest/tree/patch 核验及命令记录 |
| `verification/bundle-construction.json` | bundle 构建、唯一 ref、完整历史与 42 提交验证 |
| `verification/restore-validation.json` | 新恢复脚本全部操作、实际导入位置与成功状态 |
| `verification/smoke-output/report.json` | 全部八 case 的功能结果与逐字段数值核对 |
| `verification/smoke-output/` | smoke 的完整 worker 日志、数组及生产输出 |

这些 JSON 保留实测时的绝对路径作为记录；用于重新执行的是上述命令和
`restore.py`，不应照抄旧日志中的临时路径。原五轮正式测量仍以原包内的
`perf-audit-optimization/report.json` 为准。
