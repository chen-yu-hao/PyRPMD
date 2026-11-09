---
orphan: true
---

# T48 去除 MetaD/OPES 兼容层

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1（第五轮） |
| 前置 | 无 |
| 后续依赖本卡 | T51 |
| 建议分支 | `agent/t48-remove-compat-shims` |

## 背景

T38（分布式 multiple walkers）为了不直接编辑大文件，采用了“逐字节私有副本 + 导入时打补丁”的兼容层。
第四轮审查已把去重列为待办（`docs/roadmap.md` 的“去重”），并据此新增了规则 G15。目前的情况：

- `src/pyrpmd/workflows/metadynamics.py:43` 执行 `sys.modules[__name__] = _base`，
  用 `_metadynamics_base.py` 替换自身，并在替换前修改 `_base.METADYNAMICS_FIELDS` 与
  `_base.resolve_metadynamics`。
- `src/pyrpmd/workflows/opes.py` 把 `_opes_serial` 的全部全局名称复制进自身
  （`for _name, _value in vars(_serial).items(): globals()[_name] = _value`），
  再用 `setattr` 向 `_serial`、`_mpi`、`_async`、`_async_io`、`_common` 五个模块注入函数，
  并直接改写 `_async.atomic_savez`、`_mpi.atomic_savez`、`_async.fes_sampler` 等属性。
- `src/pyrpmd/workflows/shared_metadynamics.py` 用同样的 `setattr` 方式修补四个模块。
- `src/pyrpmd/_opes_serial.py`（442 行）与 `src/pyrpmd/_shared_metadynamics_serial.py`（56 行）
  是原始实现，`opes.py`、`shared_metadynamics.py` 再从中导入并扩展；
  `workflows/_opes_serial.py`（467 行）、`workflows/_shared_metadynamics_serial.py`（241 行）、
  `workflows/_metadynamics_base.py`（843 行）同理。

这种结构让真实的执行路径取决于导入顺序：先导入哪个模块，决定了哪个函数被替换。
它也阻碍后续改动，例如 T51 的 OPES 核压缩。

## 目标

把 T38 的改动直接写进原模块，删除副本与导入时补丁。
所有已有输入的物理身份、已有 checkpoint 格式和续算结果保持逐位不变。

## 范围

**要做：**

1. **合并模块。** 每组只保留一个实现模块：
   - 偏置：`opes.py`（含原 `_opes_serial.py` 内容）、`shared_metadynamics.py`（含原 `_shared_metadynamics_serial.py`）；
   - 工作流：`workflows/metadynamics.py`、`workflows/opes.py`、`workflows/shared_metadynamics.py`，
     把字段白名单扩展（`walkers_mode`）、分布式模式校验和分派直接写进函数体。
2. **分布式模块的依赖改为显式参数或正常导入。** `_distributed_*` 模块不再依赖被注入的
   `atomic_savez`、`fes_sampler` 等属性；需要的函数通过正常导入或参数传入。
   “导入阶段无副作用”的原意通过把副作用移到函数调用时实现，而不是在导入时改写其他模块。
3. **旧模块名。** 如果仓库内或已保存的 checkpoint 元数据引用了 `_opes_serial`、
   `_shared_metadynamics_serial`、`_metadynamics_base` 这些模块名，保留同名的薄别名模块：
   只从新位置重新导出名称（不超过约 10 行，不含任何实现代码，不修改其他模块），
   并发出 `DeprecationWarning`。没有任何引用的，直接删除。
4. **测试中的 monkeypatch 点。** 已有测试如果修补的是旧位置，改为修补新的真实位置，
   并在 PR 中逐一列出。
5. **静态检查。** 新增一个测试（可放在 `tests/test_code_structure.py`），用 AST 扫描
   `src/pyrpmd/`，拒绝以下写法：对 `sys.modules[__name__]` 赋值；
   在模块顶层把其他模块的 `vars()` 复制进 `globals()`；在模块顶层对其他模块执行 `setattr`
   或属性赋值。`potentials/python.py:37` 与 `input.py:75` 为加载用户插件而写入 `sys.modules`，
   属于合法用法，放入显式白名单并注明理由。

**不做：**

- 不改变任何数值行为、随机流、hills/kernel 存储格式或 checkpoint 布局。
- 不实现 OPES 核压缩（见 T51）。
- 不改动 `bias.py` 中 `MetadynamicsBias` 的数学实现。

## 文件所有权

- **独占：** `src/pyrpmd/opes.py`、`_opes_serial.py`、`shared_metadynamics.py`、
  `_shared_metadynamics_serial.py`、`workflows/metadynamics.py`、`workflows/_metadynamics_base.py`、
  `workflows/opes.py`、`workflows/_opes_serial.py`、`workflows/shared_metadynamics.py`、
  `workflows/_shared_metadynamics_serial.py`、`workflows/_distributed_*.py`、
  新建 `tests/test_code_structure.py`，以及修补点需要更新的相关测试。
- **共享注册点：** `workflows/__init__.py`（只在导入路径变化时调整）。

## 交付物

- 合并后的模块与（如需要的）薄别名模块。
- 结构检查测试。
- 验证记录 `docs/validation/compat-shim-removal-<日期>/README.md`：列出每个被删除的副本、
  每个改动的 monkeypatch 点，以及下面的逐位续算证据。

## 验收标准

- `src/pyrpmd/` 中不再有对 `sys.modules[__name__]` 的赋值、`globals()` 复制或跨模块的顶层 `setattr`
  （白名单除外），由结构检查测试保证。
- **逐位续算。** 在 `main@801953d` 上生成以下运行的 checkpoint，用新代码续算，
  与在 `801953d` 上不中断运行的结果逐位一致（`q`、`p`、hills 或 kernels、RNG 状态、输出数组）：
  - 独立副本 MetaD 与 well-tempered MetaD；
  - 同步 shared-walker MetaD；
  - OPES schema 1 与 schema 2 历史、explore 模式、自适应带宽；
  - `walkers_mode="mpi-sync"`（MPI 可用时）与 `"async-files"`。
  生成脚本与 SHA256 写进验证记录；大文件不进 git（G9）。
- 已有运行的 run identity 不变；已有示例 `examples/metadynamics/`、`examples/opes/` 输出不变。
- 删除的私有模块不再被任何源码或测试导入，或者以薄别名形式保留并发出 `DeprecationWarning`。
- 满足通用规则 G5–G8、G15。

## 验证命令

```bash
python -m pytest -q tests/test_code_structure.py
python -m pytest -q tests -k "metadynamics or opes or walker or shared"
python -m pytest -q -m "not campaign"
# MPI 用例由 tests/conftest.py 的 mpi_launch 启动；需要 mpi4py 与可用的 mpiexec
python -m pytest -q tests -k "mpi and (metadynamics or opes or walker)"
```
