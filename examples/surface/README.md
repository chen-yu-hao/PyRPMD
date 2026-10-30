# 表面计算示例

完整定义和适用范围见 [表面催化说明](../../docs/surface-catalysis.md)。

| 文件 | 用途 |
| --- | --- |
| `diffusion.inp` | 8-bead 周期解析模型，一阶盆地归一化迁移速率及真实回穿 |
| `fes2d.inp` | 同一模型的迁移／高度二维自由能 |
| `toy_surface.py` | 显式绑定 cell/PBC 的解析周期 Python PES，提供可核对的力 |
| `ase_emt.py` | ASE 构建 Cu(111)+Cu adatom，EMT、固定底层和活动表层的 FES |

在仓库根目录：

```bash
python -m pip install -e '.[test,ase]'
pyrpmd run examples/surface/diffusion.inp --output runs/surface-diffusion
pyrpmd run examples/surface/fes2d.inp --output runs/surface-fes2d
pyrpmd run examples/surface/ase_emt.py --output runs/surface-ase
```

所有配置均为 pilot，不代表物理收敛或目标催化剂的定量预测。真实研究需替换成
经验证的 PES，并检查 bead 数、时间步、slab 大小／厚度、固定层、盆地边界、
采样、直方图以及回穿平台。解析模型只统计单个迁移出口，不额外乘位点简并。
