# RRF：可形变物体动力学预测

RRF 用于预测物体在外部操控下的运动与形变。给定物体的历史状态和控制点运动轨迹，模型生成后续粒子位置序列，并支持将预测结果渲染为图像进行展示和评估。

本仓库提供模型训练、推理、评测、结果汇总及可视化相关代码，支持 PhysTwin 和 PGND 数据集。

## 主要功能

| 功能 | 内容 |
| --- | --- |
| 数据准备 | 读取和整理 PhysTwin、PGND 数据 |
| 模型训练 | 单场景训练、类别级训练，以及中断后继续训练 |
| 动力学预测 | 根据历史状态和控制轨迹，生成物体后续运动 |
| 泛化测试 | 跨动作、跨物体及类别级测试 |
| 几何评估 | 评估预测点云和运动轨迹 |
| 图像评估 | 计算 IoU、PSNR、SSIM、LPIPS 等指标 |
| 结果展示 | 渲染预测结果，并汇总评测表格 |
| 模型导出 | 导出用于推理的模型文件 |

## 输入与输出

**输入**包括物体历史粒子位置、控制点运动轨迹和参考形状等数据。

**输出**包括预测粒子位置序列、评测结果和汇总表格。配置好渲染环境与外观资源后，可以进一步生成物体图像及可视化结果。

## 仓库内容

| 目录或文件 | 用途 |
| --- | --- |
| `rrf/models/` | 模型代码 |
| `rrf/data/` | 数据读取与准备 |
| `rrf/configs/` | 数据路径和任务配置 |
| `rrf/train.py` | 模型训练 |
| `rrf/evaluate.py` | 模型推理与评测 |
| `rrf/pipeline.py` | 训练与评测流程入口 |
| `rrf/metrics/` | 评测指标 |
| `rrf/rendering/` | 渲染与图像评估 |
| `rrf/collect.py` | 结果汇总 |
| `scripts/` | 安装、迁移评测和模型导出工具 |
| `docs/SETUP.md` | 数据与运行环境配置说明 |

## 安装与配置

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate rrf
bash scripts/install.sh cu128

cp paths.env.example paths.env
# 将 paths.env 中的路径修改为本地数据和运行环境路径。
source paths.env
```

数据准备与渲染环境配置见 [配置说明](docs/SETUP.md)。

## PhysTwin 训练与测试

以下以 `double_stretch_zebra` 场景为例：

```bash
python -m rrf.prepare phystwin \
  --data-root "$PHYSTWIN_DATA" --out prepared/phystwin \
  --groups double_stretch_zebra

python -m rrf.pipeline \
  --task phystwin_scene \
  --data prepared/phystwin/double_stretch_zebra \
  --out runs/zebra --device cuda:0
```

## PGND 训练与测试

参照 `rrf/configs/pgnd_paths.example.json` 配置本地的 `pgnd_paths.json`，然后执行：

```bash
python -m rrf.prepare pgnd \
  --pgnd-root "$PGND_ROOT" --paths pgnd_paths.json \
  --out prepared/pgnd --tasks rope

python -m rrf.pipeline \
  --task pgnd --data prepared/pgnd/rope \
  --out runs/pgnd_rope --device cuda:0
```

支持 `rope`、`bread`、`sloth`、`box`、`cloth` 和 `paperbag` 六类物体。

## 泛化与迁移

仓库保留四组类别级配置：`main_rope_scene`、`main_cloth_scene`、`additional_cloth_scene` 和 `additional_cloth_action`。

类别级任务通过 `phystwin_scene_generalization` 或 `phystwin_action_generalization` 运行。跨动作与跨物体测试使用已训练的源模型，通过 `scripts/transfer_pairs.py` 指定源模型和目标场景。

相关配置位于 `rrf/configs/`，使用说明见 [配置文档](docs/SETUP.md)。

## 结果与可视化

运行后可获得预测轨迹、几何评测结果和汇总表格。渲染入口与图像评测的使用方式见 [渲染说明](docs/SETUP.md#simulation-and-rendering)。

模型导出示例：

```bash
python scripts/export_model.py \
  --checkpoint runs/zebra/model/final.pt --out zebra_rrf.pt
```
