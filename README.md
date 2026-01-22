# ChestVision

本项目是“医学人工智能”课程作业项目，围绕**胸部X光多标签疾病分类**任务（ChestX-ray14 风格的 14 类疾病标签）复现并改进经典 CheXNet 思路，并完成从数据划分、训练、评测到可视化的一套可复现实验流程。

最终方案为 **DacNet-CB+**：以 DenseNet-201 为骨干，结合 **Class-Balanced Focal Loss**、两阶段微调与 Mixup 增强，在测试集上取得更优的综合指标。

> 说明：本仓库用于课程学习与研究展示，不构成医疗建议或临床可用系统。

## 项目亮点

- **Patient-level 划分**：按患者维度拆分训练/验证/测试，避免同一患者出现在不同集合导致数据泄漏。
- **类别不平衡处理**：从 BCE → Focal Loss → **CB-Focal Loss**，系统探索长尾分布下的优化策略。
- **多模型对比**：Baseline（CheXNet复现）、DacNet、DacNet-CB+、ViT（对照实验）。
- **可视化完善**：ROC 曲线、训练曲线、代表性疾病混淆矩阵、模型对比表等。

## 目录结构

```
ChestVision/
├── replicate_chexnet.py         # Baseline（CheXNet复现）训练脚本
├── dacnet.py                    # DacNet（Focal Loss + AdamW 等）训练脚本
├── dacnet_cb.py                 # DacNet-CB+（CB-Focal + DenseNet-201 + Mixup）训练脚本
├── vit_transformer.py           # Vision Transformer 对照实验
├── split.py                     # Patient-level 数据划分
├── test_pretrained_dacnet.py    # 加载权重进行测试/推理
├── visualize.py                 # 生成 ROC/混淆矩阵/训练曲线 等可视化
├── requirements.txt             # 依赖（pip）
├── dataset_splits/              # 划分后的患者ID
├── checkpoints*/                # 各模型训练权重
├── results*/                    # 测试结果与训练日志
└── visualizations/              # 可视化输出（csv/png）
```

## 环境与依赖

### 方式 A：pip（推荐给课程项目快速复现）

使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

### 方式 B：pixi/conda（如果你希望用统一环境）

## 数据准备

本项目默认使用“类似 ChestX-ray14 的”数据组织方式：

- 图像目录（例如 `images/`）
- 标签文件 `filtered_labels.csv`（仓库根目录提供样例/生成后的文件）

脚本中数据路径通过各训练脚本内的配置项指定（例如 `data_dir`、`images_dir`、`csv_file`）。

> 由于公开数据集体积较大，本仓库不直接提交原始影像文件；请自行下载/放置。

## 快速开始

### 1）划分数据（Patient-level）

运行 `split.py` 后会在 `dataset_splits/` 生成训练/验证/测试患者ID。

### 2）训练模型

- Baseline（CheXNet复现）：`replicate_chexnet.py`
- 中间改进版 DacNet：`dacnet.py`
- 最终方案 DacNet-CB+：`dacnet_cb.py`
- ViT 对照：`vit_transformer.py`

训练完成后会输出：

- 模型权重：`checkpoints*/`
- 训练日志：`results*/train_history*.npy`
- 测试评测：`results*/test_results*.json`

### 3）生成可视化

运行 `visualize.py` 会在 `visualizations/` 产出：

- `roc_curves_all_diseases.png`
- `roc_curve_average.png`
- `confusion_matrices_representative.png`
- `training_curves.png`
- `model_comparison.csv`

## 实验设置

- **任务**：胸片多标签分类（14 类疾病）
- **最终模型**：DacNet-CB+（DenseNet-201）
- **关键策略**：CB-Focal Loss（β=0.99, γ=2）、Mixup（α=0.1）、两阶段微调（前 3 个 epoch 冻结 backbone）、AdamW
- **评测指标**：AUC、F1、Loss（整体与逐疾病）

## 复现提示（常见问题）

- **显存不足**：将 `batch_size` 调小；或关闭部分增强；或降低 `image_size`。
- **路径不一致**：Windows 下请优先使用绝对路径，或在脚本配置中统一 `data_dir/images_dir`。
- **结果对不上**：确认是否使用相同的 `dataset_splits/` 与随机种子（seed=42）。

## 免责声明

- 本项目仅用于课程学习与科研复现。
- 模型输出不应直接用于临床诊断或治疗决策。
