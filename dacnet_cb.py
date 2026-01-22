"""
DacNet-CB+训练脚本 - Class-Balanced DacNet with DenseNet201
在原始dacnet.py基础上，加入以下针对小规模数据集的策略：
1) 两阶段微调：先冻结特征提取网络，仅训练分类头；再解冻整体细调
2) 类别均衡Focal Loss：根据训练集统计得到每类权重，缓解严重类别不平衡
3) Mixup数据增强：在batch级别进行线性混合，提升泛化能力、减轻过拟合
4) 适度但稳定的图像增强：只采用torchvision内置增强，避免额外依赖
5) 与原实验一致的患者级划分与评估流程
"""

import os
import json
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
    densenet169,
    DenseNet169_Weights,
    densenet201,
    DenseNet201_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b4,
    EfficientNet_B4_Weights,
)
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve


# ==================== 配置 ====================
CONFIG: Dict[str, Any] = {
    "model_name": "DacNet-CB+",

    # 训练超参数
    "batch_size": 16,                # 可根据GPU显存调整
    "learning_rate": 1e-4,           # 基础学习率
    "epochs": 10,                    # 总训练轮数
    "freeze_backbone_epochs": 3,     # 前几个epoch只训练分类头
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 数据路径（保持与原始脚本一致）
    "data_dir": "../task",
    "images_dir": "../../autodl-tmp/chestxray/images",
    "csv_file": "../task/filtered_labels.csv",

    # 训练配置
    "patience": 3,                   # Early stopping耐心值
    "seed": 42,
    "image_size": 320,               # 使用略高于224的分辨率以保留更多细节

    # 基座模型选择： "densenet169"（默认）或 "densenet121" / "densenet201" / "efficientnet_b0" / "efficientnet_b4"
    "backbone": "densenet201",

    # Mixup配置（可按需关闭）
    "use_mixup": True,
    "mixup_alpha": 0.1,

    # 类别均衡Focal Loss配置
    "cb_beta": 0.99,                 # Class-Balanced公式中的beta
    "focal_gamma": 2.0,              # Focal Loss中的gamma

    # WandB配置（可选）
    "use_wandb": False,
    "wandb_project": "ChestXray-DacNet-Improve",
}


# 14种疾病列表（保持与原始脚本一致）
disease_list = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


print("=" * 60)
print("DacNet-CB+ (Class-Balanced DacNet) 配置信息")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"{key:22s}: {value}")
print(f"{'diseases':22s}: {len(disease_list)} classes")
print("=" * 60)


# ==================== 随机数种子 ====================
def set_seed(seed: int) -> None:
    """固定随机数种子，保证复现实验结果"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CONFIG["seed"])


# ==================== 数据增强 ====================
# 针对胸片的相对温和增强，避免过强几何变换破坏医学语义
image_size = CONFIG["image_size"]

transform_train = transforms.Compose(
    [
        transforms.Resize(int(image_size * 1.1)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

transform_eval = transforms.Compose(
    [
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

print("数据增强转换已准备就绪。图像尺寸:", image_size)


# ==================== 标签编码与统计 ====================
def get_label_vector(labels_str: str) -> np.ndarray:
    """
    将标签字符串转换为14维向量（多标签）
    'No Finding' 视为全0
    """
    labels = labels_str.split("|")
    if labels == ["No Finding"]:
        return np.zeros(len(disease_list), dtype=np.int64)
    return np.array([1 if disease in labels else 0 for disease in disease_list], dtype=np.int64)


def compute_class_counts(df: pd.DataFrame) -> np.ndarray:
    """
    在训练集上统计每个疾病的阳性样本数，用于构造类别均衡权重
    """
    counts = np.zeros(len(disease_list), dtype=np.int64)
    for labels_str in df["Finding Labels"]:
        counts += get_label_vector(labels_str)
    return counts


# ==================== Dataset类 ====================
class ChestXrayDataset(Dataset):
    """
    与原始脚本保持一致的Dataset实现，只修改了transform接口
    """

    def __init__(self, dataframe: pd.DataFrame, images_dir: str, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        img_name = self.dataframe.iloc[idx]["Image Index"]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels_str = self.dataframe.iloc[idx]["Finding Labels"]
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float32)

        return image, labels


# ==================== Mixup实现 ====================
def apply_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> (torch.Tensor, torch.Tensor):
    """
    在batch维度上进行Mixup：
    - 对图像做线性混合
    - 对多标签目标做同样的线性混合（保持在[0,1]区间）
    适用于BCE/Focal等基于概率的损失函数
    """
    if alpha <= 0.0:
        return inputs, targets

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    mixed_targets = lam * targets + (1.0 - lam) * targets[index]

    return mixed_inputs, mixed_targets


# ==================== 类别均衡 Focal Loss ====================
class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss
    - 使用训练集统计到的每类样本数，构造有效样本数权重
    - 再叠加Focal Loss思想，提升对少数类的关注，抑制易分类样本
    论文参考：Class-Balanced Loss Based on Effective Number of Samples (Cui et al.)
    """

    def __init__(self, class_counts: np.ndarray, beta: float = 0.99, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # 计算每个类别的有效样本数及对应权重
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)

        # 归一化，使得平均权重为1
        weights = weights / np.sum(weights) * len(class_counts)

        # 注册为buffer，自动随模型迁移到GPU
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

        print("\n类别均衡Focal Loss权重:")
        for disease, w, c in zip(disease_list, weights, class_counts):
            print(f"  {disease:20s} count={int(c):5d} | weight={w:6.3f}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 基础BCE损失（逐元素）
        bce_loss = self.bce(logits, targets)

        # Focal部分：对易分类样本降低权重
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal_factor = (1.0 - pt) ** self.gamma

        loss = bce_loss * focal_factor

        # 按类别权重加权
        loss = loss * self.class_weights.to(logits.device)

        return loss.mean()


# ==================== 模型构建 ====================
def build_model(num_classes: int, backbone_name: str) -> nn.Module:
    """
    构建可切换基座的多标签分类模型：
    - DenseNet121 / DenseNet169 / DenseNet201：与原始DacNet保持一致
    - EfficientNet-B0 / EfficientNet-B4：更高参数效率的模型
    两种基座均在分类头部分增加非线性与Dropout
    """
    print("\n构建模型...")
    backbone_name = backbone_name.lower()

    if backbone_name == "densenet121":
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier.in_features

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
    elif backbone_name == "densenet169":
        backbone = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier.in_features

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )
    elif backbone_name == "efficientnet_b0":
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # EfficientNet-B0原始classifier为 [Dropout, Linear]
        if isinstance(backbone.classifier, nn.Sequential):
            in_features = backbone.classifier[-1].in_features
        else:
            in_features = backbone.classifier.in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
    elif backbone_name == "densenet201":
        backbone = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier.in_features

        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )
    elif backbone_name == "efficientnet_b4":
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # EfficientNet-B4原始classifier为 [Dropout, Linear]
        if isinstance(backbone.classifier, nn.Sequential):
            in_features = backbone.classifier[-1].in_features
        else:
            in_features = backbone.classifier.in_features

        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"未知backbone: {backbone_name}")

    backbone = backbone.to(CONFIG["device"])
    print(f"模型加载完成，使用设备: {CONFIG['device']}，backbone={backbone_name}")
    return backbone


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    """
    控制是否冻结特征提取网络：
    - 特征提取层：除了classifier以外的所有参数
    - 冻结阶段只更新分类头参数，降低小数据集上的过拟合风险
    """
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


# ==================== 自适应阈值计算 ====================
def get_optimal_thresholds(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    为每个疾病计算最优F1阈值
    保持与原始脚本完全一致，方便结果对比
    """
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_threshold)
    return np.array(thresholds)


# ==================== 评估函数 ====================
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, desc: str = "Eval"):
    """
    评估模型，在验证集/测试集上计算：
    - 平均loss
    - 平均AUC / F1
    - 每类AUC / F1
    并打印每个疾病的详细指标，方便分析具体哪几类提升/退化
    """
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    thresholds = get_optimal_thresholds(all_labels, all_preds)

    preds_binary = (all_preds > thresholds[None, :]).astype(int)

    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(all_preds.shape[1])]
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(all_preds.shape[1])]

    avg_auc = float(np.mean(auc_scores))
    avg_f1 = float(np.mean(f1_scores))

    print(f"\n{desc} 结果:")
    for i, disease in enumerate(disease_list):
        print(
            f"  {disease:20s} AUC: {auc_scores[i]:.4f} | "
            f"F1: {f1_scores[i]:.4f} | Thresh: {thresholds[i]:.3f}"
        )
    print(f"  {'平均':20s} AUC: {avg_auc:.4f} | F1: {avg_f1:.4f}")

    return {
        "loss": running_loss / len(loader),
        "avg_auc": avg_auc,
        "avg_f1": avg_f1,
        "auc_scores": auc_scores,
        "f1_scores": f1_scores,
        "thresholds": thresholds.tolist(),
    }


# ==================== 训练函数 ====================
def train_epoch(
    epoch: int,
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    use_mixup: bool,
    mixup_alpha: float,
) -> float:
    """
    训练一个epoch：
    - 按batch迭代
    - 可选启用Mixup增强
    - 使用类别均衡Focal Loss进行反向传播
    """
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']} [Train]")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if use_mixup:
            inputs, labels = apply_mixup(inputs, labels, alpha=mixup_alpha)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})

    train_loss = running_loss / len(trainloader)
    return train_loss


# ==================== 主训练流程 ====================
def main():
    print("\n加载数据...")
    df = pd.read_csv(CONFIG["csv_file"])
    print(f"总图像数: {len(df)}")
    print(f"CSV列: {list(df.columns)}")

    images_dir = CONFIG["images_dir"]
    existing_images = set(os.listdir(images_dir))
    df = df[df["Image Index"].isin(existing_images)].reset_index(drop=True)
    print(f"存在的图像数: {len(df)}")

    # 从文件名提取患者ID：00000001_000.png -> patient_id=1
    df["Patient ID"] = df["Image Index"].apply(lambda x: int(x.split("_")[0]))

    # 使用与原始脚本一致的患者级划分
    print("\n加载数据集划分...")
    split_dir = "./dataset_splits_xhj"
    if not os.path.exists(split_dir):
        print(f"错误: 未找到 {split_dir}/ 文件夹")
        print("请先运行: python split_dataset.py")
        return

    train_patients = np.load(os.path.join(split_dir, "train_patients.npy"))
    val_patients = np.load(os.path.join(split_dir, "val_patients.npy"))
    test_patients = np.load(os.path.join(split_dir, "test_patients.npy"))
    print(
        f"✓ 数据集划分已加载 "
        f"(train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)} 患者)"
    )

    train_df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test_df = df[df["Patient ID"].isin(test_patients)].reset_index(drop=True)

    print("\n数据集划分:")
    print(f"  训练集: {len(train_df)} 图像, {len(train_patients)} 患者")
    print(f"  验证集: {len(val_df)} 图像, {len(val_patients)} 患者")
    print(f"  测试集: {len(test_df)} 图像, {len(test_patients)} 患者")

    # 统计训练集中每类阳性样本数，用于构建Class-Balanced Loss
    train_class_counts = compute_class_counts(train_df)
    print("\n训练集类别阳性样本统计:")
    for disease, c in zip(disease_list, train_class_counts):
        print(f"  {disease:20s}: {int(c):5d}")

    # 构建Dataset和DataLoader
    train_dataset = ChestXrayDataset(train_df, images_dir, transform=transform_train)
    val_dataset = ChestXrayDataset(val_df, images_dir, transform=transform_eval)
    test_dataset = ChestXrayDataset(test_df, images_dir, transform=transform_eval)

    trainloader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    print("\nDataLoader创建完成")

    # 构建模型与损失函数
    model = build_model(num_classes=len(disease_list), backbone_name=CONFIG["backbone"])

    # 初始阶段冻结backbone，只训练分类头
    freeze_backbone(model, freeze=True)
    criterion = ClassBalancedFocalLoss(
        class_counts=train_class_counts,
        beta=CONFIG["cb_beta"],
        gamma=CONFIG["focal_gamma"],
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.2
    )

    # # 创建checkpoint与结果目录
    checkpoint_dir = "./checkpoints_cb"
    results_dir = "./results_cb"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # WandB（可选）
    if CONFIG["use_wandb"]:
        import wandb
    
        wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
        wandb.watch(model, log="all")
    
    best_val_auc = 0.0
    patience_counter = 0
    train_history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    
    print("\n开始训练...")
    for epoch in range(CONFIG["epochs"]):
        # 在指定epoch之后解冻backbone，进行全网络微调
        if epoch == CONFIG["freeze_backbone_epochs"]:
            print(f"\nEpoch {epoch + 1}: 解冻backbone，开始端到端微调")
            freeze_backbone(model, freeze=False)
            # 解冻后需要让优化器包含全部参数
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CONFIG["learning_rate"],
                weight_decay=1e-5,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=1, factor=0.1
            )
    
        # 训练阶段
        train_loss = train_epoch(
            epoch=epoch,
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=CONFIG["device"],
            use_mixup=CONFIG["use_mixup"],
            mixup_alpha=CONFIG["mixup_alpha"],
        )
    
        # 验证阶段
        val_stats = evaluate(
            model=model,
            loader=valloader,
            criterion=criterion,
            device=CONFIG["device"],
            desc=f"Epoch {epoch + 1} [Val]",
        )
    
        scheduler.step(val_stats["loss"])
    
        train_history["train_loss"].append(train_loss)
        train_history["val_loss"].append(val_stats["loss"])
        train_history["val_auc"].append(val_stats["avg_auc"])
        train_history["val_f1"].append(val_stats["avg_f1"])
    
        print(f"\nEpoch {epoch + 1} 总结:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_stats['loss']:.4f}")
        print(f"  Val AUC:    {val_stats['avg_auc']:.4f}")
        print(f"  Val F1:     {val_stats['avg_f1']:.4f}")
    
        if CONFIG["use_wandb"]:
            import wandb
    
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_stats["loss"],
                    "val_auc": val_stats["avg_auc"],
                    "val_f1": val_stats["avg_f1"],
                }
            )
    
        # 保存最佳模型（按验证集平均AUC）
        if val_stats["avg_auc"] > best_val_auc:
            best_val_auc = val_stats["avg_auc"]
            patience_counter = 0
    
            # 删除旧checkpoint，只保留最新最佳模型
            # for old_file in os.listdir(checkpoint_dir):
            #     if old_file.startswith("best_model_"):
            #         os.remove(os.path.join(checkpoint_dir, old_file))
    
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_epoch{epoch + 1}_{timestamp}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ 保存最佳模型: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\nEarly stopping触发 (patience={CONFIG['patience']})")
                break
    
    print("\n训练完成!")
    
    # ==================== 测试集评估 ====================
    print("\n在测试集上评估最佳模型...")
    best_checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_")
    ]
    if not best_checkpoint_files:
        print("未找到最佳模型checkpoint，无法在测试集上评估。")
        return

    best_checkpoint = sorted(best_checkpoint_files)[-1]
    best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    print(f"加载模型: {best_checkpoint_path}")

    test_stats = evaluate(
        model=model,
        loader=testloader,
        criterion=criterion,
        device=CONFIG["device"],
        desc="Test",
    )

    # 保存结果
    results = {
        "config": CONFIG,
        "test_auc": float(test_stats["avg_auc"]),
        "test_f1": float(test_stats["avg_f1"]),
        "test_loss": float(test_stats["loss"]),
        "per_disease_auc": {
            disease: float(score)
            for disease, score in zip(disease_list, test_stats["auc_scores"])
        },
        "per_disease_f1": {
            disease: float(score)
            for disease, score in zip(disease_list, test_stats["f1_scores"])
        },
        "optimal_thresholds": {
            disease: float(thresh)
            for disease, thresh in zip(disease_list, test_stats["thresholds"])
        },
    }

    results_path = os.path.join(results_dir, "test_results_cb.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")

    # 保存训练历史
    history_path = os.path.join(results_dir, "train_history_cb.npy")
    np.save(history_path, train_history)
    print(f"训练历史已保存到: {history_path}")

    if CONFIG["use_wandb"]:
        import wandb
    
        wandb.finish()
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()
