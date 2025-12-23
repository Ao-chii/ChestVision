"""
Vision Transformer (ViT) 训练脚本
与CNN模型的关键区别：
1. 使用Transformer架构（自注意力机制）
2. 需要transformers库（Hugging Face）
3. 使用ViT专用的图像预处理
4. 对数据量更敏感（Transformer需要大量数据）
"""
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
CONFIG = {
    "model_name": "ViT_Transformer",
    "batch_size": 16,
    "learning_rate": 0.0001,  # ViT通常用1e-4
    "epochs": 20,  # ViT收敛较慢
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 数据路径
    "data_dir": ".",
    "images_dir": "./images",
    "csv_file": "./filtered_labels.csv",

    # 训练配置
    "patience": 5,
    "seed": 42,
    "image_size": 224,

    # ViT模型配置
    "vit_model": "google/vit-base-patch16-224",  # ViT-Base预训练模型

    # WandB配置（可选）
    "use_wandb": False,
    "wandb_project": "ChestXray-ViT",
}

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

print("=" * 60)
print("Vision Transformer 配置")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"{key:20s}: {value}")
print(f"{'diseases':20s}: {len(disease_list)} classes")
print("=" * 60)
print("\n注意：ViT需要transformers库，首次运行会自动下载预训练模型")

# ==================== 加载ViT预处理器和模型 ====================
print("\n加载ViT模型和预处理器...")
processor = ViTImageProcessor.from_pretrained(CONFIG["vit_model"])

# 修改ViT模型的分类头
model = ViTForImageClassification.from_pretrained(
    CONFIG["vit_model"],
    num_labels=14,  # 14类疾病
    ignore_mismatched_sizes=True  # 忽略分类头大小不匹配的警告
)
model = model.to(CONFIG["device"])

print(f"模型: {CONFIG['vit_model']}")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"设备: {CONFIG['device']}")

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

print(f"\n损失函数: BCEWithLogitsLoss")
print(f"优化器: Adam (lr={CONFIG['learning_rate']})")
print(f"调度器: ReduceLROnPlateau (patience=3)")

# ==================== 数据加载 ====================
print("\n加载数据...")
df = pd.read_csv(CONFIG["csv_file"])
print(f"总图像数: {len(df)}")

# 检查图像文件
images_dir = CONFIG["images_dir"]
existing_images = set(os.listdir(images_dir))
df = df[df['Image Index'].isin(existing_images)]
print(f"存在的图像数: {len(df)}")

# 从文件名提取Patient ID
df['Patient ID'] = df['Image Index'].apply(lambda x: int(x.split('_')[0]))
unique_patients = sorted(df['Patient ID'].unique())
print(f"唯一患者数: {len(unique_patients)}")

# 按患者划分数据集
train_val_patients, test_patients = train_test_split(
    unique_patients, test_size=0.15, random_state=CONFIG["seed"]
)
train_patients, val_patients = train_test_split(
    train_val_patients, test_size=0.15/0.85, random_state=CONFIG["seed"]
)

train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

print(f"\n数据集划分:")
print(f"  训练集: {len(train_df)} 图像, {len(train_patients)} 患者")
print(f"  验证集: {len(val_df)} 图像, {len(val_patients)} 患者")
print(f"  测试集: {len(test_df)} 图像, {len(test_patients)} 患者")

# ==================== 标签编码 ====================
def get_label_vector(labels_str):
    labels = labels_str.split('|')
    if labels == ['No Finding']:
        return [0] * len(disease_list)
    else:
        return [1 if disease in labels else 0 for disease in disease_list]

# ==================== Dataset类（ViT专用） ====================
class ViTChestXrayDataset(Dataset):
    """使用ViT预处理器的Dataset"""
    def __init__(self, dataframe, images_dir, processor, is_train=False):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.processor = processor
        self.is_train = is_train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        img_path = os.path.join(self.images_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 使用ViT预处理器（自动resize、normalize等）
        # return_tensors='pt' 返回PyTorch tensor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # 去掉batch维度

        # 标签编码
        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)

        return pixel_values, labels

# ==================== DataLoader ====================
print("\n创建ViT专用DataLoader...")
train_dataset = ViTChestXrayDataset(train_df, images_dir, processor, is_train=True)
val_dataset = ViTChestXrayDataset(val_df, images_dir, processor, is_train=False)
test_dataset = ViTChestXrayDataset(test_df, images_dir, processor, is_train=False)

trainloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                         shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                       shuffle=False, num_workers=CONFIG["num_workers"])
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

print(f"DataLoader创建完成")
print(f"注意：ViT使用专用预处理器，不需要手动定义transform")

# ==================== 自适应阈值计算 ====================
def get_optimal_thresholds(labels, preds):
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_threshold)
    return thresholds

# ==================== 评估函数 ====================
def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device)

            # ViT的输出是一个对象，需要取logits
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    thresholds = get_optimal_thresholds(all_labels, all_preds)

    preds_binary = np.zeros_like(all_preds)
    for i in range(all_preds.shape[1]):
        preds_binary[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]

    avg_auc = np.mean(auc_scores)
    avg_f1 = np.mean(f1_scores)

    print(f"\n{desc} 结果:")
    for i, disease in enumerate(disease_list):
        print(f"  {disease:20s} AUC: {auc_scores[i]:.4f} | F1: {f1_scores[i]:.4f} | Thresh: {thresholds[i]:.3f}")
    print(f"  {'平均':20s} AUC: {avg_auc:.4f} | F1: {avg_f1:.4f}")

    return {
        "loss": running_loss / len(loader),
        "avg_auc": avg_auc,
        "avg_f1": avg_f1,
        "auc_scores": auc_scores,
        "f1_scores": f1_scores,
        "thresholds": thresholds
    }

# ==================== 训练函数 ====================
def train_epoch(epoch, model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).logits  # ViT输出需要取logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})

    train_loss = running_loss / len(trainloader)
    return train_loss

# ==================== 主训练循环 ====================
print("\n开始训练 Vision Transformer...")
print("注意：ViT计算量较大，训练速度可能比DenseNet慢")

checkpoint_dir = "./checkpoints_vit"
os.makedirs(checkpoint_dir, exist_ok=True)

if CONFIG["use_wandb"]:
    import wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model, log="all")

best_val_auc = 0.0
patience_counter = 0
train_history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

for epoch in range(CONFIG["epochs"]):
    train_loss = train_epoch(epoch, model, trainloader, optimizer, criterion, CONFIG["device"])
    val_stats = evaluate(model, valloader, criterion, CONFIG["device"], desc=f"Epoch {epoch+1} [Val]")
    scheduler.step(val_stats["loss"])

    train_history["train_loss"].append(train_loss)
    train_history["val_loss"].append(val_stats["loss"])
    train_history["val_auc"].append(val_stats["avg_auc"])
    train_history["val_f1"].append(val_stats["avg_f1"])

    print(f"\nEpoch {epoch+1} 总结:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_stats['loss']:.4f}")
    print(f"  Val AUC: {val_stats['avg_auc']:.4f}")
    print(f"  Val F1: {val_stats['avg_f1']:.4f}")

    if CONFIG["use_wandb"]:
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_auc": val_stats["avg_auc"],
            "val_f1": val_stats["avg_f1"],
        })

    if val_stats["avg_auc"] > best_val_auc:
        best_val_auc = val_stats["avg_auc"]
        patience_counter = 0

        # 删除旧的checkpoint（只保留最新最佳）
        for old_file in os.listdir(checkpoint_dir):
            if old_file.startswith('best_model_'):
                os.remove(os.path.join(checkpoint_dir, old_file))

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}_{timestamp}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✓ 保存最佳模型: {checkpoint_path} (已删除旧模型)")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"\nEarly stopping触发 (patience={CONFIG['patience']})")
            break

print("\n训练完成!")

# ==================== 测试集评估 ====================
print("\n在测试集上评估最佳模型...")
best_checkpoint = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_')])[-1]
best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
model.load_state_dict(torch.load(best_checkpoint_path))
print(f"加载模型: {best_checkpoint_path}")

test_stats = evaluate(model, testloader, criterion, CONFIG["device"], desc="Test")

# 保存结果
results_dir = "./results_vit"
os.makedirs(results_dir, exist_ok=True)

results = {
    "model": "ViT_Transformer",
    "vit_version": CONFIG["vit_model"],
    "test_auc": float(test_stats["avg_auc"]),
    "test_f1": float(test_stats["avg_f1"]),
    "test_loss": float(test_stats["loss"]),
    "per_disease_auc": {disease: float(score) for disease, score in zip(disease_list, test_stats["auc_scores"])},
    "per_disease_f1": {disease: float(score) for disease, score in zip(disease_list, test_stats["f1_scores"])},
    "optimal_thresholds": {disease: float(thresh) for disease, thresh in zip(disease_list, test_stats["thresholds"])},
}

import json
with open(os.path.join(results_dir, "test_results_vit.json"), "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n结果已保存到: {results_dir}/test_results_vit.json")

np.save(os.path.join(results_dir, "train_history_vit.npy"), train_history)
print(f"训练历史已保存到: {results_dir}/train_history_vit.npy")

if CONFIG["use_wandb"]:
    wandb.finish()

print("\n全部完成!")
print("\n" + "=" * 60)
print("Vision Transformer vs CNN 特点对比:")
print("  ViT优势: 全局感受野（自注意力机制）")
print("  ViT劣势: 需要更多数据，计算量大")
print("  预期: 在数据量较少时，CNN (DacNet) 可能表现更好")
print("=" * 60)
