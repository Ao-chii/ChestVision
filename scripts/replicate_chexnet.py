"""
CheXNet Baseline训练脚本
与DacNet的关键区别：
1. 标准BCE Loss（不用Focal Loss）
2. 更高的学习率 (0.001 vs 0.00005)
3. 更简单的数据增强（去掉ColorJitter）
"""
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import numpy as np
from torchvision.models import densenet121, DenseNet121_Weights
import time

# ==================== 配置 ====================
CONFIG = {
    "model_name": "CheXNet_Baseline",
    "batch_size": 16,
    "learning_rate": 0.001,  # 比DacNet高20倍
    "epochs": 25,  # 稍微少一点
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # 数据路径
    "data_dir": ".",
    "images_dir": "/mnt/data4T-2/xhj/images",
    "csv_file": "/mnt/data4T-2/xhj/filtered_labels.csv",

    # 训练配置
    "patience": 5,
    "seed": 42,
    "image_size": 224,

    # WandB配置（可选）
    "use_wandb": False,
    "wandb_project": "ChestXray-Baseline",
}

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

print("=" * 60)
print("CheXNet Baseline 配置")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"{key:20s}: {value}")
print(f"{'diseases':20s}: {len(disease_list)} classes")
print("=" * 60)

# ==================== 数据增强 ====================
# 训练集：只用RandomHorizontalFlip，不用ColorJitter
transform_train = transforms.Compose([
    transforms.Resize(224),  # 直接resize，不用RandomResizedCrop
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("\n数据增强策略:")
print("  训练集: Resize(224) + RandomHorizontalFlip (更简单)")
print("  测试集: Resize(224)")
print("  对比DacNet: 少了RandomResizedCrop和ColorJitter")

# ==================== 模型构建 ====================
print("\n构建模型...")
model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 14)
model = model.to(CONFIG["device"])
print(f"模型: DenseNet-121 (ImageNet预训练)")
print(f"设备: {CONFIG['device']}")

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 标准BCE，不是Focal Loss
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)

print(f"\n损失函数: BCEWithLogitsLoss (标准BCE)")
print(f"优化器: Adam (lr={CONFIG['learning_rate']})")
print(f"对比DacNet: 用BCE替代Focal Loss, 学习率高20倍")

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

# ==================== Dataset类 ====================
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform=None):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)
        return image, labels

# ==================== DataLoader ====================
train_dataset = ChestXrayDataset(train_df, images_dir, transform=transform_train)
val_dataset = ChestXrayDataset(val_df, images_dir, transform=transform_test)
test_dataset = ChestXrayDataset(test_df, images_dir, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                         shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                       shuffle=False, num_workers=CONFIG["num_workers"])
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

print(f"\nDataLoader创建完成")

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
            outputs = model(inputs)
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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (i + 1)})

    train_loss = running_loss / len(trainloader)
    return train_loss

# ==================== 主训练循环 ====================
print("\n开始训练 CheXNet Baseline...")

checkpoint_dir = "./checkpoints_baseline"
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
results_dir = "./results_baseline"
os.makedirs(results_dir, exist_ok=True)

results = {
    "model": "CheXNet_Baseline",
    "test_auc": float(test_stats["avg_auc"]),
    "test_f1": float(test_stats["avg_f1"]),
    "test_loss": float(test_stats["loss"]),
    "per_disease_auc": {disease: float(score) for disease, score in zip(disease_list, test_stats["auc_scores"])},
    "per_disease_f1": {disease: float(score) for disease, score in zip(disease_list, test_stats["f1_scores"])},
    "optimal_thresholds": {disease: float(thresh) for disease, thresh in zip(disease_list, test_stats["thresholds"])},
}

import json
with open(os.path.join(results_dir, "test_results_baseline.json"), "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n结果已保存到: {results_dir}/test_results_baseline.json")

np.save(os.path.join(results_dir, "train_history_baseline.npy"), train_history)
print(f"训练历史已保存到: {results_dir}/train_history_baseline.npy")

if CONFIG["use_wandb"]:
    wandb.finish()

print("\n全部完成!")
