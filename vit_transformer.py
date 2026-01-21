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
import wandb
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor
import time

# Configuration settings
CONFIG = {
    "model": "vit_transformer",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "epochs": 20,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 数据路径（与dacnet.py保持一致）
    "data_dir": ".",
    "images_dir": "/mnt/data4T-2/xhj/images",
    "csv_file": "/mnt/data4T-2/xhj/filtered_labels.csv",
    
    "wandb_project": "ChestXray-ViT",
    "patience": 5,
    "seed": 42,
    "image_size": 224,
}

print("=" * 60)
print("ViT Transformer 训练配置")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"{key:20s}: {value}")
print("=" * 60)

# Define the model name and load feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Define transform functions
def transform_train(img):
    return feature_extractor(images=img, return_tensors='pt')['pixel_values'][0]

def transform_test(img):
    return feature_extractor(images=img, return_tensors='pt')['pixel_values'][0]

# ==================== 数据加载 ====================
print("\n加载数据...")
df = pd.read_csv(CONFIG["csv_file"])
print(f"总图像数: {len(df)}")
print(f"CSV列: {list(df.columns)}")

# 检查图像文件是否存在
images_dir = CONFIG["images_dir"]
existing_images = set(os.listdir(images_dir))
df = df[df['Image Index'].isin(existing_images)]
print(f"存在的图像数: {len(df)}")

# 从文件名提取Patient ID (00000001_000.png -> patient_id=1)
df['Patient ID'] = df['Image Index'].apply(lambda x: int(x.split('_')[0]))

# 加载预先划分好的数据集（保证所有实验一致）
print("\n加载数据集划分...")
split_dir = "./dataset_splits"
if not os.path.exists(split_dir):
    print(f"错误: 未找到 {split_dir}/ 文件夹")
    print("请先运行: python split_dataset.py")
    exit(1)

train_patients = np.load(os.path.join(split_dir, "train_patients.npy"))
val_patients = np.load(os.path.join(split_dir, "val_patients.npy"))
test_patients = np.load(os.path.join(split_dir, "test_patients.npy"))
print(f"✓ 数据集划分已加载 (train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)} 患者)")

train_df = df[df['Patient ID'].isin(train_patients)].reset_index(drop=True)
val_df = df[df['Patient ID'].isin(val_patients)].reset_index(drop=True)
test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

print(f"\n数据集划分:")
print(f"  训练集: {len(train_df)} 图像, {len(train_patients)} 患者")
print(f"  验证集: {len(val_df)} 图像, {len(val_patients)} 患者")
print(f"  测试集: {len(test_df)} 图像, {len(test_patients)} 患者")

# List of diseases we’re classifying
disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Function to convert label string to a vector
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
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 标签编码
        labels_str = self.dataframe.iloc[idx]['Finding Labels']
        label_vector = get_label_vector(labels_str)
        labels = torch.tensor(label_vector, dtype=torch.float)
        
        return image, labels

# ==================== DataLoader ====================
train_dataset = ChestXrayDataset(train_df, images_dir, transform=transform_train)
val_dataset = ChestXrayDataset(val_df, images_dir, transform=transform_test)
test_dataset = ChestXrayDataset(test_df, images_dir, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

print(f"\nDataLoader创建完成")

# Load the pre-trained model
print("\n构建ViT模型...")
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=14,
    ignore_mismatched_sizes=True
)
model = model.to(CONFIG["device"])
print(f"✓ ViT模型加载完成，使用设备: {CONFIG['device']}")

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

print(f"\n损失函数: BCEWithLogitsLoss")
print(f"优化器: Adam (lr={CONFIG['learning_rate']})")
print(f"学习率调度: ReduceLROnPlateau")

# Evaluation function
def evaluate(model, testloader, criterion, device, desc="[Test]"):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc=desc, leave=True)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    test_loss = running_loss / len(testloader)

    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    avg_auc = np.mean(auc_scores)
    for i, disease in enumerate(disease_list):
        print(f"{desc} {disease} AUC-ROC: {auc_scores[i]:.4f}")
    auc_dict = {disease_list[i]: auc_scores[i] for i in range(14)}

    preds_binary = (all_preds > 0.5).astype(int)
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]
    avg_f1 = np.mean(f1_scores)
    for i, disease in enumerate(disease_list):
        print(f"{desc} {disease} F1 Score: {f1_scores[i]:.4f}")
    f1_dict = {disease_list[i]: f1_scores[i] for i in range(14)}
    print(f"{desc} Loss: {test_loss:.4f}, Avg AUC-ROC: {avg_auc:.4f}, Avg F1 Score: {avg_f1:.4f}")
    return test_loss, avg_auc, avg_f1, auc_dict, f1_dict

# Training function
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=True)
    # Ensure progress_bar is closed properly
    try:
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / (i + 1)})
    finally:
        progress_bar.close()
    train_loss = running_loss / len(trainloader)
    return train_loss

def validate(model, valloader, criterion, device):
    val_loss, val_auc, val_f1, auc_dict, f1_dict = evaluate(model, valloader, criterion, device, desc="[Validate]")
    return val_loss, val_auc, val_f1, auc_dict, f1_dict

# Training loop with WandB and timestamped checkpoints
try:
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)
except Exception as e:
    print(f"WandB initialization failed: {e}. Continuing without WandB.")
    wandb.init(mode="disabled")

run_id = wandb.run.id
checkpoint_dir = os.path.join("models", run_id)
os.makedirs(checkpoint_dir, exist_ok=True)

best_val_auc = 0.0
patience_counter = 0

# 记录训练历史
train_history = {
    "train_loss": [],
    "val_loss": [],
    "val_auc": [],
    "val_f1": []
}

print("\n" + "="*60)
print("开始训练...")
print("="*60)

for epoch in range(CONFIG["epochs"]):
    train_loss = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
    val_loss, val_auc, val_f1, auc_dict, f1_dict = validate(model, valloader, criterion, CONFIG["device"])
    scheduler.step(val_loss)

    # 记录到历史
    train_history["train_loss"].append(train_loss)
    train_history["val_loss"].append(val_loss)
    train_history["val_auc"].append(val_auc)
    train_history["val_f1"].append(val_f1)

    # 打印 epoch 总结
    print(f"\nEpoch {epoch+1} 总结:")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Val F1: {val_f1:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "f1_dict": f1_dict,
        "auc_dict": auc_dict,
    })

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
        print(f"  ✓ 新最佳模型保存: Val AUC = {val_auc:.4f}")
    else:
        patience_counter += 1
        print(f"  ⚠ 验证集AUC未提升 (patience: {patience_counter}/{CONFIG['patience']})")
        if patience_counter >= CONFIG["patience"]:
            print("\n早停触发，训练结束。")
            break

# Evaluate the best model
print("\n" + "="*60)
print("测试最佳模型...")
print("="*60)
checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('best_model_')]
if not checkpoint_files:
    raise FileNotFoundError("No checkpoint files found. Training may not have saved any models.")
best_checkpoint_path = sorted(checkpoint_files)[-1]
print(f"加载最佳模型: {best_checkpoint_path}")
model.load_state_dict(torch.load(best_checkpoint_path))
test_loss, test_auc, test_f1, auc_dict, f1_dict = evaluate(model, testloader, criterion, CONFIG["device"], desc="[Test]")

# 上传到 WandB
wandb.log({
    "test_loss": test_loss,
    "test_auc": test_auc,
    "test_f1": test_f1,
    "test_auc_dict": auc_dict,
    "test_f1_dict": f1_dict
})

# ==================== 保存结果到本地 ====================
print("\n" + "="*60)
print("保存结果到本地...")
print("="*60)

results_dir = "./results_vit"
os.makedirs(results_dir, exist_ok=True)

# 保存测试结果（与 dacnet.py 格式一致）
import json
results = {
    "model": "ViT_Transformer",
    "test_auc": float(test_auc),
    "test_f1": float(test_f1),
    "test_loss": float(test_loss),
    "per_disease_auc": {disease: float(auc_dict[disease]) for disease in disease_list},
    "per_disease_f1": {disease: float(f1_dict[disease]) for disease in disease_list},
}

result_path = os.path.join(results_dir, "test_results_vit.json")
with open(result_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f"✓ 测试结果已保存: {result_path}")

# 保存训练历史
history_path = os.path.join(results_dir, "train_history_vit.npy")
np.save(history_path, train_history)
print(f"✓ 训练历史已保存: {history_path}")

# 保存为易读的 JSON 格式
history_json_path = os.path.join(results_dir, "train_history_vit.json")
with open(history_json_path, "w") as f:
    json.dump(train_history, f, indent=4)
print(f"✓ 训练历史(JSON)已保存: {history_json_path}")

print(f"✓ 最佳模型路径: {best_checkpoint_path}")
print(f"✓ 所有结果保存在: {results_dir}/")

wandb.finish()
print("\n训练完成！")