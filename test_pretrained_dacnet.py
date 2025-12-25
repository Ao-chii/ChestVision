"""
测试预训练DacNet模型（零样本）
加载原DacNet模型，直接在我们的测试集上评估，不进行训练
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from tqdm import tqdm
import json

# ==================== 配置 ====================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_workers": 4,
    "images_dir": "/mnt/data4T-2/xhj/images",
    "csv_file": "/mnt/data4T-2/xhj/filtered_labels.csv",

    # 原DacNet模型路径
    "pretrained_model": "/mnt/data4T-2/xhj/ChestVision/dannynet-55-best_model_20250422-211522.pth",
}

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

print("=" * 60)
print("测试预训练DacNet模型（零样本）")
print("=" * 60)
print(f"设备: {CONFIG['device']}")
print(f"预训练模型: {CONFIG['pretrained_model']}")

# ==================== 数据加载 ====================
print("\n加载数据...")
df = pd.read_csv(CONFIG["csv_file"])
images_dir = CONFIG["images_dir"]
existing_images = set(os.listdir(images_dir))
df = df[df['Image Index'].isin(existing_images)]

# 提取Patient ID
df['Patient ID'] = df['Image Index'].apply(lambda x: int(x.split('_')[0]))

# 加载统一数据集划分
split_dir = "../dataset_splits"
if not os.path.exists(split_dir):
    print(f"错误: 未找到 {split_dir}/ 文件夹")
    print("请先运行: python split_dataset.py")
    exit(1)

test_patients = np.load(os.path.join(split_dir, "test_patients.npy"))
test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)
print(f"测试集: {len(test_df)} 图像, {len(test_patients)} 患者")

# ==================== 数据集类 ====================
def get_label_vector(labels_str):
    labels = labels_str.split('|')
    if labels == ['No Finding']:
        return [0] * len(disease_list)
    else:
        return [1 if disease in labels else 0 for disease in disease_list]

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

# ==================== 数据预处理 ====================
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = ChestXrayDataset(test_df, images_dir, transform=transform_test)
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

# ==================== 加载模型 ====================
print("\n加载预训练模型...")
model = densenet121(weights=None)  # 不使用ImageNet预训练
model.classifier = nn.Linear(model.classifier.in_features, 14)

if not os.path.exists(CONFIG["pretrained_model"]):
    print(f"错误: 未找到预训练模型 {CONFIG['pretrained_model']}")
    print("请确认模型文件路径是否正确")
    exit(1)

model.load_state_dict(torch.load(CONFIG["pretrained_model"], map_location=CONFIG["device"]))
model = model.to(CONFIG["device"])
model.eval()
print(f"✓ 预训练模型加载成功")

# ==================== 评估函数 ====================
def get_optimal_thresholds(labels, preds):
    """为每个疾病计算最优F1阈值"""
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_threshold)
    return thresholds

def evaluate_model(model, loader, device):
    """评估模型"""
    model.eval()
    all_labels, all_preds = [], []

    print("\n在测试集上评估...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # 计算最优阈值
    thresholds = get_optimal_thresholds(all_labels, all_preds)

    # 使用最优阈值计算F1
    preds_binary = np.zeros_like(all_preds)
    for i in range(all_preds.shape[1]):
        preds_binary[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    # 计算指标
    auc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(14)]
    f1_scores = [f1_score(all_labels[:, i], preds_binary[:, i]) for i in range(14)]

    avg_auc = np.mean(auc_scores)
    avg_f1 = np.mean(f1_scores)

    return {
        "avg_auc": avg_auc,
        "avg_f1": avg_f1,
        "auc_scores": auc_scores,
        "f1_scores": f1_scores,
        "thresholds": thresholds
    }

# ==================== 评估 ====================
test_stats = evaluate_model(model, testloader, CONFIG["device"])

print("\n" + "=" * 60)
print("预训练模型测试结果（零样本）")
print("=" * 60)
print(f"平均 AUC: {test_stats['avg_auc']:.4f}")
print(f"平均 F1:  {test_stats['avg_f1']:.4f}")

print("\n各疾病详细结果:")
print(f"{'疾病':<25} {'AUC':<10} {'F1':<10} {'阈值':<10}")
print("-" * 60)
for i, disease in enumerate(disease_list):
    print(f"{disease:<25} {test_stats['auc_scores'][i]:<10.4f} {test_stats['f1_scores'][i]:<10.4f} {test_stats['thresholds'][i]:<10.4f}")

# ==================== 保存结果 ====================
results_dir = "./results_pretrained"
os.makedirs(results_dir, exist_ok=True)

results = {
    "model_type": "Pretrained DacNet (Zero-shot)",
    "pretrained_model_path": CONFIG["pretrained_model"],
    "test_auc": float(test_stats["avg_auc"]),
    "test_f1": float(test_stats["avg_f1"]),
    "per_disease_auc": {disease: float(score) for disease, score in zip(disease_list, test_stats["auc_scores"])},
    "per_disease_f1": {disease: float(score) for disease, score in zip(disease_list, test_stats["f1_scores"])},
    "optimal_thresholds": {disease: float(thresh) for disease, thresh in zip(disease_list, test_stats["thresholds"])},
}

with open(os.path.join(results_dir, "test_results_pretrained.json"), "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n结果已保存到: {results_dir}/test_results_pretrained.json")
print("=" * 60)
