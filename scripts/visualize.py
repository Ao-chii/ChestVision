"""
综合可视化脚本 - DacNet vs Baseline
生成：
1. ROC曲线（14个疾病 + 平均）
2. 混淆矩阵（5个代表性疾病）
3. 训练曲线（Loss、AUC、F1）
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
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm import tqdm
import json

# ==================== 配置 ====================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 16,
    "num_workers": 4,
    "images_dir": "../images",
    "csv_file": "../filtered_labels.csv",
    "seed": 42,

    # 模型路径
    "dacnet_checkpoint_dir": "./checkpoints",
    "baseline_checkpoint_dir": "./checkpoints_baseline",

    # 结果路径
    "dacnet_results_dir": "./results",
    "baseline_results_dir": "./results_baseline",

    # 输出路径
    "output_dir": "./visualizations",
}

disease_list = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("=" * 60)
print("Comprehensive Visualization - DacNet vs Baseline")
print("=" * 60)
print(f"Device: {CONFIG['device']}")
print(f"Output directory: {CONFIG['output_dir']}")

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

# ==================== Load test set ====================
print("\nLoading test set...")
df = pd.read_csv(CONFIG["csv_file"])
images_dir = CONFIG["images_dir"]
existing_images = set(os.listdir(images_dir))
df = df[df['Image Index'].isin(existing_images)]

# Extract Patient ID and split dataset (same as training)
df['Patient ID'] = df['Image Index'].apply(lambda x: int(x.split('_')[0]))
unique_patients = sorted(df['Patient ID'].unique())  # Sort for consistency

from sklearn.model_selection import train_test_split
train_val_patients, test_patients = train_test_split(
    unique_patients, test_size=0.15, random_state=CONFIG["seed"]
)
test_df = df[df['Patient ID'].isin(test_patients)].reset_index(drop=True)

print(f"Test set: {len(test_df)} images, {len(test_patients)} patients")

# 测试集transform
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = ChestXrayDataset(test_df, images_dir, transform=transform_test)
testloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                        shuffle=False, num_workers=CONFIG["num_workers"])

# ==================== Load model and predict ====================
def load_model_and_predict(checkpoint_dir, model_name):
    """Load model and predict on test set"""
    print(f"\nLoading {model_name} model...")

    # Load best checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('best_model_')]
    if not checkpoint_files:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
    print(f"Loading: {checkpoint_path}")

    # Build model
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
    model = model.to(CONFIG["device"])
    model.eval()

    # Predict
    all_labels = []
    all_probs = []

    print(f"Predicting on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc=f"{model_name} Prediction"):
            inputs = inputs.to(CONFIG["device"])
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)

    print(f"Prediction complete: {all_labels.shape}")
    return all_labels, all_probs

# Load predictions from both models
labels_true, probs_dacnet = load_model_and_predict(
    CONFIG["dacnet_checkpoint_dir"], "DacNet"
)
_, probs_baseline = load_model_and_predict(
    CONFIG["baseline_checkpoint_dir"], "Baseline"
)

# ==================== 1. ROC Curves ====================
print("\nGenerating ROC curves...")

def plot_roc_curves(labels, probs_dacnet, probs_baseline, disease_list, output_path):
    """Plot ROC curve comparison"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()

    # Plot ROC curve for each disease
    for i, disease in enumerate(disease_list):
        ax = axes[i]

        # DacNet ROC
        fpr_d, tpr_d, _ = roc_curve(labels[:, i], probs_dacnet[:, i])
        auc_d = auc(fpr_d, tpr_d)

        # Baseline ROC
        fpr_b, tpr_b, _ = roc_curve(labels[:, i], probs_baseline[:, i])
        auc_b = auc(fpr_b, tpr_b)

        # Plot
        ax.plot(fpr_d, tpr_d, 'b-', linewidth=2, label=f'DacNet (AUC={auc_d:.3f})')
        ax.plot(fpr_b, tpr_b, 'r--', linewidth=2, label=f'Baseline (AUC={auc_b:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=10)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=10)
        ax.set_title(f'{disease}', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved: {output_path}")
    plt.close()

plot_roc_curves(
    labels_true, probs_dacnet, probs_baseline, disease_list,
    os.path.join(CONFIG["output_dir"], "roc_curves_all_diseases.png")
)

# Plot average ROC curve
print("\nGenerating average ROC curve...")
def plot_average_roc(labels, probs_dacnet, probs_baseline, output_path):
    """Plot average ROC curve (Macro-Average)"""
    from scipy import interpolate

    # Calculate macro-average ROC (same as in training scripts)
    # For each disease, calculate ROC and interpolate to common FPR points
    mean_fpr = np.linspace(0, 1, 100)

    # DacNet macro-average
    tprs_d = []
    aucs_d = []
    for i in range(labels.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:, i], probs_dacnet[:, i])
        tprs_d.append(np.interp(mean_fpr, fpr, tpr))
        tprs_d[-1][0] = 0.0
        aucs_d.append(auc(fpr, tpr))

    mean_tpr_d = np.mean(tprs_d, axis=0)
    mean_tpr_d[-1] = 1.0
    mean_auc_d = np.mean(aucs_d)

    # Baseline macro-average
    tprs_b = []
    aucs_b = []
    for i in range(labels.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:, i], probs_baseline[:, i])
        tprs_b.append(np.interp(mean_fpr, fpr, tpr))
        tprs_b[-1][0] = 0.0
        aucs_b.append(auc(fpr, tpr))

    mean_tpr_b = np.mean(tprs_b, axis=0)
    mean_tpr_b[-1] = 1.0
    mean_auc_b = np.mean(aucs_b)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr_d, 'b-', linewidth=3, label=f'DacNet (AUC={mean_auc_d:.4f})')
    plt.plot(mean_fpr, mean_tpr_b, 'r--', linewidth=3, label=f'Baseline (AUC={mean_auc_b:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Average ROC Curve (Macro-Average)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Average ROC curve saved: {output_path}")
    plt.close()

plot_average_roc(
    labels_true, probs_dacnet, probs_baseline,
    os.path.join(CONFIG["output_dir"], "roc_curve_average.png")
)

# ==================== 2. Confusion Matrices ====================
print("\nGenerating confusion matrices...")

# Load optimal thresholds
with open(os.path.join(CONFIG["dacnet_results_dir"], "test_results.json"), "r") as f:
    dacnet_results = json.load(f)
dacnet_thresholds = [dacnet_results["optimal_thresholds"][d] for d in disease_list]

with open(os.path.join(CONFIG["baseline_results_dir"], "test_results_baseline.json"), "r") as f:
    baseline_results = json.load(f)
baseline_thresholds = [baseline_results["optimal_thresholds"][d] for d in disease_list]

# Select 5 representative diseases
# Based on sample count and AUC
disease_counts = labels_true.sum(axis=0)
disease_aucs_dacnet = [dacnet_results["per_disease_auc"][d] for d in disease_list]

# Select: most samples, least samples, highest AUC, lowest AUC, median AUC
indices = {
    "Most Samples": int(np.argmax(disease_counts)),
    "Least Samples": int(np.argmin(disease_counts[disease_counts > 0])),  # Exclude 0 samples
    "Highest AUC": int(np.argmax(disease_aucs_dacnet)),
    "Lowest AUC": int(np.argmin(disease_aucs_dacnet)),
    "Median AUC": int(np.argsort(disease_aucs_dacnet)[len(disease_aucs_dacnet)//2])
}

selected_diseases = {k: disease_list[v] for k, v in indices.items()}
print(f"\nSelected representative diseases:")
for desc, disease in selected_diseases.items():
    idx = disease_list.index(disease)
    print(f"  {desc}: {disease} (samples={int(disease_counts[idx])}, AUC={disease_aucs_dacnet[idx]:.3f})")

def plot_confusion_matrices(labels, probs_dacnet, probs_baseline,
                            selected_indices, disease_list, thresholds_d, thresholds_b, output_path):
    """Plot confusion matrices for selected diseases"""
    n_diseases = len(selected_indices)
    fig, axes = plt.subplots(n_diseases, 2, figsize=(10, 4*n_diseases))

    if n_diseases == 1:
        axes = axes.reshape(1, -1)

    for row, (desc, idx) in enumerate(selected_indices.items()):
        disease = disease_list[idx]

        # DacNet confusion matrix
        preds_d = (probs_dacnet[:, idx] > thresholds_d[idx]).astype(int)
        cm_d = confusion_matrix(labels[:, idx], preds_d)

        # Baseline confusion matrix
        preds_b = (probs_baseline[:, idx] > thresholds_b[idx]).astype(int)
        cm_b = confusion_matrix(labels[:, idx], preds_b)

        # Plot DacNet
        sns.heatmap(cm_d, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                    ax=axes[row, 0], cbar=True, annot_kws={"size": 14})
        axes[row, 0].set_title(f'{disease} - DacNet\n({desc})', fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel('True Label', fontsize=11)
        axes[row, 0].set_xlabel('Predicted Label', fontsize=11)

        # Plot Baseline
        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                    ax=axes[row, 1], cbar=True, annot_kws={"size": 14})
        axes[row, 1].set_title(f'{disease} - Baseline\n({desc})', fontsize=12, fontweight='bold')
        axes[row, 1].set_ylabel('True Label', fontsize=11)
        axes[row, 1].set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved: {output_path}")
    plt.close()

plot_confusion_matrices(
    labels_true, probs_dacnet, probs_baseline,
    indices, disease_list, dacnet_thresholds, baseline_thresholds,
    os.path.join(CONFIG["output_dir"], "confusion_matrices_representative.png")
)

# ==================== 3. Training Curves ====================
print("\nGenerating training curves...")

# Load training history
history_dacnet = np.load(
    os.path.join(CONFIG["dacnet_results_dir"], "train_history.npy"),
    allow_pickle=True
).item()

history_baseline = np.load(
    os.path.join(CONFIG["baseline_results_dir"], "train_history_baseline.npy"),
    allow_pickle=True
).item()

def plot_training_curves(history_dacnet, history_baseline, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs_d = range(1, len(history_dacnet["train_loss"]) + 1)
    epochs_b = range(1, len(history_baseline["train_loss"]) + 1)

    # Loss curve
    axes[0].plot(epochs_d, history_dacnet["train_loss"], 'b-', linewidth=2, label='DacNet Train')
    axes[0].plot(epochs_d, history_dacnet["val_loss"], 'b--', linewidth=2, label='DacNet Val')
    axes[0].plot(epochs_b, history_baseline["train_loss"], 'r-', linewidth=2, label='Baseline Train')
    axes[0].plot(epochs_b, history_baseline["val_loss"], 'r--', linewidth=2, label='Baseline Val')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # AUC curve
    axes[1].plot(epochs_d, history_dacnet["val_auc"], 'b-', linewidth=2, label='DacNet')
    axes[1].plot(epochs_b, history_baseline["val_auc"], 'r-', linewidth=2, label='Baseline')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('Validation AUC Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # F1 curve
    axes[2].plot(epochs_d, history_dacnet["val_f1"], 'b-', linewidth=2, label='DacNet')
    axes[2].plot(epochs_b, history_baseline["val_f1"], 'r-', linewidth=2, label='Baseline')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Validation F1 Curves', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved: {output_path}")
    plt.close()

plot_training_curves(history_dacnet, history_baseline, CONFIG["output_dir"])

# ==================== Generate comparison table ====================
print("\nGenerating model comparison table...")

comparison_table = []
for disease in disease_list:
    comparison_table.append({
        "Disease": disease,
        "DacNet_AUC": dacnet_results["per_disease_auc"][disease],
        "Baseline_AUC": baseline_results["per_disease_auc"][disease],
        "DacNet_F1": dacnet_results["per_disease_f1"][disease],
        "Baseline_F1": baseline_results["per_disease_f1"][disease],
    })

# Add average
comparison_table.append({
    "Disease": "Average",
    "DacNet_AUC": dacnet_results["test_auc"],
    "Baseline_AUC": baseline_results["test_auc"],
    "DacNet_F1": dacnet_results["test_f1"],
    "Baseline_F1": baseline_results["test_f1"],
})

df_comparison = pd.DataFrame(comparison_table)

# Save as CSV
csv_path = os.path.join(CONFIG["output_dir"], "model_comparison.csv")
df_comparison.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Comparison table saved: {csv_path}")

# Print to console
print("\n" + "=" * 80)
print("Model Comparison Table")
print("=" * 80)
print(df_comparison.to_string(index=False))

print("\n" + "=" * 60)
print("Visualization complete! Generated files:")
print("  1. roc_curves_all_diseases.png - ROC curves for 14 diseases")
print("  2. roc_curve_average.png - Average ROC curve")
print("  3. confusion_matrices_representative.png - Confusion matrices for 5 diseases")
print("  4. training_curves.png - Training process curves")
print("  5. model_comparison.csv - Model comparison table")
print("=" * 60)
