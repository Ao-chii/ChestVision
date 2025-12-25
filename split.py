"""
统一数据集划分脚本
运行一次，生成train/val/test患者列表，所有实验共用
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

CONFIG = {
    "csv_file": "../filtered_labels.csv",
    "images_dir": "../images",
    "seed": 42,
    "test_size": 0.15,   # 15% for test
    "val_size": 0.15,    # 15% for validation (from train+val)
}

print("=" * 60)
print("统一数据集划分")
print("=" * 60)

# 加载数据
print("\n加载数据...")
df = pd.read_csv(CONFIG["csv_file"])
print(f"CSV文件: {len(df)} 条记录")

# 过滤存在的图像
images_dir = CONFIG["images_dir"]
existing_images = set(os.listdir(images_dir))
df = df[df['Image Index'].isin(existing_images)]
print(f"存在的图像: {len(df)} 条记录")

# 提取Patient ID
df['Patient ID'] = df['Image Index'].apply(lambda x: int(x.split('_')[0]))
unique_patients = sorted(df['Patient ID'].unique())  # Sort for reproducibility
print(f"唯一患者数: {len(unique_patients)}")

# 划分数据集（患者级别，防止数据泄露）
print("\n按患者划分数据集...")
train_val_patients, test_patients = train_test_split(
    unique_patients, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
)
train_patients, val_patients = train_test_split(
    train_val_patients, test_size=CONFIG["val_size"]/(1-CONFIG["test_size"]),
    random_state=CONFIG["seed"]
)

# 统计信息
train_df = df[df['Patient ID'].isin(train_patients)]
val_df = df[df['Patient ID'].isin(val_patients)]
test_df = df[df['Patient ID'].isin(test_patients)]

print("\n数据集划分结果:")
print(f"  训练集: {len(train_df)} 图像, {len(train_patients)} 患者 ({len(train_patients)/len(unique_patients)*100:.1f}%)")
print(f"  验证集: {len(val_df)} 图像, {len(val_patients)} 患者 ({len(val_patients)/len(unique_patients)*100:.1f}%)")
print(f"  测试集: {len(test_df)} 图像, {len(test_patients)} 患者 ({len(test_patients)/len(unique_patients)*100:.1f}%)")

# 保存患者列表
output_dir = "./dataset_splits"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "train_patients.npy"), train_patients)
np.save(os.path.join(output_dir, "val_patients.npy"), val_patients)
np.save(os.path.join(output_dir, "test_patients.npy"), test_patients)

print(f"\n患者列表已保存到: {output_dir}/")
print(f"  - train_patients.npy")
print(f"  - val_patients.npy")
print(f"  - test_patients.npy")

# 验证
print("\n验证:")
print(f"  训练+验证+测试 = {len(train_patients) + len(val_patients) + len(test_patients)} 患者")
print(f"  总患者数 = {len(unique_patients)} 患者")
print(f"  ✓ 数据集划分完成！" if len(train_patients) + len(val_patients) + len(test_patients) == len(unique_patients) else "  ✗ 错误：患者数不匹配")