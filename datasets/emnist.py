import os
import shutil

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
def load_emnist(train=True):
    """加载EMNIST数据集，根据参数选择训练集还是测试集"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 缩放到32x32
        transforms.ToTensor()
    ])

    # Download and load EMNIST dataset
    dataset = datasets.EMNIST(root='./', split='letters', train=train, download=True, transform=transform)
    return dataset

def save_images(dataset, dataset_type):
    """保存图像到本地，格式仿照Tiny ImageNet，并区分训练集或测试集"""
    base_dir = 'emnist_output'
    dataset_dir = 'train' if dataset_type == 'train' else 'test'
    if os.path.exists(base_dir+"/"+dataset_dir):
        shutil.rmtree(base_dir+"/"+dataset_dir)
    os.makedirs(os.path.join(base_dir, dataset_dir), exist_ok=True)
    # 创建类别和图像文件夹
    for idx in range(0, 26):  # EMNIST Letters 包含26个类别（A-Z）
        folder_name = f"{idx}"
        class_dir = os.path.join(base_dir, dataset_dir, folder_name)
        os.makedirs(class_dir, exist_ok=True)

    # 保存图像
    for idx, (image, label) in enumerate(dataset):
        folder_name = f"{label-1}"  # Label转换为文件夹名称
        class_dir = os.path.join(base_dir, dataset_dir, folder_name)

        # 转换成三通道RGB图像
        # image = image.squeeze()  # 移除额外的维度
        # image_rgb = Image.fromarray((image.numpy() * 255).astype(np.uint8)).convert("RGB")
        # 移除多余的通道维度，转换为numpy array，并乘255转为uint8
        image_np = image.squeeze(0).numpy() * 255
        image_np = image_np.astype(np.uint8)
        image_pil = Image.fromarray(image_np).convert('RGB')  # 将单通道转为RGB

        # 保存图片
        image_filename = f"{idx}.JPEG"
        image_pil.save(os.path.join(class_dir, image_filename))

if __name__ == "__main__":
    # 加载和保存训练集
    emnist_train = load_emnist(train=True)
    save_images(emnist_train, 'train')

    # 加载和保存测试集
    emnist_test = load_emnist(train=False)
    save_images(emnist_test, 'val')