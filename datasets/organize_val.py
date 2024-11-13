import os
import shutil

def organize_validation_set(data_dir, val_annotations_file):
    val_dir = os.path.join(data_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, val_annotations_file)

    # 创建和训练集相同结构的验证集目录
    dest_dir = os.path.join(val_dir, "images_per_class")
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # 读取验证集标签文件
    with open(annotations_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        image, class_id = line.strip().split('\t')[:2]
        class_dir = os.path.join(dest_dir, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        # 移动图像到相应类别文件夹
        src_image_path = os.path.join(images_dir, image)
        dst_image_path = os.path.join(class_dir, image)
        shutil.move(src_image_path, dst_image_path)

    print("Validation data organized!")

if __name__ == "__main__":
    data_dir = './tiny-imagenet-200' # Update this path
    val_annotations_file = 'val_annotations.txt'
    organize_validation_set(data_dir, val_annotations_file)