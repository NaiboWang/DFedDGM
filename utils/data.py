import shutil

import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
import torch


# data_dir = os.path.join(os.environ['HOME'],"datasets")
data_dir = "./datasets"

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iCaltech101(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    class_order = np.arange(101).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iSVHN(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.SVHN(data_dir, split="train", download=True)
        test_dataset = datasets.SVHN(data_dir, split="test", download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.labels
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.labels
        )

class iEMNISTLetters(iData): # 26*3700=96k, 5 tasks, each task 5*3700=18.5k
    use_path = True # 问题出在这里！！！
    train_trsf = [
        # transforms.Resize((32, 32)),
        transforms.ToTensor()]
    test_trsf = [
        # transforms.Resize((32, 32)),
        transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    # tr = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    class_order = np.arange(26).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "{}/emnist_output/train/".format(data_dir)
        test_dir = "{}/emnist_output/test/".format(data_dir)
        # print()

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        pass

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):  # 1300*100 = 13w, 5 tasks, each task 20*1300=2.6w
    use_path = True
    train_trsf = [
        # transforms.RandomResizedCrop(224),
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(128),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dir = "{}/imagenet100/train/".format(data_dir)
        test_dir = "{}/imagenet100/val/".format(data_dir)

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class TinyImageNet200(iData):   # 200*500=10w, 5 tasks, each task=40*500=2w
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.CenterCrop(64),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "{}/tiny-imagenet-200/train/".format(data_dir)
        test_dir = "{}/tiny-imagenet-200/val/".format(data_dir)
        # print()

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)


        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

def prepare_and_save_emnist(dataset, root_dir, subset_name):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),  # Resize to 32x32
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channel
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    # 创建子文件夹
    subset_dir = os.path.join(root_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)

    # 迭代处理图像
    for i, (img, label) in enumerate(dataset):
        img = transform(img[0])  # 应用转换
        img = transforms.ToPILImage()(img)
        # 转换成float32
        img = img.convert('RGB')
        label_folder = os.path.join(subset_dir, f'{label - 1}')  # 标签-1作为文件夹名称
        os.makedirs(label_folder, exist_ok=True)
        img.save(os.path.join(label_folder, f'{i}.JPEG'))

def main():
    root = './datasets'
    output_dir = '../datasets/output_emnist'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Training data
    train_dataset = datasets.EMNIST(root="../datasets", split='letters', download=True, train=True, transform=transforms.ToTensor())
    prepare_and_save_emnist(train_dataset, output_dir, 'train')

    # Test data
    test_dataset = datasets.EMNIST(root="../datasets", split='letters', download=True, train=False, transform=transforms.ToTensor())
    prepare_and_save_emnist(test_dataset, output_dir, 'test')  # Tiny ImageNet uses 'val' for validation/test



if __name__ == '__main__':
    main()