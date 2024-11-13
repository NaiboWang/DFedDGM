import os
import pickle

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from models.df_model import Diffusion
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, _get_idata, DummyDataset
import copy, wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from torch.utils.data import Sampler
from PIL import Image
from torch.utils.data import Subset
from utils.test_set import get_test_set, train_oracle_model

EPSILON = 1e-8
T = 2

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, known_classes, total_classes, n_samples):
        self.labels = np.array(labels)
        self.n_classes = np.arange(known_classes, total_classes)
        self.n_samples = n_samples
        self.batch_size = 0
        max_samples = 0
        self.class_indices = []
        for i in self.n_classes:
            indice = np.where(self.labels == i)[0]
            np.random.shuffle(indice)
            self.class_indices.append(indice)
            if len(indice) != 0:
                self.batch_size += n_samples
            max_samples = max(max_samples, len(indice))

        self.used_indices = [0] * self.n_classes
        self.length = max_samples // n_samples
        if max_samples % n_samples != 0:
            self.length += 1

    def __iter__(self):
        self.count = 0
        while self.count < self.length:
            batch_indices = []
            for i in range(len(self.n_classes)):
                start_pos = self.used_indices[i]
                stop_pos = start_pos + self.n_samples

                if stop_pos > len(self.class_indices[i]):
                    np.random.shuffle(self.class_indices[i])
                    start_pos = 0
                    self.used_indices[i] = 0
                    stop_pos = self.n_samples

                self.used_indices[i] += self.n_samples
                batch_indices.extend(self.class_indices[i][start_pos:stop_pos])

            yield batch_indices
            self.count += 1

    def __len__(self):
        return self.length

def show_images(images):
    import matplotlib.pyplot as plt
    images = images[:10]
    # 假设你有一个包含20个PIL图像的列表pil_images
    # 设置整个图表的大小
    plt.figure(figsize=(10, 8))  # 10和8分别代表图表的宽和高，在这里可以根据需要调整

    # 通过循环遍历pil_images列表来展示每个图像
    for i, image in enumerate(images):
        plt.subplot(5, 2, i + 1)  # 创建子图。注意，matplotlib的index从1开始
        plt.imshow(image)  # 显示PIL图像
        plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 自动调整子图参数，以给定的填充
    plt.show()  # 显示整个图表

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, model, transform=None, keep=1.0, test_L=None, oracle_test_L=None, data_manager=None, oracle_model=None):
        self.images = []
        self.model = copy.deepcopy(model)

        # self.model.load_state_dict(oracle_model.state_dict())

        # Convert 300, 3, 32, 32 to 300, 32, 32, 3
        for i in range(tensors.shape[0]):  # 遍历批次中的每个图像
            img = tensors[i]  # 获取当前图像张量

            # 数值范围调整至0到255，并转换为uint8
            img = torch.clamp(img, 0, 255)  # 限制数值范围到0-255
            img = img / img.max() * 255.0  # 归一化（可选步骤，视具体情况而定）
            img_uint8 = img.to(torch.uint8)  # 转换为uint8

            # 调整通道顺序: [C, H, W] -> [H, W, C]
            img_uint8 = img_uint8.permute(1, 2, 0)

            # 将图像添加到列表中
            self.images.append(img_uint8.cpu().numpy())

        # 将images转换为numpy数组
        self.images = np.array(self.images)

        # 如果存在transform，则应用transform
        if transform:
            try:
                images = np.array([transform(image) for image in self.images])
            except:
                # Tiny Imagenet center crop不能用
                # images = np.asarray([transforms.Compose(transform.transforms[1:])(image) for image in self.images])
                # numpy to PIL
                images = [Image.fromarray(image) for image in self.images]
                images = np.array([transform(image) for image in images])
        # 预处理后，确保输入到模型的是Tensor
        images = torch.tensor(images, dtype=torch.float32).cuda()

        # # 调整图像张量的形状以匹配模型的输入期望
        # if len(images.shape) == 4 and images.shape[1] == 3:  # 如果是[batch_size, C, H, W]
        #     self.images = images.permute(0, 2, 3, 1)  # 转换为[batch_size, H, W, C]如果模型需要

        # 使用模型进行预测
        self.model.eval()  # 设置模型为评估模式
        np.set_printoptions(linewidth=np.inf)
        with torch.no_grad():  # 关闭梯度计算
            logits = self.model(images)["logits"]
            self.labels = logits.argmax(dim=1).cpu().numpy()
            # self.original_logits = logits.cpu().numpy()
            softmax_logits = F.softmax(logits, dim=1).cpu().numpy()  # 获取softmax概率
            # 计算每个样本的概率的熵
            # self.original_entropy = -np.sum(self.original_logits * np.log(self.original_logits), axis=1)
            self.entropy = -np.sum(softmax_logits * np.log(softmax_logits), axis=1)
            # 根据熵值对样本进行排序
            self.sorted_indices = np.argsort(self.entropy)
            # 更换图像和标签的顺序
            self.images_copy = self.images[self.sorted_indices]
            self.labels_copy = self.labels[self.sorted_indices]
            self.entropy_copy = self.entropy[self.sorted_indices]
            logits_copy = softmax_logits[self.sorted_indices]
            print("Entropy: ", self.entropy_copy)
            # 按logits和entropy打印
            # for i in range(len(self.entropy_copy)):
            #     print(self.sorted_indices[i], np.sort(logits_copy[i])[::-1][:10], self.entropy_copy[i], np.argsort(logits_copy[i])[::-1][0],  self.labels_copy[i], sep=', ')




            '''
            标准差和熵的结果基本相同，因此只保留熵的代码
            '''
            # 计算softmax_logits的标准差
            # self.std = np.std(softmax_logits, axis=1)
            # print("Std: ", self.std)
            # # 根据标准差对样本进行排序
            # self.sorted_indices_std = np.argsort(self.std)
            # print("Entropy sorted: ", self.sorted_indices)
            # print("Std sorted: ", self.sorted_indices_std)
            # self.images_copy2 = self.images[self.sorted_indices_std]
            # self.labels_copy2 = self.labels[self.sorted_indices_std]
            # self.std_copy = self.std[self.sorted_indices_std]
            # # 按logits和std打印
            # for i in range(len(self.std)):
            #     print(self.sorted_indices_std[i], self.std_copy[i], self.labels_copy2[i], sep=', ')



        # 保留一定比例的数据
        keep_ratio = str(keep)
        if keep >= 0:
            keep = int(keep * len(self.images_copy))
            # 从0.4开始保留数据
            keep_start = int(0 * len(self.images_copy))
            self.images = self.images_copy[:keep]
            self.labels = self.labels_copy[:keep]
            # self.images = self.images_copy[keep_start:keep]
            # self.labels = self.labels_copy[keep_start:keep]
        else:
            keep = int(-keep * len(self.images_copy))
            self.images = self.images_copy[keep:]
            self.labels = self.labels_copy[keep:]
        # with torch.no_grad():  # 关闭梯度计算
        #     # 重新计算images_copy和labels_copy
        #     if transform:
        #         images = np.array([transform(image) for image in self.images])
        #     images = torch.tensor(images, dtype=torch.float32).cuda()
        #     logits = self.model(images)["logits"]
        #     labels = logits.argmax(dim=1).cpu().numpy()
        #     softmax_logits = F.softmax(logits, dim=1).cpu().numpy()
        #     entropy = -np.sum(softmax_logits * np.log(softmax_logits), axis=1)
        #     sorted_indices = np.argsort(entropy)
        #     print(labels, self.labels, entropy, sorted_indices)
        #     # 引入Oracle模型
        #     # oracle_model = models.resnet152()
        #     # num_ftrs = oracle_model.fc.in_features
        #     # oracle_model.fc = nn.Linear(num_ftrs, 100)  # 假设num_classes是你的类别数量
        #     if oracle_model is not None:
        #         oracle_model = copy.deepcopy(oracle_model)
        #         # oracle_model.load_state_dict(torch.load(f"./oracle_91_20.pth"))
        #     else:
        #         oracle_model.load_state_dict(torch.load(f"./oracle_99_100.pth"))
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     oracle_model.to(device)
        #     # t = transforms.Compose([
        #     #     # transforms.RandomCrop((224, 224), padding=4),
        #     #     transforms.Resize((224, 224)),  # 根据模型预训练大小调整图像大小
        #     #     transforms.RandomHorizontalFlip(),
        #     #     transforms.ToTensor(),
        #     # ])
        #     # images_o = [Image.fromarray(image) if isinstance(image, np.ndarray) else image for image in self.images]
        #     # images_oracle = np.array([t(image) for image in images_o])
        #     # images_oracle = torch.tensor(images_oracle, dtype=torch.float32).cuda()
        #     # oracle_logits = oracle_model(images_oracle)
        #     # _, predicted = torch.max(oracle_logits.data, 1)
        #     # self.oracle_labels = predicted.cpu().numpy()
        #     logits = oracle_model(images)["logits"]
        #     self.oracle_labels = logits.argmax(dim=1).cpu().numpy()
        #     print("Oracle labels: ", self.oracle_labels)
        #     print("Labels: ", self.labels)
        #     # 准确率
        #     correct = (self.labels == self.oracle_labels).sum()
        #     with open(f"results_1011_{keep_ratio}.txt", 'a') as f:
        #         f.write("Accuracy: " + str(correct / len(self.labels)) + "\n")
        #         print("Accuracy: ", correct / len(self.labels))
        #         # 前20%的准确率
        #         correct = (self.labels[:int(len(self.labels) * 0.2)] == self.oracle_labels[:int(len(self.labels) * 0.2)]).sum()
        #         print("Accuracy of the first 20%: ", correct / int(len(self.labels) * 0.2))
        #         f.write("Accuracy of the first 20%: " + str(correct / int(len(self.labels) * 0.2)) + "\n")
        #
        #         # 后80%的准确率
        #         correct = (self.labels[int(len(self.labels) * 0.2):] == self.oracle_labels[int(len(self.labels) * 0.2):]).sum()
        #         print("Accuracy of the last 80%: ", correct / int(len(self.labels) * 0.8))
        #         f.write("Accuracy of the last 80%: " + str(correct / int(len(self.labels) * 0.8)) + "\n")
        #
        #         # 前40%的准确率
        #         correct = (self.labels[:int(len(self.labels) * 0.4)] == self.oracle_labels[:int(len(self.labels) * 0.4)]).sum()
        #         print("Accuracy of the first 40%: ", correct / int(len(self.labels) * 0.4))
        #         f.write("Accuracy of the first 40%: " + str(correct / int(len(self.labels) * 0.4)) + "\n")
        #
        #         # 后60%的准确率
        #         correct = (self.labels[int(len(self.labels) * 0.4):] == self.oracle_labels[int(len(self.labels) * 0.4):]).sum()
        #         print("Accuracy of the last 60%: ", correct / int(len(self.labels) * 0.6))
        #         f.write("Accuracy of the last 60%: " + str(correct / int(len(self.labels) * 0.6)) + "\n")
        #
        #         # 前60%的准确率
        #         correct = (self.labels[:int(len(self.labels) * 0.6)] == self.oracle_labels[:int(len(self.labels) * 0.6)]).sum()
        #         print("Accuracy of the first 60%: ", correct / int(len(self.labels) * 0.6))
        #         f.write("Accuracy of the first 60%: " + str(correct / int(len(self.labels) * 0.6)) + "\n")
        #
        #         # 后40%的准确率
        #         correct = (self.labels[int(len(self.labels) * 0.6):] == self.oracle_labels[int(len(self.labels) * 0.6):]).sum()
        #         print("Accuracy of the last 40%: ", correct / int(len(self.labels) * 0.4))
        #         f.write("Accuracy of the last 40%: " + str(correct / int(len(self.labels) * 0.4)) + "\n")
        #
        #         # 前80%的准确率
        #         correct = (self.labels[:int(len(self.labels) * 0.8)] == self.oracle_labels[:int(len(self.labels) * 0.8)]).sum()
        #         print("Accuracy of the first 80%: ", correct / int(len(self.labels) * 0.8))
        #         f.write("Accuracy of the first 80%: " + str(correct / int(len(self.labels) * 0.8)) + "\n")
        #         # 后20%的准确率
        #         correct = (self.labels[int(len(self.labels) * 0.8):] == self.oracle_labels[int(len(self.labels) * 0.8):]).sum()
        #         print("Accuracy of the last 20%: ", correct / int(len(self.labels) * 0.2))
        #         f.write("Accuracy of the last 20%: " + str(correct / int(len(self.labels) * 0.2)) + "\n")
        #
        #
        #
        #     # test_dataset_oracle = datasets.CIFAR100(root='../data', train=False, transform=t)
        #     # test_dataset = datasets.CIFAR100(root='../data', train=False, transform=transform)
        #
        #     # 筛选出0-19类别的数据
        #     # test_indices = [i for i, (_, label) in enumerate(test_dataset) if label < 20]
        #
        #     # test_data = Subset(test_dataset, test_indices)
        #     # test_dataset_oracle = Subset(test_dataset_oracle, test_indices)
        #
        #     # test_loader = torch.utils.data.DataLoader(test_data, batch_size=256,
        #     #                                           shuffle=False, num_workers=4)
        #     # test_loader_oracle = torch.utils.data.DataLoader(test_dataset_oracle, batch_size=256,
        #     #                                           shuffle=False, num_workers=4)
        #     # test_set = data_manager.get_dataset(
        #     #     np.arange(0, 20), source="test", mode="test"
        #     # )
        #     # test_loader = DataLoader(
        #     #     test_set, batch_size=256, shuffle=False, num_workers=4
        #     # )
        #     # test_set_oracle = data_manager.get_dataset(
        #     #     np.arange(0, 20), source="test", mode="oracle_test"
        #     # )
        #     # test_loader_oracle = DataLoader(
        #     #     test_set_oracle, batch_size=256, shuffle=False, num_workers=4
        #     # )
        #     # correct = 0
        #     # total = 0
        #     # for _, images, labels in test_loader:
        #     #     images, labels = images.to(device), labels.to(device)
        #     #     outputs = self.model(images)["logits"]
        #     #     predicts = torch.max(outputs, dim=1)[1]
        #     #     # _, predicted = torch.max(outputs.data, 1)
        #     #     # labels = logits.argmax(dim=1).cpu().numpy()
        #     #     total += labels.size(0)
        #     #     correct += (predicts == labels).sum().item()
        #     # print('Accuracy of the network on the first 2000 test images on our own model: %d %%' % (
        #     #         100 * correct / total))
        #     #
        #     # correct = 0
        #     # total = 0
        #     # for _, images, labels in test_loader:
        #     #     images, labels = images.to(device), labels.to(device)
        #     #     outputs = oracle_model(images)["logits"]
        #     #     predicts = torch.max(outputs, dim=1)[1]
        #     #     # _, predicted = torch.max(outputs.data, 1)
        #     #     # labels = logits.argmax(dim=1).cpu().numpy()
        #     #     total += labels.size(0)
        #     #     correct += (predicts == labels).sum().item()
        #     # print('Accuracy of the network on the first 2000 test images on oracle model: %d %%' % (
        #     #         100 * correct / total))
        #     # # for _, images, labels in test_loader:
        #     # #     images, labels = images.to(device), labels.to(device)
        #     # #     oracle_logits = oracle_model(images)
        #     # #
        #     # #     # oracle_logits = oracle_model(images_oracle)
        #     # #     _, predicts = torch.max(oracle_logits.data, 1)
        #     # #     # self.oracle_labels = predicted.cpu().numpy()
        #     # #
        #     # #     # predicts = torch.max(outputs.data, dim=1)[1]
        #     # #     # _, predicted = torch.max(outputs.data, 1)
        #     # #     # labels = logits.argmax(dim=1).cpu().numpy()
        #     # #     total += labels.size(0)
        #     # #     correct += (predicts == labels).sum().item()
        #     # # print('Accuracy of the network on the first 2000 test images on the oracle model: %d %%' % (
        #     # #         100 * correct / total))
        #     #
        #     # correct = 0
        #     # total = 0
        #     # test_loader, order_20 = get_test_set(transform)
        #     # for images, labels in test_loader:
        #     #     images, labels = images.to(device), labels.to(device)
        #     #     outputs = self.model(images)["logits"]
        #     #     predicts = torch.max(outputs, dim=1)[1]
        #     #     # _, predicted = torch.max(outputs.data, 1)
        #     #     # labels = logits.argmax(dim=1).cpu().numpy()
        #     #     labels = torch.tensor([order_20.index(label) for label in labels], device=device)
        #     #     total += labels.size(0)
        #     #     correct += (predicts == labels).sum().item()
        #     # print('Accuracy of the network on the 10000 test images or our own model: %d %%' % (
        #     #         100 * correct / total))
        #     #
        #     # correct = 0
        #     # total = 0
        #     # # test_loader_oracle, order_20 = get_test_set()
        #     # # for images, labels in test_loader_oracle:
        #     # #     images, labels = images.to(device), labels.to(device)
        #     # #     outputs = oracle_model(images)
        #     # #     _, predicted = torch.max(outputs.data, 1)
        #     # #     labels = torch.tensor([order_20.index(label) for label in labels], device=device)
        #     # #     total += labels.size(0)
        #     # #     correct += (predicted == labels).sum().item()
        #     # for images, labels in test_loader:
        #     #     images, labels = images.to(device), labels.to(device)
        #     #     outputs = oracle_model(images)["logits"]
        #     #     predicts = torch.max(outputs, dim=1)[1]
        #     #     # _, predicted = torch.max(outputs.data, 1)
        #     #     # labels = logits.argmax(dim=1).cpu().numpy()
        #     #     labels = torch.tensor([order_20.index(label) for label in labels], device=device)
        #     #     total += labels.size(0)
        #     #     correct += (predicts == labels).sum().item()
        #     # print('Accuracy of the network on the first 2000 test images on our own model: %d %%' % (
        #     #         100 * correct / total))
        #     # # print('Accuracy of the network on the 10000 test images of oracle model: %d %%' % (
        #     # #     100 * correct / total))
        #     # pass

    def __getitem__(self, index):
        x = self.images[index]

        if self.transform:
            # 假设transform已经可以直接应用在tensor上
            x = self.transform(x)

        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.images.shape[0]


def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    # first_batch = next(iter(train_data_loader))
    # images, labels = first_batch
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    return sorted(temp.items(),key=lambda x:x[0])



class Client_Diffusion(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.memory_size = args["memory_size"]
        image_size = 32
        image_channels = 3
        if args["dataset"] == "tiny_imagenet":
            image_size = 64
        # elif args["dataset"] == "emnist_letters":
        #     # image_size = 28
        #     image_channels = 1
        self.diffusion_models = [Diffusion(image_channels=image_channels, epochs=args["diffusion_epochs"], model_path="", dataset=None, image_size=image_size) for _ in range(args["num_users"])]
        self.local_models = {}

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        self._old_test_loader = self.test_loader
        self._old_test_loader_oracle = self.test_loader_oracle
        self._old_test_set_oracle = self.test_dataset_oracle
        self._old_test_set = self.test_dataset
        self._old_oracle_model = self._oracle_model

    def get_all_previous_dataset_by_diffusion_model(self, idx, model_path="", batch_size=None):
        if batch_size is None:
            batch_size = self.memory_size
        # batch_num = self.memory_size // 64
        # self.memory_size = 10
        idata = _get_idata(self.args["dataset"])
        # # Transforms
        test_trsf = idata.test_trsf
        common_trsf = idata.common_trsf
        trsf = transforms.Compose([*test_trsf, *common_trsf])
        if self.args["label_model"] == 'global':
            if self.args["note"].find("global_labelmodel") == -1:
                self.args["note"] += "_global_labelmodel"
            model = copy.deepcopy(self._old_network)
            # model = copy.deepcopy(self._network)
            # model.load_state_dict(self._old_network.state_dict())
            print("Load global label model")
        elif self.args["label_model"] == 'local':
            if self.args["note"].find("local_labelmodel") == -1:
                self.args["note"] += "_local_labelmodel"
            model = self.local_models[idx]
            print("Load local label model")
        if self.args["note"].find("memorysize") == -1:
            self.args["note"] += "_memorysize{}".format(self.memory_size)
        all_images = np.array([])
        all_labels = np.array([])
        keep = self.args["keep_ratio"]
        if self.args["note"].find("keep") == -1:
            self.args["note"] += "_keep{}".format(keep)
        while True:
            # batch_size = 2000
            sample_tensors = self.diffusion_models[idx].sample(model_path, log=False, batch_size=batch_size)
            previous_dataset = CustomTensorDataset(sample_tensors, model, transform=trsf, keep=keep, test_L=self._old_test_set, oracle_test_L=self._old_test_loader_oracle, data_manager=self.data_manager, oracle_model=self._old_oracle_model)
            images = previous_dataset.images
            labels = previous_dataset.labels
            all_images = np.concatenate((all_images, images), axis=0) if len(all_images) > 0 else images
            all_labels = np.concatenate((all_labels, labels), axis=0) if len(all_labels) > 0 else labels
            len_of_images = len(all_images)
            if len_of_images >= self.memory_size:
                if len_of_images > self.memory_size: # 保证数据集大小不超过memory_size
                    all_images = all_images[:self.memory_size]
                    all_labels = all_labels[:self.memory_size]
                break
        previous_dataset = DummyDataset(all_images, all_labels, trsf, use_path=False)
        return previous_dataset





    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.data_manager = data_manager
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            # appendent=self._get_memory(),   # get memory, 2000 data: 100 * 20cls[0~19]
        )
        diffusion_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="diffusion",
            # appendent=self._get_memory(),  # get memory, 2000 data: 100 * 20cls[0~19]
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        self._network.cuda()
        setup_seed(self.seed)
        # if self._cur_task == 0:
        self.train_oracle(data_manager)
        self._fl_train(train_dataset, self.test_loader, data_manager, diffusion_dataset)

    def train_oracle(self, data_manager):
        self.test_dataset_oracle = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader_oracle = DataLoader(
            self.test_dataset_oracle, batch_size=256, shuffle=False, num_workers=4
        )
        train_dataset_oracle = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        self.train_loader_oracle = DataLoader(
            train_dataset_oracle, batch_size=self.args["local_bs"], shuffle=True, num_workers=4
        )
        self._oracle_model = copy.deepcopy(self._network)
        self._oracle_model.update_fc(self._total_classes)
        self._oracle_model.cuda()
        # train_oracle_model(self._oracle_model, self.train_loader_oracle, self.test_loader_oracle, 20)

        # test_dataset_oracle = data_manager.get_dataset(
        #     np.arange(0, 100), source="test", mode="test"
        # )
        # test_loader_oracle = DataLoader(
        #     test_dataset_oracle, batch_size=256, shuffle=False, num_workers=4
        # )
        # train_dataset_oracle = data_manager.get_dataset(
        #     np.arange(0, 100), source="train", mode="train"
        # )
        # train_loader_oracle = DataLoader(
        #     train_dataset_oracle, batch_size=self.args["local_bs"], shuffle=True, num_workers=4
        # )
        # model = copy.deepcopy(self._network)
        # model.update_fc(100)
        # model.cuda()
        # self.oracle_model = model
        # train_oracle_model(model, train_loader_oracle, test_loader_oracle, 100)



    def _local_update(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                if type(images) == list:
                    images = torch.stack(images, dim=0)
                    images = images.squeeze(1)
                    labels = torch.stack(labels, dim=0)
                    labels = labels.squeeze(1)

                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter == 0 and com_id==0 : print("task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task ,client_id, total, tmp))
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        old_network_copy = copy.deepcopy(self._old_network)
        old_network_copy.update_fc(self._total_classes)
        old_network_copy.cuda()
        optimizer_old = torch.optim.SGD(old_network_copy.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                if type(images) == list:
                    images = torch.stack(images, dim=0)
                    images = images.squeeze(1)
                    labels = torch.stack(labels, dim=0)
                    labels = labels.squeeze(1)
                images, labels = images.cuda(), labels.cuda()
                # fake_targets = labels - self._known_classes
                original_output = model(images)
                output = original_output["logits"]
                features = original_output["features"]
                old_output = self._old_network(images)
                old_logits = old_output["logits"]
                old_features = old_output["features"]
                #* finetune on the new tasks
                loss_clf = F.cross_entropy(output, labels) # 目前所有的类别
                loss_kd = _KD_loss(
                    output[:, : self._known_classes], # 之前的类别，这里指的是student model的输出
                    # self._old_network(images)["logits"], # 之前的类别，这里指的是teacher model的输出
                    old_logits,
                    T,
                )
                if self.args["way"] == 3: # including kd loss
                    loss_distill = F.mse_loss(features, old_features)
                    loss = loss_clf + 3 * loss_kd + 2 * loss_distill
                    # loss = loss_clf + 25 * loss_kd
                    # loss = loss_clf + 3 * loss_kd
                elif self.args["way"] == 2: # only clf loss
                    loss = loss_clf
                    # print("only clf loss")
                optimizer.zero_grad()
                loss.backward()

                # old_network_copy.train()
                # # 确保所有参数均可求导
                # for param in old_network_copy.parameters():
                #     param.requires_grad = True
                # old_network_copy.zero_grad()
                # OLO = old_network_copy(images)["logits"]
                # loss2 = F.cross_entropy(OLO, labels)
                # loss2.backward()
                #
                # # 访问并比较两个模型最后一层的梯度
                # grad1 = list(model.parameters())[-1].grad
                # grad2 = list(old_network_copy.parameters())[-1].grad
                # # 计算梯度差异
                # grad_difference = torch.norm(grad1 - grad2)
                # print("gradient difference:", grad_difference)
                #
                # grad_difference.backward()

                optimizer.step()

                # 打印模型参数
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)
                #     break
                total += images.shape[0]
            # print("loss_clf:{}, loss_kd:{}, loss_distill:{}".format(loss_clf, loss_kd, loss_distill))
            if iter ==0 and com_id==0 : print("task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task ,client_id, total, tmp))
        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader, data_manager, diffusion_dataset):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        # Save the User Groups
        prog_bar = tqdm(range(self.args["com_round"]))
        previous_local_datasets = {}
        if self._cur_task != 0:
            for idx in range(self.args["num_users"]):
                previous_local_dataset = self.get_all_previous_dataset_by_diffusion_model(idx)
                previous_local_datasets[idx] = previous_local_dataset

        for _, com in enumerate(prog_bar):
            if self._cur_task == 0:
                if os.path.exists(os.path.join("checkpoints", "task_{}_com_{}_{}.pth".format(self._cur_task, com, self.args["training_id"]))):
                    self._network.load_state_dict(torch.load(os.path.join("checkpoints", "task_{}_com_{}.pth".format(self._cur_task, com))))
                    print("Load model from task_{}_com_{}_{}.pth".format(self._cur_task, com, self.args["training_id"]))
                    continue
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            print("Selected users:", idxs_users)
            for idx in idxs_users:
                # update local train data
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:
                    current_local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                    # previous_local_dataset = self.get_all_previous_dataset_by_diffusion_model(idx)
                    previous_local_dataset = previous_local_datasets[idx]

                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset)
                    local_dataset = DatasetSplit(local_dataset, range(local_dataset.labels.shape[0]))

                if self.args["training_sample_selection_mode"] == "balance":
                    n_samples = max(self.args["local_bs"]//self._total_classes, 2)
                    labels = local_dataset.dataset.labels[local_dataset.idxs]
                    sampler = BalancedBatchSampler(labels, 0, self._total_classes, n_samples)
                    local_train_loader = DataLoader(local_dataset, sampler=sampler)
                    if self.args["note"].find("_balanceTraining") == -1:
                        self.args["note"] += "_balanceTrainingSampler"
                else:
                    local_train_loader = DataLoader(local_dataset, batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                tmp = print_data_stats(idx, local_train_loader)
                if com !=0:
                    tmp = ""
                if self._cur_task == 0:                    
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                local_weights.append(copy.deepcopy(w))
                t_model = copy.deepcopy(self._network)
                t_model.load_state_dict(w)
                self.local_models[idx] = t_model
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            # if self._cur_task == 0:
            #     torch.save(global_weights,
            #                os.path.join("checkpoints", "task_{}_com_{}_{}.pth".format(self._cur_task, com, self.args["training_id"])))
            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})



        if self._cur_task + 1 < self.args["tasks"]:
            for idx in range(self.args["num_users"]):
                model_path = f"checkpoints_diffusion/exp{self.args['id']}_task{self._cur_task}_client{idx}_"
                local_diffusion_dataset = DatasetSplit(diffusion_dataset, user_groups[idx])
                if self.args["diffusion_mode"] == "all":
                    print("Combine previous dataset with current dataset")
                    if self._cur_task != 0:
                        generated_length = len(local_diffusion_dataset.idxs)
                        print("Generated length: ", generated_length)
                        generated_dataset = self.get_all_previous_dataset_by_diffusion_model(idx, batch_size=generated_length)
                        user_diffusion_dataset = self.combine_dataset(generated_dataset, local_diffusion_dataset)
                    else:
                        user_diffusion_dataset = local_diffusion_dataset
                    if self.args["note"].find("_alldiffusion") == -1:
                        self.args["note"] += "_alldiffusion"
                    local_diffusion_train_loader = torch.utils.data.DataLoader(user_diffusion_dataset, batch_size=64,
                                                                               shuffle=True, pin_memory=True)
                else:
                    print("Train with current dataset only")
                    if self.args["note"].find("_current_diffusion") == -1:
                        self.args["note"] += "_current_diffusion"
                    # 手动构建dataloaders, 每次从每个类别中随机抽取一些样本
                    if self.args["sample_selection_mode"] == "balance":
                        n_samples = 64 // 20
                        labels = local_diffusion_dataset.dataset.labels[local_diffusion_dataset.idxs]
                        sampler = BalancedBatchSampler(labels, self._known_classes, self._total_classes, n_samples)
                        local_diffusion_train_loader = DataLoader(local_diffusion_dataset, sampler=sampler)
                        if self.args["note"].find("_balance") == -1:
                            self.args["note"] += "_balanceSampler"
                    else:
                        local_diffusion_train_loader = torch.utils.data.DataLoader(local_diffusion_dataset,
                                                                                   batch_size=64, shuffle=True,
                                                                                   pin_memory=True)
                self.diffusion_models[idx].train_without_sample(local_diffusion_train_loader, model_path, batch_size=16)

    def combine_dataset(self, pre_dataset, cur_dataset):
        pre_labels = pre_dataset.labels
        pre_data = pre_dataset.images

        idx = cur_dataset.idxs
        cur_labels = cur_dataset.dataset.labels[idx]
        cur_data = cur_dataset.dataset.images[idx]

        if len(cur_data.shape) == 1: # 目录中读取的图片，不是多维ndarray， 针对TinyImageNet
            # 读取图像文件并转换为数组
            cur_images = []
            for file_path in cur_data:
                with Image.open(file_path) as img:
                    if file_path.find("emnist") != -1:
                        img = img.resize((32, 32))
                    elif file_path.find("tiny") != -1:
                        img = img.resize((64, 64))  # 确保和 pre_data 尺寸一致
                    img_array = np.array(img)
                    if img_array.shape == (64, 64):  # 如果是黑白图像，转换为彩色
                        img_array = np.stack([img_array] * 3, axis=-1)
                    elif img_array.shape == (32, 32):
                        img_array = np.stack([img_array] * 3, axis=-1)
                    cur_images.append(img_array)
            cur_data = np.array(cur_images)


        combined_data = np.concatenate((cur_data, pre_data), axis=0)
        combined_label = np.concatenate((cur_labels, pre_labels), axis=0)

        idata = _get_idata(self.dataset_name)
        _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
        trsf = transforms.Compose([*_train_trsf, *_common_trsf])
        combined_dataset = DummyDataset(combined_data, combined_label, trsf, use_path=False)

        return combined_dataset





def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
