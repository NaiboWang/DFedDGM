import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, TinyImageNet200, iEMNISTLetters, iSVHN
import torch, copy
import os, pdb, random
import numpy as np
import torch.backends.cudnn as cudnn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if type(item) == int:
            idx, image, label = self.dataset[self.idxs[item]]
        else:
            image = []
            label = []
            idx = []
            for index in item:
                i, img, lab = self.dataset[self.idxs[index]]
                image.append(img)
                label.append(lab)
                idx.append(i)
        return idx, image, label


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(y_train, beta=0.4, n_parties=5):
    data_size = y_train.shape[0]
    if beta == 0:  # for iid
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif beta > 0:  # for niid
        min_size = 0
        min_require_size = 1
        # label = np.unique(y_train).shape[0]
        labels = np.unique(y_train)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in labels:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(  # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    # record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        if dataset_name == "emnist_letters":
            if init_cls == 5:
                self._increments = [5, 5, 5, 5, 6]
            elif init_cls == 2:
                self._increments = [3, 2, 3, 2, 3, 2, 3, 2, 3, 3]
        print("Increments: {}".format(self._increments))  # [20, 20, 20, 20, 20] if init_cls=20, task = 5

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_class_order(self):
        return self._class_order

    def get_dataset(
            self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        elif mode == "oracle_test":
            trsf = transforms.Compose([
                # transforms.RandomCrop((224, 224), padding=4),
                transforms.Resize((224, 224)),  # 根据模型预训练大小调整图像大小
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif mode == "oracle_train":
            trsf = transforms.Compose([
                # transforms.RandomCrop((224, 224), padding=4),
                transforms.Resize((224, 224)),  # 根据模型预训练大小调整图像大小
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif mode == "diffusion":
            # trsf = transforms.Compose([*self._test_trsf])
            trsf = transforms.Compose([
                transforms.ToTensor(),
            ]
            )
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
            self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order CIFAR100: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
            self._class_order = order
            # Map indices
            self._train_targets = _map_new_class_index(
                self._train_targets, self._class_order
            )  # 把_train_targets中的类别映射到新的_class_order类别上，即如果原来的类别是19，而19在_class_order中的索引是3，那么新的类别就是3，即把所有的原来为19的类别都映射为3
            self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        else:
            order = idata.class_order
            self._class_order = order

        # print(self._class_order)



    """
    这个语句是用 Python 语言中的 NumPy 库编写的。NumPy 是一个用于科学计算的库，提供了强大的数组对象和相关的操作。这个特定的语句执行的是在一个 NumPy 数组 y 中查找所有满足特定条件的元素的位置（索引），并返回这些位置组成的一个数组。
    让我们分步骤地解释这个语句：
    y >= low_range： - 生成一个布尔数组，其中每个位置的值根据数组 y 中相应位置的值是否大于或等于 low_range 的值进行设置。如果 y 中的某个值大于或等于 low_range，在结果布尔数组中的相应位置将是 True，否则为 False。
    y < high_range： - 同样地，这会生成一个布尔数组，其中每个位置的值取决于数组 y 中相应位置的值是否小于 high_range 的值。满足条件的为 True，否则为 False。
    np.logical_and(...)： - 这个函数将两个布尔数组作为输入，并执行逐元素的逻辑“与”（AND）操作。只有两个输入布尔数组在相同位置的值都为 True 时，结果布尔数组在该位置的值才为 True。这里就是查找 y 中同时满足 y >= low_range 和 y < high_range 这两个条件的元素。
    np.where(...)： - 这个 NumPy 函数接受一个布尔数组作为输入，并返回一个包含满足条件（即值为 True）的元素索引的元组。在这个例子中，元组中只有一个元素，因为 y 是一维数组。
    [0]： - 由于 np.where(...) 返回的是一个元组，而我们只需要这个元组中的第一个元素（也是唯一的一个，因为我们处理的是一维数组），所以使用 [0] 来获取这个元素。这样会得到一个索引数组，其中包含所有满足条件的元素的索引。
    总结：这个语句返回的是一个数组，包含所有在 y 中的值处于 [low_range, high_range) 区间内的元素的索引。
    """

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            try:
                image = self.trsf(Image.fromarray(self.images[idx]))
            except:
                image = self.trsf(torch.tensor(self.images[idx]))  # 生成的样本
        label = self.labels[idx]

        return idx, image, label


"""
order = ['a', 'b', 'c', 'd']
y = ['d', 'b', 'a']

indexes = map(lambda x: order.index(x), y)
print(list(indexes))  # 输出 [3, 1, 0]
在这个例子中，indexes 是 [3, 1, 0]，因为在 order 中 'd' 的索引是 3，'b' 的索引是 1，而 'a' 的索引是 0。
"""


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "svhn":
        return iSVHN()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "tiny_imagenet":
        return TinyImageNet200()
    elif name == "emnist_letters":
        return iEMNISTLetters()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

# def accimage_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
#     accimage is available on conda-forge.
#     """
#     import accimage

#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     """
#     Ref:
#     https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
#     """
#     from torchvision import get_image_backend

#     if get_image_backend() == "accimage":
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)
