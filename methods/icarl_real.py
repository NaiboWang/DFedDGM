import numpy as np
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from methods.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed, _get_idata, DummyDataset
import copy, wandb
from PIL import Image
from torchvision import transforms

EPSILON = 1e-8
T = 2


class TempDataset:
    def __init__(self, images, labels):
        self.dataset = DummyDataset(images, labels, None)
        self.idxs = np.arange(len(images))


def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a, b):
        temp = dict()
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp

    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    return sorted(temp.items(), key=lambda x: x[0])


class iCaRL_real(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.memory_size = args["memory_size"]
        self.exemplar_sets = []
        self.class_mean_sets = []
        for i in range(args["num_users"]):
            self.exemplar_sets.append({})
            self.class_mean_sets.append({})

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def get_all_previous_dataset(self, data_manager, idx):
        # for second task, self._cur_task=1
        bgn_cls, end_cls = 0, self.each_task  # each_task=20 for 5 tasks in cifar100
        train_dataset = data_manager.get_dataset(
            np.arange(bgn_cls, end_cls),
            source="train",
            mode="train",
        )
        setup_seed(self.seed)
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        all_previous_dataset = DatasetSplit(train_dataset, user_groups[idx])
        # for third task
        for i in range(2, self._cur_task + 1):  # 2-4
            setup_seed(self.seed)
            bgn_cls += self.each_task  # 20-40
            end_cls += self.each_task
            train_dataset_next = data_manager.get_dataset(
                np.arange(bgn_cls, end_cls),
                source="train",
                mode="train",
            )
            user_groups_next = partition_data(train_dataset_next.labels, beta=self.args["beta"],
                                              n_parties=self.args["num_users"])
            tmp_dataset = DatasetSplit(train_dataset_next, user_groups_next[idx])  # <utils.data_manager.DummyDataset>
            all_previous_dataset = self.combine_dataset(all_previous_dataset, tmp_dataset, 0)  # combine two datasets
            all_previous_dataset = DatasetSplit(all_previous_dataset, range(all_previous_dataset.labels.shape[0]))
            # 2417->   all_previous_dataset.idxs[0:4]= [9013, 7479, 5185, 7241]
        return all_previous_dataset

    def get_memory_buffer(self, idx):
        exemplar_set = self.exemplar_sets[idx]
        images = []
        labels = []
        for key in exemplar_set.keys():
            images += exemplar_set[key]
            labels += [key] * len(exemplar_set[key])

        dataset = TempDataset(images, labels)
        return dataset

    def _construct_exemplar_set(self, images, len_of_set, idx, class_id):
        if images.shape[0] != 0:
            idata = _get_idata(self.dataset_name)
            _train_trsf, _common_trsf = idata.train_trsf, idata.common_trsf
            trsf = transforms.Compose([transforms.ToTensor(), *_common_trsf])
            class_mean, feature_extractor_output = self.compute_class_mean(images, trsf)
            exemplar = []
            now_class_mean = np.zeros((1, len(class_mean)))

            for i in range(len_of_set):
                # shape：batch_size*512
                x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
                # shape：batch_size
                x = np.linalg.norm(x, axis=1)
                index = np.argmin(x)
                now_class_mean += feature_extractor_output[index]
                exemplar.append(images[index])

            print("the size of exemplar: %s" % (str(len(exemplar))))
            self.exemplar_sets[idx][class_id] = exemplar
        else:
            print("No images in class %d" % class_id)
            self.exemplar_sets[idx][class_id] = []

    def _reduce_exemplar_sets(self, m, idx):
        exemplar_set = self.exemplar_sets[idx]
        for index in exemplar_set.keys():
            exemplar_set[index] = exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(exemplar_set[index]))))

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).cuda()
        model = copy.deepcopy(self._network)
        model.train()
        features = model(x)["features"]
        feature_extractor_output = F.normalize(features.detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),  # get memory, 2000 data: 100 * 20cls[0~19]
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
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        self._network.cuda()
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader, data_manager, diffusion_dataset)
        m = int(self.memory_size / self._total_classes)
        for idx in range(self.args["num_users"]):
            self._reduce_exemplar_sets(m, idx)
            for i in range(self._total_classes - data_manager.get_task_size(self._cur_task), self._total_classes):
                print('Construct class %s examplars for client %s' % (i, idx))
                idxs = self.train_datasets[idx].idxs
                images = self.train_datasets[idx].dataset.images[idxs]
                labels = self.train_datasets[idx].dataset.labels[idxs]
                images = images[labels == i]
                len_of_set = min(len(images), m)
                self._construct_exemplar_set(images, len_of_set, idx, i)

    def _local_update(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter == 0 and com_id == 0: print(
                "task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task, client_id, total,
                                                                                      tmp))
        return model.state_dict()

    def _local_finetune(self, model, train_data_loader, client_id, tmp, com_id):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        for iter in range(self.args["local_ep"]):
            total = 0
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss_clf = F.cross_entropy(output, labels)  # 目前所有的类别
                loss_kd = _KD_loss(
                    output[:, : self._known_classes],  # 之前的类别
                    self._old_network(images)["logits"],
                    T,
                )
                if self.args["way"] == 3:  # including kd loss
                    loss = loss_clf + loss_kd
                elif self.args["way"] == 2:  # only clf loss
                    loss = loss_clf
                    # print("only clf loss")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += images.shape[0]
            if iter == 0 and com_id == 0: print(
                "task_id:{}, client_id: {}, local dataset size: {}, labels:{}".format(self._cur_task, client_id, total,
                                                                                      tmp))
        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader, data_manager, diffusion_dataset):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        self.train_datasets = {}

        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                # update local train data
                if self._cur_task == 0:
                    local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                else:
                    current_local_dataset = DatasetSplit(train_dataset, user_groups[idx])
                    # previous_local_dataset = self.get_all_previous_dataset(data_manager, idx)
                    previous_local_dataset = self.get_memory_buffer(idx)

                    local_dataset = self.combine_dataset(previous_local_dataset, current_local_dataset)
                    local_dataset = DatasetSplit(local_dataset, range(local_dataset.labels.shape[0]))
                original_dataset = DatasetSplit(diffusion_dataset, user_groups[idx])
                self.train_datasets[idx] = original_dataset
                local_train_loader = DataLoader(local_dataset, batch_size=self.args["local_bs"], shuffle=True,
                                                num_workers=4)
                tmp = print_data_stats(idx, local_train_loader)
                if com != 0:
                    tmp = ""
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                else:
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, idx, tmp, com)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = ("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc, ))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})

    def combine_dataset(self, pre_dataset, cur_dataset):
        # correct
        pre_labels = pre_dataset.dataset.labels
        pre_data = pre_dataset.dataset.images

        idx = cur_dataset.idxs
        cur_labels = cur_dataset.dataset.labels[idx]
        cur_data = cur_dataset.dataset.images[idx]

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
