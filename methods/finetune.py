import pickle

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from sklearn.metrics import confusion_matrix

# init_epoch = 200
# com_round = 100  
# num_users = 5 # 5, 
# frac = 1 # 




# local_bs = 128  # cifar100, 5w, 5 tasks, 1w for each task, 2k for each client
# local_ep = 5
# batch_size = 128
# num_workers = 4

tau=1


def print_data_stats(client_id, train_data_loader):
    # pdb.set_trace()
    def sum_dict(a,b):
        temp = dict()
        # | 并集
        for key in a.keys() | b.keys():
            temp[key] = sum([d.get(key, 0) for d in (a, b)])
        return temp
    temp = dict()
    for batch_idx, (_, images, labels) in enumerate(train_data_loader):
        unq, unq_cnt = np.unique(labels, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        temp = sum_dict(tmp, temp)
    print(client_id, sorted(temp.items(),key=lambda x:x[0]))





def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class Finetune(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.acc = []

    def after_task(self):
        self._known_classes = self._total_classes
        self.pre_loader = self.test_loader
        self._old_network = self._network.copy().freeze()


    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""
        KLDiv = nn.KLDivLoss(reduction="batchmean")
        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self._total_classes)
        pred_probs = F.log_softmax(logits / tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self._total_classes)
            dg_probs = torch.softmax(dg_logits / tau, dim=1)

        loss = (tau ** 2) * KLDiv(pred_probs, dg_probs)

        return loss


    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(   #* get the data for one task
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        ) # 这一步将训练数据集所有的self._known_classes到self._total_classes的数据都取出来生成一个新的数据集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        setup_seed(self.seed)
        self._fl_train(train_dataset, self.test_loader)
        

        # if self._cur_task == 0:
        #     # self._fl_train(train_dataset, self.test_loader)
        #     # torch.save(self._network.state_dict(), 'finetune.pkl')
        #     # print("save checkpoint >>>")

        #     self._network.cuda()
        #     state_dict = torch.load('finetune.pkl')
        #     self._network.load_state_dict(state_dict)
        #     test_acc = self._compute_accuracy(self._network, self.test_loader)
        #     print("For task 1, loading ckpt, acc:{}".format(test_acc))

        #     # return 
        # else:
        #     # return 
        #     acc = self._compute_accuracy(self._old_network, self.pre_loader)
        #     print("loading ckpt, acc:{}".format(acc))
            
        #     self._fl_train(train_dataset, self.test_loader)

        

    # def _local_update(self, model, train_data_loader):
    #     model.train()
    #     cp_model =  copy.deepcopy(model)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #     for iter in range(local_ep):
    #         for batch_idx, (_, images, labels) in enumerate(train_data_loader):
    #             images, labels = images.cuda(), labels.cuda()
    #             output = model(images)["logits"]
    #             loss_ce = F.cross_entropy(output, labels)
    #             with torch.no_grad():
    #                 dg_logits = cp_model(images.detach())["logits"]
    #             # only learn from out-distribution knowledge, overcome local forgetting
    #             loss_ntd = self._ntd_loss(output, dg_logits, labels)
    #             loss = loss_ce + loss_ntd 
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #     return model.state_dict()

    def _local_update(self, model, train_data_loader, idx):
        # print_data_stats(idx, train_data_loader)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output = model(images)["logits"]
                loss = F.cross_entropy(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()


    def per_cls_acc(self, val_loader, model):
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for i, (_, input, target) in enumerate(val_loader):
                input, target = input.cuda(), target.cuda()
                # compute output
                output = model(input)["logits"]
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)

        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)

        cls_acc = cls_hit / cls_cnt
        """
        这四行代码是用Python语言编写的，并且使用了数据科学和机器学习中的常用库NumPy来处理数据。这些代码行的目的是从预测和真实目标值中计算出混淆矩阵，并且计算每个类别的分类准确率。这些代码通常在评估分类模型的性能时使用。下面逐行解析代码：

        cf = confusion_matrix(all_targets, all_preds).astype(float): 这一行调用confusion_matrix函数创建一个混淆矩阵cf，传入的参数是真实的目标值all_targets和模型的预测值all_preds。混淆矩阵是一个方阵，其维度等于分类任务中的类别个数。矩阵中的每个元素cf[i, j]表示真实类别为i的样本被预测为类别j的次数。通过.astype(float)，将混淆矩阵的数据类型转换为浮点数类型，这通常是为了后续的计算方便，尤其是涉及到需要小数的计算时。
        
        cls_cnt = cf.sum(axis=1): 这条语句通过sum函数沿着混淆矩阵的横轴（即axis=1）进行求和，得到一个一维数组cls_cnt。其中，每个元素cls_cnt[i]代表真实类别为i的样本总数。实际上，它是计算每个类别的样本数。
        
        cls_hit = np.diag(cf): 这一行使用NumPy库的diag函数提取混淆矩阵的对角线元素，并将其保存在cls_hit中。对角线上的元素反映了正确预测的数量，也就是每个类别正确分类的样本数。
        
        cls_acc = cls_hit / cls_cnt: 在这行代码中，计算所有类别的分类准确率。它通过将正确分类的样本数cls_hit与真实样本数cls_cnt进行逐元素的除法得到一个一维数组cls_acc。cls_acc[i]表示第i类别的准确率。这是通过计算每个类别的正确预测数除以该类别的样本总数得到的。
        
        结合起来看，这串代码为每个类别提供了一个准确率的度量，这可以帮助理解模型在不同类别上的性能如何，特别是在类别不平衡的数据集中，这种性能指标非常重要。
        """
        return cls_acc
        # pdb.set_trace()
        # out_cls_acc = 'Per Class Accuracy: %s' % ((np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        # print(out_cls_acc)
        

        

    def _local_finetune(self, model, train_data_loader, idx):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # print_data_stats(idx, train_data_loader)
        for iter in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                fake_targets = labels - self._known_classes # 把原来的标签减去已知的类别数，得到新的标签，如原来的标签是[36, 37, 38, 39, 40]，已知的类别数是20，那么新的标签就是[16, 17, 18, 19, 20]
                output = model(images)["logits"]
                #* finetune on the new tasks
                loss = F.cross_entropy(output[:, self._known_classes:], fake_targets) # 只看新的类别的loss和acc，如_known_classes=20，那么只看20-39的类别的loss和acc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # self.per_cls_acc(self.test_loader, model)

        return model.state_dict()

    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        cls_acc_list = []
        # 第一个任务有0-19类别的数据，分给args.num_users去训练，每个用户得到的数据是non-iid的，根据dirichlet分布beta进行设定；然后再进行第二个任务的训练
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        with open(f'user_groups_{self.args["dataset"]}_task{self.args["tasks"]}_{self.args["num_users"]}_beta{self.args["beta"]}_{self._cur_task}_seed{self.args["seed"]}.pkl', "wb") as f:
            pickle.dump(user_groups, f)
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                if self._cur_task == 0: # 如果是刚开始的任务，就直接训练
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader, idx)
                else: # 如果不是第一个任务，就进行finetune
                    w = self._local_finetune(copy.deepcopy(self._network), local_train_loader, idx)
                local_weights.append(copy.deepcopy(w))
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            if com % 1 == 0:
                cls_acc = self.per_cls_acc(self.test_loader, self._network)
                cls_acc_list.append(cls_acc)

                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
        acc_arr = np.array(cls_acc_list)
        acc_max = acc_arr.max(axis=0) # 按列取最大值，记录了每个类别在所有轮次中的最大准确率
        if self._cur_task == 4:
            acc_max = self.per_cls_acc(self.test_loader, self._network)
        print("For task: {}, acc list max: {}".format(self._cur_task, acc_max))
        self.acc.append(acc_max)



