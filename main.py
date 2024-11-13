import argparse
import copy
import datetime
import json
import time

import numpy as np
import wandb, os
from torch.utils.data import Dataset, DataLoader
from methods.client_diffusion import Client_Diffusion
from methods.fedprox import FedProx
from methods.icarl_real import iCaRL_real
from methods.lander import LANDER
from methods.ours import OURS
from utils.data_manager import DataManager, setup_seed
from utils.toolkit import count_parameters
from methods.finetune import Finetune
from methods.icarl import iCaRL
from methods.lwf import LwF
from methods.ewc import EWC
from methods.target import TARGET
import warnings
from dbconfig import *
warnings.filterwarnings('ignore')


def get_learner(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        return iCaRL(args)
    elif name == "icarl_real":
        return iCaRL_real(args)
    elif name == "ewc":
        return EWC(args)
    elif name == "lwf":
        return LwF(args)
    elif name == "finetune":
        return Finetune(args)
    elif name == "fedprox":
        return FedProx(args)
    elif name == "target":
        return TARGET(args)
    elif name == "lander":
        return LANDER(args)
    elif name == "ours_server_diffusion":
        return OURS(args)
    elif name == "ours_client_diffusion":
        return Client_Diffusion(args)
    else:
        assert 0
        




def train(args, argsn):
    print(args)
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(     
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    args["class_order"] = data_manager.get_class_order()
    learner = get_learner(args["method"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    max_accuracy = []
    last_accuracy = []
    # train for each task
    start_time = time.time()
    print("Start time: ", start_time)
    for task in range(data_manager.nb_tasks):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network), 
            count_parameters(learner._network, True))) 
        learner.incremental_train(data_manager) # train for one task
        cnn_accy, nme_accy = learner.eval_task()
        _known_classes = learner._known_classes
        _total_classes = learner._total_classes
        test_dataset_current = data_manager.get_dataset(
            np.arange(_known_classes, _total_classes), source="test", mode="test"
        )
        test_loader_current = DataLoader(
            test_dataset_current, batch_size=256, shuffle=False, num_workers=4
        )
        global_model = copy.deepcopy(learner._network)
        current_accuracy = learner._compute_accuracy(global_model, test_loader_current)
        max_accuracy.append(current_accuracy)
        print("Current accuracy: ", current_accuracy)

        learner.after_task()

        print("CNN: {}".format(cnn_accy["grouped"]))
        cnn_curve["top1"].append(cnn_accy["top1"])
        print("CNN top1 curve: {}".format(cnn_curve["top1"]))

    end_time = time.time()
    print("End time: ", end_time)
    total_time = end_time - start_time
    print("Total time in seconds: ", total_time)
    # Last Acc Test
    _known_classes = 0
    for t in range(argsn.tasks):
        _cur_task = t
        _total_classes = _known_classes + data_manager.get_task_size(
            _cur_task
        )
        print("Testing on {}-{}".format(_known_classes, _total_classes))
        test_dataset_current = data_manager.get_dataset(
            np.arange(_known_classes, _total_classes), source="test", mode="test"
        )
        test_loader_current = DataLoader(
            test_dataset_current, batch_size=256, shuffle=False, num_workers=4
        )
        acc = learner._compute_accuracy(global_model, test_loader_current)
        print(f"total_accuracy_{t}: {acc}")
        last_accuracy.append(acc)
        _known_classes = _total_classes
    forgetting = sum([max_accuracy[i] - last_accuracy[i] for i in range(argsn.tasks)]) / argsn.tasks
    print('forgetting:', forgetting)
    print('max_accuracy:', max_accuracy)
    saved = {
        "id": argsn.id,
        "note": argsn.note,
        "total_time": total_time,
        "way": argsn.way,
        "dataset": argsn.dataset,
        "model": argsn.net,
        "num_users": argsn.num_users,
        "method": argsn.method,
        "tasks": argsn.tasks,
        # "partition": argsn.partition,
        "cnn_curve": cnn_curve,
        "top1s": cnn_curve["top1"],
        "top1": cnn_accy["top1"],
        "mean_avg": float(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])),
        "last_accuracy": last_accuracy,
        "max_accuracy": max_accuracy,
        "forgetting": forgetting,
        "beta": argsn.beta,
        "com_round": argsn.com_round,
        "seed": argsn.seed,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": str(argsn),
    }
    json.dump(saved, open("jsons/{}_{}_{}_{}_{}.json".format(argsn.net, argsn.method, argsn.dataset, argsn.num_users, id), "a"))
    db("FCL_Main").insert_one(saved)




def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=0, help='1 for using wandb')
    parser.add_argument('--id', type=int, default=-1, help='1 for resume training')
    parser.add_argument('--training_id', type=int, default=-1, help='')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="TARGET", help='wandb project')
    parser.add_argument('--group', type=str, default="exp1", help='wandb group')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default="0", help='gpu id')
    parser.add_argument('--note', type=str, default="", help='note for this run')
    parser.add_argument('--shuffle', type=int, default=1, help='shuffle the data')
    

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="", help='choose a learner')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_users', type=int, default=10, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--local_ep', type=int, default=5, help='local training epochs')
    parser.add_argument('--beta', type=float, default=0.5, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--nums', type=int, default=8000, help='the num of synthetic data')
    parser.add_argument('--kd', type=int, default=25, help='for kd loss')
    parser.add_argument('--memory_size', type=int, default=2000, help='the num of real data per task')
    parser.add_argument('--way', type=int, default=3, help='Way to use UNet')
    parser.add_argument('--label_model', type=str, default="global", help='label model type: local or global')
    parser.add_argument('--diffusion_mode', type=str, default="only", help='diffusion mode: all or only current task')
    parser.add_argument('--diffusion_epochs', type=int, default=201, help='diffusion epochs')
    parser.add_argument('--sample_selection_mode', type=str, default='balance', help='sample selection mode: balance or random')
    parser.add_argument('--training_sample_selection_mode', type=str, default='balance',
                        help='sample selection mode for classifier training: balance or random')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='keep ratio for generated samples when deal with entropy')

    # Data-free Generation
    parser.add_argument('--lr_g', default=2e-3, type=float, help='learning rate of generator')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='synthetic data batch size')
    parser.add_argument('--bn', default=1.0, type=float, help='parameter for batchnorm regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='parameter for similarity')
    parser.add_argument('--adv', default=1.0, type=float, help='parameter for diversity')
    parser.add_argument('--nz', default=256, type=int, help='output size of noisy nayer')
    parser.add_argument('--warmup', default=10, type=int,
                        help='number of epoches generator only warmups not stores images')
    parser.add_argument('--syn_round', default=40, type=int, help='number of synthetize round.')
    parser.add_argument('--g_steps', default=40, type=int, help='number of generation steps.')

    # Client Training
    parser.add_argument('--num_worker', type=int, default=4, help='number of worker for dataloader')
    parser.add_argument('--mulc', type=str, default="fork", help='type of multi process for dataloader')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--syn_bs', default=1, type=int, help='number of old synthetic data in training, 1 for similar to local_bs')
    parser.add_argument('--local_lr', default=0.01, type=float, help='learning rate for optimizer')

    # LANDER
    parser.add_argument('--r', default=0.015, type=float, help='LTE center radius')
    parser.add_argument('--ltc', default=5, type=float, help='lamda_ltc parameter for LTE center')
    parser.add_argument('--pre', type=float, default=0.4, help='alpha_pre for distilling from previous task')
    parser.add_argument('--cur', type=float, default=0.2, help='alpha_cur for current task training')

    parser.add_argument('--type', default=-1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--syn', default=1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--spec', type=str, default="t1", help='choose a model')

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = args_parser()
    args.num_class = 200 if args.dataset=="tiny_imagenet" else 100
    if args.dataset == "cifar10":
        args.num_class = 10
    elif args.dataset == "emnist_letters":
        args.num_class = 26
    elif args.dataset == "svhn":
        args.num_class = 10
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls
    args.gpu = "cuda"
    if args.id == -1:
        args.id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    if args.training_id == -1:
        args.training_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"
    if args.method == "target":
        dir = "run"
        if not os.path.exists(dir):
            os.makedirs(dir) 
        args.save_dir = os.path.join(dir, args.group+"_"+args.exp_name)
    elif args.method == "lander":
        dir = "run_lander"
        args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)
    elif args.method == "ours":
        dir = "run_ours"
        if not os.path.exists(dir):
            os.makedirs(dir)
        args.save_dir = os.path.join(dir, args.group+"_"+args.exp_name+"_{}".format(args.way)+"_{}".format(args.note))
        if os.path.exists(args.save_dir):
            # 删除文件夹
            import shutil
            shutil.rmtree(args.save_dir)
    
    # if args.wandb == 1:
    #     wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
    args_v = vars(args)
    
    train(args_v, args)

