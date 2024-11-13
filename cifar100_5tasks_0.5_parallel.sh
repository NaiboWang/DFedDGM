# 切换目录到 ~/exps/FCL/TARGET
cd ~/exps/FCL/TARGET

# 并行运行任务
CUDA_VISIBLE_DEVICES=1 python main.py --group=5tasks_cifar100 --method=finetune --tasks=5 --beta=0.5 --dataset cifar100 --seed 0 &
CUDA_VISIBLE_DEVICES=1 python main.py --group=5tasks_cifar100 --method=lander --tasks=5 --beta=0.3 --dataset cifar100 --seed 0 &
CUDA_VISIBLE_DEVICES=2 python main.py --group=5tasks_cifar100 --method=target --tasks=5 --beta=0.5 --dataset cifar100 --seed 0 --nums=8000 --kd=25 --exp_name=25_kd_8k_data &

# 切换目录到 ~/exps/FCL/MFCL 并运行任务
cd ~/exps/FCL/MFCL
python main.py --dataset=CIFAR100 --seed 0 --method=MFCL --num_clients=10 --path=datasets --gpuID 3 --n_tasks=5 --beta=0.5 &

# 返回 ~/exps/FCL/TARGET 目录并并行运行其他任务
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=4 python main.py --group=5tasks_cifar100 --method=fedprox --tasks=5 --beta=0.5 --dataset cifar100 --seed 0 &
CUDA_VISIBLE_DEVICES=5 python main.py --group=5tasks_cifar100 --method=lwf --tasks=5 --beta=0.5 --dataset cifar100 --seed 0 &
CUDA_VISIBLE_DEVICES=6 python main.py --group=5tasks_cifar100 --method=ewc --tasks=5 --beta=0.5 --dataset cifar100 --seed 0 &
CUDA_VISIBLE_DEVICES=7 python main.py --group=5tasks_cifar100 --method=ours_client_diffusion --tasks=5 --beta=0.5 --memory_size 1000 --num_users 10 --dataset cifar100 --id 11080 --note 3kdloss --label_model global --sample_selection_mode balance --keep_ratio 1.0 --way 3 --seed 0 &