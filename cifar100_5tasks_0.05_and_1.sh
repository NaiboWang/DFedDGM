cuda=1
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=finetune --tasks=5 --beta=0.05 --dataset cifar100
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=target --tasks=5  --beta=0.05 --dataset cifar100 --nums=8000 --kd=25 --exp_name=25_kd_8k_data
cd ~/exps/FCL/MFCL
python main.py --dataset=CIFAR100 --method=MFCL --num_clients=10 --path=datasets --gpuID $cuda --n_tasks=5 --beta=0.05
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=fedprox --tasks=5 --beta=0.05 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=lwf --tasks=5 --beta=0.05 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=ewc --tasks=5 --beta=0.05 --dataset cifar100

cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=finetune --tasks=5 --beta=1 --dataset cifar100
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=target --tasks=5  --beta=1 --dataset cifar100 --nums=8000 --kd=25 --exp_name=25_kd_8k_data
cd ~/exps/FCL/MFCL
python main.py --dataset=CIFAR100 --method=MFCL --num_clients=10 --path=datasets --gpuID $cuda --n_tasks=5 --beta=1
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=fedprox --tasks=5 --beta=1 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=lwf --tasks=5 --beta=1 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_cifar100 --method=ewc --tasks=5 --beta=1 --dataset cifar100

# CUDA_VISIBLE_DEVICES=2 python main.py --group=5tasks_cifar100 --method=ours_client_diffusion --tasks=5 --beta=0.05 --memory_size 2000 --num_users 10 --dataset cifar100 --id 1080 --note 3kdloss  --label_model global --sample_selection_mode balance --keep_ratio 1.0 --way 3

# CUDA_VISIBLE_DEVICES=3 python main.py --group=5tasks_cifar100 --method=ours_client_diffusion --tasks=5 --beta=1 --memory_size 2000 --num_users 10 --dataset cifar100 --id 1090 --note 3kdloss  --label_model global --sample_selection_mode balance --keep_ratio 1.0 --way 3
