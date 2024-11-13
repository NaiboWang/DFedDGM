cuda=5
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=10tasks_cifar100 --method=finetune --tasks=10 --beta=0.5 --dataset cifar100
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=10tasks_cifar100 --method=target --tasks=10  --beta=0.5 --dataset cifar100 --nums=8000 --kd=25 --exp_name=25_kd_8k_data
cd ~/exps/FCL/MFCL
python main.py --dataset=CIFAR100 --method=MFCL --num_clients=10 --path=datasets --gpuID $cuda --n_tasks 10
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=10tasks_cifar100 --method=fedprox --tasks=10 --beta=0.5 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=10tasks_cifar100 --method=lwf --tasks=10 --beta=0.5 --dataset cifar100
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=10tasks_cifar100 --method=ewc --tasks=10 --beta=0.5 --dataset cifar100
