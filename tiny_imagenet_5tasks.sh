cuda=6
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_tinyImagenet --method=finetune --tasks=5 --beta=0.5 --dataset tiny_imagenet
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_tinyImagenet --method=target --tasks=5  --beta=0.5 --dataset tiny_imagenet --nums=8000 --kd=25 --exp_name=25_kd_8k_data
cd ~/exps/FCL/MFCL
python main.py --dataset=tiny_imagenet --method=MFCL --num_clients=10 --n_tasks=5 --gpuID $cuda
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_tinyImagenet --method=fedprox --tasks=5 --beta=0.5 --dataset tiny_imagenet
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_tinyImagenet --method=lwf --tasks=5 --beta=0.5 --dataset tiny_imagenet
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_tinyImagenet --method=ewc --tasks=5 --beta=0.5 --dataset tiny_imagenet