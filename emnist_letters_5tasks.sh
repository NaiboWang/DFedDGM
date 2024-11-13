cuda=6
#cd ~/exps/FCL/TARGET
#CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_emnist_letters --method=finetune --tasks=5 --beta=0.5 --dataset emnist_letters
cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_emnist_letters --method=target --tasks=5  --beta=0.5 --dataset emnist_letters --nums=8000 --kd=25 --exp_name=25_kd_8k_data
#cd ~/exps/FCL/MFCL
#python main.py --dataset=emnist_letters --method=MFCL --num_clients=10 --gpuID $cuda --n_tasks=5
#cd ~/exps/FCL/TARGET
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_emnist_letters --method=fedprox --tasks=5 --beta=0.5 --dataset emnist_letters
CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_emnist_letters --method=lwf --tasks=5 --beta=0.5 --dataset emnist_letters
#CUDA_VISIBLE_DEVICES=$cuda python main.py --group=5tasks_emnist_letters --method=ewc --tasks=5 --beta=0.5 --dataset emnist_letters
