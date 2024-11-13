# Reproducing
We test the code on V100 GPU with pytorch: 
```
torch==2.0.1
torchvision==0.15.2
```

## Baseline
Here, we provide a simple example for different methods. 
For example, for `cifar100-5tasks`, please run the following commands to test the model performance with non-IID (`$\beta=0.5$`) data.

```
#!/bin/bash
# method= ["finetue", "lwf", "ewc", "icarl"]

CUDA_VISIBLE_DEVICES=0 python main.py --wandb=1 --group=5tasks_cifar100 --method=$method --tasks=5 --beta=0.5
```

### Ours
```
CUDA_VISIBLE_DEVICES=0 python main.py --wandb=1 --group=5tasks_cifar100 --method=ours_client_diffusion --tasks=5  --beta=0.5 --nums=8000 --kd=25 --exp_name=25_kd_8k_data
```