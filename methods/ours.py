import numpy as np
import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from methods.base import BaseLearner
from utils.data_manager import partition_data, DatasetSplit, average_weights, setup_seed
import copy, wandb
from abc import ABC
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image
from models.diffusion_model import DiffusionGenerator


dataset = "cifar100"

if dataset =="cifar100":
    synthesis_batch_size = 4
    sample_batch_size = 4
    g_steps=20
    is_maml=1
    kd_steps=400    
    warmup=20
    # lr_g=0.002
    lr_g=2e-5
    lr_z=0.01
    oh=0.5
    T=20.0
    act=0.0
    adv=1.0
    bn=10.0
    reset_l0=1
    reset_bn=0
    bn_mmt=0.9
    syn_round=200
    tau=1
    data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**dict(data_normalize)),
    ])
    
else:
    synthesis_batch_size = 256
    sample_batch_size = 256
    g_steps=50  
    is_maml=0   
    kd_steps=400     
    warmup=20
    lr_g=0.0002 
    lr_z=0.01   
    oh=0.1  
    T=5     
    act=0.0 
    adv=1.0 
    bn=0.1  
    reset_l0=0 
    reset_bn=0 
    bn_mmt=0.9  
    syn_round=200  
    tau=1
    data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**dict(data_normalize)),
    ])

    

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                # random.shuffle(files)
                files = files[:nums]
                # files = files[20*256:20*256+nums]       # discard the ealry-stage data
                # files = files[-nums:]  # 40*256 
            for f in files:
                if f.endswith(pos):
                    images.append( os.path.join( dirpath, f ) )
    return images


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if tar_p.grad is not None:
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


def reset_l0_fun(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def save_image_batch(imgs, output, col=None, size=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                        self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data)

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False)
        self._idx+=1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()

def show_image(img, title=""):
    from matplotlib import pyplot as plt
    """Helper function to display an image"""
    img = img.detach().cpu().numpy()
    img = img.clip(0, 1)
    # img = img.cpu().numpy()
    plt.imshow(img.transpose(1, 2, 0))
    plt.title(title)
    plt.show()

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class GlobalSynthesizer(ABC):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                    init_dataset=None, iterations=100, lr_g=2e-5,
                    synthesis_batch_size=128, sample_batch_size=128, 
                    adv=0.0, bn=1, oh=1,
                    save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                    normalizer=None, distributed=False, lr_z = 0.01,
                    warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                    is_maml=1, args=None, n_steps = 1000):
        super(GlobalSynthesizer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ismaml = is_maml
        self.args = args

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])


        self.aug = transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
        
        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt) )
                # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).cuda()

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    def synthesize(self, targets=None):
        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120+self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)
        
        best_inputs = None
        # z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
        # z.requires_grad = True
        x = torch.randn([self.synthesis_batch_size, 3, 32, 32]).cuda()
        x.requires_grad = True
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.cuda()

        fast_generator = self.generator.clone()
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [x], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        optimizer = torch.optim.Adam(fast_generator.parameters(), lr=self.lr_g, betas=[0.5, 0.999])
        '''
            三种方式：
            1. UNet直接生成图像
                1.1 x保持不变
                1.2 x每一轮都变一下
            2. UNet经过一步即可生成完美噪音，去除之后得到原图
            3. 需要多步去噪音操作
        '''
        copy_generator = fast_generator.clone()
        copy_generator.eval()
        for it in range(self.iterations):
            # inputs = fast_generator(x)
            # 这里生成的x我们默认是经过1000步的扩散的，所以我们需要将x进行逆扩散
            # 训练时，只有第一步需要计算梯度，后面的步骤都不需要计算梯度
            t = self.n_steps - 1
            ts = x.new_full((self.synthesis_batch_size,), t, dtype=torch.long)
            if self.args["way"] == 1:
                input = fast_generator(x, ts) # 这里一定要用input=，不能用x=
            elif self.args["way"] >= 2:
                eps_theta = fast_generator(x, ts)
                # with torch.no_grad():
                alpha_bar = gather(self.alpha_bar, ts)
                alpha = gather(self.alpha, ts)
                eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
                mean = 1 / (alpha ** 0.5) * (x - eps_coef * eps_theta)
                var = gather(self.sigma2, ts)
                eps = torch.randn(x.shape, device=x.device)
                input = mean + (var ** .5) * eps
                if self.args["way"] == 3:
                    # 这里是否能够这样做？
                    # 其余的步骤都不需要计算梯度
                    # input_t = input.clone()
                    copy_generator.load_state_dict(fast_generator.state_dict())
                    for t_ in range(self.n_steps-1):
                        t = self.n_steps - t_ - 1
                        ts = input.new_full((self.synthesis_batch_size,), t, dtype=torch.long)
                        with torch.no_grad(): # 只能写在这，如果写在外面，会导致input的梯度不更新
                            eps_theta = copy_generator(input, ts)
                            alpha_bar = gather(self.alpha_bar, ts)
                            alpha = gather(self.alpha, ts)
                            eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
                            var = gather(self.sigma2, ts)
                            eps = torch.randn(input.shape, device=input.device)
                        mean = 1 / (alpha ** 0.5) * (input - eps_coef * eps_theta)
                        input = mean + (var ** .5) * eps
                        # if t_ % 100 == 0:
                            # print("t: ", t_, "/", self.n_steps-1)
            inputs_aug = self.aug(input) # crop and normalize
            # writer = SummaryWriter('runs/simple_model')
            # writer.add_graph(fast_generator, input)
            if it == 0:
                originalMeta = input
            t_out = self.teacher(inputs_aug)["logits"]
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.cuda()

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)
            if self.adv>0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            # loss = loss_oh
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = input.data.cpu() # mem, cpu
                    print("Best Inputs at it: ", it, " loss: ", loss.item())
                    # save_data = best_inputs.clone()
                    # vutils.save_image(save_data[:200], 'real_{}.png'.format(dataset), normalize=True, scale_each=True, nrow=20)

            for (f_para, t_para) in zip(fast_generator.parameters(), self.generator.parameters()):
                print("Before", it, "/", self.iterations, " loss: ", loss.item())
                print("Parameters:", f_para.data[0][0][0][0], t_para.data[0][0][0][0])
                print("x:", x[0][0][0][0])
                break
            optimizer.zero_grad()
            loss.backward()
            for (f_para, t_para) in zip(fast_generator.parameters(), self.generator.parameters()):
                break



            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations-1): self.meta_optimizer.step()

            optimizer.step()
            for (f_para, t_para) in zip(fast_generator.parameters(), self.generator.parameters()):
                print("After")
                print("Parameters:", f_para.data[0][0][0][0], t_para.data[0][0][0][0])
                print("x:", x[0][0][0][0])
                break

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (x, targets)
        end = time.time()

        self.data_pool.add( best_inputs )       # add a batch of data


        

'''
    这个 weight_init 函数是用于初始化神经网络中各种层类型权重(weight)的函数。该函数设计用于与 PyTorch 的神经网络模块一同使用。通过将 weight_init 函数应用到 PyTorch 模型的实例上（如通过 model.apply(weight_init)），可以根据每一种层的类别初始化其参数。
    
    函数中使用 isinstance() 检查传入的模块 m 是否为特定的层类型，并根据层的类型调用不同的初始化函数。以下是按层类型分类的初始化策略：
    
    卷积层 (nn.Conv1d, nn.Conv2d, nn.Conv3d):
    nn.Conv1d: 使用正态分布 (init.normal_) 初始化权重和偏置（bias）。
    nn.Conv2d: 使用Xavier正态分布 (init.xavier_normal_) 初始化权重，使用正态分布初始化偏置。
    nn.Conv3d: 和nn.Conv2d相同。
    
    转置卷积层 (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
    
    nn.ConvTranspose1d: 使用正态分布初始化权重和偏置。
    nn.ConvTranspose2d: 使用Xavier正态分布初始化权重，使用正态分布初始化偏置。
    nn.ConvTranspose3d: 和nn.ConvTranspose2d相同。
    
    批量归一化层 (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d):
    
    初始化权重为均值为1，标准差为0.02的正态分布。
    偏置初始化为常数0。
    
    全连接层 (nn.Linear):
    
    权重使用Xavier正态分布初始化。
    偏置使用正态分布初始化。
    
    循环神经网络 (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell):
    
    对于多维参数，使用正交初始化 (init.orthogonal_)。
    对于其他参数，使用正态分布初始化。
    其中，init 通常为 torch.nn.init 模块的别名，该模块包含了一系列初始化方法。
    
    初始化网络参数是深度学习中的一个重要步骤，它可以影响网络的收敛速度以及最终的性能。不同类型的初始化方法适合不同的层类型和激活函数。例如，Xavier初始化（也称为Glorot初始化）通常用于与sigmoid和tanh激活函数搭配使用的网络层，而正态和正交初始化常用于循环神经网络的权重初始化，以帮助避免梯度消失或爆炸问题。
'''
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    init是nn.Module的模块，用于初始化权重
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

'''
这个Python函数名为refine_as_not_true，其目的是在给定的logits（通常是机器学习模型的未缩放输出）与对应的目标标签后处理，将目标类别的logits排除以便在多分类问题中仅处理非目标（即非正确答案）的类别logits。

函数接收三个参数： - logits：这是一个PyTorch张量，通常大小是(batch_size, num_classes)，表示模型的未缩放输出。 - targets：一个大小为(batch_size)的张量，包含每个样本的正确类别的整数标签。 - num_classes：一个整数，表示类别的数量。

函数的处理流程如下： 1. nt_positions = torch.arange(0, num_classes).cuda(): 创建一个从0到num_classes-1的整数的一维张量，并将其移到GPU（前提是CUDA可用）。

nt_positions = nt_positions.repeat(logits.size(0), 1): 对这个张量进行重复，以便于每个批次中的样本都有一份，将它的维度变为(batch_size, num_classes)。

nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]: 这是一个过滤操作，去除掉每个样本对应目标类别的索引。这里，targets被重塑成(batch_size, 1)并与nt_positions的每行比较。不等于对应targets的位置被保留。结果是一个扁平化的一维张量，大小是batch_size * (num_classes - 1)。

nt_positions = nt_positions.view(-1, num_classes - 1): 将上一步得到的一维张量重塑成(batch_size, num_classes - 1)。此时，对于每个样本来说，这个二维张量只包含非目标类别的索引。

logits = torch.gather(logits, 1, nt_positions): 使用torch.gather函数根据nt_positions提取logits张量中的条目。对于每个样本，仅在那些非目标的类别上收集logits。

最后，函数返回修改后的logits张量，现在每个样本都只包含其对应的非目标类别的logits。

例如，在一个5分类问题中，如果有一个批次大小为2的样本，targets可能是 [2, 3]，这个函数将排除掉索引为2的类别logit对第一个样本，和索引为3的类别logit对第二个样本，只保留其他非目标类别的logits。

这一处理有助于排除对后续计算不感兴趣的类别，例如在计算对数似然或者其他特定于类别的损失函数时，有时需要排除正确类别的影响。
'''
def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits


class OURS(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
                
    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()

    def kd_train(self, student, teacher, criterion, optimizer):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data() 
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().cuda()  
            with torch.no_grad():
                t_out = teacher(images)["logits"]
            s_out = student(images.detach())["logits"]
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()
            loss_s.backward()
            optimizer.step()

    def data_generation(self):
        nz = 256
        img_size = 32 if self.args["dataset"] == "cifar100" else 64
        if self.args["dataset"] == "imagenet100": img_size = 128 
            
        img_shape = (3, 32, 32) if self.args["dataset"] == "cifar100" else (3, 64, 64)
        if self.args["dataset"] == "imagenet100": img_shape = (3, 128, 128) #(3, 224, 224)
        # generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).cuda()
        generator = DiffusionGenerator().cuda()
        student = copy.deepcopy(self._network)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir) 
        synthesizer = GlobalSynthesizer(copy.deepcopy(self._network), student, generator,
                    nz=nz, num_classes=self._total_classes, img_size=img_shape, init_dataset=None,
                    save_dir=tmp_dir,
                    transform=train_transform, normalizer=normalizer,
                    synthesis_batch_size=synthesis_batch_size, sample_batch_size=sample_batch_size,
                    iterations=g_steps, warmup=warmup, lr_g=lr_g, lr_z=lr_z,
                    adv=adv, bn=bn, oh=oh,
                    reset_l0=reset_l0, reset_bn=reset_bn,
                    bn_mmt=bn_mmt, is_maml=is_maml, args=self.args)
        
        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=0.0001,
                            momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        for it in range(syn_round):
            synthesizer.synthesize() # generate synthetic data
            if it >= warmup:
                self.kd_train(student, self._network, criterion, optimizer) # kd_steps
                test_acc = self._compute_accuracy(student, self.test_loader)
                print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                    self._cur_task, it + 1, syn_round, test_acc,))
                scheduler.step()
                # wandb.log({'Distill {}, accuracy'.format(self._cur_task): test_acc})


        print("For task {}, data generation completed! ".format(self._cur_task))  

            
    def get_syn_data_loader(self):
        if self.args["dataset"] =="cifar100":
            dataset_size = 50000
        elif self.args["dataset"] == "tiny_imagenet":
            dataset_size = 100000
        elif self.args["dataset"] == "imagenet100":
            dataset_size = 130000 
        iters = math.ceil(dataset_size / (self.args["num_users"]*self.args["tasks"]*self.args["local_bs"]))
        syn_bs = int(self.nums/iters)
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task-1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True,
            num_workers=4, pin_memory=True, )
        return syn_data_loader

    def get_all_syn_data(self): # 有点歧义，应该不是所有的，而是cur_task的
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)
        return loader



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
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=4
        )
        setup_seed(self.seed)
        if self._cur_task == 0 and (not os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir)
        if self._cur_task != 0: 
            # get syn_data for old tasks
            self.syn_data_loader = self.get_syn_data_loader()
        
        # for all tasks
        self._fl_train(train_dataset, self.test_loader, self._cur_task)
        if self._cur_task+1 != self.tasks:
            self.data_generation()


    def _local_update(self, model, train_data_loader):
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



    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id):
        # global print_flag
        model.train()
        teacher.eval()
        # w_l = [0,80,100,100,150] # id=1,2,3,4 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for it in range(self.args["local_ep"]):
            iter_loader = enumerate(zip((train_data_loader), (self.syn_data_loader)))
            total_local = 0.0
            total_syn = 0.0
            for batch_idx, ((_, images, labels), syn_input) in iter_loader:
                images, labels, syn_input = images.cuda(), labels.cuda(), syn_input.cuda()
                fake_targets = labels - self._known_classes
                output = model(images)["logits"]
                # for new tasks
                loss_ce = F.cross_entropy(output[:, self._known_classes :], fake_targets)
                s_out = model(syn_input)["logits"]
                with torch.no_grad():
                    t_out = teacher(syn_input.detach())["logits"]
                    total_syn += syn_input.shape[0]
                    total_local += images.shape[0]
                # for old task
                loss_kd = _KD_loss(
                    s_out[:, : self._known_classes],   # logits on previous tasks
                    t_out.detach(),
                    2,
                )
                loss = loss_ce + self.args["kd"]*loss_kd 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return model.state_dict(), total_syn, total_local


    def _fl_train(self, train_dataset, test_loader, task_id):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            if os.path.exists(os.path.join("checkpoints", "task_{}_com_{}.pth".format(self._cur_task, com))) and self._cur_task == 0:
                self._network.load_state_dict(torch.load(os.path.join("checkpoints", "task_{}_com_{}.pth".format(self._cur_task, com))))
                print("Load model from task_{}_com_{}.pth".format(self._cur_task, com))
                if com == 99:
                    stat = self._network.state_dict()
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = ("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                        self._cur_task, com + 1, self.args["com_round"], test_acc, ))
                    prog_bar.set_description(info)
                continue
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                if self._cur_task == 0:
                    w = self._local_update(copy.deepcopy(self._network), local_train_loader)
                else:
                    w, total_syn, total_local = self._local_finetune(self._old_network, copy.deepcopy(self._network), 
                        local_train_loader, self._cur_task, idx)
                    if com == 0 and self._cur_task == 1:
                        print("\t \t client {}, local dataset size:{},  syntheic data size:{}".format(idx, total_local, total_syn))

                local_weights.append(copy.deepcopy(w))
                del local_train_loader, w
                torch.cuda.empty_cache() 
                
            # update global weights
            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)
            # Save the model
            torch.save(global_weights, os.path.join("checkpoints", "task_{}_com_{}.pth".format(self._cur_task, com)))

            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

