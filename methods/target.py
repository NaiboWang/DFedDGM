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
from abc import ABC
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image


dataset = "cifar100"

if dataset =="cifar100":
    synthesis_batch_size = 256
    sample_batch_size = 256
    g_steps=10  
    is_maml=1
    kd_steps=400    
    warmup=20
    lr_g=0.002
    lr_z=0.01
    oh=0.5
    T=20.0
    act=0.0
    adv=1.0
    bn=10.0
    reset_l0=1
    reset_bn=0
    bn_mmt=0.9
    syn_round= 10 
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

    """
        在Python中，__repr__ 是一个特殊的方法，用于定义对象的“官方”字符串表示形式。这个表示形式通常应该是明确的，并且尽可能地表达出足够信息，以便使用这个字符串表示可以重新创建出那个对象。
    
        例如，若有一个点（Point）类，它有两个属性：x 和 y，表示点在二维空间中的横纵坐标。它的 __repr__ 方法可能看起来是这样的：
        
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
            def __repr__(self):
                return f"Point({self.x}, {self.y})"
         
        当你创建一个Point对象并在解释器中打印它，或者使用内置的 repr() 函数时，Python就会调用这个 __repr__ 方法：
        
        p = Point(1, 2)
        print(p)                           # 输出：Point(1, 2)
        print(repr(p))                     # 输出：Point(1, 2)
         
        __repr__ 不仅有助于调试，因为它提供了一个对象状态的快照，而且在需要的时候，它也可以用于实现对象的自我复制，例如，通过 eval(repr(obj)) 方式。
        
        注意，还有一个相关的方法叫 __str__，它也是用于定义对象的字符串表示，但 __str__ 的目的是可读性，给用户看的，而不是为了明确地表示出对象的全部信息用于再现对象。 如果一个对象没有定义 __str__ 方法，那么Python会回退到使用 __repr__ 作为其字符串表示。
    """
    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

"""
这个Python函数pack_images的目的是将一批图像打包到一个单一的大图像中，这样就可以在单个网格中一起可视化它们。这种方法在机器学习和计算机视觉中很常见，尤其是在监测模型输出或进行数据探索时。

函数的输入参数包括： 
- images: 一个包含多个图像的Numpy数组或图像列表（形状为N x C x H x W 或 N x H x W x C）。 
- col: 指定每行应该有多少列图片。如果不指定，默认创建一个近似正方形的布局（尽可能等分），使得行数和列数大致相等。 
- channel_last: 一个布尔值，如果设置为True，表示输入图像Numpy数组是采用通道在最后的格式（N x H x W x C）。默认情况下，函数期望的输入是通道在前的格式（N x C x H x W）。 
- padding: 一个整数，指定图片间隔的像素数。

函数执行以下步骤：
如果images是一个列表或元组，那么它将被转换成一个Numpy数组。
如果channel_last参数为True，那么输入图像会被从通道在最后的格式转变为通道在前的格式。
函数确认输入图像是一个四维数组（即有一个批次维度和三个空间维度：批次、通道、高、宽）。
然后函数计算每行应该显示多少图像，如果未指定列数col，则使用图像数量的平方根来估算。
接下来，根据行数、列数、图像尺寸以及填充来确定打包图像的总体尺寸。
创建一个足够大的空图像（pack）来包含所有小图像，用0（一般代表黑色）填充。
函数遍历每个图像，并计算每个图像在pack中的位置，然后将其复制到相应位置。
函数返回这个大的打包图像。
这个函数能够处理灰度图像（C=1）或RGB图像（C=3），具体取决于输入图像的通道数。最终返回的打包图像pack可以很方便地使用图像处理库来显示或保存为文件。
"""

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

"""
reptile_grad 函数似乎是为了在机器学习环境下，使用 PyTorch 库而设计的。这个函数可能是为了实施 Reptile 元学习算法的一个操作，这是一种简化版的模型快速适应（model-agnostic meta-learning，简称 MAML）。
这个函数接收两个参数 src 和 tar。这两个参数预期是包含可学习参数（比如神经网络的权重）的对象，这些对象具有.parameters()方法，该方法返回一个迭代器，在 PyTorch 中通常是模型参数的迭代器。
函数的核心操作流程如下：
对 src 和 tar 的参数进行遍历：用 zip(src.parameters(), tar.parameters()) 一一对应这些参数。
检查 src 中每个参数的梯度 (p.grad) 是否为 None。如果是 None，就初始化一个与参数同形状的全零变量，并将其放置到 CUDA（GPU）上以进行加速计算。
更新 src 中的参数梯度：p.grad.data.add_ 是一个原地操作（in-place operation），这意味着它会直接修改 p.grad.data 的值，而不是创建一个新的变量。
每个参数梯度都加上 p.data - tar_p.data 的结果，并且乘以一个学习率 alpha。在你提供的代码中，这个学习率被设置为 67。这个操作看起来是在尝试减小 src 参数与 tar 参数之间的差异。
注意：此代码只是对 PyTorch 自动梯度引擎的简单封装，用来自定义梯度更新策略。其在具体的深度学习模型中是如何适用的，则需要更加详细的背景信息来解释。此外，代码的注释 # , alpha=40 表示原始代码可能在测试不同的学习率，但这一部分在当前提供的代码中已经被注释掉。
"""
def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


"""
函数 fomaml_grad 也是一个用于机器学习和 PyTorch 库的函数，它的作用看起来与一阶模型无关元学习（First-Order Model-Agnostic Meta-Learning，简称 FOMAML）算法有关。FOMAML 是 MAML（Model-Agnostic Meta-Learning）算法的一种变体，它省略了二阶微分项以减少计算复杂性，特别是在执行后向传播计算梯度时。
代码详解如下：
src 和 tar 参数预期分别是源模型和目标模型的对象。这两个对象通过 .parameters() 方法提供对其参数的引用，通常来说这两个对象都是深度神经网络模型。
代码通过 zip(src.parameters(), tar.parameters()) 同时迭代两个模型的参数。
如果 src 参数对象中的梯度属性 .grad 是 None （即尚未初始化），则创建一个与参数同形状的全零变量，并且将其转移到 GPU 上，为后续的梯度累加作准备。
p.grad.data.add_(tar_p.grad.data) 这一行将 tar 中对应参数的梯度累加到 src 参数的梯度上。这意味着，目标模型的梯度被直接用于修改源模型的梯度。
注释中的 # , alpha=0.67 一样可能表示先前存在一个实验性尝试，用来乘以一个缩放因子（学习率），但在最终的代码中这一部分已经不再使用。
FOMAML 相较于 MAML 节省计算资源的特性使其在特定场景下更加受欢迎，尤其是当二阶梯度项的准确性对于最终性能的影响不大时。
在实际情境中，fomaml_grad 函数通常会被用在元学习训练循环的梯度更新步骤中，以反映从多个任务中学来的知识。函数的作用是将训练完成后任务模型的梯度回传，更新初始模型（即源模型）的参数梯度。
"""
def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


"""
在Python中，reset_l0_fun 函数看起来是用来重置一个神经网络模型中特定层的权重和偏差的函数。为了能够更好地分析这个函数，首先需要确认它是使用了 PyTorch 这个深度学习库，从代码中调用 nn.init.normal_ 和 nn.init.constant_ 函数可以看出这一点。
让我们逐步分析一下这个函数：
函数定义：reset_l0_fun(model) 定义了一个名为 reset_l0_fun 的函数，该函数接受一个参数 model。这个 model 应该是一个 PyTorch 神经网络模型。
循环结构：for n, m in model.named_modules(): 这一行开始了一个循环，遍历模型中的所有模块。在 PyTorch 中，named_modules() 方法返回一个迭代器，包含模型中所有模块的名字（n）和模块本身（m）。
条件语句：if n == "l1.0" or n == "conv_blocks.0": 通过列表示，这行代码检查当前模块的名字是否为 "l1.0" 或者 "conv_blocks.0"。如果是的话，下面的初始化代码将会被执行。
权重初始化：nn.init.normal_(m.weight, 0.0, 0.02) 这行初始化模块 m 的权重，使用均值为 0.0，标准差为 0.02 的正态分布。这是 PyTorch 中的就地操作（in-place operation）。normal_ 表示将权重初始化为正态分布。
偏差初始化：nn.init.constant_(m.bias, 0) 这一行用一个常数 0 初始化模块 m 的偏差。
从函数名和代码行为来推断，这个函数可能是用于重置神经网络某个初始化过程中的第一层 (l1.0) 和可能是某种卷积块 (conv_blocks.0) 的权重和偏置。这样的重置可能是为了实验上的需求、调试、或者在训练开始前要确保从同一预设的初始化分布开始。这是在深度学习中常见的做法，特别是当进行比较敏感于初始化的实验时。
"""
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


"""
这个类 DeepInversionHook 是一个 PyTorch 前向钩子的实现，设计用于跟踪神经网络层中的特征统计并基于这些统计计算损失。它旨在通过强制实现特征分布的匹配来进行正则化。具体来说，该钩子追踪层的特征的均值和方差，并通过计算目标分布与实际分布之间的距离来感知损失。

类的详细分析如下：

__init__(self, module, mmt_rate): 类的初始化方法，接收两个参数：module 是被附加钩子的模块（通常是神经网络的某一层），mmt_rate 是一个动量率，用于更新统计数据的移动平均。
hook = module.register_forward_hook(self.hook_fn): 在给定的 module 上注册一个前向钩子，当 module 执行前向传播时，会调用 hook_fn。
hook_fn(self, module, input, output): 是钩子的回调函数。它计算输入特征图的平均值和方差，并用这些统计数据与模块的运行统计数据（running_mean 和 running_var）进行比较，计算它们之间的 L2 范数作为一个正则化损失项 (r_feature)。
update_mmt(self): 更新统计数据的动量。如果是第一次，就初始化 self.mmt，否则就用当前计算的均值和方差更新移动平均。
remove(self): 移除注册的钩子。
总的来说，这个类是深度反演（Deep Inversion）技术的一个组成部分，用于在没有访问原始训练数据的情况下，通过网络内部的特征分布来构建合成图像。这样做可以帮助理解网络对于输入数据统计数据的依赖，并且在某些情况下有助于模型的正则化和训练。
"""
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

    """
    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
    这一行获取输入数据的通道数 nch。这是通过 input[0].shape[1] 来实现的，其中 input 是到达前向钩子的数据的元组，input[0] 是实际的张量数据。
        
        mean = input[0].mean([0, 2, 3])
    对输入数据的批次、高度、宽度轴（索引 0, 2, 3）求平均，只保留通道维度的均值。这个均值将用来计算损失。
        
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
    首先对输入数据进行 permute 来交换维度，使通道维度变为第一个维度。然后 flattens 除通道维度外的其他维度，这样每个通道变成一维数组。最后对每个通道的一维数组计算方差（指定 unbiased=False 表示方差计算不应该使用无偏估计器）。
        if self.mmt is None:
            ...
        else:
            ...
     
    这部分代码检查是否已经存在移动平均的统计量（self.mmt）。如果没有（self.mmt is None），则直接使用模块的当前统计量计算损失；如果有，则使其参与到损失计算中。
    其中损失的计算是通过以下方式进行的：
    
    r_feature = torch.norm(module.running_var.data - var, 2) + \
                torch.norm(module.running_mean.data - mean, 2)
     
    如果 self.mmt 还未初始化，损失 r_feature 是由当前层的方差（module.running_var.data）和均值（module.running_mean.data）与输入特征的实时方差和均值之间的 L2 范数（Euclidean distance）相加得到的。
    mean_mmt, var_mmt = self.mmt
    r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)
     
    如果 self.mmt 已经初始化，损失 r_feature 会考虑动量更新。此时，损失是当前统计量和经过动量率平滑后的输入特征统计量之间的 L2 范数。
    最后，我们有：
    
    self.r_feature = r_feature
    self.tmp_val = (mean, var)
     
    上述计算出的损失被赋值给 self.r_feature，以便之后可以访问它，并进行后续损失计算。同时，当前输入的均值和方差被暂时存储在 self.tmp_val 中，用于执行动量更新。
    """
    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            """
            在PyTorch 中，running_mean 和 running_var 是在使用 torch.nn.BatchNorm 层时用来跟踪数据的均值和方差的参数。Batch Normalization（批标准化）是一种用于深度学习的技术，可以加速深度网络的训练，同时使其对初始化配置不那么敏感。

            在批标准化中，算法会对每一批数据进行归一化处理，确保数据的均值为0，方差为1。这通常有助于避免训练过程中的内部协变量偏移（Internal Covariate Shift），并可以允许使用更高的学习率以及减少对初始化的依赖。
            
            running_mean 和 running_var 在训练过程中是动态计算和更新的。它们分别累计了在训练过程中见到的所有批次数据的均值和无偏方差（采用移动平均的方式计算）。当模型转入评估模式（model.eval()）时，这些累积的均值和方差将被用来对数据进行标准化，因为在推理过程中我们不再有整个的批次数据。这意味着我们不能计算当前批次的均值和方差，所以使用训练过程中累积的统计量来代替。
            """
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    """
    update_mmt 方法
    def update_mmt(self):
        mean, var = self.tmp_val
     
    从临时变量 self.tmp_val 中获取最近计算的均值和方差。
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
     
    如果之前没有动量统计量，则用当前的均值和方差直接初始化它。否则提取现有的动量均值和方差。
    接下来是动量更新部分：
    
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                        self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )
     
    这里对均值和方差统计量进行动量更新。更新的动量统计量是之前统计量的 self.mmt_rate 比例和当前统计量的 (1-self.mmt_rate) 比例组合的加权平均。
    这样，经过 hook_fn 和 update_mmt 的操作，DeepInversionHook 可以追踪并更新模块的特征统计量，从而在不需要实际输入数据的情况下，通过正则化来引导合成数据的生成，这在数据隐私或者数据不足的场景中非常有用。
    """
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


"""
这个类是一个神经网络模型定义，属于生成对抗网络（GAN）中的生成器（Generator）部分。生成对抗网络通常包含两部分：生成器（Generator）和判别器（Discriminator）。生成器的目标是生成尽可能真实的图像，而判别器的目标是区分真实图像和生成的图像。

对于给出的 Generator 类，它使用 PyTorch 框架定义了一个生成器网络。这里是该类定义的简要概述：

__init__ 方法初始化生成器。它接受以下参数：
nz：噪声向量的维度，这是生成器的输入。
ngf：生成器特征映射的大小。
img_size：生成图像的尺寸。
nc：生成图像的通道数，对于彩色图像，通常是 3（表示 RGB）。

init_size 是图像初始尺寸的一个内部参数，它由原始图像大小除以 4 得到。这是因为网络会逐渐上采样输入噪声向量以生成最终图像。

l1 是一个序列层，其中包含一个线性层，将噪声向量映射到一组特征上。

conv_blocks 是一个序列化的层堆叠，包括批量归一化（BatchNorm）、上采样（Upsample）、二维卷积层（Conv2d）、泄露的线性整流函数（LeakyReLU），逐渐上采样特征映射并转换成最终的图像。

forward 方法定义了该网络的前向传递。给定一个噪声向量 z，通过序列化的线性层，它首先将这个噪声处理成一个特征图，然后通过卷积块将其处理成最终的图像。

clone 方法用于创建该生成器的一个克隆，复制其权重并返回这个新克隆的生成器，通常用于参数隔离和避免在原始模型上作出破坏性改动。

在 GAN 训练过程中，生成器会接收一个从某种分布（如高斯分布）中随机生成的噪声向量作为输入，通过网络生成一张图像，然后这张图像会被送给判别器进行真伪判定。根据判别器的反馈，生成器会调整其参数以生成更加真实的图像。在这个过程中，生成器学会生成与训练数据集相似的新图像，而判别器的目标是逐渐变得更擅长区分真实图像和生成器产生的伪造图像。
"""

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


"""
定义的 kldiv 函数使用 Python 语言编写，它计算了两个概率分布之间的Kullback-Leibler散度（KL散度），这是测量一个概率分布相对于另一个参考概率分布的不同程度的一种手段。

让我们逐行分析这段代码：

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
 
def kldiv(...): 是函数定义的起始，kldiv 是函数名。
logits 通常指未归一化的模型预测结果，是一个张量（Tensor），可以看成是神经网络最后一层的输出，未经过softmax处理。
targets 是期望的概率分布，通常是实际的标签进行one-hot编码后的结果，或另一个预测分布。
T 是温度参数，默认值为1.0，它用来平滑概率分布。当 T 大于1时，概率分布会变得更加平缓；而当 T 小于1时，分布会变得更加尖锐。
reduction 定义了在计算完所有样本的KL散度之后如何在批次上进行约简（降维）操作，'batchmean' 意味着将所有样本的散度加和后除以样本总数，以获得平均值。
    q = F.log_softmax(logits/T, dim=1)
 
F.log_softmax(logits/T, dim=1) 计算 logits 经过温度缩放后的log-softmax值，即对每个logit除以温度 T 后，应用 softmax 函数然后取对数。参数 dim=1 指定了在哪个维度上进行softmax运算，这里是在每个样本的输出向量上进行。
    p = F.softmax(targets/T, dim=1)
 
F.softmax(targets/T, dim=1) 计算 targets 经过温度缩放后的softmax值，相似地，将 targets 的每一个元素除以温度 T，然后应用softmax函数。这里 targets 通常表示标签的概率分布。
    return F.kl_div(q, p, reduction=reduction) * (T*T)
 
F.kl_div(q, p, reduction=reduction) 使用上述计算的 q（logits的log_softmax值）和 p（targets的softmax值）来计算KL散度。KL散度衡量两个概率分布的差异。
最后，KL散度的结果乘以 (T*T) 以补偿前面在概率分布上进行的温度缩放操作。
这个 kldiv 函数能够在机器学习模型的训练过程中用作自定义损失函数，尤其当你希望模型的输出分布尽可能贴近某个目标分布时。一种典型的应用场景是在知识蒸馏中，使用预训练的大型模型的输出来指导小型模型的训练。
"""

def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

"""
这段代码定义了一个名为 KLDiv 的类，它是 nn.Module 的子类，表明 KLDiv 是一个PyTorch模块，可以集成到神经网络架构中以计算Kullback-Leibler散度。在PyTorch框架中，自定义层和损失函数通常通过继承 nn.Module 来创建。

下面是逐行分析：

class KLDiv(nn.Module):
 
class KLDiv(nn.Module): 通过继承 nn.Module 类来定义一个名为 KLDiv 的新类。
    def __init__(self, T=1.0, reduction='batchmean'):
 
def __init__(self, T=1.0, reduction='batchmean'): 此方法是 KLDiv 类的构造器，用来初始化该模块的实例，它接受两个参数：温度 T 和约简方式 reduction。
        super().__init__()
 
调用父类 nn.Module 的构造器，确保所有继承自父类的初始化操作都能正确完成。
        self.T = T
        self.reduction = reduction
 
这两行将传递给构造器的参数(T 和 reduction)分别存储在模块的实例变量中，供后续使用。
接下来定义 forward 方法：

    def forward(self, logits, targets):
 
forward 函数定义了模块的前向传播逻辑。在PyTorch中，当模块被调用时，实际上是在调用其 forward 方法。
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)
 
这一行调用之前定义的 kldiv 函数，将传入 KLDiv 模块的 logits 和 targets 作为参数，并使用类内部存储的温度参数 self.T 和约简方式 self.reduction。然后返回计算得到的KL散度。
整体来看，KLDiv 类提供了一个方便的封装，使得计算KL散度更加模块化，在PyTorch网络架构中用起来非常方便，可以将其作为损失函数直接与其他组件整合使用。
"""

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


"""
这个类是GlobalSynthesizer，基于ABC类。它的主要作用是为神经网络模型（如分类任务中的教师模型和学生模型）生成数据，使用梯度下降的方式训练生成模型来合成逼真的图像数据。以下是对类中方法及属性的中文解读：

class GlobalSynthesizer(ABC):
 
定义了一个名为GlobalSynthesizer的类，该类继承于ABC抽象基类。

def __init__(self, teacher, student, generator, nz, num_classes, img_size, ...
 
类的构造函数初始化多个属性和超参数：

teacher: 教师模型，用于产生目标输出以指导生成器。
student: 学生模型，其输出与教师模型的输出进行比较以进行知识蒸馏。
generator: 生成器模型，用于生成新的图像数据。
nz: 噪声向量的维度数。
num_classes: 类别数目，用于分类任务。
img_size: 图像的尺寸。
init_dataset: 初始化的数据集（此处未在函数体内使用）。
iterations: 合成一个批次数据所进行迭代的次数。
lr_g: 生成器的学习率。
synthesis_batch_size: 合成数据的批次大小。
sample_batch_size: 采样数据的批次大小。
adv: 用于对抗性蒸馏的权重。
bn: 批归一化损失的权重。
oh: one-hot 编码损失的权重。
save_dir: 存储目录地址。
transform: 数据变换函数。
autocast: 自动数据类型转换功能（此处未在函数体内使用）。
use_fp16: 是否使用半精度浮点数（FP16）。
normalizer: 数据归一化的函数。
distributed: 是否为分布式训练。
lr_z: 潜在噪声向量的学习率。
warmup: 预热迭代次数。
reset_l0: 是否重置第0层。
reset_bn: 是否重置批归一化层。
bn_mmt: 批归一化的动量。
is_maml: 是否使用模型无关的元学习（MAML）。
args: 其他参数。
super(GlobalSynthesizer, self).__init__()
 
调用超类ABC的构造函数。

其他属性初始化包括创建图像池、数据变换和优化器等。钩子函数用于监视教师模型中批归一化层的行为。

def synthesize(self, targets=None):
 
synthesize函数用于生成图像数据。可以传递目标类（targets），否则会随机选取。函数内用于生成图像和进行迭代优化的过程。

synthesize方法的核心是通过梯度下降来优化噪声向量z，使得通过生成器fast_generator生成的图像能够获得低成本（即低损失值）。这里用到了批归一化损失、one-hot编码损失以及对抗性蒸馏损失来指导生成过程。

在MAML训练情境中，会使用一个内部循环来更新fast_generator的参数，并通过外部循环调整原始生成器的参数。

最终，生成的图像和对应的标签会被存储到数据池中，供后续使用。

注意：代码中有些部分如reset_l0_fun、fomaml_grad和reptile_grad等函数在给出的代码片段中未定义，因此它们的具体行为依赖于在类定义外部的具体定义。此外，代码分析基于给出的代码片段，实际在运行时可能依赖于更多上下文信息。
"""
class GlobalSynthesizer(ABC):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                    init_dataset=None, iterations=100, lr_g=0.1,
                    synthesis_batch_size=128, sample_batch_size=128, 
                    adv=0.0, bn=1, oh=1,
                    save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                    normalizer=None, distributed=False, lr_z = 0.01,
                    warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                    is_maml=1, args=None):
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


    """
        synthesize 函数是GlobalSynthesizer类的一个方法，它的目标是生成逼真的图像数据并训练生成器。以下是对该函数每一行代码的逐行分析：
    
        def synthesize(self, targets=None):
         
        定义了一个名为synthesize的方法，该方法接受一个可选参数targets，如果提供的话，会指定要生成的图像类别的标签。
        
            self.ep+=1
            self.student.eval()
            self.teacher.eval()
         
        self.ep+=1: 给ep属性加一，ep可能表示当前是第几个epoch。
        self.student.eval() 和 self.teacher.eval(): 将学生模型和教师模型设置为评估模式。在评估模式下，模型会关闭一些特有于训练模式的操作（比如dropout和batch normalization的更新）。
            best_cost = 1e6
         
        初始化best_cost为一个很大的数值（例如 1e6），这将用于记录迭代过程中遇到的最佳（最低）损失值。
        
            if (self.ep == 120+self.ep_start) and self.reset_l0:
                reset_l0_fun(self.generator)
         
        如果满足某个特定的epoch条件，并且设置了重置第0层（reset_l0）为真，则调用外部定义的reset_l0_fun函数重置生成器的特定层。
        
            best_inputs = None
            z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
            z.requires_grad = True
         
        best_inputs = None: 初始化变量best_inputs为None，稍后将存储生成的最优图像数据。
        z = torch.randn(...): 初始化噪声向量z，其形状由合成批次大小synthesis_batch_size和噪声向量维度nz决定，这是生成图像的输入。
        z.requires_grad = True: 开启噪声向量z的梯度计算，用于梯度下降。
            if targets is None:
                targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            else:
                targets = targets.sort()[0]  # sort for better visualization
            targets = targets.cuda()
         
        如果没有提供targets，则随机生成表示图像类别的目标向量。如果提供了，对其进行排序以便更好地可视化。之后将targets移到GPU上。
        
            fast_generator = self.generator.clone()
         
        克隆当前的生成器self.generator，创建一个fast_generator副本，用于优化过程中的快速迭代。
        
            optimizer = torch.optim.Adam([
                {'params': fast_generator.parameters()},
                {'params': [z], 'lr': self.lr_z}
            ], lr=self.lr_g, betas=[0.5, 0.999])
         
        为fast_generator的参数和噪声向量z创建一个Adam优化器。给z单独设置一个学习率self.lr_z。
        
            for it in range(self.iterations):
         
        开始迭代优化的循环，迭代的次数由self.iterations决定。
        
                inputs = fast_generator(z)
         
        用噪声向量z通过fast_generator生成图像inputs。
        
                inputs_aug = self.aug(inputs)  # crop and normalize
         
        对生成的图像应用数据增强，这可能包括随机裁剪和归一化等操作。
        
                if it == 0:
                    originalMeta = inputs
         
        如果是第一次迭代，保留原始生成的图像inputs，可能用于后续步骤，尽管在给出的代码片段中originalMeta没有被进一步使用。
        
                t_out = self.teacher(inputs_aug)["logits"]
         
        将增强后的图像inputs_aug输入到教师模型中获取分类结果的对数几率输出logits。
        
                if targets is None:
                    targets = torch.argmax(t_out, dim=-1)
                    targets = targets.cuda()
         
        如果最初未提供targets，则根据教师模型的输出选择最有可能的类别作为目标。
        
                loss_bn = sum([h.r_feature for h in self.hooks])
         
        计算所有批归一化层钩子的正则化特征损失，并累加。
        
                loss_oh = F.cross_entropy(t_out, targets)
         
        计算targets对应的交叉熵损失，这个被乘上了self.oh权重。
        
                if self.adv > 0 and (self.ep >= self.ep_start):
                    s_out = self.student(inputs_aug)["logits"]
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean()
                else:
                    loss_adv = loss_oh.new_zeros(1)
         
        如果设置了对抗性蒸馏，计算学生模型和教师模型输出之间的对抗损失。使用KL散度作为学生和教师模型输出之间的相似性度量，乘以上述代码段中定义的mask以及self.adv权重。
        
                loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
         
        将所有损失加权求和，得到总的损失函数loss。
        
                with torch.no_grad():
                    if best_cost > loss.item() or best_inputs is None:
                        best_cost = loss.item()
                        best_inputs = inputs.data.cpu()
         
        在不跟踪梯度的前提下，判断当前损失是否是迄今为止的最小损失。如果是，则更新best_cost和best_inputs，将最好的生成图片保存下来。
        
                optimizer.zero_grad()
                loss.backward()
         
        优化器梯度清零，并对损失函数进行反向传播。
        
                if self.ismaml:
                    if it==0: self.meta_optimizer.zero_grad()
                    fomaml_grad(self.generator, fast_generator)
                    if it == (self.iterations-1): self.meta_optimizer.step()
         
        对于模型无关元学习（MAML），进行特殊的梯度处理。每次迭代首先清零元优化器的梯度，调用fomaml_grad来适应快速变化的任务，并在最后一次迭代时对元优化器进行一步更新。
        
                optimizer.step()
         
        对噪声向量z和快速生成器fast_generator应用一步梯度下降。
        
            if self.bn_mmt != 0:
                for h in self.hooks:
                    h.update_mmt()
         
        如果批归一化的动量不为零，为每个钩子更新动量。
        
            if not self.ismaml:
                self.meta_optimizer.zero_grad()
                reptile_grad(self.generator, fast_generator)
                self.meta_optimizer.step()
         
        如果不是使用MAML，那么使用另一种元学习策略（可能是REPTILE）来更新生成器。
        
            self.student.train()
            self.prev_z = (z, targets)
            end = time.time()
         
        这部分代码恢复学生模型的训练模式，记录当前的噪声向量z和目标targets，记录当前时间（尽管代码中并没有定义time）。
        
            self.data_pool.add(best_inputs)
         
        将生成的最优图像best_inputs加入到数据池中。
        
        整体来说，这个synthesize函数使用了一种迭代的方式，通过对输入噪声向量z的梯度下降来优化生成器的参数，使其生成与教师模型输出匹配的图像，并存储达到最低损失函数值时的图像。同时，此代码还引入了MAML或其他元学习策略来进一步指导模型的优化。
        
        reptile_grad 函数
        def reptile_grad(src, tar):
            for p, tar_p in zip(src.parameters(), tar.parameters()):
                if p.grad is None:
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40
         
        reptile_grad是REPTILE元学习算法的梯度更新函数。该函数对源模型（src）和目标模型（tar）的参数执行操作，通常src是基本模型，tar是训练过程中的临时模型。 - for循环遍历src的参数和tar的参数。 - 如果src中的参数p没有梯度，则首先在GPU上初始化一个与p相同大小的全零变量作为p.grad。 - 使用原始模型参数p.data减去临时模型参数tar_p.data的差值来更新梯度。这里使用了add_函数，并带有一个步长系数alpha=67。
        
        fomaml_grad 函数
        def fomaml_grad(src, tar):
            for p, tar_p in zip(src.parameters(), tar.parameters()):
                if p.grad is None:
                    p.grad = Variable(torch.zeros(p.size())).cuda()
                p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67
         
        fomaml_grad是First-Order Model-Agnostic Meta-Learning (FOMAML)的梯度更新方法，一个简化版本的MAML算法。该函数直接将临时模型tar的梯度加到源模型src的梯度上。 - 循环处理每一对参数。 - 如果src中的参数p没有梯度，则在GPU上为其初始化一个零梯度。 - 将tar的梯度加到src的梯度上。这里的实现忽略了原始MAML的二阶导数部分，只考虑一阶导数更新。
        
        reset_l0_fun 函数
        def reset_l0_fun(model):
            for n,m in model.named_modules():
                if n == "l1.0" or n == "conv_blocks.0":
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    nn.init.constant_(m.bias, 0)
         
        reset_l0_fun函数用来重新初始化模型中特定的层（"l1.0"或"conv_blocks.0"）。 - 遍历模型model的所有具名模块。 - 如果模块名是指定的层，则： - 使用均值为0.0、标准差为0.02的正态分布重置权重m.weight。 - 将偏置m.bias重置为0。
        
        在synthesize函数中的使用
        reptile_grad用于如果不是使用MAML时，它用于在synthesize函数末尾将经过一轮训练迭代的fast_generator模型参数的变化应用到原始generator模型上，以指导长期的元学习过程。
        fomaml_grad在使用MAML时用于元优化步骤，它将在所有迭代后更新原始模型通过元学习步骤的梯度，从而用于优化目标任务性能。
        reset_l0_fun可能在特定的训练周期，作为一种重置机制，有助于生成器跳出局部极小或是开始新的搜索轨迹。
        这三个函数的存在，说明GlobalSynthesizer类利用了元学习策略来同时优化生成器在单个任务（如图像合成）上的短期性能，以及通过不断学习各种任务的能力来提升其长期适应性。通过不断迭代和调整，生成器能够更好地模仿训练数据分布，提高合成图像的质量，并辅助教师模型和学生模型之间的知识转移。
    """
    def synthesize(self, targets=None):
        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120+self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)
        
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
        z.requires_grad = True
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.cuda()

        fast_generator = self.generator.clone()
        """
            在PyTorch中，torch.optim.Adam 是一种实现Adam优化算法的类。这个优化器通常用于训练神经网络。
            betas 是Adam优化器的一个超参数，它是一个长度为2的元组，其中包含了两个用于计算梯度以及梯度平方的指数移动平均的衰减率。betas的两个元素通常被记为(beta1, beta2)。
            beta1：它控制了一阶矩估计的指数衰减。这实际上是预测的偏差矫正项，也就是所有以前梯度的加权平均。一般设置为0.9左右。
            beta2：它控制了二阶矩估计的指数衰减。这对应于梯度的缩放因子，也就是以前所有梯度的平方的加权平均。一般设置为0.999左右。
            在你提供的代码片段中，betas=[0.5, 0.999]表示beta1被设定为0.5，而beta2被设定为0.999。这些值决定了一阶和二阶估计值如何随着时间衰减，影响优化器如何结合过去的梯度来更新权重。
            Optimizer在这里配置了两组不同的参数组：
            fast_generator.parameters()：这代表了你模型中的生成器的所有可训练参数，该组参数使用self.lr_g作为学习率。
            [z]：这应该是一个张量或者张量列表，表示可学习的潜在变量，为它们指定了不同的学习率self.lr_z。这种情况通常在希望对模型中特定参数使用不同学习率策略时使用。
            在初始化Adam优化器时，你可以为不同的参数组指定不同的学习率和其他优化设置。这很有用，例如，当你希望对网络中的一些部分使用不同的学习速度时，或者像在训练GANs时，对生成器和判别器使用不同的学习率。
        """
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            if it == 0:
                originalMeta = inputs
            t_out = self.teacher(inputs_aug)["logits"]
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.cuda()

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets )
            if self.adv>0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.cpu() # mem, cpu
                    # save_data = best_inputs.clone()
                    # vutils.save_image(save_data[:200], 'real_{}.png'.format(dataset), normalize=True, scale_each=True, nrow=20)


            optimizer.zero_grad()
            loss.backward()
            # for (f_para, t_para) in zip(fast_generator.parameters(), self.generator.parameters()):
            #     print("Parameters:", f_para.data[0][0], t_para.data[0][0])
            #     print("z:", z[0][0])
            #     break

            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations-1): self.meta_optimizer.step()

            optimizer.step()
            for (f_para, t_para) in zip(fast_generator.parameters(), self.generator.parameters()):
                print("Parameters:", f_para.data[0][0], t_para.data[0][0])
                print("z:", z[0][0])
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
        self.prev_z = (z, targets)
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


class TARGET(BaseLearner):
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
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).cuda()
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
        '''
        CosineAnnealingLR 是 PyTorch 的 torch.optim.lr_scheduler 模块中的一个类，用于实现余弦退火的学习率调整策略。

        代码片段的说明如下： - scheduler：这是变量名称，用于存储学习率调度器的实例。 - torch.optim.lr_scheduler.CosineAnnealingLR：这是 PyTorch 中负责实现余弦退火学习率调度策略的类。 - optimizer：这个参数应该是 PyTorch 优化器的实例（例如，torch.optim.SGD、torch.optim.Adam 等），该学习率调度器将调整此优化器的学习率。 - 200：这是名为 T_max 的参数，它指定完整的余弦周期的迭代次数（即，多少个epoch后学习率会恢复到其初始值）。简单来说，它定义了学习率调整的周期长度。 - eta_min=2e-4：这是学习率退火到的最小值。在余弦退火周期内，学习率会从初始学习率下降到这个最小值，然后再升回初始学习率。
        
        在训练神经网络时，使用CosineAnnealingLR调度器可以使学习率根据余弦函数规律下降和上升，这往往能够帮助模型更好地收敛，并避免陷入局部最优解。在达到每个周期（T_max）的末尾，学习率会重置，允许模型可能在新的周期开始时从某种程度上“重新开始”学习过程。
        '''
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
        if self.args["dataset"] =="cifar100" or self.args["dataset"] == "emnist_letters":
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
        self._fl_train(train_dataset, self.test_loader)
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


    def _fl_train(self, train_dataset, test_loader):
        self._network.cuda()
        user_groups = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        prog_bar = tqdm(range(self.args["com_round"]))
        for _, com in enumerate(prog_bar):
            local_weights = []
            m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            for idx in idxs_users:
                local_train_loader = DataLoader(DatasetSplit(train_dataset, user_groups[idx]), 
                    batch_size=self.args["local_bs"], shuffle=True, num_workers=4)
                if com == 0:
                    print_data_stats(idx, local_train_loader)
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

            if com % 1 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info=("Task {}, Epoch {}/{} =>  Test_accuracy {:.2f}".format(
                    self._cur_task, com + 1, self.args["com_round"], test_acc,))
                prog_bar.set_description(info)
                if self.wandb == 1:
                    wandb.log({'Task_{}, accuracy'.format(self._cur_task): test_acc})
        


'''
这个函数是一个实现知识蒸馏(Knowledge Distillation, 简称KD)中常用的损失函数。在知识蒸馏的过程中，一个训练好的复杂模型（教师模型）被用来指导另一个模型（学生模型）进行学习。损失函数的目的是最小化学生模型预测的 softened log probabilities 和教师模型的 softened probabilities 之间的差异。

参数解释: - pred: 学生模型的原始输出 logits。 - soft: 教师模型的原始输出 logits。 - T: 温度参数，用来调节softmax操作的平滑程度。

函数步骤解析: 1. pred = torch.log_softmax(pred / T, dim=1): 将学生模型的 logits 除以温度 T，然后通过softmax和对数操作来获得每个类别的 softened log probabilities。softmax操作是在第二个维度(dim=1)，即对每一个样本的 logits 进行处理。

soft = torch.softmax(soft / T, dim=1): 将教师模型的 logits 除以温度 T 后，通过softmax操作来获得 softened probabilities。同样是对每个样本的 logits 操作。

loss = -1 * torch.mul(soft, pred).sum() / pred.shape[0]: 计算 softened probabilities 和 softened log probabilities 的加权负对数似然损失（交叉熵）。torch.mul(soft, pred) 计算学生模型的 softened log probabilities 与教师模型的 softened probabilities 的 element-wise 乘积。之后对这个乘积求和(sum)来获得整个批次的损失，然后除以批次中的样本数量(pred.shape[0])得到每个样本的平均损失，这里的 -1 表示取负值。

总结: 这个 _KD_loss 函数计算的是知识蒸馏中的损失，侧重于让学生模型学习到教师模型预测结果的软化信息，其中参数 T 可以控制这种软化的程度。通过最小化这个损失，学生模型可以从教师模型的logits中提取知识，即便是当教师模型不非常确定其预测时（即对于某些类别具有较低的预测置信度）。
'''
def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

