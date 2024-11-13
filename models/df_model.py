import os
from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from models.denoise_diffusion import DenoiseDiffusion
from models.unet import UNet
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class Diffusion():
    device: torch.device = 'cuda'

    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    image_size: int = 32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]

    # Number of time steps $T$
    n_steps: int = 1_000
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 5e-5 # original 2e-5

    # Number of training epochs
    epochs: int = 1_000

    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, image_channels, epochs, model_path, dataset=None, image_size=32):
        self.dataset = dataset
        self.image_channels = image_channels
        self.epochs = epochs
        self.image_size = image_size
        # Create $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)
        if model_path != "": # Load the model
            self.eps_model.load_state_dict(torch.load(model_path))

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        if self.dataset is not None:
            # Create dataloader
            self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=True)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)

        # Image logging
        tracker.set_image("sample", True)

    def sample(self, model_path="", log=True, batch_size=16):
        """
        ### Sample images
        """
        if model_path != "": # Load the model
            self.eps_model.load_state_dict(torch.load(model_path))
        with torch.no_grad():
            # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
            x = torch.randn([batch_size, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

            # Remove noise for $T$ steps
            for t_ in monit.iterate('Sample', self.n_steps):
                # $t$
                t = self.n_steps - t_ - 1
                # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
                x = self.diffusion.p_sample(x, x.new_full((batch_size,), t, dtype=torch.long))

            # Log samples
            if log:
                tracker.save('sample', x)
            return x

    def train(self, model_path=""):
        """
        ### Train
        """
        if model_path != "": # Load the model
            self.eps_model.load_state_dict(torch.load(model_path))
        # Iterate through the dataset
        for data in monit.iterate('Train', self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # detect the type of data
            if type(data) == list:
                data = data[1] # for federated learning data split
                if type(data) == list:
                    data = torch.stack(data, dim=0)
                    data = data.squeeze(1)
                assert len(list(data.shape)) == 4
            # Move data to device
            data = data.to(self.device)

            # 尝试展示图片
            # plt.imshow(data[0].permute(1, 2, 0).cpu().numpy()); plt.show()

            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            tracker.save('loss', loss)

    def run(self, model_path=""):
        """
        ### Training loop
        """
        existing_epochs = 0
        if model_path != "": # Load the model
            self.eps_model.load_state_dict(torch.load(model_path))
            existing_epochs = int(model_path.split('eps_model')[1].split('.pth')[0])
            print("Existing epochs: ", existing_epochs)
        for epoch in monit.loop(self.epochs):
            # Train the model
            self.train()
            # Sample some images
            self.sample()
            # New line in the console
            tracker.new_line()
            # Save the model
            # experiment.save_checkpoint()
            torch.save(self.eps_model.state_dict(), 'eps_model{}.pth'.format(epoch+existing_epochs))

    def train_without_sample(self, dataloader=None, model_path="", batch_size=16):
        """
        ### Train without sampling
        """
        if dataloader is not None: # Load new dataloader
            self.data_loader = dataloader
        save_interval = 50
        real_epoch = 0
        for i in range(self.epochs):
            model_next_save_path = model_path+'eps_model{}.pth'.format(real_epoch + save_interval)
            if os.path.exists(model_next_save_path):
                print("Model exists, skip saving and training for the next {} epochs".format(save_interval))
                self.diffusion.eps_model.load_state_dict(torch.load(model_next_save_path))
                real_epoch += save_interval
            if i >= real_epoch:
                model_real_path = model_path + 'eps_model{}.pth'.format(real_epoch)
                if not os.path.exists(model_real_path):
                    self.train()
                    # self.sample(batch_size=batch_size)
                    tracker.new_line()
                    if real_epoch % save_interval == 0:
                        torch.save(self.eps_model.state_dict(), model_real_path)
                    print("Epoch: {}/{}".format(real_epoch, self.epochs))
                else:
                    print("Skip training for epoch {}".format(real_epoch))
                real_epoch += 1
            else:
                print("Skip training for epoch {}".format(i))


class CelebADataset(torch.utils.data.Dataset):
    """
    ### CelebA HQ dataset
    """

    def __init__(self, image_size: int):
        super().__init__()

        # CelebA images folder
        folder = lab.get_data_path() / 'celebA'
        # List of files
        self._files = [p for p in folder.glob(f'**/*.jpg')]

        # Transformations to resize the image and convert to tensor
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self._files)

    def __getitem__(self, index: int):
        """
        Get an image
        """
        img = Image.open(self._files[index])
        return self._transform(img)


# @option(Diffusion.dataset, 'CelebA')
def celeb_dataset(c: Diffusion):
    """
    Create CelebA dataset
    """
    return CelebADataset(c.image_size)


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


# @option(Diffusion.dataset, 'MNIST')
def mnist_dataset(c: Diffusion):
    """
    Create MNIST dataset
    """
    return MNISTDataset(c.image_size)


class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    """
    ### CIFAR-10 dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


# @option(Diffusion.dataset, 'CIFAR10')
def cifar10_dataset(c: Diffusion):
    return CIFAR10Dataset(c.image_size)


class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    """
    ### CIFAR-100 dataset
    """

    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


# @option(Diffusion.dataset, 'CIFAR100')
def cifar100_dataset(c: Diffusion):
    return CIFAR100Dataset(c.image_size)

def main():
    dtset = CIFAR100Dataset(32)
    model_path = ""
    diffusion = Diffusion(dataset=dtset, image_channels=3, epochs=200, model_path=model_path)
    diffusion.run(model_path='')

def sample():
    dtset = CIFAR100Dataset(32)
    # generate a list of 10, 20, ..., 1300
    ids = list(range(0, 600, 20))
    for id in tqdm(ids):
        model_path = "../checkpoints_diffusion/exp20240328113622468183_task0_client0_eps_model{}.pth".format(id)
        # Create configurations
        diffusion = Diffusion(dataset=dtset, image_channels=3, epochs=200, model_path=model_path)
        batch_num = 1
        for i in range(batch_num):
            x = diffusion.sample(model_path, log=False, batch_size=16)
            pil_images = []  # 存储转换后的PIL图像
            for j in range(x.shape[0]):  # 遍历批次中的每个图像
                img = x[j]  # 获取当前图像张量

                # 数值范围调整至0到255，并转换为uint8
                img = torch.clamp(img, 0, 255)  # 限制数值范围到0-255
                img = img / img.max() * 255.0  # 归一化（可选步骤，视具体情况而定）
                img_uint8 = img.to(torch.uint8)  # 转换为uint8

                # 调整通道顺序: [C, H, W] -> [H, W, C]
                img_uint8 = img_uint8.permute(1, 2, 0)

                # 转换为PIL图像
                pil_img = Image.fromarray(img_uint8.cpu().numpy())

                # 将PIL图像添加到列表中
                pil_images.append(pil_img)
            import matplotlib.pyplot as plt

            # 假设你有一个包含20个PIL图像的列表pil_images
            # 设置整个图表的大小
            plt.figure(figsize=(10, 8))  # 10和8分别代表图表的宽和高，在这里可以根据需要调整
            plt.title(f'Image {id}')

            # 通过循环遍历pil_images列表来展示每个图像
            for i, image in enumerate(pil_images):
                plt.subplot(4, 4, i + 1)  # 创建子图。注意，matplotlib的index从1开始
                plt.imshow(image)  # 显示PIL图像
                plt.axis('off')  # 关闭坐标轴


            plt.tight_layout()  # 自动调整子图参数，以给定的填充
            plt.savefig("../images/7e_5_{}.png".format(id), dpi=600)
            plt.show()  # 显示整个图表
    return x

def sample_whole():
    total = 400
    dtset = CIFAR100Dataset(32)
    # generate a list of 10, 20, ..., 1300
    id = 1000
    num_clients = 10
    num_tasks = 5
    for task in range(num_tasks - 1):
        for c in range(num_clients):
            model_path = f"../checkpoints_diffusion/exp{id}_task{task}_client{c}_eps_model{total}.pth"
            # Create configurations
            diffusion = Diffusion(dataset=dtset, image_channels=3, epochs=200, model_path=model_path)
            batch_num = 1
            for i in range(batch_num):
                x = diffusion.sample(model_path, log=False, batch_size=16)
                pil_images = []  # 存储转换后的PIL图像
                for j in range(x.shape[0]):  # 遍历批次中的每个图像
                    img = x[j]  # 获取当前图像张量

                    # 数值范围调整至0到255，并转换为uint8
                    img = torch.clamp(img, 0, 255)  # 限制数值范围到0-255
                    img = img / img.max() * 255.0  # 归一化（可选步骤，视具体情况而定）
                    img_uint8 = img.to(torch.uint8)  # 转换为uint8

                    # 调整通道顺序: [C, H, W] -> [H, W, C]
                    img_uint8 = img_uint8.permute(1, 2, 0)

                    # 转换为PIL图像
                    pil_img = Image.fromarray(img_uint8.cpu().numpy())

                    # 将PIL图像添加到列表中
                    pil_images.append(pil_img)
                import matplotlib.pyplot as plt

                # 假设你有一个包含20个PIL图像的列表pil_images
                # 设置整个图表的大小
                plt.figure(figsize=(10, 8))  # 10和8分别代表图表的宽和高，在这里可以根据需要调整
                plt.title(f'Image {model_path.split("/")[-1].split(".")[0]}')

                # 通过循环遍历pil_images列表来展示每个图像
                for i, image in enumerate(pil_images):
                    plt.subplot(4, 4, i + 1)  # 创建子图。注意，matplotlib的index从1开始
                    plt.imshow(image)  # 显示PIL图像
                    plt.axis('off')  # 关闭坐标轴

                plt.tight_layout()  # 自动调整子图参数，以给定的填充
                image_path = "../images/{}.png".format(model_path.split("/")[-1].split(".")[0])
                plt.savefig(image_path, dpi=600)
                plt.show()  # 显示整个图表
    return x

def sample_simple():
    dtset = CIFAR100Dataset(32)
    # generate a list of 10, 20, ..., 1300
    model_path = "../checkpoints_diffusion/exp2000_task0_client0_eps_model200.pth"
    # Create configurations
    diffusion = Diffusion(dataset=dtset, image_channels=3, epochs=200, model_path=model_path)
    batch_num = 1
    for i in range(batch_num):
        x = diffusion.sample(model_path, log=False, batch_size=16)
        pil_images = []  # 存储转换后的PIL图像
        for j in range(x.shape[0]):  # 遍历批次中的每个图像
            img = x[j]  # 获取当前图像张量

            # 数值范围调整至0到255，并转换为uint8
            img = torch.clamp(img, 0, 255)  # 限制数值范围到0-255
            img = img / img.max() * 255.0  # 归一化（可选步骤，视具体情况而定）
            img_uint8 = img.to(torch.uint8)  # 转换为uint8

            # 调整通道顺序: [C, H, W] -> [H, W, C]
            img_uint8 = img_uint8.permute(1, 2, 0)

            # 转换为PIL图像
            pil_img = Image.fromarray(img_uint8.cpu().numpy())

            # 将PIL图像添加到列表中
            pil_images.append(pil_img)
        import matplotlib.pyplot as plt

        # 假设你有一个包含20个PIL图像的列表pil_images
        # 设置整个图表的大小
        plt.figure(figsize=(10, 8))  # 10和8分别代表图表的宽和高，在这里可以根据需要调整
        plt.title(f'Image {model_path.split("/")[-1].split(".")[0]}')

        # 通过循环遍历pil_images列表来展示每个图像
        for i, image in enumerate(pil_images):
            plt.subplot(4, 4, i + 1)  # 创建子图。注意，matplotlib的index从1开始
            plt.imshow(image)  # 显示PIL图像
            plt.axis('off')  # 关闭坐标轴
            # 不显示坐标轴
            plt.axis('off')


        plt.tight_layout()  # 自动调整子图参数，以给定的填充
        plt.savefig("../images/7e_5_{}.png".format(id), dpi=600)
        plt.show()  # 显示整个图表
    return x

if __name__ == '__main__':
    # main()
    x = torch.randn([1, 3, 64, 64],
                    device='cuda')
    # 把x画出来
    plt.imshow(x[0].permute(1, 2, 0).cpu().numpy()); plt.show()
    sample_simple()
    # sample_whole()