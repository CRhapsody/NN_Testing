import torch
import torch.nn as nn
import torch.optim as optim

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 这里简化定义，实际应用中应根据需求设计网络结构
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 更多层...
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
    def forward(self, x):
        return self.main(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 同样简化定义
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 更多层...
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # 输出概率
        )
        
    def forward(self, x, adv_noise=None):
        if adv_noise is not None:
            x = torch.cat([x, adv_noise], dim=1)
        return self.main(x).view(-1)

# 初始化模型和优化器
generator = Generator()
discriminator = Discriminator()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# load dataset
label = 0
images_dataset, noise_dataset = torch.load('/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{}.pt'.format(label))
dataset = torch.utils.data.TensorDataset(images_dataset, noise_dataset)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 定义损失函数
criterion = nn.BCELoss()

# 训练循环示例
num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):  # 假设data_loader是你的数据加载器
        # 训练判别器
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # 真实图像和噪声
        real_images = images.to(device)
        noise = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        # 生成对抗性噪声
        gen_noise = generator(real_images)
        
        # 计算判别器损失
        real_loss = criterion(discriminator(real_images, gen_noise.detach()), real_labels)
        fake_loss = criterion(discriminator(real_images, gen_noise), fake_labels)
        d_loss = real_loss + fake_loss
        
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        # 目标是最大化D(G(z))，但为了优化方便，我们最小化-log(D(G(z)))
        g_loss = criterion(discriminator(real_images, gen_noise), real_labels)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # 打印损失等信息
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")