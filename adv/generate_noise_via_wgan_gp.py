import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import imageio
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import numpy as np
import torch.optim as optim
from pgd_attack import threshold_tensor_exterme_image, threshold_tensor_exterme



# Define the Generator class, inheriting from nn.Module
class Generator(nn.Module):
    """
    The Generator class is responsible for generating images from latent space vectors.

    Attributes:
        img_size (tuple): The size of the output image (channels, height, width).
        latent_dim (int): The dimension of the latent space vector.
        dim (int): The base dimension of the model.
    """
    def __init__(self, img_size, latent_dim, dim = None):
        """
        Initialize the Generator.

        Parameters:
            img_size (tuple): The size of the output image.
            latent_dim (int): The dimension of the latent space.
            dim (int): The base dimension of the feature maps.
        """
        super(Generator, self).__init__()

        self.dim = dim
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (self.img_size[1] / 16, self.img_size[2] / 16)

        # Define the module that transforms latent vectors into feature maps
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, int(8 * dim)),
            nn.ReLU()
        )



        # Define the module that transforms feature maps into images
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim+self.img_size[0], 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),

            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),

            # nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            # nn.ReLU(),
            # nn.BatchNorm2d(dim),

            # batch_size x 3 x 512 x 512 -> batch_size x 3 x 32 x 32
            # 2 layer
            nn.Conv2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, 3, 4, 2, 1),


            nn.Sigmoid()
        )

    def forward(self, latent_data, input_data):
        """
        Defines the forward pass.

        Parameters:
            input_data (Tensor): The input data, a tensor from the latent space.

        Returns:
            Tensor: The generated image tensor.
        """
        # Map the latent vector to feature maps
        # Map latent into appropriate size for transposed convolutions
        latent_feature = self.latent_to_features(latent_data).unsqueeze(-1).unsqueeze(-1)
        latent_feature = latent_feature.expand(-1, -1, *self.img_size[1:])
        # Concatenate noise and embedded labels along the channel dimension
        x = torch.cat([latent_feature, input_data], dim=1)
        
        # Generate the image through the feature-to-image module
        # Return generated image
        x = self.features_to_image(x)
        

        return x

    def sample_latent(self, num_samples):
        """
        Generates random latent vectors.

        Parameters:
            num_samples (int): The number of latent vectors to generate.

        Returns:
            Tensor: A tensor containing the generated latent vectors.
        """
        return torch.randn((num_samples, self.latent_dim))
    

# Define the Discriminator class, inheriting from nn.Module
class Discriminator(nn.Module):
    """
    The Discriminator class is responsible for judging the authenticity of images.

    Attributes:
        img_size (tuple): The size of the input image (height, width, channels).
        dim (int): The base dimension of the model.
    """
    def __init__(self, img_size, dim = 16):
        """
        Initialize the Discriminator.

        Parameters:
            img_size (tuple): The size of the input image.
            dim (int): The base dimension of the feature maps.
        """
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.dim = dim

        # Define the module that transforms images into feature maps
        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[0], dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid()
        )

        # Calculate the size of the output after the last convolution
        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * (img_size[1] / 16) * (img_size[2] / 16)
        output_size = int(output_size)
        # Define the module that transforms feature maps into a probability
        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        """
        Defines the forward pass.

        Parameters:
            input_data (Tensor): The input image tensor.

        Returns:
            Tensor: A scalar value indicating the probability of the input image being real.
        """
        batch_size = input_data.size()[0]
        # Transform the image into feature maps
        x = self.image_to_features(input_data)
        # Flatten the feature maps into a vector
        x = x.view(batch_size, -1)
        # Output the probability of the input being real
        return self.features_to_prob(x)
    

class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        """ """
        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        # Calculate probabilities on real and generated data
        data = Variable(data)
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data)
        d_generated = self.D(generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)
        # if self.use_cuda:
        #     generated_data = generated_data.cuda()
        # Calculate loss and optimize
        d_generated = self.D(generated_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            input_data, noise_data = data
            if self.use_cuda:
                input_data = input_data.cuda()
                noise_data = noise_data.cuda()
            self.num_steps += 1
            self._critic_train_iteration(noise_data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(input_data)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        if save_training_gif:
            # Fix latents to see how image generation improves during training
            fixed_latents = Variable(self.G.sample_latent(10))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()
            training_progress_images = []
            # load the dataset and save first 10 images
            noise_fix = None
            for i, (data, noise) in enumerate(data_loader):
                if i == 0:
                    noise_fix = noise
                    # save first 10 noise as IMAGE
                    a_list = []
                    for j in range(10):
                        
                        a = threshold_tensor_exterme(noise[j], epsilon=8,exterme=(0,255))
                        a_list.append(a.unsqueeze(0))
                    a = torch.cat(a_list, dim=0)
                    noise_arrange_img_grid = make_grid(a.cpu().detach())
                    noise_arrange_img_grid = np.transpose(((noise_arrange_img_grid + 0.5)*255).clamp(0.,255.).numpy().astype(np.uint8)
                                        , (1, 2, 0))
                    imageio.imwrite('noise_arrange_img_grid_{}.png'.format(j), noise_arrange_img_grid)
                    # a = threshold_tensor_exterme(noise_fix, epsilon=8,exterme=(0,255))
                    # noise_arrange_img_grid = make_grid(a.cpu().detach())
                    # noise_arrange_img_grid = np.transpose(((noise_arrange_img_grid + 0.5)*255).clamp(0.,255.).numpy().astype(np.uint8)
                    #                     , (1, 2, 0))
                    # imageio.imwrite('noise_arrange_img_grid.png', noise_arrange_img_grid)
                    if self.use_cuda:
                        noise_fix = noise_fix.cuda()
                    break


        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            if save_training_gif:
                # Generate batch of images and convert to grid
                img_grid = make_grid(self.G(fixed_latents, noise_fix).cpu().detach())
                # Convert to numpy and transpose axes to fit imageio convention
                # i.e. (width, height, channels)
                img_grid = np.transpose(((img_grid + 0.5)*255).clamp(0.,255.).numpy().astype(np.uint8)
                                        , (1, 2, 0))
                # Add image grid to training progress
                training_progress_images.append(img_grid)

        if save_training_gif:
            imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
                            training_progress_images)
        # save loss as a graph
        import matplotlib.pyplot as plt
        plt.plot(self.losses['D'], label='D')
        plt.plot(self.losses['GP'], label='GP')
        plt.plot(self.losses['G'], label='G')
        plt.plot(self.losses['gradient_norm'], label='gradient_norm')
        plt.legend()
        plt.savefig('loss.png')
        


    def sample_generator(self, num_samples):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        data_samples = Variable(torch.randn(num_samples, *self.G.img_size))
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
            data_samples = data_samples.cuda()
        generated_data = self.G(latent_samples, data_samples)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
    
label = 0
# images_dataset, noise_dataset = torch.load(f'/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{label}.pt')
images_dataset, noise_dataset = torch.load(f'/home/chizm/Testing/data/noise_extreme/{label}/cifar10_resnet18_pgd_noise_label_{label}.pt')
dataset = torch.utils.data.TensorDataset(images_dataset, noise_dataset)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)


img_size = (3,32,32)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 1000
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available(),print_every=1)
trainer.train(data_loader, epochs, save_training_gif=True)

# Save models
name = 'cifar10_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
