import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
# add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models import ResNet18
from utils.for_model import check_state

class LinfPGDAttack(object):
    def __init__(self, model, epsilon, alpha = None, step = 10):
        self.model = model
        self.epsilon = epsilon
        # self.alpha = epsilon / 4 if alpha is None else alpha
        self.step = step
        self.alpha = alpha
    

    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, value = None):
        if value is None:
            self._alpha = self.epsilon / 4
        else:
            assert value > 0
            self._alpha = value

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.step):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def attack(self, x, y, model):
        model_copied = copy.deepcopy(model)
        model_copied.eval()
        self.model = model_copied
        adv = self.perturb(x, y)
        return adv


# Observation：adv noise of dataset cifar10 and model resnet18
def lookup_noise(label):
    '''
    :return: image,noise,label
    '''
    save_list = []
    model = ResNet18()
    model.load_state_dict(check_state(torch.load('/home/chizm/Testing/model_file/resnet18_pgd_adversarial_training.pt')))
    model = model.to('cuda')
    model.eval()
    adversary = LinfPGDAttack(model, epsilon=8/255, step=10)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)
    for data, target in test_loader:
        data, target = data.to('cuda'), target.to('cuda')
        adv = adversary.attack(data, target, model)
        # whether the image is adversarial
        with torch.no_grad():
            output = model(adv)
            pred = output.argmax(dim=1, keepdim=True)
            if pred != target and target == label:
                save_list.append((data, adv-data))
                if len(save_list) >= 100:
                    break
            else:
                continue
    # concate the image and noise
    image = torch.cat([i[0] for i in save_list], dim=0)
    noise = torch.cat([i[1] for i in save_list], dim=0)
    torch.save((image, noise), '/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{}.pt'.format(label))


# load the image and noise as Image
# def tensor_to_image(tensor):
#     '''
#     :param tensor: tensor
#     :return: image
#     '''
#     tensor = tensor.squeeze(0)
#     tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

#     return Image.fromarray(tensor)

# def noise_tensor_to_image(epsilon,tensor):
#     tensor = tensor.squeeze(0)
#     tensor = 255*tensor.mul(255/(2*epsilon)).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     return Image.fromarray(tensor)

# def noise_tensor_to_image_theshold(epsilon,tensor,theshold):
#     '''
#     :param epsilon: the epsilon of the noise
#     :param tensor: tensor
#     :param theshold: the theshold of the noise in image, tuple(min, max)
#     '''
#     tensor = tensor.squeeze(0)
#     tensor = 255*tensor.mul(255/(2*epsilon)).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     # theshold the noise
#     tensor = tensor * (tensor >= theshold[0]) * (tensor <= theshold[1])
#     return Image.fromarray(tensor)
# def read_noise_theshold(label,theshold):
#     '''
#     :param label: label of the image
#     :param theshold: the theshold of the noise in image, tuple(min, max)
#     '''
#     image, noise = torch.load('/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{}.pt'.format(label))
#     image_list = [None] * len(image)
#     noise_list = [None] * len(noise)
#     for i in range(len(image)):
#         image_list[i] = tensor_to_image(image[i])
#         noise_list[i] = noise_tensor_to_image_theshold(8, noise[i],theshold)
#         combine_image = Image.new('RGB', (64, 32))
#         combine_image.paste(image_list[i], (0, 0))
#         combine_image.paste(noise_list[i], (32, 0))
#         combine_image.show()



# def convert_image_and_noise(label):
#     '''
#     :return: image,noise
#     '''
#     image, noise = torch.load('/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{}.pt'.format(label))
#     tensor_to_image_transform = transforms.ToPILImage()
#     image_list = [None] * len(image)
#     noise_list = [None] * len(noise)
#     # save the image and noise
#     for i in range(len(image)):
#         image_list[i] = tensor_to_image_transform(image[i])
#         noise_list[i] = noise_tensor_to_image(8, noise[i])
#         # save the image and noise as one picture
#         combine_image = Image.new('RGB', (64, 32))
#         combine_image.paste(image_list[i], (0, 0))
#         combine_image.paste(noise_list[i], (32, 0))
#         combine_image.save(f'/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{label}_{i}.png')



def preprocess_tensor(tensor, epsilon=0):
    """
    Preprocesses a PyTorch tensor to an image-ready format.
    Optionally scales by epsilon for noise visualization.
    """
    tensor = tensor.squeeze(0)
    scaled_tensor = tensor if epsilon == 0 else 255*tensor.mul(255/(2*epsilon)).add_(0.5)
    clamped_tensor = scaled_tensor.clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8) if scaled_tensor.shape[0] == 3 else scaled_tensor.clamp_(0, 255).to('cpu', torch.uint8)
    return clamped_tensor

def tensor_to_image(tensor, epsilon=0):
    """Converts a preprocessed tensor into a PIL Image."""
    processed_tensor = preprocess_tensor(tensor, epsilon).numpy()
    return Image.fromarray(processed_tensor)

def pix_tensor_nomalize(tensor, epsilon = None):
    '''
    :param tensor: tensor
    :param epsilon: epsilon of the noise, if None, the range of the tensor is [-1,1]; else the range is [-epsilon, epsilon]
    '''
    tensor = tensor.squeeze(0)
    if epsilon is not None:
        tensor = tensor.div(255).sub(0.5).mul(2*epsilon/255).clamp_(-epsilon, epsilon)
    else:
        tensor = tensor.div(255).sub(0.5).clamp_(-1, 1)
    return tensor

def threshold_tensor_interval(tensor, epsilon=0, threshold=(0, 255)):
    """Applies thresholding to a preprocessed tensor before converting to image."""
    processed_tensor = preprocess_tensor(tensor, epsilon)
    thresholded_tensor = processed_tensor * (processed_tensor >= threshold[0]) * (processed_tensor <= threshold[1])
    return Image.fromarray(thresholded_tensor.numpy())

def threshold_tensor_exterme(tensor, epsilon=0, exterme=(0, 255)):
    """Applies extreme to a preprocessed tensor before converting to image."""
    processed_tensor = preprocess_tensor(tensor, epsilon)
    # only retain the extreme value of the tensor <=min or >=max, others are set to RGB as (139,125,107)
    # For image(H,W,C), judge every C channel of each pixel
    H, W, C = processed_tensor.shape
    exterme_min = torch.tensor(exterme[0])
    exterme_max = torch.tensor(exterme[1])
    for i in range(H):
        for j in range(W):
            # judge every C channel of each pixel
            if (torch.Tensor.abs((processed_tensor[i][j] - exterme_min))).sum() <= 255:
                processed_tensor[i][j] = torch.tensor([0,0,0])
            elif (torch.Tensor.abs((processed_tensor[i][j] - exterme_max))).sum() <= 255:
                processed_tensor[i][j] = torch.tensor([255,255,255])
            else:
                processed_tensor[i][j] = torch.tensor([139,125,107])

    return processed_tensor

def threshold_tensor_exterme_image(tensor, epsilon=0, exterme=(0, 255)):
    """Applies extreme to a preprocessed tensor before converting to image."""
    processed_tensor = threshold_tensor_exterme(tensor, epsilon, exterme)
    return Image.fromarray(processed_tensor.numpy())

def load_data(label):
    """Loads image and noise data from a specified label."""
    path = f'/home/chizm/Testing/data/noise/cifar10_resnet18_pgd_noise_label_{label}.pt'
    return torch.load(path)

def combine_images(image, noise, width=64):
    """Combines two images horizontally."""
    combined_image = Image.new('RGB', (width, 32))
    combined_image.paste(image, (0, 0))
    combined_image.paste(noise, (width // 2, 0))
    return combined_image

def process_and_save_images(label, epsilon=8, threshold=None,extreme=None):
    """Processes and saves images with optional noise thresholding or extreme."""
    image, noise = load_data(label)
    pil_to_image_transform = transforms.ToPILImage()
    # assert threshold and extreme are mutually exclusive
    assert threshold is None or extreme is None
    # mkdir
    Path(f'/home/chizm/Testing/data/noise_extreme/{label}').mkdir(parents=True, exist_ok=True)
    
    for i in range(len(image)):
        img = pil_to_image_transform(image[i])

        
        if threshold is not None:
            noisy_img = threshold_tensor_interval(noise[i], epsilon, threshold)
        elif extreme is not None:
            noisy_img = threshold_tensor_exterme(noise[i], epsilon, extreme)
        else:
            noisy_img = tensor_to_image(noise[i], epsilon)
        
        noisy_img.save('test.png')
        combined = combine_images(img, noisy_img)
        combined.save(f'/home/chizm/Testing/data/noise_extreme/{label}/cifar10_resnet18_pgd_noise_label_{label}_{i}.png')
        # If you want to display instead of saving, uncomment the line below
        # combined.show()

def save_image_and_noise_extreme_aspt(label,epsilon=8,extreme=(0, 255)):
    image, noise = load_data(label)
    pil_to_image_transform = transforms.ToPILImage()
    Path(f'/home/chizm/Testing/data/noise_extreme/{label}').mkdir(parents=True, exist_ok=True)
    image_list = []
    noise_list = []

    for i in range(len(image)):
        # noise_list.append().unsqueeze(0)
        noise_list.append(pix_tensor_nomalize(threshold_tensor_exterme(noise[i], epsilon, extreme), epsilon = None).unsqueeze(0))
        image_list.append(image[i].unsqueeze(0))
        # tensor_to_image(noise_list[i], epsilon).save(f'/home/chizm/Testing/cifar10_resnet18_pgd_noise_label_{label}_{i}.png')
        # Image.fromarray(theshold_tensor_exterme(noise[i], epsilon, extreme).numpy()).save(f'/home/chizm/Testing/cifar10_resnet18_pgd_noise_label_{label}_{i}.png')
    image_cat = torch.cat(image_list, dim=0)
    noise_cat = torch.cat(noise_list, dim=0)

    torch.save((image_cat, noise_cat.permute(0, 3, 1, 2)
                ), f'/home/chizm/Testing/data/noise_extreme/{label}/cifar10_resnet18_pgd_noise_label_{label}.pt')

# process_and_save_images(0, epsilon=8, extreme=(0, 255)) #

# convert_image_and_noise(0)

# lookup_noise(0)   
save_image_and_noise_extreme_aspt(0,extreme=(0,255))


