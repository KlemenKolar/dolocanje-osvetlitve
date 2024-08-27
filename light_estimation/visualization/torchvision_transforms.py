import torchvision.transforms as transforms
from PIL import Image
from evaluation.utils import load_to_hdf5
from estimation.enums import DataMode
import torch
from matplotlib import pyplot as plt
import numpy as np

def to_grayscale(image):
    transform = transforms.Grayscale()
    #sample = torch.from_numpy(np.array(sample).swapaxes(0, 2).swapaxes(0, 1)).float()
    return np.array(transform(image))

def adjust_brightness(image, brightness_factor):
    transform = transforms.ColorJitter(brightness=brightness_factor)
    return transform(image)

def adjust_contrast(image, contrast_factor):
    transform = transforms.ColorJitter(contrast=contrast_factor)
    return transform(image)

def apply_gaussian_blur(image, kernel_size):
    transform = transforms.GaussianBlur(kernel_size)
    return transform(image)

def add_random_noise(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.04)], p=1),
        #transforms.ToPILImage(),
    ])
    return np.array(transform(image)).swapaxes(0, 2).swapaxes(0, 1)

if __name__ == "__main__":
    images,_,_ = load_to_hdf5('F:\Klemen_diploma\light_estimation\estimation\dataset\LED128x128.hdf5', data_mode=DataMode.RADIANS)
    pil_images = [Image.fromarray(image) for image in images]
    grayscale_image = to_grayscale(pil_images[0])
    brightness_image = adjust_brightness(pil_images[2], 0.9)
    contrast_image = adjust_contrast(pil_images[2], 6)
    blur_image = apply_gaussian_blur(pil_images[3], 5)
    noise_image = add_random_noise(pil_images[4])
    plt.imshow(noise_image) #add cmap='gray' argument if image is grayscale
    plt.show()
    plt.imshow(images[4])
    plt.show()
    #grayscale_image.show()
    #brightness_image.show()
    #contrast_image.show()
    #blur_image.show()
    #noise_image.show()