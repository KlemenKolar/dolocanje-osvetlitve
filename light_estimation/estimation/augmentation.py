import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class RandomBrightness:
    def __init__(self, mean=1.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        brightness_factor = random.gauss(self.mean, self.std)
        return F.adjust_brightness(img, brightness_factor)

class RandomContrast:
    def __init__(self, mean=1.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        contrast_factor = random.gauss(self.mean, self.std)
        return F.adjust_contrast(img, contrast_factor)

class RandomSaturation:
    def __init__(self, mean=1.0, std=0.2):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        saturation_factor = random.gauss(self.mean, self.std)
        return F.adjust_saturation(img, saturation_factor)

class RandomHue:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        hue_factor = random.gauss(self.mean, self.std)
        return F.adjust_hue(img, hue_factor)
    

def test():
    transform = transforms.Compose([
        RandomBrightness(mean=1.0, std=0.2),
        RandomContrast(mean=1.0, std=0.2),
        RandomSaturation(mean=1.0, std=0.2),
        RandomHue(mean=0.0, std=0.1),
    ])

    from PIL import Image
    from evaluation.utils import load_to_hdf5
    from estimation.enums import DataMode
    images, light_data, ambient = load_to_hdf5("/home/klemen/light_estimation/estimation/dataset/LED128x128.hdf5", DataMode.DISCRETE) # Replace with your dataset path
    for i, image in enumerate(images):
        if i > 20:
            break
        image = Image.fromarray(image)
        augmented_img = transform(image)

        image.save(f"/home/klemen/light_estimation/visualization/image_augment/{i}_old.jpg")
        augmented_img.save(f"/home/klemen/light_estimation/visualization/image_augment/{i}_new.jpg")
