from matplotlib import pyplot as plt
from evaluation.utils import load_to_hdf5
from estimation.enums import DataMode
import numpy as np
import os
import json
from PIL import Image

def plot_dataset():
    images,_,_ = load_to_hdf5('F:\Klemen_diploma\light_estimation\estimation\dataset\LED128x128.hdf5', data_mode=DataMode.RADIANS)
    for i in range(len(images)):
        plt.imshow(images[i+20000])
        print(f'image {i}')
        plt.show()


def subplot_dataset_short():
    images,_,_ = load_to_hdf5('F:\Klemen_diploma\light_estimation\estimation\dataset\Only_Material_Dataset_Shorter.hdf5', data_mode=DataMode.RADIANS)
    plt.suptitle("Podatkovna množica Synthetic_128_128_short", fontsize=14)
    plt.subplot(2, 3, 1)
    plt.imshow(images[0])
    plt.subplot(2, 3, 2)
    plt.imshow(images[1])
    plt.subplot(2, 3, 3)
    plt.imshow(images[4])
    plt.subplot(2, 3, 4)
    plt.imshow(images[20013])
    plt.subplot(2, 3, 5)
    plt.imshow(images[5])
    plt.subplot(2, 3, 6)
    plt.imshow(images[9])
    plt.show()


def subplot_dataset_big():
    images,_,_ = load_to_hdf5('F:\Klemen_diploma\light_estimation\estimation\dataset\LED128x128.hdf5', data_mode=DataMode.RADIANS)
    plt.suptitle("Podatkovna množica SynthA", fontsize=14)
    plt.subplot(2, 3, 1)
    plt.imshow(images[39+20000])
    plt.subplot(2, 3, 2)
    plt.imshow(images[5+20000])
    plt.subplot(2, 3, 3)
    plt.imshow(images[8+20000])
    plt.subplot(2, 3, 4)
    plt.imshow(images[44+20000])
    plt.subplot(2, 3, 5)
    plt.imshow(images[47+20000])
    plt.subplot(2, 3, 6)
    plt.imshow(images[53+20000])
    plt.show()


def subplot_dataset_test():
    images,_,_ = load_to_hdf5('F:\Klemen_diploma\light_estimation\estimation\dataset\LED128x128_test_1.hdf5', data_mode=DataMode.RADIANS)
    plt.suptitle("Podatkovna množica Synthetic_128x128_test", fontsize=14)
    plt.subplot(2, 2, 1)
    plt.imshow(images[0])
    plt.subplot(2, 2, 2)
    plt.imshow(images[1])
    plt.subplot(2, 2, 3)
    plt.imshow(images[2])
    plt.subplot(2, 2, 4)
    plt.imshow(images[111])
    plt.show()


def subplot_dataset_test_real():
    images = [None] * 100
    dirs = ["F:\Klemen_diploma\light_estimation\evaluation\img/angle_aruco", "F:\Klemen_diploma\light_estimation\evaluation\img/angle"]
    for directory in dirs:
        for image_path in sorted(os.listdir(directory)):
            number, ext = image_path.split(sep=".")
            image_path = os.path.join(directory, image_path)
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = np.array(image)
            images[int(number)-1] = image

    images = np.array(images)

    plt.suptitle("Podatkovna množica realnih slik", fontsize=14)
    plt.subplot(2, 2, 1)
    plt.imshow(images[58])
    plt.subplot(2, 2, 2)
    plt.imshow(images[50])
    plt.subplot(2, 2, 3)
    plt.imshow(images[69])
    plt.subplot(2, 2, 4)
    plt.imshow(images[98])
    plt.show()


if __name__ == '__main__':
    #plot_dataset()
    subplot_dataset_big()
    #subplot_dataset_test()
    #subplot_dataset_test_real()
