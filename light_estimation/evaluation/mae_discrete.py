import numpy

from estimation.model import efficient_net_b3_discrete
import torch
import h5py
from utils import load_to_hdf5
from estimation.enums import DataMode
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error
from utils import compare_discrete, print_each, compare_discrete2
import os
import json


def synthetic():
    model = efficient_net_b3_discrete()
    model.load_state_dict(torch.load("../estimation/models/efficient_net_b3_discrete-2642024/model_28", map_location=torch.device("cpu")))

    (
        input_data,
        truth_data,
        ambient
    ) = load_to_hdf5("../estimation/dataset/Only_Material_Dataset_Shorter.hdf5", DataMode.DISCRETE)

    model.eval()

    preds = model(torch.from_numpy(input_data.swapaxes(1, 3).swapaxes(2, 3)).float())

    preds = [preds[0].detach().numpy(), preds[1].detach().numpy()]

    preds_bin, truth_bin = compare_discrete(preds, truth_data)

    maeA = mean_absolute_error(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0])
    maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])

    for predicted, truth in zip(preds_bin, truth_bin):
        print(f'predicted: {predicted[0]}, {predicted[1]} | truth: {truth[0]}, {truth[1]}')

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')


def real():
    model = efficient_net_b3_discrete()
    model.load_state_dict(torch.load("../estimation/models/efficient_net_b3_discrete-2642024/model_28", map_location=torch.device("cpu")))

    model.eval()

    images = [None] * 100
    dirs = ["img/angle_aruco", "img/angle"]
    for directory in dirs:
        for image_path in sorted(os.listdir(directory)):
            number, ext = image_path.split(sep=".")
            image_path = os.path.join(directory, image_path)
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = np.array(image)
            images[int(number) - 1] = image

    images = np.array(images)
    # np.random.shuffle(images)
    labels = [None] * 100
    for label_path in sorted(os.listdir("labels")):
        number, ext = label_path.split(sep=".")
        label_path = os.path.join("labels", label_path)
        f = open(label_path)
        json_data = json.load(f)
        labels[int(number) - 1] = json_data["pos"]


    # np.random.shuffle(labels)
    preds = model(torch.from_numpy(images.swapaxes(1, 3).swapaxes(2, 3)).float())
    preds = [preds[0].detach().numpy(), preds[1].detach().numpy()]

    preds_bin, truth_bin = compare_discrete2(preds, labels)

    print_each(preds_bin, truth_bin)

    maeA = mean_absolute_error(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0])
    maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')


if __name__ == "__main__":
    # real()
    synthetic()
