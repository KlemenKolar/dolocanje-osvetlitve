import numpy

from estimation.model import efficient_net_b3_discrete, efficient_net_b7_discrete, efficient_net_b5_discrete
import torch
from evaluation.utils import load_to_hdf5
from estimation.enums import DataMode
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error
from evaluation.utils import compare_discrete, print_each, compare_discrete2, circular_mae
import os
import json
from estimation.config import img_dir_path, labels_dir_path


def synthetic():
    model = efficient_net_b3_discrete()
    model.load_state_dict(torch.load("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-nopretrained-shortdataset/model_21idnopretrained-shortdataset", map_location=torch.device("cpu")))

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


def real(path_to_model, model=efficient_net_b3_discrete()):
    #model = efficient_net_b3_discrete()
    try:
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device("cpu")))

        model.eval()

        images = [None] * 51
        dirs = [f"{img_dir_path}/angle"] # f"{img_dir_path}/angle_aruco",
        for directory in dirs:
            for image_path in sorted(os.listdir(directory)):
                number, ext = image_path.split(sep=".")
                image_path = os.path.join(directory, image_path)
                image = Image.open(image_path)
                image = image.resize((128, 128))
                image = np.array(image)
                images[int(number) - 50] = image

        images = np.array(images)
        # np.random.shuffle(images)
        labels = [None] * 51
        for label_path in sorted(os.listdir(labels_dir_path)):
            number, ext = label_path.split(sep=".")
            label_path = os.path.join(labels_dir_path, label_path)
            f = open(label_path)
            json_data = json.load(f)
            labels[int(number) - 50] = json_data["pos"]


        # np.random.shuffle(labels)
        preds = model(torch.from_numpy(images.swapaxes(1, 3).swapaxes(2, 3)).float())
        preds = [preds[0].detach().numpy(), preds[1].detach().numpy()]

        preds_bin, truth_bin = compare_discrete2(preds, labels)

        print_each(preds_bin, truth_bin)

        maeA = circular_mae(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0], 32)
        maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])

        print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
    except:
        maeA = maeE = 100_000_000
    finally:
        return maeA, maeE


if __name__ == "__main__":
    real("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-big-dataset-greyscale/model_32", model=efficient_net_b3_discrete())
    # synthetic()
