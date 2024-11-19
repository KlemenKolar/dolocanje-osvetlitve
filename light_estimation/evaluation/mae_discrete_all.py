import numpy

from estimation.model import efficient_net_b3_discrete, efficient_net_b7_discrete, efficient_net_b5_discrete, efficient_net_b3_heatmap_no_bottleneck
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
import time

random_num = np.random.randint(100)


def synthetic():
    model = efficient_net_b3_discrete()
    model.load_state_dict(torch.load("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-nopretrained-shortdataset/model_21idnopretrained-shortdataset", map_location=torch.device("cpu")))

    (
        input_data,
        truth_data,
        ambient
    ) = load_to_hdf5("/home/klemen/light_estimation/estimation/dataset/LED_old128x128.hdf5", DataMode.DISCRETE)

    model.eval()

    preds = model(torch.from_numpy(input_data.swapaxes(1, 3).swapaxes(2, 3)).float())

    preds = [preds[0].detach().numpy(), preds[1].detach().numpy()]

    preds_bin, truth_bin = compare_discrete(preds, truth_data)

    maeA = mean_absolute_error(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0])
    maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])

    for predicted, truth in zip(preds_bin, truth_bin):
        print(f'predicted: {predicted[0]}, {predicted[1]} | truth: {truth[0]}, {truth[1]}')

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')


def real(path_to_model, model=efficient_net_b3_discrete(), estimate_inference=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    images = [None] * 100
    dirs = [f"{img_dir_path}/angle_aruco", f"{img_dir_path}/angle"]
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
    dirs = ["/home/klemen/light_estimation/evaluation/labels", "/home/klemen/light_estimation/evaluation/angle_aruco_labels"]
    for directory in dirs:
        for label_path in sorted(os.listdir(directory)):
            number, ext = label_path.split(sep=".")
            label_path = os.path.join(directory, label_path)
            f = open(label_path)
            json_data = json.load(f)
            labels[int(number) - 1] = json_data["pos"]


    inputs = torch.from_numpy(images.swapaxes(1, 3).swapaxes(2, 3)).float()
    if estimate_inference:
        inputs = inputs[random_num].unsqueeze(0)

    inputs.to(device)

    time_sum = 0
    if estimate_inference:
        for i in range(100):
            start_time = time.time()
            preds = model(inputs)
            end_time = time.time()
            time_sum += end_time-start_time
        print(f'Duration of inference: {time_sum/100} sec')

    else:
        preds = model(inputs)
        preds = [preds[0].detach().numpy(), preds[1].detach().numpy()]

        preds_bin, truth_bin = compare_discrete(preds, labels)

        print(preds_bin)
        print(truth_bin)
        #preds_bin, truth_bin = compare_discrete(preds, labels)

        #print_each(preds_bin, truth_bin)

        maeA = circular_mae(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0], 32)
        maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])

        print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
        return maeA, maeE
    

def get_results(model_dir, model_backbone):
    for model in os.listdir(model_dir):
        maeA, maeE = real(os.path.join(model_dir, model), model=model_backbone)
        with open(os.path.join(model_dir, 'results.txt'), 'a') as file:
            # Write the content to the file
            file.write(f'{model} --> MAE azimuth: {maeA}, MAE elevation: {maeE}\n')


if __name__ == "__main__":
    #real("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-big-dataset-classic/model_2", model=efficient_net_b3_discrete())
    #get_results("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-big-dataset-discrete-bbins8", model_backbone=efficient_net_b3_discrete(a_bins=32, b_bins=8)
    #synthetic()
    #get_results("/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-big-dataset-brightness", model_backbone=efficient_net_b3_discrete())
    pass
