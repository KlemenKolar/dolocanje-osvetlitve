import os
from estimation.model import efficient_net_b3, efficient_net_b5_discrete, efficient_net_b7
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error
import json
from estimation.config import labels_dir_path, img_dir_path

def real(model_path, model=None):
    model = efficient_net_b7()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

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
            images[int(number)-1] = image

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
            labels[int(number)-1] = json_data["pos"]

    labels = np.array(labels)
    # np.random.shuffle(labels)
    preds = model(torch.from_numpy(images.swapaxes(1, 3).swapaxes(2, 3)).float())
    preds[:, 0] =  preds[:, 0] * np.pi * 2
    preds[:, 1] = preds[:, 1] * (np.pi / 2) 
    #print(preds)
    print(np.array(labels))

    maeA = mean_absolute_error(labels[:, 0], preds[:, 0].detach().numpy())
    maeE = mean_absolute_error(labels[:, 1], preds[:, 1].detach().numpy())

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')

    return maeA, maeE


def get_results(model_dir, model_backbone):
    for model in os.listdir(model_dir):
        maeA, maeE = real(os.path.join(model_dir, model), model=model_backbone)
        with open(os.path.join(model_dir, 'results.txt'), 'a') as file:
            # Write the content to the file
            file.write(f'{model} --> MAE azimuth: {maeA}, MAE elevation: {maeE}\n')


if __name__ == "__main__":
    real("/home/klemen/light_estimation/estimation/models/efficient_net_b7-big-dataset-/model_2")
    #get_results("/home/klemen/light_estimation/estimation/models/efficient_net_b7-big-dataset-", efficient_net_b7())




