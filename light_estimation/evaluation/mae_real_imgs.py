import os

import numpy

from estimation.model import efficient_net_b3
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error
import json

if __name__ == "__main__":
    model = efficient_net_b3()
    model.load_state_dict(torch.load("F:\Klemen_diploma\light_estimation\estimation\models_old\model_20231024_125920_28", map_location=torch.device("cpu")))

    model.eval()

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
    # np.random.shuffle(images)
    labels = [None] * 100
    for label_path in sorted(os.listdir("F:\Klemen_diploma\light_estimation\evaluation\labels")):
        number, ext = label_path.split(sep=".")
        label_path = os.path.join("F:\Klemen_diploma\light_estimation\evaluation\labels", label_path)
        f = open(label_path)
        json_data = json.load(f)
        labels[int(number)-1] = json_data["pos"]

    labels = np.array(labels)
    # np.random.shuffle(labels)
    preds = model(torch.from_numpy(images.swapaxes(1, 3).swapaxes(2, 3)).float())
    preds[:, 0] =  preds[:, 0] * np.pi * 2
    preds[:, 1] = preds[:, 1] * (np.pi / 2) 
    print(preds)
    print(labels)

    maeA = mean_absolute_error(labels[:, 0], preds[:, 0].detach().numpy())
    maeE = mean_absolute_error(labels[:, 1], preds[:, 1].detach().numpy())

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')




