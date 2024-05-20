import numpy

from estimation.model import efficient_net_b3
import torch
import h5py
from estimation.train import load_data_hdf5
from estimation.enums import DataMode
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    model = efficient_net_b3()
    model.load_state_dict(torch.load("../estimation/models/model_20231228_143259_54", map_location=torch.device("cpu")))

    (
        images_train,
        images_test,
        light_data_train,
        light_data_test,
        ambient_data_train,
        ambient_data_test
    ) = load_data_hdf5("../estimation/dataset/LED128x128_test_1.hdf5", DataMode.RADIANS)

    model.eval()
    """
    image = np.array(Image.open("../estimation/real_images/real_scene_128_2.jpg"))

    pred = model(torch.from_numpy(np.expand_dims(image, axis=0).swapaxes(1, 3).swapaxes(2, 3)).float())
    truth = light_data_train[0]
    
    print(f'prediction: {pred}, truth: {truth}')
    """
    input_data = numpy.concatenate((images_train, images_test), axis=0)
    truth_data = numpy.concatenate((light_data_train, light_data_test), axis=0)

    preds = model(torch.from_numpy(input_data.swapaxes(1, 3).swapaxes(2, 3)).float())
    print(light_data_train[:, 0])
    print(preds[:,0])
    maeA = mean_absolute_error(truth_data[:, 0], preds[:, 0].detach().numpy())
    maeE = mean_absolute_error(truth_data[:, 1], preds[:, 1].detach().numpy())

    for predicted, truth in zip(preds, truth_data):
        print(f'predicted: {predicted[0]}, {predicted[1]} | truth: {truth[0]}, {truth[1]}')

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')

    # for image in images_train:
    #     plt.imshow(image)
    #     plt.show()




