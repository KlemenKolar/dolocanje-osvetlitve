import numpy

from estimation.model import efficient_net_b3_heatmap, efficient_net_b7_heatmap_no_bottleneck2, efficient_net_b5_heatmap, efficient_net_b7_heatmap, efficient_net_b3_heatmap_no_bottleneck2
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error
from evaluation.utils import print_each, compare_heatmap, circular_mae, visualize_heatmap, visualize_reference_heatmap, get_adjacent_indices_circular, normalize_array
import os
import json
from estimation.config import img_dir_path, labels_dir_path
import time

random_num = np.random.randint(100)


def real(path_to_model, model=efficient_net_b3_heatmap(), sigma=1, estimate_inference=False, metric='mae'):
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


    # np.random.shuffle(labels)

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
        preds = preds.detach().numpy()
        preds_bin, truth_bin = compare_heatmap(preds, labels, 32, 16)

        if metric == 'mae':
            #print_each(preds_bin, truth_bin)
        
            maeA = circular_mae(numpy.array(truth_bin)[:, 0], numpy.array(preds_bin)[:, 0], 32)
            varianceA = np.var(abs(numpy.array(truth_bin)[:, 0]) - numpy.array(preds_bin)[:, 0])

            maeE = mean_absolute_error(numpy.array(truth_bin)[:, 1], numpy.array(preds_bin)[:, 1])
            varianceE = np.var(abs(numpy.array(truth_bin)[:, 1]) - numpy.array(preds_bin)[:, 1])

            #visualize_heatmap(preds, preds_bin, truth_bin, path_to_model.split("/")[6])
            #visualize_reference_heatmap(truth_bin, path_to_model.split("/")[6], sigma)

            print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
            print(f'Variance azimuth: {varianceA}, variance elevation: {varianceE}')

        elif metric == 'subpixel':
            maeA = 0
            maeE = 0
            for i, pred in enumerate(preds_bin):
                indices = get_adjacent_indices_circular(pred[0], pred[1], 32, 16, 2)
                truth = truth_bin[i]
                subpixel_maeA = 0
                subpixel_maeE = 0
                weight_sum = 0
                preds_i = normalize_array(preds[i])
                for j, ind in enumerate(indices):
                    weight = preds_i[int(ind[0] * 16 + ind[1])]
                    weight_sum += weight
                    subpixel_maeA += circular_mae(numpy.array(truth)[0], numpy.array([ind[0]]), 32) * weight
                    subpixel_maeE += circular_mae(numpy.array(truth)[1], numpy.array([ind[1]]), 32) * weight
                subpixel_maeA = subpixel_maeA / weight_sum
                subpixel_maeE = subpixel_maeE / weight_sum
                maeA += subpixel_maeA
                maeE += subpixel_maeE
            maeA = maeA / (i + 1)
            maeE = maeE / (i + 1)
            print(f'Subpixel MAE azimuth: {maeA}, Subpixel MAE elevation: {maeE}')

        return maeA, maeE


    

def get_results(model_dir, model_backbone):
    for model in os.listdir(model_dir):
        maeA, maeE = real(os.path.join(model_dir, model), model=model_backbone)
        with open(os.path.join(model_dir, 'results.txt'), 'a') as file:
            # Write the content to the file
            file.write(f'{model} --> MAE azimuth: {maeA}, MAE elevation: {maeE}\n')


if __name__ == "__main__":
    real("/home/klemen/light_estimation/estimation/models/efficient_net_b5_heatmap-big-dataset-heatmap-vectortrain/model_9", model=efficient_net_b5_heatmap(), estimate_inference=True)
    real("/home/klemen/light_estimation/estimation/models/efficient_net_b7_heatmap-big-dataset-heatmap-crossent/model_2", model=efficient_net_b7_heatmap(), estimate_inference=True)
    #get_results("/home/klemen/light_estimation/estimation/models/efficient_net_b7_heatmap_no_bottleneck2-big-dataset-heatmap-crossent-interpolate-gauss-sigma3-aug-saturation", model_backbone=efficient_net_b7_heatmap_no_bottleneck2())
    pass
