from estimation.config import h_dataset, w_dataset
from estimation.model import (efficient_net_b3_discrete, efficient_net_b3_heatmap, efficient_net_b3,
                               efficient_net_b7_heatmap_no_bottleneck2)
import torch
from estimation.enums import DataMode
import numpy as np
from PIL import Image
from evaluation.utils import visualize_heatmap

def evaluate_single(img, model_path, datamode, a_bins=32, b_bins=16, img_id=None, dest_dir=None):
    img = np.resize(img, (h_dataset, w_dataset, 3))
    match datamode:
        case DataMode.DISCRETE:
            model = efficient_net_b3_discrete()
        case DataMode.RADIANS:
            model = efficient_net_b3()
        case DataMode.HEATMAP:
            model = efficient_net_b7_heatmap_no_bottleneck2()

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    img = np.array(img)
    img = torch.from_numpy(img.swapaxes(0, 2).swapaxes(1, 2)).float().unsqueeze(0)
    pred = model(img)

    match datamode:
        case DataMode.DISCRETE:
            pred = [pred[0].detach().numpy(), pred[1].detach().numpy()]
            return [np.argmax(pred[0]) * (2*np.pi / a_bins), np.argmax(pred[1]) * ((1.05864) / b_bins)] # 1.05864 je max vrednost kota beta v množici LED128x128.hdf5
        case DataMode.RADIANS:
            pred = [pred[0].detach().numpy(), pred[1].detach().numpy()]
            return pred
        case DataMode.HEATMAP:
            pred = pred.detach().numpy()
            index = np.argmax(pred)
            a_angle = np.floor(index / b_bins)
            b_angle = (index % b_bins)
            #visualize_heatmap([pred], dest_dir, img_id, a_bins, b_bins)
            return [a_angle * (2*np.pi / a_bins), b_angle * ((1.05864) / b_bins)]


if __name__ == "__main__":
    #img = cv2.imread("/home/klemen/light_estimation/visualization/calibration_images/SamsungA54/20240827_184913.jpg")
    #img = Image.open("/home/klemen/light_estimation/visualization/calibration_images/SamsungA54/20240827_184913.jpg")
    #evaluate_single(img, "/home/klemen/light_estimation/estimation/models/efficient_net_b3_discrete-big-dataset-discrete-brightness-p1-br01/model_0", DataMode.DISCRETE)
    pass