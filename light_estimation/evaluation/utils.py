import numpy as np
from estimation.enums import DataMode
import h5py
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import KBinsDiscretizer

# Helper functions


def compare_discrete(preds, truth):
    preds_bin = []
    if preds:
        for pred0, pred1 in zip(preds[0], preds[1]):
            preds_bin.append([np.argmax(pred0), np.argmax(pred1)])

    truth_bin = []
    if truth:
        dict32 = {((i + 1) / 32) * 2 * np.pi: i for i in np.arange(32)}
        dict16 = {((i + 1) / 16) * np.pi / 2: i for i in np.arange(16)}
        for true in truth:
            truth_bin.append([dict32[min(dict32, key=lambda x:abs(x-(true[0]*2*np.pi)))], dict16[min(dict16, key=lambda x:abs(x-(true[1]*np.pi/2)))]])

    return preds_bin, truth_bin


def compare_discrete2(preds, truth):

    preds_bin = []
    if preds:
        for pred0, pred1 in zip(preds[0], preds[1]):
            preds_bin.append([np.argmax(pred0), np.argmax(pred1)])
    truth = np.array(truth)
    truth_bin = truth
    a_angle_discretizer = KBinsDiscretizer(
        n_bins=len(preds[0][0]),
        encode='ordinal',
        strategy='uniform'
    )
    truth_bin[:, 0] = a_angle_discretizer.fit_transform(
        truth[:, 0].reshape(-1, 1)
    ).flatten()

    b_angle_discretizer = KBinsDiscretizer(
        n_bins=len(preds[1][0]),
        encode='ordinal',
        strategy='uniform'
    )
    truth_bin[:, 1] = b_angle_discretizer.fit_transform(
        truth[:, 1].reshape(-1, 1)
    ).flatten()

    return preds_bin, truth_bin


def print_each(preds, truth_data, names=None):
    for predicted, truth in zip(preds, truth_data):
        print(f'predicted: {predicted[0]}, {predicted[1]} | truth: {truth[0]}, {truth[1]}')


def load_to_hdf5(
        hdf5_dataset_path: str,
        data_mode: DataMode,
        a_bins: int = 32,
        b_bins: int = 16
) -> tuple:
    with h5py.File(hdf5_dataset_path, 'r') as file:
        images = np.array(file['images'], dtype=np.uint8)
        ambient = np.array(file['ambient'], dtype=float)
        light_angles = np.array(file['light_angles'], dtype=float)

        if data_mode == DataMode.DISCRETE:
            n = images.shape[0]
            light_data = np.zeros((n, 2))

            a_angle_discretizer = KBinsDiscretizer(
                n_bins=a_bins,
                encode='ordinal',
                strategy='uniform'
            )
            light_data[:, 0] = a_angle_discretizer.fit_transform(
                light_angles[:, 0].reshape((n, 1))
            ).flatten()

            b_angle_discretizer = KBinsDiscretizer(
                n_bins=b_bins,
                encode='ordinal',
                strategy='uniform'
            )
            light_data[:, 1] = b_angle_discretizer.fit_transform(
                light_angles[:, 1].reshape((n, 1))
            ).flatten()

        else:
            light_dataset_name = {
                DataMode.RADIANS: 'light_angles',
                DataMode.STEREOGRAPHIC: 'stereographic_light_angles'
            }[data_mode]

            light_data = np.array(
                file[light_dataset_name],
                dtype=float
            )

            if data_mode == DataMode.RADIANS:
                light_data[:, 0] /= 2 * np.pi
                light_data[:, 1] /= np.pi / 2

    return images, light_data, ambient
