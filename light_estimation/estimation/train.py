import argparse
import os
import time
from PIL import Image

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import pyplot as plt

from estimation.config import (checkpoints_dir_path, dataset_dir_path,
                    dataset_size_stamp, default_dataset,
                    hdf5_dataset_path, logs_dir_path,
                    model_architecture, model_dir_path, test_size,
                    train_test_split_random_state)
from estimation.enums import DataMode
from estimation.model import create_model
from estimation.utils import check_create_dir
from loss import *


def load_data_hdf5(
        hdf5_dataset_path: str = hdf5_dataset_path,
        data_mode: DataMode = DataMode.RADIANS,
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

    return train_test_split(
        images,
        light_data,
        ambient,
        test_size=test_size,
        random_state=train_test_split_random_state
    )


def load_data_npy(
        images_dataset_path: str,
        light_pos_data_dataset_path: str,
        ambient_dataset_path: str,
        data_mode: DataMode = DataMode.RADIANS
) -> tuple:
    images = np.load(f'{images_dataset_path}.npy', mmap_mode='r')
    light_data = np.load(f'{light_pos_data_dataset_path}.npy')
    ambient_data = np.load(f'{ambient_dataset_path}.npy')

    if data_mode == DataMode.RADIANS:
        light_data[:, 0] /= 2 * np.pi
        light_data[:, 1] /= np.pi / 2

    return train_test_split(
        images,
        light_data,
        ambient_data,
        test_size=test_size,
        random_state=train_test_split_random_state
    )


def load_data(
        dataset_path: str,
        data_mode: DataMode,
        a_bins: int = 32,
        b_bins: int = 16
):
    hfd5_dataset_filename = os.path.basename(hdf5_dataset_path)

    if hfd5_dataset_filename in os.listdir(dataset_path):
        print("HDF5 dataset detected", os.path.abspath(
            f"{dataset_path}/{hfd5_dataset_filename}"
        ))

        return load_data_hdf5(
            hdf5_dataset_path=os.path.abspath(
                f"{dataset_path}/{hfd5_dataset_filename}"
            ),
            data_mode=data_mode,
            a_bins=a_bins,
            b_bins=b_bins
        )

    else:
        print("Legacy .npy datasets detected")
        print(*[
            f"{dataset_path}/images{dataset_size_stamp}",
            f"{dataset_path}/light_data_pos{dataset_size_stamp}",
            f"{dataset_path}/ambient{dataset_size_stamp}"
        ])
        return load_data_npy(
            images_dataset_path=f"{dataset_path}/images{dataset_size_stamp}",
            light_pos_data_dataset_path=f"{dataset_path}/light_data_pos{dataset_size_stamp}",
            ambient_dataset_path=f"{dataset_path}/ambient{dataset_size_stamp}",
            data_mode=data_mode
        )


class Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            pil_image = Image.fromarray(sample)
            sample = np.array(self.transform(pil_image))
            sample = torch.from_numpy(np.array(sample).swapaxes(0, 2).swapaxes(0, 1)).float()
            #plt.imshow(sample, cmap='gray')
            #plt.show()
        return sample, label


def train(
        run_id: str = None,
        model_architecture: str = None,
        dataset_path: str = None,
        model_path: str = None,
        weights: str = None,
        data_mode: DataMode = DataMode.RADIANS,
        learning_rate: float = 0.00001,
        batch_size: int = 32,
        epochs: int = 200,
        a_bins: int = 32,
        b_bins: int = 16,
        load_model: str = None
):
    """Trains the CNN and saves the model.

    Args:
        model_architecture (str, optional): Which architecture to use. Architectures are defined in model.py. Defaults to model_architecture.
        images_dataset_path (str, optional): Path to where images dataset is saved. Defaults to images_dataset_path.
        light_dataset_path (str, optional): Path to where light dataset (labels) is saved. Defaults to light_dataset_path.
        ambient_dataset_path (str, optional): Path to where ambient dataset (labels) is saved. Defaults to ambient_dataset_path.
    """

    check_create_dir(model_dir_path)
    check_create_dir(logs_dir_path)
    check_create_dir(checkpoints_dir_path)

    (
        images_train,
        images_test,
        light_data_train,
        light_data_test,
        ambient_data_train,
        ambient_data_test
    ) = load_data(
        dataset_path=dataset_path,
        data_mode=data_mode,
        a_bins=a_bins,
        b_bins=b_bins
    )

    print(images_train.shape, images_test.shape, sep='\n')

    if load_model is not None:
        print(f'loading model: {load_model}')
        model = models.efficientnet_b3(pretrained=True)
        checkpoint = torch.load(load_model)
        model.load_state_dict(checkpoint)

    else:
        model = create_model(
            model_architecture,
            weights=weights,
            data_mode=data_mode,
            a_bins=a_bins,
            b_bins=b_bins
        )

        loss_function = {
            DataMode.RADIANS: nn.MSELoss(),
            DataMode.STEREOGRAPHIC: nn.MSELoss(),
            DataMode.DISCRETE: {
                "angle_a": nn.CrossEntropyLoss(),
                "angle_b": nn.CrossEntropyLoss(),
            }
        }[data_mode]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Define your transformations
        transform = transforms.Compose([
            # Randomly apply grayscale conversion
            #transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=1),
            # Randomly apply brightness adjustment
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.2),
            # Randomly apply contrast adjustment
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.2),
            # Convert to tensor
            transforms.ToTensor(),
            # Randomly apply noise
            # transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.04)], p=0.2),
            # Randomly apply Gaussian blur
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2)
        ])

        patience = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(light_data_train.shape)

    if data_mode == DataMode.DISCRETE:
        train_dataset = Dataset(images_train, light_data_train, transform=transform)
        validation_dataset = Dataset(images_test, light_data_test, transform=transform)

        training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(epochs):
            start_time = time.time()
            print('EPOCH {}:'.format(epoch_number + 1))

            model.train(True)
            model.to(device)

            running_loss = 0.
            last_loss = 0.
            for i, (inputs, labels) in enumerate(training_loader):
                # Every data instance is an input + label pair
                inputs = torch.from_numpy(np.array(inputs).swapaxes(1, 3).swapaxes(2, 3)).float()
                labels = labels.long()

                inputs, labels = inputs.to(device), labels.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                # loss = loss_function["angle_a"](outputs[0], labels[:, 0]) + loss_function["angle_b"](outputs[1], labels[:, 1])
                loss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100  # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.

            avg_loss = last_loss
            running_vloss = 0.0

            model.eval()

            with torch.no_grad():
                for i, (vinputs, vlabels) in enumerate(validation_loader):
                    vinputs = torch.from_numpy(np.array(vinputs).swapaxes(1, 3).swapaxes(2, 3)).float()
                    vlabels = vlabels.long()
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    #vloss = loss_function(voutputs, vlabels)
                    # vloss = loss_function["angle_a"](voutputs[0], vlabels[:, 0]) + loss_function["angle_b"](voutputs[1], vlabels[:, 1])
                    vloss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_dir = f'models/{model_architecture}-{timestamp}'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                model_path = f'{model_dir}/model_{epoch_number}'
                torch.save(model.state_dict(), model_path)
                patience = 10
            else:
                patience -= 1

            if patience == 0:
                print("Early stopping, no improvement for too long.")
                break
            epoch_number += 1

            end_time = time.time()
            print(f'Epoch {epoch_number} elapsed time: {round((end_time - start_time) / 60, 1)} minutes')
    else:
        train_dataset = Dataset(images_train, light_data_train)
        validation_dataset = Dataset(images_test, light_data_test)

        training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        epoch_number = 0

        best_vloss = 1_000_000.

        for epoch in range(epochs):
            start_time = time.time()
            print('EPOCH {}:'.format(epoch_number + 1))

            model.train(True)
            model.to(device)

            running_loss = 0.
            last_loss = 0.
            for i, (inputs, labels) in enumerate(training_loader):
                # Every data instance is an input + label pair
                inputs = torch.from_numpy(np.array(inputs).swapaxes(1, 3).swapaxes(2, 3)).float()
                labels = labels.float()

                inputs, labels = inputs.to(device), labels.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                loss = loss_function(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()

                # Gather data and report
                running_loss += loss.item()
                if i % 100 == 99:
                    last_loss = running_loss / 100  # loss per batch
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    running_loss = 0.
            avg_loss = last_loss
            running_vloss = 0.0

            model.eval()

            with torch.no_grad():
                for i, (vinputs, vlabels) in enumerate(validation_loader):
                    vinputs = torch.from_numpy(np.array(vinputs).swapaxes(1, 3).swapaxes(2, 3)).float()
                    vlabels = vlabels.float()
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = loss_function(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_dir = f'models/{model_architecture}-{timestamp}'
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                model_path = f'{model_dir}/model_{epoch_number}'
                torch.save(model.state_dict(), model_path)
                patience = 10
            else:
                patience -= 1

            if patience == 0:
                print("Early stopping, no improvement for too long.")
                break
            epoch_number += 1

            end_time = time.time()
            print(f'Epoch {epoch_number} elapsed time: {round((end_time - start_time) / 60, 1)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-id',
        help='run id',
        type=str,
        required=False
    )

    parser.add_argument(
        '-m', '--model',
        help='which archtitecture to use for training, if not specified it is read from config',
        default=model_architecture
    )

    parser.add_argument(
        '-d', '--dataset',
        help='which dataset to use for training',
        default=default_dataset
    )

    parser.add_argument(
        '-w', '--weights',
        help="path for pretrained weights file",
        default=None
    )

    parser.add_argument(
        '--data_mode',
        help='data mode',
        choices=[x.value for x in DataMode],
        default=DataMode.DISCRETE.value
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200
    )

    parser.add_argument(
        '--a_bins',
        type=int,
        default=32
    )

    parser.add_argument(
        '--b_bins',
        type=int,
        default=16
    )

    parser.add_argument(
        '--load_model',
        help='id for model to load',
        default=None
    )

    args = parser.parse_args()

    print(*[f'{x}: {y}' for x, y in vars(args).items()], sep='\n')

    model = args.model + (
        '_discrete' if args.data_mode == DataMode.DISCRETE.value else ''
    )

    load_model = args.load_model and f'{model_dir_path}/{args.load_model}'

    train(
        run_id=args.id,
        model_architecture=model,
        dataset_path=f"{dataset_dir_path}",  # /{args.dataset}
        model_path=f'{model_dir_path}/{args.id}',
        weights=args.weights,
        data_mode=DataMode(args.data_mode),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        a_bins=args.a_bins,
        b_bins=args.b_bins,
        load_model=load_model
    )
