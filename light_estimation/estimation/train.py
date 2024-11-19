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
from estimation.augmentation import *

from estimation.config import (checkpoints_dir_path, dataset_dir_path,
                    dataset_size_stamp, default_dataset,
                    hdf5_dataset_path, logs_dir_path,
                    model_architecture, model_dir_path, test_size,
                    train_test_split_random_state)
from estimation.enums import DataMode
from estimation.model import create_model
from estimation.utils import check_create_dir, get_index_from_heatmap
from estimation.interpolate import get_interpolated_matrix, get_interpolated_vector, get_interpolated_gauss_matrix
import torch.nn.functional as F
import mlflow
from estimation.loss import *
import json


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

        if data_mode == DataMode.DISCRETE or data_mode == DataMode.HEATMAP:
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


def log_training_parameters(id, model_architecture, dataset_path, model_path, weights, data_mode, learning_rate,
                             batch_size, epochs, a_bins, b_bins, load_model, transform, epoch_reached, model,
                             loss_history, vloss_history, full_elapsed_time):
    with mlflow.start_run():
        mlflow.log_param("id", id)
        mlflow.log_param("model_architecture", model_architecture)
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("weights", weights)
        mlflow.log_param("data_mode", data_mode)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("a_bins", a_bins)
        mlflow.log_param("b_bins", b_bins)
        mlflow.log_param("load_model", load_model)
        mlflow.log_param("transform", transform)
        mlflow.log_param("erpoch_reached", epoch_reached)
        mlflow.log_param("loss_history", loss_history)
        mlflow.log_param("vloss_history", vloss_history)
        mlflow.log_param("full_elapsed_time", full_elapsed_time)
        mlflow.pytorch.log_model(model, f'model_{epoch_reached}id{id}')


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
            #sample = torch.from_numpy(np.array(sample).swapaxes(0, 2).swapaxes(1, 2)).float()
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
        learning_rate: float = 0.0002,
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
            },
            #DataMode.HEATMAP: nn.CrossEntropyLoss(),
            #DataMode.HEATMAP: CircularDistanceLoss(a_bins=a_bins)
            DataMode.HEATMAP: nn.KLDivLoss(reduction='batchmean'),
        }[data_mode]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        transform = transforms.Compose([
            #RandomBrightness(mean=1.0, std=0.2),
            #RandomContrast(mean=1.0, std=0.2),
            #RandomSaturation(mean=1.0, std=0.2),
            #RandomHue(mean=0.0, std=0.1),
        ])

        #transform = transforms.Compose([
        #    RandomHue(0.0, std=0.1)
        #])

        transform = None

        patience = 10

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        loss_history = []
        vloss_history = [] 

        full_elapsed_time = 0

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

                inputs = inputs.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                # Compute the loss and its gradients
                labels = labels.to(device)
                loss = loss_function["angle_a"](outputs[0], labels[:, 0]) + loss_function["angle_b"](outputs[1], labels[:, 1])

                #softmax_labels_a = get_interpolated_vector(a_bins, labels[:, 0].numpy()) kldivergence
                #softmax_labels_b = get_interpolated_vector(b_bins, labels[:, 0].numpy())
                #softmax_outputs_a = F.log_softmax(outputs[0], dim=1)
                #softmax_outputs_b = F.log_softmax(outputs[1], dim=1)
                #softmax_labels_a, softmax_outputs_a, softmax_labels_b, softmax_outputs_b = softmax_labels_a.to(device), softmax_outputs_a.to(device), softmax_labels_b.to(device), softmax_outputs_b.to(device)
                #loss = loss_function["angle_a"](softmax_outputs_a, softmax_labels_a) + loss_function["angle_b"](softmax_outputs_b, softmax_labels_b)

                # loss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(outputs, labels)
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
                    vinputs = vinputs.to(device)
                    voutputs = model(vinputs)
                    #vloss = loss_function(voutputs, vlabels)
                    vlabels = vlabels.to(device)
                    vloss = loss_function["angle_a"](voutputs[0], vlabels[:, 0]) + loss_function["angle_b"](voutputs[1], vlabels[:, 1])

                    #softmax_vlabels_a = get_interpolated_vector(a_bins, labels[:, 0].numpy())
                    #softmax_vlabels_b = get_interpolated_vector(b_bins, labels[:, 0].numpy())
                    #softmax_voutputs_a = F.log_softmax(outputs[0], dim=1)
                    #softmax_voutputs_b = F.log_softmax(outputs[1], dim=1)
                    #softmax_vlabels_a, softmax_voutputs_a, softmax_vlabels_b, softmax_voutputs_b = softmax_vlabels_a.to(device), softmax_voutputs_a.to(device), softmax_vlabels_b.to(device), softmax_voutputs_b.to(device)
                    #vloss = loss_function["angle_a"](softmax_voutputs_a, softmax_vlabels_a) + loss_function["angle_b"](softmax_voutputs_b, softmax_vlabels_b)

                    # vloss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            loss_history.append(avg_loss)
            vloss_history.append(avg_vloss)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_dir = os.path.join(model_dir_path, f'{model_architecture}-{run_id}')
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
            epoch_time = round((end_time - start_time) / 60, 1)
            full_elapsed_time += epoch_time
            print(f'Epoch {epoch_number} elapsed time: {epoch_time} minutes')

    elif data_mode == DataMode.HEATMAP:
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

                labels_merged = labels[:, 0] * b_bins + labels[:, 1]

                inputs = inputs.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)

                '''outputs = outputs.cpu()

                # Extract angles from the output matrices
                outputs =  torch.tensor(np.array([torch.unravel_index(torch.argmax(output), output.shape) for output in outputs]).astype(np.float32), requires_grad=True)

                outputs = outputs.to(device)'''

                # Compute the loss and its gradients
                # loss = loss_function["angle_a"](outputs[:, 0], labels[:, 0]) + loss_function["angle_b"](outputs[:, 1], labels[:, 1]) na ta način odkomentiraš zgornjo kodo in v layers.py vračaš return heatmap.view(-1, self.a_bins, self.b_bins)
                # loss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(outputs, labels)
                

                #1. uncomment if you want to use KLDivergenceloss softmax
                #softmax_labels = get_interpolated_matrix(a_bins, b_bins, labels_merged, zero=True)
                #softmax_outputs = F.log_softmax(outputs, dim=1)
                #softmax_labels, softmax_outputs = softmax_labels.to(device), softmax_outputs.to(device)
                #loss = loss_function(softmax_outputs, softmax_labels)
                
                #2.uncomment for crossentropyloss
                #labels_merged, outputs = labels_merged.to(device), outputs.to(device) 
                #loss = loss_function(outputs, labels_merged)

                #3. uncomment for circulardistanceloss
                #outputs_indexes = torch.tensor(get_index_from_heatmap(outputs.cpu(), b_bins), requires_grad=True)
                #outputs_indexes, labels = outputs_indexes.to(device), labels.to(device)
                #loss = loss_function(outputs_indexes, labels)

                #4. uncomment for KLDiv/crossent loss gauss normalized
                labels = get_interpolated_gauss_matrix(a_bins, b_bins, labels_merged, sigma=3)
                softmax_outputs = F.log_softmax(outputs, dim=1)
                labels, softmax_outputs = labels.to(device), softmax_outputs.to(device)
                loss = loss_function(softmax_outputs, labels)

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
                    vlabels_merged = vlabels[:, 0] * b_bins + vlabels[:, 1]
                    vinputs = vinputs.to(device)
                    voutputs = model(vinputs)
                    '''voutputs = voutputs.cpu()
                    voutputs =  torch.tensor(np.array([torch.unravel_index(torch.argmax(voutput), voutput.shape) for voutput in voutputs]).astype(np.float32), requires_grad=True)
                    voutputs = voutputs.to(device)
                    #vloss = loss_function(voutputs, vlabels)
                    vloss = loss_function["angle_a"](voutputs[:, 0], vlabels[:, 0]) + loss_function["angle_b"](voutputs[:, 1], vlabels[:, 1])'''
                    # vloss = CustomDiscreteLoss(a_bins=a_bins, b_bins=b_bins, weight_losses=False)(voutputs, vlabels)
                    
                    #1. uncomment for crossentropyloss
                    #vlabels_merged = vlabels_merged.to(device)
                    #vloss = loss_function(voutputs, vlabels_merged)

                    #2. uncomment for kldiv loss
                    #softmax_vlabels = get_interpolated_matrix(a_bins, b_bins, vlabels_merged, zero=True)
                    #softmax_voutputs = F.log_softmax(voutputs, dim=1)
                    #softmax_vlabels, softmax_voutputs = softmax_vlabels.to(device), softmax_voutputs.to(device)
                    #vloss = loss_function(softmax_voutputs, softmax_vlabels)

                    #3. uncomment for circular distance loss
                    #voutputs_indexes = torch.tensor(get_index_from_heatmap(outputs.cpu(), b_bins), requires_grad=True)
                    #voutputs_indexes = voutputs_indexes.to(device), vlabels.to(device)
                    #vloss = loss_function(voutputs_indexes, vlabels)

                    #4. uncomment for KLDiv/crossent loss gauss normalized
                    vlabels = get_interpolated_gauss_matrix(a_bins, b_bins, vlabels_merged, sigma=3)
                    softmax_voutputs = F.log_softmax(voutputs, dim=1)
                    vlabels, softmax_voutputs = vlabels.to(device), softmax_voutputs.to(device)
                    vloss = loss_function(softmax_voutputs, vlabels)

                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            loss_history.append(avg_loss)
            vloss_history.append(avg_vloss)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_dir = os.path.join(model_dir_path, f'{model_architecture}-{run_id}')
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
            epoch_time = round((end_time - start_time) / 60, 1)
            full_elapsed_time += epoch_time
            print(f'Epoch {epoch_number} elapsed time: {epoch_time} minutes')  
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
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            loss_history.append(avg_loss)
            vloss_history.append(avg_vloss)

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_dir = os.path.join(model_dir_path, f'{model_architecture}-{run_id}')
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
            epoch_time = round((end_time - start_time) / 60, 1)
            full_elapsed_time += epoch_time
            print(f'Epoch {epoch_number} elapsed time: {epoch_time} minutes')

    loss_history=json.dumps(loss_history)
    vloss_history=json.dumps(vloss_history)    
    log_training_parameters(
        id=run_id,
        model_architecture=model_architecture,
        dataset_path=dataset_dir_path,
        model_path=model_dir,
        weights=weights,
        data_mode=DataMode(data_mode),
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        a_bins=a_bins,
        b_bins=b_bins,
        load_model=load_model,
        transform=transform,
        epoch_reached=epoch_number,
        model=model,
        loss_history=json.dumps(loss_history),
        vloss_history=json.dumps(vloss_history),
        full_elapsed_time=full_elapsed_time,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-id',
        help='run id',
        type=str,
        required=False,
        default='No_id'
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
        default=None,
        #default=models.EfficientNet_B7_Weights.IMAGENET1K_V1 # Pretrained weights, you can put None if you don't want this
    )

    parser.add_argument(
        '--data_mode',
        help='data mode',
        choices=[x.value for x in DataMode],
        default=DataMode.RADIANS.value
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

    parser.add_argument(
        '--add_model_arg',
        help='additional argument for model architecture',
        choices=['no_bottleneck', 'no_bottleneck2', 'no_bottleneck_no_relu'],
        default=''
    )

    args = parser.parse_args()

    print(*[f'{x}: {y}' for x, y in vars(args).items()], sep='\n')

    model = args.model + (
        '_discrete'  if args.data_mode == DataMode.DISCRETE.value else '_heatmap' if args.data_mode == DataMode.HEATMAP.value else ''
    )

    model = model + (
        f'_{args.add_model_arg}'  if args.add_model_arg != '' else ''
    )

    load_model = args.load_model and f'{model_dir_path}/{args.load_model}'

    train(
        run_id=args.id,
        model_architecture=model,
        dataset_path=f"{dataset_dir_path}", #/{args.dataset}",
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
