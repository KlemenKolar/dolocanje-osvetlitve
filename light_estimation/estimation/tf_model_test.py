import tensorflow as tf
from sklearn.metrics import mean_absolute_error, explained_variance_score
from train import load_data_hdf5
from enums import DataMode
import numpy
from PIL import Image
import os
import json


def print_each(preds, truth_data):
    for predicted, truth in zip(preds, truth_data):
        print(f'predicted: {predicted[0]}, {predicted[1]} | truth: {truth[0]}, {truth[1]}')

def resize_all(inputs, new_size: tuple):
    new_inputs = []
    for input in inputs:
        input = Image.fromarray(input)
        input = input.resize(new_size)
        new_inputs.append(numpy.array(input))

    return tf.image.convert_image_dtype(numpy.array(new_inputs), dtype=tf.float32)


def compare_discrete(preds, truth):
    preds_bin = []
    for pred0, pred1 in zip(preds[0], preds[1]):
        preds_bin.append([numpy.argmax(pred0), numpy.argmax(pred1)])

    dict32 = {((i+1)/32) * 2 * numpy.pi: i for i in numpy.arange(32)}
    dict16 = {((i+1)/16) * numpy.pi/2: i for i in numpy.arange(16)}

    truth_bin = []
    for true in truth:
        truth_bin.append([dict32[min(dict32, key=lambda x:abs(x-(true[0]*2*numpy.pi)))], dict16[min(dict16, key=lambda x:abs(x-(true[1]*numpy.pi/2)))]])

    return preds_bin, truth_bin


def evaluate_synthetic(dataset_path, model_path, predict_individually=False, print_all=False):
    model = tf.saved_model.load(model_path)

    (
        images_train,
        images_test,
        light_data_train,
        light_data_test,
        ambient_data_train,
        ambient_data_test
    ) = load_data_hdf5(dataset_path, DataMode.RADIANS)

    input_data = numpy.concatenate((images_train, images_test), axis=0)
    # input_data = resize_all(input_data, (128, 128)) odvisno kaj zahteva model
    truth_data = numpy.concatenate((light_data_train, light_data_test), axis=0)

    if predict_individually:
        preds = []
        for i, input_sample in enumerate(input_data):
            # preds.append(model(numpy.expand_dims(input_sample, axis=0)))
            preds.append(model(input_sample))
            print(i)
        preds = numpy.squeeze(numpy.array(preds))
    else:
        preds = model(input_data)

    preds, truth_data = compare_discrete(preds, truth_data)

    maeA = mean_absolute_error(numpy.array(truth_data)[:, 0], numpy.array(preds)[:, 0])
    maeE = mean_absolute_error(numpy.array(truth_data)[:, 1], numpy.array(preds)[:, 1])

    if print_all:
        print_each(preds, truth_data)

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')


def evaluate_real(model_path, print_all=False):
    model = tf.saved_model.load(model_path)

    images = [None] * 100
    dirs = ["img/angle_aruco", "img/angle"]
    for directory in dirs:
        for image_path in sorted(os.listdir(directory)):
            number, ext = image_path.split(sep=".")
            image_path = os.path.join(directory, image_path)
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = numpy.array(image)
            images[int(number) - 1] = image

    images = resize_all(images, (64, 64))
    images = numpy.array(images)
    # np.random.shuffle(images)
    labels = [None] * 100
    for label_path in sorted(os.listdir("labels")):
        number, ext = label_path.split(sep=".")
        label_path = os.path.join("labels", label_path)
        f = open(label_path)
        json_data = json.load(f)
        labels[int(number) - 1] = json_data["pos"]

    labels = numpy.array(labels)
    # np.random.shuffle(labels)
    preds = model(images)
    preds = preds.numpy()
    preds[:, 0] *= 2 * numpy.pi
    preds[:, 1] *= (numpy.pi/2)

    maeA = mean_absolute_error(labels[:, 0], preds[:, 0])
    maeE = mean_absolute_error(labels[:, 1], preds[:, 1])

    # varA = explained_variance_score(labels[:, 0], preds[:, 0])
    # varE = explained_variance_score(labels[:, 1], preds[:, 1])

    if print_all:
        print_each(preds, labels)

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
    # print(f'Explained variance azimuth: {varA}, Explained variance elevation: {varE}')

    f = open("results_discrete.txt", "w", encoding="utf-8")
    f.write(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
    f.close()


if __name__ == "__main__":
    evaluate_synthetic(
        "/home/klemen/light_estimation/estimation/dataset/LED128x128.hdf5",
        "../estimation/models/f36a6f67-4402-4081-9eeb-6480ccf24a43",
        predict_individually=True, print_all=True)
    # evaluate_real("../estimation/models/sequential", print_all=True)
