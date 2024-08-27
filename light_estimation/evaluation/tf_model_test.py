import tensorflow as tf
from sklearn.metrics import mean_absolute_error, explained_variance_score
from estimation.train import load_data_hdf5
from estimation.enums import DataMode
import numpy
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from utils import compare_discrete, print_each, compare_discrete2


def resize_all(inputs, new_size: tuple):
    new_inputs = []
    for input in inputs:
        input = Image.fromarray(input)
        input = input.resize(new_size)
        new_inputs.append(numpy.array(input))

    return tf.image.convert_image_dtype(numpy.array(new_inputs), dtype=tf.float32)


def evaluate_synthetic(dataset_path, model_path, predict_batch_size=None, print_all=False):
    # model = tf.saved_model.load(model_path)
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

    input_data = tf.cast(input_data, tf.float32)

    if predict_batch_size is not None:
        preds = []
        predsA = []
        predsE = []
        temp = None
        i = 0
        while i < len(input_data):
            # preds.append(model(numpy.expand_dims(input_sample, axis=0)))
            # preds.append(model(input_data[i:i+predict_batch_size]))
            temp = model(input_data[i:i+predict_batch_size])
            predsA.append(temp[0].numpy())
            predsE.append(temp[1].numpy())
            print(i)
            i += predict_batch_size

        #preds = preds[0].numpy() + preds[1].numpy()
        #preds = numpy.squeeze(numpy.array(preds))
        predsA = numpy.array(predsA)
        predsA = predsA.reshape(-1, predsA.shape[-1])
        predsE = numpy.array(predsE)
        predsE = predsE.reshape(-1, predsE.shape[-1])
        preds.append(predsA)
        preds.append(predsE)

    else:
        preds = model(input_data)

    preds, truth_data = compare_discrete(preds, truth_data)

    maeA = mean_absolute_error(numpy.array(truth_data)[:, 0], numpy.array(preds)[:, 0])
    maeE = mean_absolute_error(numpy.array(truth_data)[:, 1], numpy.array(preds)[:, 1])

    if print_all:
        print_each(preds, truth_data)

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')

    f = open("results_discrete.txt", "a", encoding="utf-8")
    f.write(f'\n----------------------\nMAE azimuth: {maeA}, MAE elevation: {maeE} - with model: {os.path.basename(model_path)} on dataset: {os.path.basename(dataset_path)}')
    f.close()


def evaluate_real(model_path, print_all=False):
    model = tf.saved_model.load(model_path)

    images = [None] * 100
    dirs = ["F:\Klemen_diploma\light_estimation\evaluation\img/angle_aruco", "F:\Klemen_diploma\light_estimation\evaluation\img/angle"]
    for directory in dirs:
        for image_path in sorted(os.listdir(directory)):
            number, ext = image_path.split(sep=".")
            image_path = os.path.join(directory, image_path)
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = numpy.array(image)
            images[int(number) - 1] = image

    images = resize_all(images, (128, 128))
    images = numpy.array(images)
    # np.random.shuffle(images)
    labels = [None] * 100
    for label_path in sorted(os.listdir("F:\Klemen_diploma\light_estimation\evaluation\labels")):
        number, ext = label_path.split(sep=".")
        label_path = os.path.join("F:\Klemen_diploma\light_estimation\evaluation\labels", label_path)
        f = open(label_path)
        json_data = json.load(f)
        labels[int(number) - 1] = json_data["pos"]

    labels = numpy.array(labels)
    # np.random.shuffle(labels)
    #for image in images:
    #    plt.imshow(image)
    #    plt.show()
    preds = model(images)
    '''preds = preds.numpy()
    preds[:, 0] *= 2 * numpy.pi
    preds[:, 1] *= (numpy.pi/2)'''
    for pred in preds[0]:
        print(tf.argmax(pred)) 

    preds, labels = compare_discrete2(preds, labels)

    maeA = mean_absolute_error(numpy.array(labels)[:, 0], numpy.array(preds)[:, 0])
    maeE = mean_absolute_error(numpy.array(labels)[:, 1], numpy.array(preds)[:, 1])

    # varA = explained_variance_score(labels[:, 0], preds[:, 0])
    # varE = explained_variance_score(labels[:, 1], preds[:, 1])

    if print_all:
        print_each(preds, labels)

    print(f'MAE azimuth: {maeA}, MAE elevation: {maeE}')
    # print(f'Explained variance azimuth: {varA}, Explained variance elevation: {varE}')

    f = open("results_discrete.txt", "a", encoding="utf-8")
    f.write(f'\n\nMAE azimuth: {maeA}, MAE elevation: {maeE} - with model: {os.path.basename(model_path)} on dataset: {dirs}')
    f.close()


if __name__ == "__main__":
    # evaluate_synthetic(
    #     "../estimation/dataset/LED128x128_test_1.hdf5",
    #     "../estimation/models/f36a6f67-4402-4081-9eeb-6480ccf24a43",
    #     predict_batch_size=1000, print_all=True)  # f36a6f67-4402-4081-9eeb-6480ccf24a43 model je tist iz članka
    evaluate_real("F:\Klemen_diploma\light_estimation\estimation\models_old\\f36a6f67-4402-4081-9eeb-6480ccf24a43", print_all=True)
