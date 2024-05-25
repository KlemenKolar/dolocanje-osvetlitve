import os

root_dir = os.environ.get('LIGHT_ESTIMATION_ROOT_DIR') or os.path.dirname(
    os.path.abspath(__file__))

# camera_rotation_steps - camera rotation steps
camera_rotation_steps = 5
# light_rotation_steps - light rotation steps
light_rotation_steps = 5
# light_elevation_steps - light elevation steps
light_elevation_steps = 5
# camera_elevation_steps - camera elevation steps
camera_elevation_steps = 5
# ambient_strength_steps - ambient strength steps
ambient_strength_steps = 5
# randomization_steps - steps used for randomization of scene objects
randomization_steps = 15


# cycles_samples - max samples when rendering with cycles
cycles_samples = 256
# camera_name - name of the Camera object in scenes used for generating images
camera_name = 'Camera'
# light_name - name of the Light object in scenes used for generating images
light_name = 'Light'
# plane_name - name of the Plane object in scenes used for generating images
plane_name = 'Plane'
# geometry_nodes_object_name - name of the Object with geomtery nodes modifier
geometry_nodes_object_name = 'Geo nodes'
# objects_collection_name - name of the collection from which rendered objects are sampled
objects_collection_name = 'Shapes'
# w - width and h - height of rendered image
w = h = 512
# gpu_id - which gpu to use for rendering
gpu_id = 'CUDA_TITAN X (Pascal)_0000:06:00'
gpu_id = 'CUDA_NVIDIA GeForce RTX 2070_0000:26:00'


# w_dataset - scaled width, h_dataset - scaled height
w_dataset = h_dataset = 128


# scene_dir_path - path to the folder where blender scenes are stored
scene_dir_path = f'{root_dir}/scene'
# img_dir_path - path to the folder where rendered images are stored
img_dir_path = f'{root_dir}/img'
# test_img_dir_path - path to the folder where real life pictures used for testing are stored
test_img_dir_path = f'{root_dir}/test_img'
# labels_dir_path - path to the folder where image labels are stored
labels_dir_path = f'{root_dir}/labels'
# dataset_dir_path - path where dataset files (.npy) are stored
dataset_dir_path = f'{root_dir}/dataset'
# model_dir_path - path where tensorflow models are stored
model_dir_path = f'{root_dir}/models'
# weights_dir_path - path where weights are stored
weights_dir_path = f'{model_dir_path}/weights'
# plot_dir - path to the folder where plots are saved
plot_dir_path = f'{root_dir}/plots'
# logs_dir_path - path to tensorboard logs
logs_dir_path = f'{root_dir}/logs'
# checkpoints_dir_path - path to checkpoints saved during training
checkpoints_dir_path = f'{root_dir}/checkpoints'

# dataset_size_stamp - width and height of dataset in string format used for identifying datasets
dataset_size_stamp = f'{w_dataset}x{h_dataset}'
# hdf5_dataset_path - path for hdf5 dataset (LightEstimationDataset)
#hdf5_dataset_path = f'{dataset_dir_path}/LED{dataset_size_stamp}.hdf5'
hdf5_dataset_path = f'{dataset_dir_path}/Only_Material_Dataset_Shorter.hdf5'
# images_dataset_path - path for images dataset
images_dataset_path = f'{dataset_dir_path}/images{dataset_size_stamp}'
# light_dataset_path - path for light position dataset
light_dataset_path = f'{dataset_dir_path}/light_data_pos{dataset_size_stamp}'
# ambient_dataset_path - path for ambient amount dataset
ambient_dataset_path = f'{dataset_dir_path}/ambient{dataset_size_stamp}'


# model_architecure - determines which model architecture is used
model_architecture = 'efficient_net_b3'
# model_path - path where tensorflow model is saved after training
model_path = f'{model_dir_path}/{model_architecture}'
# train_test_split_random_state - integer used for pseudo random shuffle of data
train_test_split_random_state = 223
# weights - weights used for training keras networks
weights = None
# batch_size
batch_size = 32
# epochs
epochs = 200
# default_dataset
default_dataset = 'CY46K'
# test_size
test_size = 0.1
