import os

root_dir = os.path.dirname(os.path.abspath(__file__))


# printable_dir_path - folder where aruco markers for identifying are stored
printable_dir_path = f'{root_dir}/printable'
# img_dir_path - path to the folder where captured images are stored
img_dir_path = f'{root_dir}/img'
# labels_dir_path - path to the folder where image labels are stored
labels_dir_path = f'{root_dir}/labels'
# plot_dir_path - path to the folder where plots are stored
plot_dir_path = f'{root_dir}/plots'
# dataset_dir_path - path where dataset files (.npy) are stored
dataset_dir_path = f'{root_dir}/dataset'
# dump_dir_path - path where heroku dump json files are stored
dump_dir_path = f'{root_dir}/dumps'


# angle_aruco_img_dir_path - path to the folder with images for angular evaluation - with aruco marker
angle_aruco_img_dir_path = f'{img_dir_path}/angle_aruco'
# angle_img_dir_path - path to the folder with images for angular evaluation - without aruco marker
angle_img_dir_path = f'{img_dir_path}/angle'
# ambient_img_dir_path - path to the folder with images for ambient evaluation
ambient_img_dir_path = f'{img_dir_path}/ambient'


# images_dataset_path - path for images dataset
images_dataset_path = f'{dataset_dir_path}/images_evaluate'
# light_dataset_path - path for light position dataset
light_dataset_path = f'{dataset_dir_path}/light_data_pos_evaluate'
# ambient_dataset_path - path for ambient amount dataset
ambient_dataset_path = f'{dataset_dir_path}/ambient_evaluate'
# images_filenames_path - path for filenames of evalutaion images
images_filenames_path = f'{dataset_dir_path}/filenames'
# hdf5_dataset_path - path for hdf5 dataset for evaluation
hdf5_dataset_path = f'{dataset_dir_path}/evaluation_32_16.hdf5'


# aruco_printable_markers_path - path to the file where generated aruco markers are stored
aruco_printable_markers_path = f'{printable_dir_path}/markers.pdf'


# tmp_filename_gopro - temporary file used to save gopro image
tmp_filename_gopro = 'tmp_gopro.jpg'
# tmp_filename_dslr - temporary file used to save dslr image
tmp_filename_dslr = 'tmp_dslr.jpg'


# camera_marker_id - id of the marker attached to the camera
camera_marker_id = 4
# light_marker_id - id of the marker attached to the light
light_marker_id = 1
# center_marker_id - id of the marker attached to the center
center_marker_id = 2


# following values were measured by hand and they will differ on a different setup
# pixel_to_meter_fac - 1 meter in pixels (gopro).
pixel_to_meter_fac = 657.25
# light_height - height of the light marker in meters
light_height = 1.4
# center_height - height of the center marker in meters
center_height = 0.72
