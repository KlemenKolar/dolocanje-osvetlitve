import os

import cv2

root_dir = os.path.dirname(os.path.abspath(__file__))

# imsize - size to which all images are resized
imsize = tuple([1000] * 2)


# calibration_images_dir_path - folder where camera calibration photos are stored
#   for each different camera_name there is a subfolder
calibration_images_dir_path = f'{root_dir}/calibration_images'
# aruco_images_dir_path - folder where photos of aruco markers are stored
#   these photos are used for visualization
#   for each different camera_name there is a subfolder
aruco_images_dir_path = f'{root_dir}/aruco_marker_images'
# calibration_dir_path - folder where calibration results stored in form of .npy files
#   files are: cameraMatrix.npy  distCoeff.npy  rvecs.npy  tvecs.npy
#   for each different camera_name there is a subfolder
calibration_dir_path = f'{root_dir}/calibration'
# printable_dir_path - folder where aruco markers and calibration pictures for printing are stored
#   newly generated aruco markers (generate_aruco.py) are stored here
printable_dir_path = f'{root_dir}/printable'
# scene_dir_path - folder where blender scenes for rendering are stored
scene_dir_path = f'{root_dir}/scene'
# rendered_images_dir_path - folder where rendered images are stored
rendered_images_dir_path = f'{root_dir}/render'
# plot_dir_path - path to the folder where plots are stored
plot_dir_path = f'{root_dir}/plots'


# current_camera_name - name of the current camera
#   used to choose the right subfolder in main directories
current_camera_name = 'luka_phone'


# aruco_dict_id - which aruco predefined dictionary is used
aruco_dict_id = cv2.aruco.DICT_6X6_100
# aruco_printable_markers_path - path to the file where generated aruco markers are stored
aruco_printable_markers_path = f'{printable_dir_path}/markers.pdf'
# chessboard_grid_size - number of rows and cols in the calibration chessboard
chessboard_grid_size = (9, 6)
