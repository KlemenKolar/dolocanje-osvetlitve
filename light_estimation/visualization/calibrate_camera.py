import argparse
import glob
import sys

import cv2
import numpy as np
from estimation.utils import check_create_dir

from visualization.config import (calibration_dir_path,
                                  calibration_images_dir_path,
                                  chessboard_grid_size, current_camera_name,
                                  imsize)
from visualization.utils import crop_center_square


def calibrate_camera(
    calibration_images_dir_path: str = calibration_images_dir_path,
    calibration_dir_path: str = calibration_dir_path,
    current_camera_name: str = current_camera_name,
    imsize: tuple[int] = imsize,
    chessboard_grid_size: tuple[int] = chessboard_grid_size
):
    """Calibrate camera with a number of images of a checkerboard pattern.

    Args:
        calibration_images_dir_path (str, optional): Directory with calibration images. Defaults to calibration_images_dir_path.
        calibration_dir_path (str, optional): Directory where calibration results are saved. Defaults to calibration_dir_path.
        current_camera_name (str, optional): Name of the current camera. Defaults to current_camera_name.
        imsize (tuple[int], optional): Size to which images are resized. Defaults to imsize.
        chessboard_grid_size (tuple[int], optional): Chessboard grid size. Defaults to chessboard_grid_size.
    """

    check_create_dir(calibration_images_dir_path)
    check_create_dir(calibration_dir_path)
    check_create_dir(f'{calibration_dir_path}/{current_camera_name}')

    c_n, c_m = chessboard_grid_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_points, detected_points = [], []

    # initialize checkboard coordinates (0,0), (0,1), (0,2) ...
    chessboard_coords = np.zeros((c_n*c_m, 3), np.float32)
    chessboard_coords[:, :2] = np.mgrid[0:c_n, 0:c_m].T.reshape(-1, 2)

    for image_path in glob.glob(f'{calibration_images_dir_path}/{current_camera_name}/*'):
        img = cv2.imread(image_path)
        img = crop_center_square(img)
        img = cv2.resize(img, imsize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_grid_size)

        if ret:
            corners_fine = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            chessboard_points.append(chessboard_coords)
            detected_points.append(corners_fine)

    chessboard_points = np.array(chessboard_points).astype(np.float32)
    detected_points = np.array(detected_points).astype(np.float32)

    ret, cameraMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(
        chessboard_points,
        detected_points,
        imsize,
        None, None
    )

    np.save(
        f'{calibration_dir_path}/{current_camera_name}/cameraMatrix', cameraMatrix)
    np.save(f'{calibration_dir_path}/{current_camera_name}/distCoeff', distCoeff)
    np.save(f'{calibration_dir_path}/{current_camera_name}/rvecs', rvecs)
    np.save(f'{calibration_dir_path}/{current_camera_name}/tvecs', tvecs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--camera-name',
        help='override camera_name, if not specified it is read from config'
    )

    args = parser.parse_args()

    calibrate_camera(
        current_camera_name=args.camera_name or current_camera_name)
