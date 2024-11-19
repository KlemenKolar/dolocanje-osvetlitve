import argparse
import glob
from datetime import datetime
from textwrap import indent
from typing import Any
from urllib.parse import quote

import bpy
import cv2
import h5py
import numpy as np
from estimation.enums import DataMode
from estimation.config import model_dir_path, model_path
from estimation.generate_images import force_gpu_render, get_override
from estimation.generate_images_random import set_ambient
from estimation.utils import check_create_dir, empty_create_dir, normalize
from evaluation.config import dataset_dir_path, img_dir_path
from evaluation.evaluate import evaluate_single
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.special import softmax
from sklearn.preprocessing import KBinsDiscretizer

from visualization.config import (aruco_dict_id, aruco_images_dir_path,
                                  calibration_dir_path, current_camera_name,
                                  imsize, rendered_images_dir_path,
                                  scene_dir_path)
from visualization.utils import crop_center_square

if True:
    import os
    
    # otherwise blenderproc throws an exception because the script is not being run in an
    # blender python environment. we just need one function, so in this case this doesn't matter.
    os.environ['OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT'] = '1'

    from blenderproc.python.camera.CameraUtility import \
        set_intrinsics_from_K_matrix


def center_object():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern='Object')
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.select_pattern(pattern='Camera')

    bpy.ops.object.parent_set()

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern='Object')

    bpy.context.selected_objects[0].location = (0, 0, 0)
    bpy.context.selected_objects[0].rotation_euler = (0, 0, 0)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern='Camera')
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    bpy.ops.object.select_all(action='DESELECT')


def estimate_and_render(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    ambient: float,
    scene_path: str,
    img_id: str,
    render_plain: bool = True,
    model_path: str = None,
    rendered_images_dir_path: str = rendered_images_dir_path,
    a_angle_discretizer: KBinsDiscretizer = None,
    b_angle_discretizer: KBinsDiscretizer = None,
    datamode: str = DataMode.HEATMAP

):
    """Performs light estimation and renders the result.

    Args:
        img (np.ndarray): Input image on which estimation is performed.
        rvec (np.ndarray): Rotation vector of the detected aruco marker. This vector is in the opencv rodrigues form.
        tvec (np.ndarray): Translation vector of the detected aruco marker.
        K (np.ndarray): Camera matrix K
        scene_path (str): Path to the blender scene used for rendering
        rendered_image_path (str): Path where rendered result is saved to.
    """

    bpy.ops.wm.open_mainfile(filepath=scene_path)

    set_intrinsics_from_K_matrix(K, imsize[0], imsize[1])

    bpy.context.scene.camera.rotation_euler = (np.pi, 0, 0)
    bpy.context.scene.camera.location = (0, 0, 0)

    rot = Rotation.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('xyz')

    bpy.data.objects['Object'].rotation_euler = rot
    bpy.data.objects['Object'].location = tvec
    bpy.data.objects['Object'].scale = [2] * 3

    center_object()

    # a, b = [x[0][0] for x in evaluate_single(
    #     img,
    #     model_in=model,
    #     normalize=False
    # )]

    pred = evaluate_single(
        img,
        model_path=model_path,
        datamode=datamode,
        img_id=img_id,
        dest_dir=rendered_images_dir_path,
    )

    a = pred[0] / (2 * np.pi)
    b = pred[1] / (1.05864)

    print('predicted light_angles: ', a * 360, b * 60.65)

    bpy.data.objects['Light'].location = [
        *bpy.data.objects['Camera'].location[:2], 0]

    bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'

    bpy.ops.object.select_pattern(pattern='Light')
    override = get_override()
    with bpy.context.temp_override(window=override["window"], screen=override["screen"], area=override["area"]):
        bpy.ops.transform.rotate(
            value=-b * 1.05864,
            orient_axis='Y',
            center_override=(0, 0, 0)
        )

    if bpy.data.objects['Light'].location[2] < 0:
        override = get_override()
        with bpy.context.temp_override(window=override["window"], screen=override["screen"], area=override["area"]):
            bpy.ops.transform.rotate(
                value=b * (np.pi),
                orient_axis='Y',
                center_override=(0, 0, 0)
            )

    bpy.context.scene.tool_settings.transform_pivot_point = 'CURSOR'

    bpy.ops.object.select_pattern(pattern='Light')
    override = get_override()
    with bpy.context.temp_override(window=override["window"], screen=override["screen"], area=override["area"]):
        bpy.ops.transform.rotate(
            value=- a * (2 * np.pi),
            orient_axis='Z',
            center_override=(0, 0, 0)
        )

    bpy.context.scene.render.film_transparent = True
    set_ambient(ambient)
    print(f'ambient set to: {ambient}')

    date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    temp_filepath = f'F:\\Klemen_diploma\\light_estimation_leon\\light_estimation\\visualization\\scene\\light_estimation_render_{date_str}.png'

    bpy.context.scene.render.filepath = temp_filepath

    force_gpu_render()

    bpy.ops.render.render(write_still=True)

    photo = Image.fromarray(img)
    render = Image.open(temp_filepath)

    photo.paste(render, (0, 0), render)
    photo.save(f'{rendered_images_dir_path}/{img_id}_estimated.png')

    os.remove(temp_filepath)

    if render_plain:
        bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[0].default_value = 0.7
        bpy.data.objects['Light'].hide_render = True

        bpy.ops.render.render(write_still=True)

        photo = Image.fromarray(img)
        render = Image.open(temp_filepath)

        photo.paste(render, (0, 0), render)
        photo.save(f'{rendered_images_dir_path}/{img_id}_plain.png')

        os.remove(temp_filepath)


def visualize_single(
    image_path: str,
    img_id_num: int,
    imsize: tuple,
    marker_dict: Any,
    cameraMatrix: np.ndarray,
    distCoeff: np.ndarray,
    scene_path: list[str],
    draw_markers: bool,
    model_path: str,
    rendered_images_dir_path: str,
    ambient: float,
    a_angle_discretizer,
    b_angle_discretizer,
    datamode: str = DataMode.HEATMAP
):
    img = cv2.imread(image_path)
    img = crop_center_square(img)
    img = cv2.resize(img, imsize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _rejectedImgPoints = cv2.aruco.detectMarkers(
        gray, marker_dict)

    if ids is None or len(ids) == 0:
        print(f'No markers detected for image {image_path}')
        return

    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
        corners, 4.5, cameraMatrix, distCoeff
    )

    rvec, tvec = rvecs[0], tvecs[0]

    if draw_markers:
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        img = cv2.aruco.drawAxis(img, cameraMatrix, distCoeff, rvec, tvec, 4.5)

    scene_name = os.path.basename(scene_path).split('.')[0]
    image_name = os.path.basename(image_path).split('.')[0]
    img_id = f'{scene_name}_{img_id_num}_{image_name}'

    print(img_id, Rotation.from_matrix(
        cv2.Rodrigues(rvec)[0]).as_euler('xyz'), tvec)

    estimate_and_render(
        img,
        rvec.ravel(),
        tvec.ravel(),
        cameraMatrix,
        0.2,
        scene_path,
        img_id=img_id,
        model_path=model_path,
        rendered_images_dir_path=rendered_images_dir_path,
        a_angle_discretizer=a_angle_discretizer,
        b_angle_discretizer=b_angle_discretizer,
        datamode=datamode
    )


def visualize(
    aruco_images_dir_path: str = aruco_images_dir_path,
    calibration_dir_path: str = calibration_dir_path,
    current_camera_name: str = current_camera_name,
    aruco_dict_id: str = aruco_dict_id,
    imsize: tuple[int] = imsize,
    scene_dir_path: str = scene_dir_path,
    rendered_images_dir_path: str = rendered_images_dir_path,
    draw_markers: bool = False,
    image_paths: list[str] = None,
    scene_paths: list[str] = None,
    model_path: str = model_path,
    datamode: str = DataMode.HEATMAP
):
    """ renders 3d objects from blender scene onto images in aruco_marker_images_dir using 
        light estimation from each image

    Args:
        aruco_images_dir_path (str): folder where photos of aruco markers are stored
        calibration_dir_path (str): folder where calibration results stored in form of .npy files
        current_camera_name (str): name of the current camera
        aruco_dict_id (str): which aruco predefined dictionary is used
        imsize (tuple[int]): size to which all images are resized
        scene_dir_path (str): folder where blender scenes for rendering are stored
        rendered_images_dir_path (str): folder where rendered results are stored
        draw_markers (bool): draw markers on the image before rendering. defaults to False
    """

    if not os.path.isdir(aruco_images_dir_path):
        os.mkdir(aruco_images_dir_path)
        os.mkdir(f'{aruco_images_dir_path}/{current_camera_name}')

    check_create_dir(rendered_images_dir_path)

    marker_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

    if image_paths is None:
        image_paths = glob.glob(
            f'{aruco_images_dir_path}/{current_camera_name}/*')

    if scene_paths is None:
        scene_paths = glob.glob(f'{scene_dir_path}/*.blend')

    cameraMatrix = np.load(
        f'{calibration_dir_path}/{current_camera_name}/cameraMatrix.npy')
    distCoeff = np.load(
        f'{calibration_dir_path}/{current_camera_name}/distCoeff.npy')

    with h5py.File(f'{dataset_dir_path}/all.hdf5') as file:
        filenames = [*file['filenames']]
        ambient = np.array(file['ambient']).flatten()
        ligth_angles = np.array(file['light_angles'])
        discrete_light_angles = np.zeros_like(ligth_angles)
        a_angle_discretizer = KBinsDiscretizer(
            n_bins=32,
            encode='ordinal',
            strategy='uniform'
        )
        discrete_light_angles[:, 0] = a_angle_discretizer.fit_transform(
            ligth_angles[:, 0].reshape((ambient.shape[0], 1))
        ).flatten()

        b_angle_discretizer = KBinsDiscretizer(
            n_bins=16,
            encode='ordinal',
            strategy='uniform'
        )
        discrete_light_angles[:, 1] = b_angle_discretizer.fit_transform(
            ligth_angles[:, 1].reshape((ambient.shape[0], 1))
        ).flatten()

    for i, image_path in enumerate(image_paths):
        image_filename = os.path.basename(image_path)

        indices = [i for i, x in enumerate(filenames)
                   if quote(image_filename) in str(x)]

        idx = indices[0] if len(indices) == 1 else -1

        for scene_path in scene_paths:
            visualize_single(
                image_path=image_path,
                img_id_num=i,
                imsize=imsize,
                marker_dict=marker_dict,
                cameraMatrix=cameraMatrix,
                distCoeff=distCoeff,
                scene_path=scene_path,
                draw_markers=draw_markers,
                model_path=model_path,
                rendered_images_dir_path=rendered_images_dir_path,
                ambient=ambient[idx] if idx >= 0 else 0,
                a_angle_discretizer=a_angle_discretizer,
                b_angle_discretizer=b_angle_discretizer, datamode=datamode
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--img',
        nargs='+',
        default=["20240902_185610.jpg", "20240902_185621.jpg", "20240902_205634.jpg", "20240902_205737.jpg"],
        help='visualize only specific images'
    )

    parser.add_argument(
        '-s', '--scene',
        nargs='+',
        default=None,
        help='visualize only specific scenes'
    )

    parser.add_argument(
        '-c', '--camera',
        help='override current camera (for calibration)',
        default='SamsungA54'
    )

    parser.add_argument(
        '-m', '--model',
        default='efficient_net_b7_heatmap_no_bottleneck2-big-dataset-heatmap-KLdiv-interpolate-gauss\model_4'
    )

    args = parser.parse_args()

    img_dir_path = 'F:\Klemen_diploma\images_to_render'

    print(args.img)

    image_paths = [
        f'{img_dir_path}/{img_path.split(".")[0]}.jpg' for img_path in args.img] if args.img else None
    scene_paths = [
        f'{scene_dir_path}/{scene_path.split(".")[0]}.blend' for scene_path in args.scene] if args.scene else None
    camera_name = args.camera or current_camera_name

    visualize(
        model_path=f'{model_dir_path}/{args.model}',
        image_paths=image_paths,
        scene_paths=scene_paths,
        rendered_images_dir_path=f'{rendered_images_dir_path}/{camera_name}/new_images',
        current_camera_name=camera_name
    )
