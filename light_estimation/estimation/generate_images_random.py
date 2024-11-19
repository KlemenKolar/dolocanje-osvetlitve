import argparse
import glob
import json
import os
from functools import reduce
from itertools import chain
from math import ceil, floor, log
from typing import Generator

import bpy
import numpy as np

from estimation.config import (camera_name, cycles_samples,
                               geometry_nodes_object_name, gpu_id, h,
                               img_dir_path, labels_dir_path, light_name,
                               objects_collection_name, plane_name,
                               scene_dir_path, w)
from estimation.utils import (LightData, ObjectData, SampleData,
                              angle_between_2d, angle_between_3d,
                              check_create_dir, empty_create_dir)

os.environ['OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT'] = '1'


def select_object(pattern: str):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern=pattern)
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]


def get_pos(name: str) -> np.ndarray:
    select_object(name)
    return np.array([*bpy.context.selected_objects[0].location])


def set_pos(name: str, new_pos: np.ndarray):
    select_object(name)
    bpy.context.selected_objects[0].location = new_pos


def set_rot(name: str, new_rot: np.ndarray):
    select_object(name)
    bpy.context.selected_objects[0].rotation_euler = new_rot


def force_gpu_render(gpu_id: str = None):
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()
    cprefs.compute_device_type = 'CUDA'

    for device in cprefs.devices:
        if device.type == 'CUDA' and (gpu_id is None or device.id == gpu_id):
            device.use = True
        else:
            device.use = False


def init_scene(scene_path: str, gpu_id: str = None):
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    force_gpu_render(gpu_id=gpu_id)

    bpy.context.tool_settings.transform_pivot_point = 'CURSOR'
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    # bpy.context.scene.cycles.samples = cycles_samples


def render_image(img_dir_path: str, sample_id: str):
    bpy.context.scene.render.filepath = f'{img_dir_path}/{sample_id}.jpg'
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h

    bpy.ops.render.render(write_still=True)


def rotate_camera(angle: float, override):
    select_object(camera_name)
    bpy.ops.transform.rotate(
        override,
        value=angle,
        center_override=(0, 0, 0),
        orient_axis='Z',
        orient_type='GLOBAL'
    )


def elevate_camera(angle: float, override):
    select_object(camera_name)

    bpy.context.scene.transform_orientation_slots[0].type = 'LOCAL'

    bpy.ops.transform.rotate(
        override,
        value=angle,
        orient_axis='X',
        orient_type='LOCAL',
        center_override=(0, 0, 0)
    )

    bpy.context.scene.transform_orientation_slots[0].type = 'GLOBAL'


def reset_camera(d: float):
    set_pos(camera_name, [0, -d, 0])
    set_rot(camera_name, [np.pi/2, 0, 0])


def reset_light(d: float):
    set_pos(light_name, [0, -d, 0])


def rotate_light(angle: float, override):
    select_object(light_name)
    bpy.ops.transform.rotate(
        override,
        value=angle,
        orient_axis='Z',
        center_override=(0, 0, 0)
    )


def elevate_light(angle: float, override):
    select_object(light_name)

    bpy.context.scene.transform_orientation_slots[0].type = 'LOCAL'

    bpy.ops.transform.rotate(
        override,
        value=angle,
        orient_axis='X',
        orient_type='LOCAL',
        center_override=(0, 0, 0)
    )

    bpy.context.scene.transform_orientation_slots[0].type = 'GLOBAL'


def set_ambient(value: float):
    bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[0].default_value = value


def set_geo_nodes_seed(seed: int):
    try:
        bpy.data.node_groups["Geometry Nodes"].nodes["Seed"].outputs[0].default_value = seed
    except:
        pass


def set_light_shadow_soft_size(value: float):
    select_object(light_name)
    bpy.context.selected_objects[0].data.shadow_soft_size = value


def set_light_power(value: float):
    select_object(light_name)
    bpy.context.selected_objects[0].data.energy = value


def set_plane_material(material_name: str):
    select_object(plane_name)
    i = [x.name for x in bpy.data.objects[plane_name].material_slots].index(
        material_name)

    bpy.ops.object.editmode_toggle()
    bpy.context.object.active_material_index = i
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.editmode_toggle()


def set_objects_materials(material_names: list[str]):
    for material_name, obj in zip(
        material_names,
        bpy.data.collections[objects_collection_name].objects
    ):
        obj.material_slots[0].material = bpy.data.materials.get(material_name)


def get_objects_materials(visible_collection: str) -> list[str]:
    def get_nested_object_materials(obj: bpy.types.Object):
        return [x.name for x in obj.material_slots] + list(chain(*[get_nested_object_materials(x) for x in obj.children]))

    objects = bpy.data.collections[visible_collection].objects

    return list(set(reduce(lambda a, b: [*a, *b], [get_nested_object_materials(obj) for obj in objects])))


def get_environment_texture() -> str:
    et = bpy.context.scene.world.node_tree.nodes.get('Environment Texture')
    return et.image.name


def set_visible_collection(name: str):
    for collection in bpy.data.collections[objects_collection_name].children:
        collection.hide_render = collection.name != name
        collection.hide_viewport = collection.name != name


def get_override():
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    area = [x for x in screen.areas if x.type == 'VIEW_3D'][0]

    return {'window': window, 'screen': screen, 'area': area}


def check_required(scene_name: str):
    if not objects_collection_name in bpy.data.collections:
        return f'No collection named {objects_collection_name} found in scene {scene_name}.'
    if not camera_name in bpy.data.objects:
        return f'No camera named {camera_name} found in scene {scene_name}.'
    if not plane_name in bpy.data.objects:
        return f'No object named {plane_name} found in scene {scene_name}.'
    if not light_name in bpy.data.objects:
        return f'No light named {light_name} found in scene {scene_name}.'
    if not geometry_nodes_object_name in bpy.data.objects:
        return f'No object named {geometry_nodes_object_name} found in scene {scene_name}.'


def render_images_and_save_data(
    scenes: list[str] = None,
    scene_dir_path: str = scene_dir_path,
    n: int = 0,
    img_dir_path: str = img_dir_path,
    labels_dir_path: str = labels_dir_path,
    seed: int = 123,
    gpu_id: str = None,
    materials: bool = True
):
    """Renders images and saves them along with labels.

    Args:
        scenes (list[str], optional): List of blender scenes used for rendering. Defaults to all scenes in scene_dir_path.
        scene_dir_path (str, optional): Directory in which to look for blender scenes. Defaults to scene_dir_path.
        n (int, optional): Number of samples to generate. Defaults to 0.
        img_dir_path (str, optional): Path to directory where images are going to be stored. Defaults to img_dir_path.
        labels_dir_path (str, optional): Path to directory where labels are going to be stored. Defaults to labels_dir_path.
        seed (int, optional): Seed used to create RNG, that way generation is reproducible. Defaults to 123.
        gpu_id (str, optional): Id of the GPU used for rendering, useful for systems with more than one gpu. Defaults to None in which case all CUDA devices are going to be used.
    """

    check_create_dir(scene_dir_path)
    check_create_dir(img_dir_path)
    check_create_dir(labels_dir_path)

    scenes = glob.glob(
        f'{scene_dir_path}/*.blend') if scenes is None else scenes
    existing_ids = set([img_filename.split('.')[0]
                       for img_filename in os.listdir(img_dir_path)])
    rng = np.random.default_rng(seed=seed)
    per_scene = n // len(scenes)
    i_just, j_just = ceil(log(len(scenes), 10)), ceil(log(per_scene, 10))

    for i, scene_path in enumerate(scenes):
        init_scene(scene_path=scene_path, gpu_id=gpu_id)

        error_msg = check_required(scene_path)
        if error_msg is not None:
            print(error_msg)
            continue

        override = get_override()
        plane_materials = [
            x.name for x in bpy.data.objects[plane_name].material_slots]
        collections = [
            x.name for x in bpy.data.collections[objects_collection_name].children]

        for j in range(per_scene):
            sample_id = f'{str(i).rjust(i_just, "0")}_{str(j).rjust(j_just, "0")}'

            sample_data = SampleData(
                sample_id=sample_id,
                camera=ObjectData(
                    a=rng.random() * np.pi * 2,
                    b=rng.random() * (np.pi / 2),
                    d=rng.random() * 20 + 10,
                ),
                light=LightData(
                    a=rng.random() * np.pi * 2,
                    b=rng.random() * (np.pi / 2) + 0.01,
                    d=rng.random() * 20 + 10,
                    shadow_soft_size=rng.random() * 5,
                    power=rng.random() * 10000 + 3000
                ),
                geo_nodes_seed=ceil(rng.random() * n),
                ambient=rng.random() * 0.7,
                plane_material=plane_materials[floor(
                    rng.random() * len(plane_materials))],
                visible_collection=collections[floor(
                    rng.random() * len(collections))]
            )

            if sample_id not in existing_ids:
                reset_camera(sample_data.camera.d)
                elevate_camera(sample_data.camera.b, override)
                rotate_camera(-sample_data.camera.a, override)

                reset_light(sample_data.light.d)
                elevate_light(sample_data.light.b, override)
                rotate_light(-sample_data.camera.a -
                             sample_data.light.a, override)
                set_light_shadow_soft_size(sample_data.light.shadow_soft_size)
                set_light_power(sample_data.light.power)

                # set_geo_nodes_seed(sample_data.geo_nodes_seed)
                set_visible_collection(sample_data.visible_collection)

                set_ambient(sample_data.ambient)

                if materials:
                    set_plane_material(sample_data.plane_material)

                render_image(img_dir_path, sample_id)

                sample_data.camera.pos = [*get_pos(camera_name)]
                sample_data.light.pos = [*get_pos(light_name)]
                sample_data.objects_materials = get_objects_materials(
                    sample_data.visible_collection
                ) if materials else []
                sample_data.environment_texture = get_environment_texture()

                sample_data.save(labels_dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--scenes',
        help='list of scenes which are used for rendering',
        nargs='*'
    )

    parser.add_argument(
        '-sdir', '--scene-dir-path',
        help='directory containing scenes used for rendering',
    )

    parser.add_argument(
        '-n',
        help='number of generated images',
        type=int
    )

    parser.add_argument(
        '--seed',
        help='seed used for random data',
        type=int
    )

    parser.add_argument(
        '--gpu',
        help='gpu id used for rendering'
    )

    parser.add_argument(
        '--no-materials',
        help='disable material interactions',
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    render_images_and_save_data(
        scenes=args.scenes,
        n=args.n,
        seed=args.seed,
        scene_dir_path=args.scene_dir_path or scene_dir_path,
        gpu_id=args.gpu,
        materials=not args.no_materials
    )
