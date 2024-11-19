import glob
import json
import os
from math import prod
from typing import Callable

import bpy
import numpy as np

from estimation.config import (ambient_strength_steps, camera_elevation_steps,
                               camera_name, camera_rotation_steps,
                               cycles_samples, gpu_id, h, img_dir_path,
                               labels_dir_path, light_elevation_steps,
                               light_name, light_rotation_steps,
                               randomization_steps, scene_dir_path, w)
from estimation.utils import (angle_between_2d, angle_between_3d,
                              check_create_dir)


def select_object(pattern: str):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_pattern(pattern=pattern)


def get_pos(name: str) -> np.ndarray:
    select_object(name)
    return np.array([*bpy.context.selected_objects[0].location])


def set_pos(name: str, new_pos: np.ndarray):
    select_object(name)
    bpy.context.selected_objects[0].location = new_pos


def force_gpu_render():
    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    prefs = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.get_devices()
    cprefs.compute_device_type = "CUDA"

    for device in cprefs.devices:
        if device.type == "CUDA" and device.id == gpu_id:
            device.use = True
        else:
            device.use = False


def set_pivot_point(value):
    # bpy.context.scene.tool_settings.transform_pivot_point = value

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces[0].pivot_point = value
            area.spaces[0].cursor_location = (0, 0, 0)
            return


def init_scene(scene_path: str):
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    force_gpu_render()

    bpy.context.tool_settings.transform_pivot_point = "CURSOR"
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = cycles_samples


def render_image(img_dir_path: str, sample_id: str):
    bpy.context.scene.render.filepath = f"{img_dir_path}/{sample_id}.jpg"
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h

    bpy.ops.render.render(write_still=True)


def get_current_light_pos_data():
    camera_pos = get_pos(camera_name)
    light_pos = get_pos(light_name)

    a = angle_between_2d(camera_pos, light_pos)
    b = angle_between_3d([*light_pos[:2], 0], light_pos)

    return [a, b]


def get_ambient_mix_value():
    return (
        bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[0].default_value
    )


def rotate_camera(angle: float, override):
    select_object(camera_name)
    bpy.ops.transform.rotate(
        override,
        value=angle,
        center_override=(0, 0, 0),
        orient_axis="Z",
        orient_type="GLOBAL",
    )


def elevate_camera(angle: float, override):
    select_object(camera_name)

    bpy.context.scene.transform_orientation_slots[0].type = "LOCAL"

    bpy.ops.transform.rotate(
        override,
        value=angle,
        orient_axis="X",
        orient_type="LOCAL",
        center_override=(0, 0, 0),
    )

    bpy.context.scene.transform_orientation_slots[0].type = "GLOBAL"


def rotate_light(angle: float, override):
    select_object(light_name)
    bpy.ops.transform.rotate(
        override, value=angle, orient_axis="Z", center_override=(0, 0, 0)
    )


def set_light_height_offset(offset: float):
    camera_pos = get_pos(camera_name)
    light_pos = get_pos(light_name)

    set_pos(light_name, [*light_pos[:2], camera_pos[2] + offset])


def set_mix_shader_value(value: float):
    bpy.data.worlds["World"].node_tree.nodes["Mix Shader"].inputs[
        0
    ].default_value = value


def set_random_seed_geo_nodes(seed):
    bpy.data.node_groups["Geometry Nodes"].nodes["Distribute Points on Faces"].inputs[
        6
    ].default_value = seed


def get_override():
    window = bpy.context.window_manager.windows[0]
    screen = window.screen
    area = [x for x in screen.areas if x.type == "VIEW_3D"][0]

    return {"window": window, "screen": screen, "area": area}


def save_label_data(img_id: str, json_data: object, labels_dir_path: str):
    with open(f"{labels_dir_path}/{img_id}.json", "w") as file:
        file.write(json.dumps(json_data))


class IterationData:
    def __init__(
        self,
        steps: list[int],
        functions: list[Callable[[int], None]],
        rendered_img_ids: list[str],
        img_dir_path: str,
        labels_dir_path: str,
    ) -> None:
        self.steps = steps
        self.functions = functions
        self.rendered_img_ids = rendered_img_ids
        self.img_dir_path = img_dir_path
        self.labels_dir_path = labels_dir_path


def render(f: list[int], scene_name: str, data: IterationData):

    iter_id = "".join([f"_{i:0>2}" for i in f])
    img_id = f"{scene_name}{iter_id}"

    if img_id not in data.rendered_img_ids:
        render_image(data.img_dir_path, img_id)

        json_data = {
            "pos": get_current_light_pos_data(),
            "ambient": get_ambient_mix_value(),
        }

        save_label_data(img_id, json_data, data.labels_dir_path)

    else:
        print(f"Image {img_id} has already been rendered.")


def iterate_nested_steps(x: int, f: list[int], scene_name: str, data: IterationData):
    if x >= len(data.steps):
        return render(f, scene_name, data)
    for i in range(data.steps[x]):
        data.functions[x](i)
        iterate_nested_steps(x + 1, f + [i], scene_name, data)


def render_images_and_save_data(
    camera_rotation_steps: int = camera_rotation_steps,
    light_rotation_steps: int = light_rotation_steps,
    light_elevation_steps: int = light_elevation_steps,
    camera_elevation_steps: int = camera_elevation_steps,
    ambient_strength_steps: int = ambient_strength_steps,
    randomization_steps: int = randomization_steps,
    scene_dir_path: str = scene_dir_path,
    img_dir_path: str = img_dir_path,
    labels_dir_path: str = labels_dir_path,
) -> bool:
    """Renders images and saves them along with labels.

    Args:
        camera_rotation_steps (int, optional): Number of steps in one full rotation of camera around the center of scene. Defaults to camera_rotation_steps.
        light_rotation_steps (int, optional): Number of steps in one full rotation of light around the center of scene. Defaults to light_rotation_steps.
        light_elevation_steps (int, optional): Number of times light elevates for each step. Defaults to light_elevation_steps.
        camera_elevation_steps (int, optional): Number of times camera elevates for each step. Defaults to light_elevation_steps.
        ambient_strength_steps (int, optional): Number of steps of ambient mix value (0.0 to 1.0). Defaults to ambient_strength_steps.
        scene_dir_path (str, optional): Directory with blender scenes used for rendering. Defaults to scene_dir_path.
        img_dir_path (str, optional): Directory where rendered images are saved. Defaults to img_dir_path.
        labels_dir_path (str, optional): Directory where json labels are saved. Defaults to labels_dir_path.

    Returns:
        (bool): Returns True if everything was succesful.
    """

    check_create_dir(scene_dir_path)
    check_create_dir(img_dir_path)
    check_create_dir(labels_dir_path)

    scenes = glob.glob(f"{scene_dir_path}/*.blend")

    steps = (
        camera_elevation_steps,
        randomization_steps,
        light_elevation_steps,
        camera_rotation_steps,
        light_rotation_steps,
        ambient_strength_steps,
    )

    functions = [
        lambda i: elevate_camera(angle=0.2, override=get_override()),
        lambda i: set_random_seed_geo_nodes(i),
        lambda i: set_light_height_offset(i),
        lambda i: rotate_camera(
            angle=2 * np.pi / camera_rotation_steps, override=get_override()
        ),
        lambda i: rotate_light(
            angle=2 * np.pi / light_rotation_steps, override=get_override()
        ),
        lambda i: set_mix_shader_value(
            i / ambient_strength_steps if ambient_strength_steps > 0 else 0
        ),
    ]

    rendered_img_ids = [
        img_filename.split(".")[0] for img_filename in os.listdir(img_dir_path)
    ]

    data = IterationData(
        steps=steps,
        functions=functions,
        rendered_img_ids=rendered_img_ids,
        img_dir_path=img_dir_path,
        labels_dir_path=labels_dir_path,
    )
    n_final_generated_images = prod(steps) * len(scenes)
    n_already_generated_images = len(os.listdir(img_dir_path))

    print(f"Final number of generated images: {n_final_generated_images}")
    print(f"Already generated images: {n_already_generated_images}")
    print(
        f"Images to generate: {n_final_generated_images - n_already_generated_images}"
    )

    print("-" * 40)
    print(f"{img_dir_path=}")
    print(f"{labels_dir_path=}")
    print("scenes:", *[" " * 4 +
          scene_path for scene_path in scenes], sep="\n")
    print("-" * 40)

    if input("Proceed? (y/n):").lower().startswith("n"):
        return False

    for scene_path in scenes:
        init_scene(scene_path)

        scene_name = os.path.basename(scene_path).split(".")[0]

        iterate_nested_steps(0, [], scene_name, data)

    return True


if __name__ == "__main__":
    render_images_and_save_data()
