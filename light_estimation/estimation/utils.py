import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from math import atan2
from shutil import rmtree
from typing import List
import torch

import numpy as np


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a)


def empty_create_dir(dir_path: str):
    if os.path.isdir(dir_path):
        rmtree(dir_path)
    os.mkdir(dir_path)


def check_create_dir(dir_path: str):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def angle_between_3d(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    [x1, y1], [x2, y2] = v1[:2], v2[:2]
    a = atan2(x1*y2 - y1*x2, x1*x2 + y1*y2)
    return (a if a > 0 else 2 * np.pi + a)


def angles_to_stereographic(a: float, b: float):
    m = np.tan((np.pi / 2 - b) / 2)
    sx = m * np.cos(a)
    sy = m * np.sin(a)

    return sx, sy


def stereographic_to_angles(sx: float, sy: float):
    l = np.sqrt(sx**2 + sy**2)
    a = np.arcsin(sy / l)
    b = np.pi / 2 - 2 * np.arctan(l)

    return a, b


def get_index_from_heatmap(outputs, b_bins):
    indexes = None
    for output in outputs:
        heatmap_index = np.argmax(output.detach().numpy())
        if indexes is None:
            indexes = np.array([np.floor(heatmap_index / b_bins), heatmap_index % b_bins])
        else:
            indexes = np.vstack((indexes, np.array([np.floor(heatmap_index / b_bins), heatmap_index % b_bins])))

    return torch.from_numpy(indexes)
        

@dataclass
class ObjectData:
    a: float
    b: float
    d: float
    pos: List[float] = field(init=False)


@dataclass
class LightData(ObjectData):
    shadow_soft_size: float
    power: float


@dataclass
class SampleData:
    sample_id: str
    camera: ObjectData
    light: LightData
    geo_nodes_seed: int
    ambient: float
    plane_material: str
    visible_collection: str
    objects_materials: List[str] = field(init=False)
    environment_texture: str = field(init=False)

    def save(self, labels_dir_path: str):
        with open(f'{labels_dir_path}/{self.sample_id}.json', 'w') as file:
            file.write(json.dumps(asdict(self)))


def list_gpus():
    import bpy
    scene = bpy.context.scene
    scene.cycles.device = 'GPU'

    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()
    cprefs.compute_device_type = 'CUDA'

    for device in cprefs.devices:
        print(f'device.id={device.id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--list_gpus',
        help='list gpu ids detected by bpy',
        action='store_true',
    )

    args = parser.parse_args()

    if args.list_gpus:
        list_gpus()
