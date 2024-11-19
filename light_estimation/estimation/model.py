from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models

from estimation.config import h_dataset, w_dataset, weights
from estimation.enums import DataMode
from estimation.layers import (light_angles_head_discrete, light_angles_head, light_angles_head_heatmap,
                               light_angles_head_heatmap_no_bottleneck, light_angles_head_heatmap_no_bottleneck_no_relu,
                               light_angles_head_heatmap_no_bottleneck2)


def efficient_net_b3(
    weights: str = None,
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head(base_model)

    return light_angles


def efficient_net_b3_discrete(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head_discrete(base_model, a_bins, b_bins)

    return light_angles

def efficient_net_b5_discrete(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b5(
        weights=weights,
    )
    light_angles = light_angles_head_discrete(base_model, a_bins, b_bins)

    return light_angles

def efficient_net_b7_discrete(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b7(
        weights=weights,
    )
    light_angles = light_angles_head_discrete(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b3_heatmap(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b3_heatmap_no_bottleneck(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b3_heatmap_no_bottleneck_no_relu(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck_no_relu(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b3_heatmap_no_bottleneck2(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck2(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b5_heatmap(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b5(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b5_heatmap_no_bottleneck(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b5(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b5_heatmap_no_bottleneck2(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b5(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck2(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b5_heatmap_no_bottleneck_no_relu(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b5(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck_no_relu(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b7_heatmap(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b7(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b7_heatmap_no_bottleneck(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b7(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b7_heatmap_no_bottleneck_no_relu(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b7(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck_no_relu(base_model, a_bins, b_bins)

    return light_angles


def efficient_net_b7_heatmap_no_bottleneck2(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b7(
        weights=weights,
    )
    light_angles = light_angles_head_heatmap_no_bottleneck2(base_model, a_bins, b_bins)

    return light_angles


def create_model(
    model_architecture: str,
    weights=weights,
    data_mode: DataMode = DataMode.RADIANS,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    if model_architecture not in globals():
        raise Exception(
            f'Model architecture \'{model_architecture}\' is not defined.'
        )

    if data_mode == DataMode.DISCRETE:
        return globals()[model_architecture](
            weights=weights,
            a_bins=a_bins,
            b_bins=b_bins
        )
    elif data_mode == DataMode.HEATMAP:
        return globals()[model_architecture](
            weights=weights,
            a_bins=a_bins,
            b_bins=b_bins
        )
    else:
        return globals()[model_architecture](
            weights=weights,
        )
