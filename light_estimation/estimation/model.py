from typing import Callable

import torch
import torch.nn as nn
import torchvision.models as models

from estimation.config import h_dataset, w_dataset, weights
from estimation.enums import DataMode
from estimation.layers import light_angles_head_discrete, light_angles_head


def efficient_net_b3(
    weights: str = None,
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights
    )
    light_angles = light_angles_head(base_model)

    return light_angles


def efficient_net_b3_discrete(
    weights: str = None,
    a_bins: int = 32,
    b_bins: int = 16
) -> nn.Module:
    base_model = models.efficientnet_b3(
        weights=weights
    )
    light_angles = light_angles_head_discrete(base_model, a_bins, b_bins)

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
    else:
        return globals()[model_architecture](
            weights=weights,
        )
