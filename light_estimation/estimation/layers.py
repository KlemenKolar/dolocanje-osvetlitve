import torch
import torch.nn as nn


class LightAnglesHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 256)
        self.Layer2 = nn.Linear(256, 64)
        self.Layer3 = nn.Linear(64, 64)
        self.Layer4 = nn.Linear(64, 2)  # n=2, ker ni diskretno in napovedujemo azimutski in elevacijski kot...
        self.R = nn.ReLU()

    def forward(self, x):  # x je slika formata 128 x 128
        x = self.R(self.base_model(x))
        x = self.R(self.Layer1(x))
        x = self.R(self.Layer2(x))
        x = self.Layer3(x)
        x = self.Layer4(x)
        return x


class LightAnglesHeadDiscrete(nn.Module):
    def __init__(self, base_model, a_bins, b_bins):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 256)
        self.Layer2 = nn.Linear(256, 64)
        self.Layer3 = nn.Linear(64, 64)
        self.Layer_a = nn.Linear(64, a_bins)
        self.Layer_b = nn.Linear(64, b_bins)
        self.R = nn.ReLU()

    def forward(self, x):  # x je slika formata 128 x 128
        x = self.R(self.base_model(x))
        x = self.R(self.Layer1(x))
        x = self.R(self.Layer2(x))
        x = self.Layer3(x)
        a_output = self.Layer_a(x)
        b_output = self.Layer_b(x)
        return a_output, b_output


class LightAnglesHeadHeatmap(nn.Module):
    def __init__(self, base_model, a_bins, b_bins):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 256)
        self.Layer2 = nn.Linear(256, 64)
        self.Layer3 = nn.Linear(64, 64)
        self.Layer_heatmap = nn.Linear(64, a_bins * b_bins)
        self.R = nn.ReLU()
        self.a_bins = a_bins
        self.b_bins = b_bins

    def forward(self, x):  # x je slika formata 128 x 128
        x = self.R(self.base_model(x))
        x = self.R(self.Layer1(x))
        x = self.R(self.Layer2(x))
        x = self.Layer3(x)
        heatmap = self.Layer_heatmap(x)
        #heatmap =  heatmap.view(-1, self.a_bins, self.b_bins)
        return heatmap
    

class LightAnglesHeadHeatmapNoBottleneck(nn.Module):
    def __init__(self, base_model, a_bins, b_bins):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 1000)
        self.Layer_heatmap = nn.Linear(1000, a_bins * b_bins)
        self.R = nn.ReLU()
        self.a_bins = a_bins
        self.b_bins = b_bins

    def forward(self, x):
        x = self.R(self.base_model(x))
        x = self.Layer1(x)
        heatmap = self.Layer_heatmap(x)
        return heatmap
    

class LightAnglesHeadHeatmapNoBottleneckNoReLu(nn.Module):
    def __init__(self, base_model, a_bins, b_bins):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 1000)
        self.Layer_heatmap = nn.Linear(1000, a_bins * b_bins)
        self.R = nn.ReLU()
        self.a_bins = a_bins
        self.b_bins = b_bins

    def forward(self, x):
        x = self.base_model(x)
        x = self.Layer1(x)
        heatmap = self.Layer_heatmap(x)
        return heatmap
    

class LightAnglesHeadHeatmapNoBottleneck2(nn.Module):
    def __init__(self, base_model, a_bins, b_bins):
        super().__init__()
        self.base_model = base_model
        self.Layer1 = nn.Linear(1000, 1000)
        self.Layer2 = nn.Linear(1000, 512)
        self.layer3 = nn.Linear(512, 512)
        self.Layer_heatmap = nn.Linear(512, a_bins * b_bins)
        self.R = nn.ReLU()
        self.a_bins = a_bins
        self.b_bins = b_bins

    def forward(self, x):
        x = self.R(self.base_model(x))
        x = self.R(self.Layer1(x))
        x = self.R(self.Layer2(x))
        x = self.layer3(x)
        heatmap = self.Layer_heatmap(x)
        return heatmap


def light_angles_head(base_model):
    model = LightAnglesHead(base_model)
    return model


def light_angles_head_discrete(base_model, a_bins, b_bins):
    model = LightAnglesHeadDiscrete(base_model, a_bins, b_bins)
    return model


def light_angles_head_heatmap(base_model, a_bins, b_bins):
    model = LightAnglesHeadHeatmap(base_model, a_bins, b_bins)
    return model

def light_angles_head_heatmap_no_bottleneck(base_model, a_bins, b_bins):
    model = LightAnglesHeadHeatmapNoBottleneck(base_model, a_bins, b_bins)
    return model

def light_angles_head_heatmap_no_bottleneck2(base_model, a_bins, b_bins):
    model = LightAnglesHeadHeatmapNoBottleneck2(base_model, a_bins, b_bins)
    return model

def light_angles_head_heatmap_no_bottleneck_no_relu(base_model, a_bins, b_bins):
    model = LightAnglesHeadHeatmapNoBottleneckNoReLu(base_model, a_bins, b_bins)
    return model