import torch
import torch.nn as nn
import numpy as np


class CircularLossDiscrete(nn.Module):
    def __init__(self, n_bins=32):
        super(CircularLossDiscrete, self).__init__()
        self.n_bins = n_bins

    def forward(self, predicted_probs, true_angles):
        predicted_angle = torch.argmax(predicted_probs) * (360 / self.n_bins)

        error = (predicted_angle - true_angles) % 360
        error = torch.min(error, 360 - error)
        return torch.mean(error)


class CircularLoss(nn.Module):
    def __init__(self):
        super(CircularLoss, self).__init__()

    def forward(self, predicted_angles, true_angles):
        error = (predicted_angles - true_angles) % 360
        error = torch.min(error, 360 - error)
        return torch.mean(error)


class CustomDiscreteLoss(nn.Module):
    def __init__(self, a_bins=32, b_bins=16, weight_losses=False):
        super(CustomDiscreteLoss, self).__init__()
        self.a_bins = a_bins
        self.b_bins = b_bins
        self.weight_losses = weight_losses

    def forward(self, preds, labels):
        # Extract true and predicted angles
        preds_a = torch.argmax(preds[0], dim=1).float()
        preds_a.requires_grad = True
        preds_b = torch.argmax(preds[1], dim=1).float()
        preds_b.requires_grad = True
        # Transform exact angles to integer bin values
        labels_a = labels[:, 0].float()
        labels_a.requires_grad = True
        labels_b = labels[:, 1].float()
        labels_b.requires_grad = True

        # Calculate azimuth and elevation square errors
        angular_diff = torch.abs(preds_a - labels_a)
        L_azimuth = torch.square(torch.minimum(angular_diff, self.a_bins - 1 - angular_diff))

        L_elevation = torch.square(preds_b - labels_b)

        # Weight factors
        alpha = beta = 1
        if self.weight_losses:
            alpha = (self.b_bins - 1 - labels_b) / self.b_bins
            beta = labels_b / self.b_bins

        # Combined loss
        loss = alpha * L_azimuth + beta * L_elevation
        return loss.mean()
    

class CircularDistanceLoss(nn.Module):
    def __init__(self, a_bins):
        super(CircularDistanceLoss, self).__init__()
        self.a_bins = a_bins

    def forward(self, pred, true):
        # Separate X and Y coordinates
        pred_x, pred_y = pred[:, 0], pred[:, 1]
        true_x, true_y = true[:, 0], true[:, 1]
        
        # Calculate X distance
        d_x = torch.abs(pred_x - true_x)
        
        # Calculate Y distance considering circularity
        d_y = torch.abs(pred_y - true_y)
        d_y_circular = torch.min(d_y, self.a_bins - d_y)
        
        # Combine distances
        loss = d_x ** 2 + d_y_circular ** 2
        
        # Return mean loss over batch
        return loss.mean()
