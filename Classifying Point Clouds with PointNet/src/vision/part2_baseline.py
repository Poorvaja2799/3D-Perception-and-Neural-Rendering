from typing import Tuple

import torch
from torch import nn


class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction
    '''
    def __init__(self,
        classes: int,
        voxel_resolution=4,
        mode="count"
    ) -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: Whether to count the number of points per voxel ("count") or just check binary occupancy ("occupancy")
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()

        self.classifier = None
        self.voxel_resolution = None
        self.mode = None

  

        # store config
        self.voxel_resolution = voxel_resolution
        self.mode = mode

        in_features = voxel_resolution ** 3
        # single linear classifier
        self.classifier = nn.Linear(in_features, classes)

  


    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting the number of points per voxel.
        This is effectively taking a 3D histogram for every item in a batch.

        Hint: 
        1) torch.histogramdd will be useful here

        Args:
        -   x: tensor of shape (B, N, in_dim)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3), indicating the percentage of points that landed in each voxel
        '''

        counts = None

  

        B, N, D = x.shape
        R = self.voxel_resolution

        # normalize coordinates from [-1,1] to [0,1]
        coords = (x + 1.0) / 2.0

        # scale to [0, R) then floor to get bin indices 0..R-1
        scaled = coords * R
        idxs = torch.floor(scaled).long()
        idxs = torch.clamp(idxs, 0, R - 1)

        counts = torch.zeros((B, R ** 3), dtype=torch.float, device=x.device)

        for b in range(B):
            linear_idx = idxs[b, :, 0] * (R * R) + idxs[b, :, 1] * R + idxs[b, :, 2]
            binc = torch.bincount(linear_idx, minlength=R ** 3).to(dtype=torch.float)
            # normalize to percentage of points
            if N > 0:
                counts[b] = binc / float(N)
            else:
                counts[b] = binc

  

        return counts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model. Make sure you handle the case where the mode 
        is set to "occupancy" by thresholding the result of count_points on zero.

        Args:
            x: tensor of shape (B, N, 3), where B is the batch size and N is the number of points per
               point cloud
        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   None, just used to allow reuse of training infrastructure
        '''

        class_outputs = None

  

        feats = self.count_points(x)
        if self.mode == "occupancy":
            feats = (feats > 0).to(dtype=torch.float)

        class_outputs = self.classifier(feats)

  

        return class_outputs, None
