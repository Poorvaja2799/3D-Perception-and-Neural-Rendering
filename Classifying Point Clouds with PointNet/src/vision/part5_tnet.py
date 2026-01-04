from typing import Tuple

import torch
from vision.part3_pointnet import PointNet
from torch import nn


class TNet(nn.Module):

    def __init__(self,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for TNet to define layers.

        Hint: The architecture is almost the same as your PointNet, just with a different
              output dimension

        Just like with PointNet, you will need to repeat the first hidden dim.
        See mlp(64, 64) in the diagram. Furthermore, you will want to include
        a BatchNorm1d after each layer in the encoder except for the final layer
        for easier training.


        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points.
        -   hidden_dims: The dimensions of the encoding MLPs. This is similar to
                         that of PointNet
        -   regression_dims: The dimensions of regression MLPs. This is similar
                         to the classifier dims in PointNet
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.regression_head = None
        self.in_dim = None

  

        self.in_dim = in_dim

        h1, h2, h3 = hidden_dims

        # Encoder layers (repeat first hidden dim)
        self.enc_lin1 = nn.Linear(in_dim, h1)
        self.enc_bn1 = nn.BatchNorm1d(h1)

        self.enc_lin2 = nn.Linear(h1, h1)
        self.enc_bn2 = nn.BatchNorm1d(h1)

        self.enc_lin3 = nn.Linear(h1, h2)
        self.enc_bn3 = nn.BatchNorm1d(h2)

        self.enc_lin4 = nn.Linear(h2, h3)

        # Regression head similar to classifier but outputs in_dim*in_dim
        out_dim = in_dim * in_dim
        self.regression_head = nn.Sequential(
            nn.Linear(h3, regression_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(regression_dims[0], regression_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(regression_dims[1], out_dim)
        )

        self.pts_per_obj = pts_per_obj

  


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the T-Net. Compute the transformation matrices, but do not apply them yet.
        The forward pass is the same as that of your original PointNet, except for:
        1) Adding an identity matrix (be sure to set the device to x.device)
        2) Reshaping the output

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   transform_matrices: tensor of shape (B, in_dim, in_dim) containing transformation matrices
                       These will be used to transform the point cloud.
        '''

        transform_matrices = None

  

        B, N, D = x.shape

        # Layer 1
        out = self.enc_lin1(x)
        out = out.permute(0, 2, 1)
        out = self.enc_bn1(out)
        out = out.permute(0, 2, 1)
        out = torch.relu(out)

        # Layer 2
        out = self.enc_lin2(out)
        out = out.permute(0, 2, 1)
        out = self.enc_bn2(out)
        out = out.permute(0, 2, 1)
        out = torch.relu(out)

        # Layer 3
        out = self.enc_lin3(out)
        out = out.permute(0, 2, 1)
        out = self.enc_bn3(out)
        out = out.permute(0, 2, 1)
        out = torch.relu(out)

        # Layer 4
        out = self.enc_lin4(out)  # (B, N, h3)

        # Global max pool
        global_feat, _ = torch.max(out, dim=1)  # (B, h3)

        reg = self.regression_head(global_feat)  # (B, in_dim*in_dim)

        # initialize identity and add
        id_mat = torch.eye(self.in_dim, device=reg.device, dtype=reg.dtype).view(1, -1)
        reg = reg + id_mat

        transform_matrices = reg.view(B, self.in_dim, self.in_dim)

  

        return transform_matrices


class PointNetTNet(nn.Module):

    def __init__(
        self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        tnet_hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        tnet_regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet with T-Net. The main difference between our
        original PointNet model and this one is the addition of a T-Net to predict
        a transform to apply to the input point cloud.

        Hint:
        1) Think about how to drectly reuse your PointNet implementation from earlier

        Args:
        -   classes: Number of output classes
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   tnet_hidden_dims: The dimensions of the encoding MLPs for T-Net
        -   tnet_regression_dims: The dimensions of the regression MLPs for T-Net
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.tnet = None
        self.point_net = None

  

        # create TNet and PointNet instances
        self.tnet = TNet(in_dim=in_dim, hidden_dims=tnet_hidden_dims, regression_dims=tnet_regression_dims, pts_per_obj=pts_per_obj)
        self.point_net = PointNet(classes=classes, in_dim=in_dim, hidden_dims=hidden_dims, classifier_dims=classifier_dims, pts_per_obj=pts_per_obj)

  


    def apply_tnet(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the transformation matrices by passing x into T-Net, and
        compute the transformed points by batch matrix multiplying x by the
        transformation matrices.

        Hint: Use torch.bmm for batched matrix multiplication. Multiply x by
        the transformation matrix rather than the other way around.

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   x_transformed: tensor of shape (B, pts_per_obj, 3) containing the
                           transformed point clouds per object.
        '''
        x_transformed = None

  
        # compute transform matrices
        transforms = self.tnet(x)

        # x: (B, pts, in_dim), transforms: (B, in_dim, in_dim)
        x_transformed = torch.bmm(x, transforms)

        return x_transformed


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Hint:
        1) Apply the T-Net transforms via apply_tnet
        2) Use your original PointNet architecture on the transformed pointcloud

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''
        class_outputs = None
        encodings = None

  

        # Apply TNet to get transformed points
        x_trans = self.apply_tnet(x)

        # Pass transformed points through PointNet
        class_outputs, encodings = self.point_net(x_trans)

  

        return class_outputs, encodings

