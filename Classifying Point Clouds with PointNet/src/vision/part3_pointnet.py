from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.

        Hint: See the modified PointNet architecture diagram from the pdf.
        You will need to repeat the first hidden dim (see mlp(64, 64) in the diagram).
        Furthermore, you will want to include a BatchNorm1d after each layer in the encoder
        except for the final layer for easier training.

        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points. This parameter is 3 by default for
                    for the basic PointNet.
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.classifier_head = None

  

        # Encoder: per-point MLPs
        h1, h2, h3 = hidden_dims

        # Define linear layers for encoder
        self.enc_lin1 = nn.Linear(in_dim, h1)
        self.enc_bn1 = nn.BatchNorm1d(h1)

        self.enc_lin2 = nn.Linear(h1, h1)
        self.enc_bn2 = nn.BatchNorm1d(h1)

        self.enc_lin3 = nn.Linear(h1, h2)
        self.enc_bn3 = nn.BatchNorm1d(h2)

        # final encoder layer -> outputs hidden_dims[-1]
        self.enc_lin4 = nn.Linear(h2, h3)

        # Classifier head: 1024 -> 512 -> 256 -> classes
        self.classifier_head = nn.Sequential(
            nn.Linear(h3, classifier_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(classifier_dims[1], classes)
        )

        # Store pts_per_obj (not strictly necessary but useful)
        self.pts_per_obj = pts_per_obj

  


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''

        class_outputs = None
        encodings = None

  

        B, N, _ = x.shape

        # Layer 1
        out = self.enc_lin1(x)  # (B,N,h1)
        # BatchNorm1d expects (B, C, L)
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

        # Layer 4 (final encoder)
        out = self.enc_lin4(out)  # (B,N,h3)
        encodings = out

        # Global max pool over points
        global_feat, _ = torch.max(encodings, dim=1)  # (B, h3)

        class_outputs = self.classifier_head(global_feat)

  

        return class_outputs, encodings
