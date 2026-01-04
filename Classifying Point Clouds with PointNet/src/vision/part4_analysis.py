from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision.part2_baseline import Baseline
from vision.part3_pointnet import PointNet
from vision.part5_tnet import PointNetTNet


def get_critical_indices(model: Union[PointNet, PointNetTNet], pts: torch.Tensor) -> np.ndarray:
    '''
    Finds the indices of the critical points in the given point cloud. A
    critical point is a point that contributes to the global feature (i.e
    a point whose calculated feature has a maximal value in at least one 
    of its dimensions)
    
    Hint:
    1) Use the encodings returned by your model
    2) Make sure you aren't double-counting points since points may
       contribute to the global feature in more than one dimension

    Inputs:
        model: The trained model
        pts: (model.pad_size, 3) tensor point cloud representing an object

    Returns:
        crit_indices: (N,) numpy array, where N is the number of critical pts

    '''
    crit_indices = None

    model.eval()
    with torch.no_grad():
        # Ensure pts shaped (1, N, 3)
        if pts.dim() == 2:
            inp = pts.unsqueeze(0)
        else:
            inp = pts

        _, encodings = model(inp)

        # encodings expected shape (B, N, C)
        enc = encodings[0]  # (N, C)

        # For each channel/dimension, find the point index that has the maximum
        # value. This gives an array of length C with indices in [0, N-1].
        _, argmax = torch.max(enc, dim=0)  # (C,)

        # Unique indices are the critical points (avoid double-counting)
        crit_idx = torch.unique(argmax)

        crit_indices = crit_idx.cpu().numpy()

    model.train()

    return crit_indices

    
def get_confusion_matrix(
    model: Union[Baseline, PointNet, PointNetTNet], 
    loader: DataLoader, 
    num_classes: int,
    normalize: bool=True, 
    device='cpu'
) -> np.ndarray:
    '''
    Builds a confusion matrix for the given models predictions
    on the given dataset. 
    
    Recall that each ground truth label corresponds to a row in
    the matrix and each predicted value corresponds to a column.

    A confusion matrix can be normalized by dividing entries for
    each ground truch prior by the number of actual isntances the
    ground truth appears in the dataset. (Think about what this means
    in terms of rows and columns in the matrix) 

    Hint:
    1) Generate list of prediction, ground-truth pairs
    2) For each pair, increment the correct cell in the matrix
    3) Keep track of how many instances you see of each ground truth label
       as you go and use this to normalize 

    Args: 
    -   model: The model to use to generate predictions
    -   loader: The dataset to use when generating predictions
    -   num_classes: The number of classes in the dataset
    -   normalize: Whether or not to normalize the matrix
    -   device: If 'cuda' then run on GPU. Run on CPU by default

    Output:
    -   confusion_matrix: a numpy array with shape (num_classes, num_classes)
                          representing the confusion matrix
    '''

    model.eval()
    confusion_matrix = None

    # Initialize matrix and counts
    cm = np.zeros((num_classes, num_classes), dtype=float)
    gt_counts = np.zeros((num_classes,), dtype=float)

    with torch.no_grad():
        for batch in loader:
            # loader can be a DataLoader yielding (pts, labels) or a list of tuples
            pts, labels = batch

            # ensure tensors
            if isinstance(pts, torch.Tensor):
                inp = pts
            else:
                inp = torch.tensor(pts)

            if isinstance(labels, torch.Tensor):
                gts = labels
            else:
                gts = torch.tensor(labels)

            # forward pass
            outputs, _ = model(inp)

            preds = torch.argmax(outputs, dim=1).cpu()
            gts = gts.view(-1).cpu()

            for gt, pred in zip(gts.tolist(), preds.tolist()):
                cm[int(gt), int(pred)] += 1.0
                gt_counts[int(gt)] += 1.0

    # Normalize rows if requested
    if normalize:
        # Avoid division by zero
        for i in range(num_classes):
            if gt_counts[i] > 0:
                cm[i, :] = cm[i, :] / gt_counts[i]

    confusion_matrix = cm
    model.train()

    return confusion_matrix
