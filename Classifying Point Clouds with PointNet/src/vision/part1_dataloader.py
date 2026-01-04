import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class Argoverse(Dataset):

    def load_path_with_classes(self, split: str, data_root: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Builds (path, class) pairs by pulling all txt files found under
        the data_root directory under the given split. Also builds a list
        of all labels we have provided data for.

        Each of the classes have a total of 200 point clouds numbered from 0 to 199.
        We will be using point clouds 0-169 for the train split and point clouds 
        170-199 for the test split. This gives us a 85/15 train/test split.

        Args:
        -   split: Either train or test. Collects (path, label) pairs for the specified split
        -   data_root: Root directory for training and testing data
        
        Output:
        -   pairs: List of all (path, class) pairs found under data_root for the given split 
        -   class_list: List of all classes present in the dataset *sorted in alphabetical order*
        """

        pairs = []
        class_list = []

  

        # collect class folders under data_root
        if not os.path.isdir(data_root):
            return pairs, class_list

        # list class directories and sort alphabetically
        entries = [e for e in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, e))]
        entries.sort()
        class_list = entries

        # determine index ranges for splits
        if split == "train":
            start_idx, end_idx = 0, 169
        elif split == "test":
            start_idx, end_idx = 170, 199
        else:
            raise ValueError("split must be 'train' or 'test'")

        for cls in class_list:
            cls_dir = os.path.join(data_root, cls)
            for idx in range(start_idx, end_idx + 1):
                fname = f"{idx}.txt"
                fpath = os.path.join(cls_dir, fname)
                if os.path.exists(fpath):
                    pairs.append((fpath, cls))

  

        return pairs, class_list


    def get_class_dict(self, class_list: List[str]) -> Dict[str, int]:
        """
        Creates a mapping from classes to labels. For example, [Animal, Car, Bus],
        would map to {Animal:0, Bus:1, Car:2}. *Note: for consistency, we sort the
        input classes in alphabetical order before creating the mapping (gradescope)
        tests will probably fail if you forget to do so*

        Args:
        -   class_list: List of classes to create mapping from

        Output: 
        -   classes: dictionary containing the class to label mapping
        """

        classes = dict()

  

        if class_list is None:
            return classes

        # ensure alphabetical order for deterministic mapping
        sorted_classes = sorted(class_list)
        for idx, cls in enumerate(sorted_classes):
            classes[cls] = idx

  

        return classes
    

    def __init__(self, split: str, data_root: str, pad_size: int) -> None:
        """
        Initializes the dataset. *Hint: Use the functions above*

        Args:
        -   split: Which split to pull data for. Either train or test
        -   data_root: The root of the directory containing all the data
        -   pad_size: The number of points each point cloud should contain when
                      when we access them. This is used in the pad_points function.

        Variables:
        -   self.instances: List of (path, class) pairs
        -   class_dict: Mapping from classes to labels
        -   pad_size: Number of points to pad each point cloud to
        """
        super().__init__()
        
        file_label_pairs, classes = self.load_path_with_classes(split, data_root)
        self.instances = file_label_pairs
        self.class_dict = self.get_class_dict(classes)
        self.pad_size = pad_size


    def get_points_from_file(self, path: str) -> torch.Tensor:
        """
        Returns a tensor containing all of the points in the given file

        Args:
        -   path: Path to the file that we should extract points from

        Output:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the file
        """

        pts = None

  

        points = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Expect three floats per line
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                points.append([x, y, z])

        if len(points) == 0:
            pts = torch.empty((0, 3), dtype=torch.float)
        else:
            pts = torch.tensor(points, dtype=torch.float)

  

        return pts

    def pad_points(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Pads pts to have pad_size points in it. Let p1 be the first point in 
        the tensor. We want to pad pts by adding p1 to the end of pts until 
        it has size (pad_size, 3). 

        Args:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the tensor

        Output: 
        -   pts_full: A tensor of shape (pad_size, 3)
        """

        pts_full = None

  

        if pts is None:
            return torch.empty((self.pad_size, 3), dtype=torch.float)

        N = pts.shape[0]
        if N >= self.pad_size:
            pts_full = pts[: self.pad_size].clone()
            return pts_full

        # Need to pad by repeating first point
        if N == 0:
            # nothing to repeat; return zeros
            pts_full = torch.zeros((self.pad_size, 3), dtype=torch.float)
            return pts_full

        first = pts[0].unsqueeze(0)  # (1,3)
        repeats = self.pad_size - N
        pad_block = first.repeat(repeats, 1)
        pts_full = torch.cat([pts, pad_block], dim=0)

  
        
        return pts_full

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the (points, label) pair at the given index.

        Hint: 
        1) get info from self.instances
        2) use get_points_from_file and pad_points

        Args:
        -   i: Index to retrieve

        Output:
        -   pts: Points contained in the file at the given index
        -   label: Tensor containing the label of the point cloud at the given index
        """

        pts = None
        label = None

  

        path, cls = self.instances[i]
        pts = self.get_points_from_file(path)
        pts = self.pad_points(pts)
        lbl = self.class_dict[cls]
        label = torch.tensor(lbl, dtype=torch.long)

  

        return pts, label

    def __len__(self) -> int:
        """
        Returns number of examples in the dataset

        Output: 
        -    l: Length of the dataset
        """
        
        l = None

  

        l = len(self.instances)

  

        return l
