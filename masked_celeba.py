import numpy as np
import os
from collections import namedtuple
import PIL
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class CelebA(datasets.CelebA):
    """Custom dataset for image inpainting at CelebA.
    """

    def __init__(self, data_root, transform, proportion, download=False, split = "train",):
        """Custom dataset for image inpainting at CelebA.

        Args:
            data_root (str): Path to dataset
            transform (torchvision.transforms): transform for dataset 
            proportion (int): Desired proportion of image to be obscured
            download (bool): Whether or not to download the dataset
            split (str): train, val, test
        """
        
        np.random.seed(222)
        
        super().__init__(root = data_root, download = download, split = split, transform=transform)
        
        
        assert isinstance(proportion, float)
        if isinstance(proportion, float):
            self.proportion = proportion
            
    def _get_rect(self, sample):
        """
            Get item from the CelebA dataset with masked image
        """
        # image is first element in sample
        image = np.array(sample)
        masked_image = image.copy()

        h, w = image.shape[:2]
        
        # get whether the obscured region is a row or column
        axis = np.random.randint(low = 0, high = 2)
        
        # if axis is 0, then the rectangle is horizontal
        rec_w = int(w * self.proportion * axis + h * self.proportion * (1 - axis))
        rec_l = int(w * (1 - axis) + h * axis)

        if axis == 0:
            start = np.random.randint(0, h - rec_w)
            masked_image[start: start + rec_w,
                  0: rec_l] = 0
        else:
            start = np.random.randint(0, w - rec_l)
            masked_image[0: rec_l,
                  start: start + rec_w] = (0, 0, 0) 
            
        return masked_image
            
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
            Get item from the CelebA dataset with masked image
        """
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        
        mask = PIL.Image.fromarray(self._get_rect(X))

        if self.transform is not None:
            X = self.transform(X)
            mask = self.transform(mask)

        return X, mask
    
    def display(self, index: int) -> None:
        """
            Display the masked and ground truth image at an index 
        """

        mask, X = self.__getitem__(index)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Mask')
        ax1.imshow(mask)
        ax2.set_title('Transformed Image')
        ax2.imshow(X)
