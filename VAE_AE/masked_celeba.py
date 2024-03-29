import numpy as np
import os
from collections import namedtuple
import PIL
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from noise import pnoise2

class CelebA(datasets.CelebA):
    """Custom dataset for image inpainting at CelebA.
    """

    def __init__(self, data_root, transform, proportion, download=False, split = "train", noise = 'perlin'):
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

        self.noise = noise

        self.proportion = proportion

        if self.noise == 'perlin':

            # initialize matrix to vectorize perlin noise
            orig_img_shape = (1000, 1000)
            tups = np.empty(orig_img_shape[0] * orig_img_shape[1], dtype=tuple)
            ind = 0
            for i in range(orig_img_shape[0]):
                for j in range(orig_img_shape[1]):
                    tups[ind] = (i, j)
                    ind += 1

            self.ind_matrix =tups

            self.perlin_masks = self._pregenerate_perlin(3, (1000, 1000))

        
        
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
        
        # if axis is 1, then the rectangle is horizontal
        rec_w = int(w * self.proportion * axis + h * self.proportion * (1 - axis))

        rec_l = int(w * (1 - axis) + h * axis)

        if axis == 0:
            start = np.random.randint(0, h - rec_w)
            masked_image[start: start + rec_w,
                  0: rec_l,:] = 0
        else:
            start = np.random.randint(0, h - rec_w)
            masked_image[0: rec_l,
                  start: start + rec_w, :] = 0
            
        return masked_image

    def _binarize(self, image_data, at=0.25):
        """
        Threshold normalized array into values of 0 and 1
        """
        q = np.quantile(image_data, at)
        return (image_data>q)*1

    def _pregenerate_perlin(self, num_perlin, shape, scale = 100, octaves = 6,
        persistence = 0.5,
        lacunarity = 2.0,
        seed = None):

        perlin_masks = np.zeros((num_perlin, shape[0], shape[1]))

        for i in range(num_perlin):
            if not seed:
                seed = np.random.randint(0, 100)

            g = lambda ind: pnoise2(ind[0] / scale,
                                    ind[1] / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=222)

            # arr = np.zeros(shape)
            # for i in range(shape[0]):
            #     for j in range(shape[1]):
            #         arr[i][j] = pnoise2(i / scale,
            #                             j / scale,
            #                             octaves = octaves,
            #                             persistence = persistence,
            #                             lacunarity = lacunarity,
            #                             repeatx = 1024,
            #                             repeaty = 1024,
            #                             base = seed)
            pn = np.vectorize(g)
            arr = pn(self.ind_matrix)
            arr = arr.reshape(shape)
            max_arr = np.max(arr)
            min_arr = np.min(arr)
            norm_me = lambda x: (x - min_arr) / (max_arr - min_arr)
            norm_me = np.vectorize(norm_me)
            arr = norm_me(arr)

            perlin_masks[i, :, :] = self._binarize(arr, at = self.proportion)

        return perlin_masks



    def _get_perlin(self, shape = (200, 200),
        scale = 100, octaves = 6,
        persistence = 0.5,
        lacunarity = 2.0,
        seed = None):
        """
        Get mask generated with perlin noise. 0 indicates background and 1 indicates the mask.
        """

       # get random sample from perlin list
        mask_ind = np.random.randint(0, self.perlin_masks.shape[0] - 1)

        mask_start_x = np.random.randint(0, self.perlin_masks.shape[1] - shape[0] - 1)
        mask_end_x = mask_start_x + shape[0]

        mask_start_y = np.random.randint(0, self.perlin_masks.shape[2] - shape[1] - 1)
        mask_end_y = mask_start_y + shape[1]

        return(self.perlin_masks[mask_ind, mask_start_x: mask_end_x, mask_start_y:mask_end_y])

            
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
            Get item from the CelebA dataset with masked image
        """
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        
        if self.noise == 'perlin':
          mask = self._get_perlin(shape = (np.array(X).shape[0], np.array(X).shape[1]))
          mask = PIL.Image.fromarray(np.invert(mask.dtype('uint8')))
        else:
          mask = PIL.Image.fromarray(self._get_rect(X))

        if self.transform is not None:
            X = self.transform(X)
            mask = self.transform(mask)

        return X, mask
    
    def display(self, index: int) -> None:
        """
            Display the masked and ground truth image at an index 
        """

        X, mask = self.__getitem__(index)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Mask')
        ax1.imshow(mask.permute(2,1, 0))
        ax2.set_title('Transformed Image')
        ax2.imshow(X.permute(2,1, 0))
