import numpy as np
import os
from collections import namedtuple
import PIL
from noise import pnoise2
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

class CelebA(datasets.CelebA):
    """
    Custom dataset for image inpainting at CelebA.
    """

    def __init__(self, data_root, transform, proportion, noise = 'perlin', download=False, split = "train",):
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

        self.noise = noise


    def _get_rect(self, shape):
        """
            Generate image mask with random rectangles that are either horizontal or vertical.
        """

        mask = np.zeros(shape)


        h, w = mask.shape[:2]

        # get whether the obscured region is a row or column
        axis = np.random.randint(low=0, high=2)

        # if axis is 1, then the rectangle is horizontal
        rec_w = int(w * self.proportion * axis + h * self.proportion * (1 - axis))

        rec_l = int(w * (1 - axis) + h * axis)

        if axis == 0:
            start = np.random.randint(0, h - rec_w)
            mask[start: start + rec_w,
            0: rec_l] = 1
        else:
            start = np.random.randint(0, h - rec_w)
            mask[0: rec_l,
            start: start + rec_w] = 1

        return mask

    def _binarize(self, image_data, at=0.25):
        """
        Threshold normalized array into values of 0 and 1
        """
        q = np.quantile(image_data, at)
        return (image_data<q)*1




    def _get_perlin(self, shape = (200, 200),
        scale = 100, octaves = 6,
        persistence = 0.5,
        lacunarity = 2.0,
        seed = None):
        """
        Get mask generated with perlin noise. 0 indicates background and 1 indicates the mask.
        """
        if not seed:
            seed = np.random.randint(0, 100)

        arr = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                arr[i][j] = pnoise2(i / scale,
                                    j / scale,
                                    octaves = octaves,
                                    persistence = persistence,
                                    lacunarity = lacunarity,
                                    repeatx = 1024,
                                    repeaty = 1024,
                                    base = seed)

        max_arr = np.max(arr)
        min_arr = np.min(arr)
        norm_me = lambda x: (x - min_arr) / (max_arr - min_arr)
        norm_me = np.vectorize(norm_me)
        arr = norm_me(arr)

        return arr

    def _add_mask(self, image, mask):
        """

        :param image:
        :type image:
        :param mask:
        :type mask:
        :return:
        :rtype:
        """
        masked_image = np.array(image.copy())
        masked_image[mask == 1] = (0, 0, 0)

        return masked_image




    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
            Get item from the CelebA dataset with masked image
        """
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.noise == 'perlin':
            perlin_mask = self._get_perlin(shape = (np.array(X).shape[0], np.array(X).shape[1]))
            mask = self._binarize(perlin_mask)
        else:
            mask = self._get_rect(shape = (np.array(X).shape[0], np.array(X).shape[1]))

        masked_image = PIL.Image.fromarray(self._add_mask(X, mask))

        if self.transform is not None:
            X = self.transform(X)
            masked_image = self.transform(masked_image)

        return X, masked_image

    def display(self, index: int) -> None:
        """
            Display the masked and ground truth image at an index
        """

        X, mask = self.__getitem__(index)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title('Mask')
        ax1.imshow(mask.permute(1,2, 0))
        ax2.set_title('Transformed Image')
        ax2.imshow(X.permute(1,2, 0))
