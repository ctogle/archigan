import os
import cv2
import numpy as np
from tqdm import tqdm
#import argparse
#from multiprocessing import Pool


class ArchiPipeline:
    """

    Args:
        layers (list):
        stages (list):

    """

    def __init__(self, layers, stages):
        self.layers = layers
        self.stages = stages

    @staticmethod
    def combineAB(path_A, path_B, path_AB):
        """Combine images at `path_A` and `path_B` into new image at `path_AB`
            from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py#L8
        """
        im_A = cv2.imread(path_A, 1)
        im_B = cv2.imread(path_B, 1)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

    @staticmethod
    def splits(N, val=0.05, test=0.05, train=0.9):
        """Create list of list of indices for splitting data"""
        index = np.random.permutation(N)
        piles = [[], [], []]
        thresholds = [int(x * N) for x in (val, test, train)]
        pile = 0
        for j in index:
            piles[pile].append(j)
            if len(piles[pile]) > thresholds[pile]:
                pile += 1
        return piles

    def setup_training(self, directory):
        """Prepare combined images and splits for training"""
        for u, v in self.stages:
            x, y = self.layers[u], self.layers[v]
            z = '__'.join((os.path.basename(x), os.path.basename(y)))
            z = os.path.join(directory, z)
            os.makedirs(z, exist_ok=True)
            fs = os.listdir(x)
            splits = self.splits(len(fs))
            splitnames = ('val', 'test', 'train')
            with tqdm(total=len(fs)) as pbar:
                for split, splitname in zip(splits, splitnames):
                    split = [fs[j] for j in split]
                    os.makedirs(os.path.join(z, splitname), exist_ok=True)
                    pbar.set_description(f'{len(split)} samples in {splitname}')
                    for f in split:
                        g = os.path.join(x, f)
                        h = os.path.join(y, f)
                        l = os.path.join(z, splitname, f)
                        self.combineAB(g, h, l)
                        pbar.update(1)
