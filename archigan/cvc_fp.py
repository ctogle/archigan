"""Utilities for the CVC-FP dataset

    `parse_CVC_FP_svg` parses SVG files available from:
        http://dag.cvc.uab.es/resources/floorplans/

    `FootprintInputLayer/RepartitionInputLayer` can represent the mapping from
    building footprint to footprint with boundary doors/windows

    `RepartitionInputLayer/RepartitionOutputLayer` can represent the mapping from
    building footprint with boundary doors/windows to fully partitioned floor plan

"""
import cv2
import numpy as np
from svglib.svglib import svg2rlg # requires light modification to retain some info
from collections import defaultdict
from archigan.datalayer import Layer


def parse_CVC_FP_svg(path, classes):
    """Parser for SVGs form the CVC-FP dataset:
        http://dag.cvc.uab.es/resources/floorplans/
    """
    drawing = svg2rlg(path)
    byclass = defaultdict(list)
    for cls in classes:
        for py in drawing.contents[0].contents:
            if py._class == cls:
                loop = list(zip(py.points[::2], py.points[1::2]))
                byclass[py._class].append(loop)
    sample = {
        'byclass': byclass,
        'height': drawing.height,
        'width': drawing.width,
    }
    return sample


class FootprintInputLayer(Layer):

    labels = dict([
        ('Footprint', 5),
    ])

    def __call__(self, byclass, height, width):
        mask = self.norm_mask(self.mask(byclass, height, width))
        layer = mask.max() - mask
        return layer


class RepartitionInputLayer(Layer):

    labels = dict([
        ('Door', 1),
        ('Window', 2),
        ('Parking', 3),
        ('Room', 5),
    ])

    def __call__(self, byclass, height, width):
        # start with the inverse of the mask
        # blend in annotations which touch the boundary of the footprint
        mask = self.norm_mask(self.mask(byclass, height, width))
        layer = mask.max() - mask
        for cls, label in self.labels.items():
            for py in byclass[cls]:
                # determine if a dilated mask of py intersects the mask
                # which implies its a feature on the boundary of the footprint
                pymask = self.mask({cls: [py]}, height, width)
                kernel = np.ones((3, 3), np.uint8)
                pyfilt = cv2.erode(pymask.copy(), kernel, iterations=1)
                pyfilt = (((mask) * (pyfilt.max() - pyfilt)).max() > 0)
                if pyfilt or cls == 'Room':
                    weight = (pymask == 0).astype(int)
                    layer = weight * label + (1 - weight) * layer
        return layer


class RepartitionOutputLayer(Layer):

    labels = dict([
        ('Door', 1),
        ('Window', 2),
        ('Parking', 3),
        ('Room', 4),
        ('Wall', 5),
    ])

    def __call__(self, byclass, height, width):
        mask = self.norm_mask(self.mask(byclass, height, width))
        layer = mask.max() - mask
        # start with the inverse of the mask
        # blend in annotations which touch the boundary of the footprint
        for cls, label in self.labels.items():
            for py in byclass[cls]:
                # determine if a dilated mask of py intersects the mask
                # which implies its a feature on the boundary of the footprint
                pymask = self.mask({cls: [py]}, height, width)
                kernel = np.ones((3, 3), np.uint8)
                pymask = cv2.erode(pymask, kernel, iterations=3)
                # blend a mask of py with the layer using a binary weight
                weight = (pymask == 0).astype(int)
                layer = weight * label + (1 - weight) * layer
        return layer
