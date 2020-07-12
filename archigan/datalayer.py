import io
import os
import cv2
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import numpy as np
#import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas


class Layer:
    """Container-like class for representing either an
    input or output image directory for training a pix2pix model"""

    @staticmethod
    def save_image(path, image, cmap='gist_earth_r'):
        """Plot/save an image at a path via matplotlib"""
        n_cols = n_rows = 1
        n_pixels = 256
        dpi_of_monitor = 96 # HARDCODED DPI VALUE FROM MY OLD DELL LAPTOP...
        figsize = (n_pixels * n_cols / dpi_of_monitor,
                   n_pixels * n_rows / dpi_of_monitor)
        f, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
        f.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(image, cmap=cmap, vmin=0, vmax=None)
        f.savefig(path, dpi=dpi_of_monitor)
        plt.close(f)

    @classmethod
    def samples_to_imgs(cls, parsed_svgs, directory=None, show_instead=False):
        """Create folder of images for image-translation from a list of SVG paths"""
        layer = cls()
        directory = os.getcwd() if directory is None else directory
        directory = os.path.join(directory, layer.name)
        os.makedirs(directory, exist_ok=True)
        for svg, sample in tqdm(parsed_svgs.items(), desc=layer.name):
            filename = os.path.basename(svg)
            if '.' in filename:
                filename = filename[:filename.rfind('.')]
            path = os.path.join(directory, filename)
            byclass = sample['byclass']
            height, width = sample['height'], sample['width']
            image = layer(byclass, height, width)
            if show_instead:
                layer.show(layer=image)
            else:
                layer.save_image(path, image)

    def plot_loop(self, ax, loop, lw=2, ls='-', col='k', fill=None):
        """Plot the polygon `loop`"""
        if fill is not None:
            xs, ys = zip(*loop)
            ax.fill(xs, ys, fill)
        for i in range(len(loop)):
            (ux, uy), (vx, vy) = loop[i - 1], loop[i]
            ax.plot([ux, vx], [uy, vy], lw=lw, ls=ls, color=col)

    def show_style(self, ax, style, byclass):
        """Plot the data in `byclass` based on specification in `style`"""
        ax.axis('off')
        ax.set_aspect(1)
        for cls in byclass:
            col, ls = style[cls]
            for part in byclass[cls]:
                self.plot_loop(ax, part, lw=2, ls=ls, col=col, fill=col)

    def figure_buffer(self, fig, dpi=180):
        """Create np.array representation from matplotlib figure buffer"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def mask(self, byclass, height, width, margin=0, figsize=(10, 10), dpi=180):
        """Create a mask to find the boundary of the sample"""
        # make ~binary mask using available classes
        style = {cls: ('k', '-') for cls in byclass}
        fig = Figure(figsize=figsize)
        fig.tight_layout(pad=0)
        fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvas(fig)
        ax = fig.subplots(1, 1)
        self.show_style(ax, style, byclass)
        ax.set_xlim(0 - margin, height + margin)
        ax.set_ylim(0 - margin, width  + margin)
        canvas.draw()
        mask = self.figure_buffer(fig, dpi=dpi)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # fill in the gaps via:
        # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
        _, thresholded = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY_INV);
        floodfilled = thresholded.copy()
        h, w = thresholded.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(floodfilled, mask, (0, 0), 255);
        mask = cv2.bitwise_not(thresholded | cv2.bitwise_not(floodfilled))
        return mask

    def norm_mask(self, mask):
        """Renormalize mask by the current label set maximum"""
        lmax = max(self.labels.values())
        return (mask * (lmax / mask.max())).astype(int)

    def show(self, byclass=None, height=None, width=None, layer=None,
             ax=None, figsize=(10, 10), cmap='gist_earth_r'):
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)
        if layer is None:
            layer = self(byclass, height, width)
        ax.imshow(layer, cmap=cmap, vmin=0, vmax=5)
        return ax

    labels = {'Void': 0}

    def __init__(self, labels=None, name=None):
        self.labels = self.__class__.labels if labels is None else labels
        self.name = self.__class__.__name__ if name is None else name

    def __call__(self, byclass, height, width):
        """Create the layer as an image (subclasses must override)

        Args:
            byclass (dict): Polygon data organized by class label
            height (int): Upper limit for x-axis of image
            width (int): Upper limit for y-axis of image

        Returns:
            np.array image representation of the layer

        """
        raise NotImplementedError
