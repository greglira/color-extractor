import numpy as np
import skimage.morphology as skm
from skimage.filters import gaussian
from skimage.color import rgb2hsv
from skimage import img_as_ubyte
import skin_detector
import matplotlib.pyplot as plt

from .task import Task


class Skin(Task):
    def __init__(self, settings=None):
        """
        Skin is detected using color ranges.

        The possible settings are:
            - skin_type: The type of skin most expected in the given images.
              The value can be 'general' or 'none'. If 'none' is given the
              an empty mask is returned.
              (default: 'general')
        """
        if settings is None:
            settings = {}

        super(Skin, self).__init__(settings)
        self._k = skm.disk(1, np.bool)
        self._lo = self._settings['low_thr']
        self._up = self._settings['high_thr']

    def get(self, img):
        t = self._settings['skin_type']
        if t == 'general':
            img = rgb2hsv(img)
            return self._range_mask(img)
        elif t == 'none':
            return np.zeros(img.shape[:2], np.bool)
        else:
            raise NotImplementedError('Only general type is implemented')

    def _range_mask(self, img):
        mask = np.all((img >= self._lo) & (img <= self._up), axis=2)

        # Smooth the mask.
        skm.binary_opening(mask, selem=self._k, out=mask)
        return gaussian(mask, 0.8, multichannel=True) != 0

    @staticmethod
    def _default_settings():
        return {
            'skin_type': 'general',
            'low_thr':  np.array([0, 0.23, 0.31], np.float64),
            'high_thr':  np.array([0.1, 0.68, 1.], np.float64)
        }
