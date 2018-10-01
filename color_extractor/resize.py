import numpy as np
from skimage.transform import resize

from .task import Task


class Resize(Task):
    """
    Resizes and crops given images to the specified shape. As most fashion
    have the subject centered, cropping may help reducing the background
    and help discarding background from foreground.
    Note that the background detection algorithm relies heavily on the
    corners, if the cropping is too important, the object itself may be
    disregarded.
    """
    def __init__(self, settings=None):
        """
        The possible settings are:
            - crop: The crop ratio to use. `1' means no cropping. A floating
              point number between `0' and `1' is expected.
              (default: 0.90)

            - shape: The height of the resized image. The ratio between height
              and width is kept.
              (default: 100)
        """
        if settings is None:
            settings = {}

        super(Resize, self).__init__(settings)

    def get(self, img, crop_location=None, crop_ratio=None):
        """Returns `img` cropped and resized."""
        crop_location = crop_location if crop_location else self._default_settings()['crop_location']
        crop_ratio = crop_ratio if crop_ratio else self._default_settings()['crop']
        return self._resize(self._crop(img, crop_location, crop_ratio))

    def _resize(self, img):
        src_h, src_w = img.shape[:2]
        dst_h = self._settings['rows']
        dst_w = int((dst_h / src_h) * src_w)
        return resize(img, (dst_h, dst_w))

    @staticmethod
    def _crop(img, crop_loc, crop_ratio):
        src_h, src_w = img.shape[:2]
        dst_h, dst_w = int(src_h * crop_ratio), int(src_w * crop_ratio)
        if crop_loc == "top":
            top_h = int(src_h * 0.05)
            rm_w = (src_w - dst_w) // 2
            return img[top_h:top_h + dst_h, rm_w:rm_w + dst_w].copy()
        elif crop_loc == "center":
            rm_h, rm_w = (src_h - dst_h) // 2, (src_w - dst_w) // 2
            return img[rm_h:rm_h + dst_h, rm_w:rm_w + dst_w].copy()

    @staticmethod
    def _default_settings():
        return {
            'crop': 0.90,
            'crop_location': 'center',
            'rows': 100,
        }
