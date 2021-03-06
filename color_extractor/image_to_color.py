import numpy as np

from .back import Back
from .cluster import Cluster
from .name import Name
from .resize import Resize
from .selector import Selector
from .skin import Skin
from .task import Task


class ImageToColor(Task):
    def __init__(self, samples, labels, settings=None):

        if settings is None:
            settings = {}

        super(ImageToColor, self).__init__(settings)
        self.skin_back_thres = self._settings['skin_back_thres']
        self._resize = Resize(self._settings['resize'])
        self._back = Back(self._settings['back'])
        self._skin = Skin(self._settings['skin'])
        self._cluster = Cluster(self._settings['cluster'])
        self._selector = Selector(self._settings['selector'])
        self._name = Name(samples, labels, self._settings['name'])

    def reset_skin(self, strategy, low_thr=None, up_thr=None):
        self._skin.change_strategy(strategy)
        self._skin.change_thresholds(low_thr, up_thr)

    def get(self, img, image_settings={}):
        crop_location = image_settings['crop_loc'] if 'crop_loc' in image_settings.keys() else None
        crop_ratio = image_settings['crop_ratio'] if 'crop_ratio' in image_settings.keys() else None
        resized = self._resize.get(img, crop_location=crop_location, crop_ratio=crop_ratio)
        back_mask = self._back.get(resized)

        skin_strategy = "general" if image_settings['remove_skin'] else "none"
        self._skin.change_strategy(skin_strategy)
        skin_mask = self._skin.get(resized)
        mask = self.handle_back_skin_proportions(back_mask, skin_mask)

        filtered_img = resized[~mask]
        k, labels, clusters_centers = self._cluster.get(filtered_img)
        centers = self._selector.get(k, labels, clusters_centers)
        colors = [self._name.get(c) for c in centers]
        flattened = list({c for l in colors for c in l})

        # This is to count the size of each label
        distinct_labels, counts = np.unique(labels, return_counts=True)

        if self._settings['debug'] is None:
            return flattened

        colored_labels = np.zeros((labels.shape[0], 3), np.float64)
        for i, c in enumerate(clusters_centers):
            colored_labels[labels == i] = np.percentile(filtered_img[labels == i], 70, axis=0)

        clusters = np.zeros(resized.shape, np.float64)
        clusters[~mask] = colored_labels

        # Sending cluster centers instead of
        return clusters_centers, dict(zip(distinct_labels, counts)), {
                    'resized': resized,
                    'back': back_mask,
                    'skin': skin_mask,
                    'clusters': clusters
                }

    def handle_back_skin_proportions(self, back_mask, skin_mask):
        back_size = np.size(back_mask) - np.count_nonzero(back_mask)
        skin_size = np.count_nonzero(skin_mask)
        if skin_size / back_size > self.skin_back_thres:
            # Either a person is naked or clothing is skin-colored, no need to apply skin filter :)
            mask = back_mask
        else:
            mask = back_mask | skin_mask
        return mask

    @staticmethod
    def _default_settings():
        return {
            'resize': {},
            'back': {},
            'skin': {},
            'cluster': {},
            'selector': {},
            'name': {},
        }

    @staticmethod
    def _default_image_settings():
        return {
            'crop': True,
            'remove_skin': True
        }
