import numpy as np
from PIL import Image, ImageOps

_MAX_LEVEL = 10.

policy_v0 = [
    dict(
        type='Translate', prob=0.6, level=4, axis='x',
        replace=(128, 128, 128)),
    dict(type='Equalize', prob=0.8, level=10),
    dict(
        type='TranslateOnlyBBox',
        prob=0.2,
        level=2,
        axis='y',
        replace=(128, 128, 128)),
    dict(type='Cutout', prob=0.8, level=8, replace=(128, 128, 128)),
    dict(type='Shear', prob=1.0, level=2, replace=(128, 128, 128), axis='y'),
    dict(
        type='TranslateOnlyBBox',
        prob=0.6,
        level=6,
        axis='y',
        replace=(128, 128, 128)),
    dict(type='Rotate', prob=0.6, level=10, replace=(128, 128, 128)),
    dict(type='Color', prob=1.0, level=6)
]


def auto_augment(img, gt_bboxes, policies):
    for policy in policies:
        policy = policy.copy()
        p = eval(policy.pop('type'))(**policy)
        img, gt_bboxes = p(img, gt_bboxes)
    return img, gt_bboxes


class Translate:

    def __init__(self, level, prob, replace, axis):
        self.level = level
        self.prob = prob
        self.replace = replace
        self.axis = axis

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        pixels = self.level_to_arg()
        img = self.translate_img(img, pixels, self.replace, self.axis)
        bboxes = self.translate_bbox(bboxes, pixels, self.axis, img.shape[0],
                                     img.shape[1])
        return img, bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 250.
        level = random_negative(level)
        return level

    @staticmethod
    def translate_img(img, pixels, replace, axis):
        assert axis in ('x', 'y')
        if axis == 'x':
            trans = (1, 0, pixels, 0, 1, 0)
        else:
            trans = (1, 0, 0, 0, 1, pixels)
        img = Image.fromarray(img)
        img = img.transform(img.size, Image.AFFINE, trans, fillcolor=replace)
        return np.array(img)

    @staticmethod
    def translate_bbox(bboxes, pixels, axis, img_height, img_width):
        assert axis in ('x', 'y')
        bboxes = bboxes.copy()
        if axis == 'x':
            bboxes[:, 0] = (bboxes[:, 0] - pixels).clip(
                min=0, max=img_width - 1)
            bboxes[:, 2] = (bboxes[:, 2] - pixels).clip(
                min=0, max=img_width - 1)
        else:
            bboxes[:, 1] = (bboxes[:, 1] - pixels).clip(
                min=0, max=img_height - 1)
            bboxes[:, 3] = (bboxes[:, 3] - pixels).clip(
                min=0, max=img_height - 1)
        return bboxes


class TranslateOnlyBBox(Translate):

    def __call__(self, img, bboxes):
        pixels = self.level_to_arg()
        for bbox in bboxes[np.random.permutation(bboxes.shape[0])]:
            if np.random.rand() < self.prob / 3.:
                bbox = bbox.astype(np.int)
                x1, y1, x2, y2 = bbox
                patch = img[y1:y2 + 1, x1:x2 + 1]
                patch = self.translate_img(patch, pixels, self.replace,
                                           self.axis)
                img[y1:y2 + 1, x1:x2 + 1] = patch
        return img, bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 120.
        level = random_negative(level)
        return level


class Cutout:

    def __init__(self, level, prob, replace):
        self.level = level
        self.prob = prob
        self.replace = replace

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        cutout_size = self.level_to_arg()
        img = img.copy()
        h, w, _ = img.shape
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        x1 = max(0, x - cutout_size)
        y1 = max(0, y - cutout_size)
        x2 = min(w, x + cutout_size)
        y2 = min(h, y + cutout_size)
        img[y1:y2, x1:x2, :] = np.array(
            self.replace)[np.newaxis, np.newaxis, :]
        return img, bboxes

    def level_to_arg(self):
        return int((self.level / _MAX_LEVEL) * 100)


class Shear:

    def __init__(self, level, prob, replace, axis):
        self.level = level
        self.prob = prob
        self.replace = replace
        self.axis = axis

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        level = self.level_to_arg()
        if self.axis == 'x':
            trans = (1, level, 0, 0, 1, 0)
        else:
            trans = (1, 0, 0, level, 1, 0)
        image = Image.fromarray(img)
        image = image.transform(
            image.size,
            Image.AFFINE,
            trans,
            resample=Image.NEAREST,
            fillcolor=self.replace)

        bboxes = self.shear_bbox(bboxes, level, self.axis, img.shape[0],
                                 img.shape[1])
        return np.array(image), bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 0.3
        level = random_negative(level)
        return level

    def shear_bbox(self, bboxes, level, axis, image_height, image_width):
        new_bboxes = np.zeros_like(bboxes)
        for i in range(bboxes.shape[0]):
            new_bboxes[i] = self._shear_bbox(bboxes[i], level, axis,
                                             image_height, image_width)
        return new_bboxes

    @staticmethod
    def _shear_bbox(bbox, level, axis, image_height, image_width):
        x1, y1, x2, y2 = bbox
        coordinates = np.stack([[y1, x1], [y1, x2], [y2, x1], [y2, x2]])

        if axis == 'x':
            translation_matrix = np.stack([[1, 0], [-level, 1]])
        else:
            translation_matrix = np.stack([[1, -level], [0, 1]])
        translation_matrix = translation_matrix
        new_coords = np.matmul(translation_matrix, np.transpose(coordinates))

        y1 = np.clip(new_coords[0, :].min(), 0, image_height - 1)
        x1 = np.clip(new_coords[1, :].min(), 0, image_width - 1)
        y2 = np.clip(new_coords[0, :].max(), 0, image_height - 1)
        x2 = np.clip(new_coords[1, :].max(), 0, image_width - 1)

        x1, y1, x2, y2 = _check_bbox_area(x1, y1, x2, y2)
        return np.array([x1, y1, x2, y2])


class Rotate:

    def __init__(self, level, prob, replace):
        self.level = level
        self.prob = prob
        self.replace = replace

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        degree = self.level_to_arg()
        bboxes = bboxes.copy()
        h, w, _ = img.shape
        img = Image.fromarray(img)
        img = img.rotate(degree, fillcolor=self.replace)

        new_bboxes = np.zeros_like(bboxes)
        for i in range(len(new_bboxes)):
            new_bboxes[i] = self._rotate_bbox(bboxes[i], degree, h, w)

        return np.array(img), new_bboxes

    def level_to_arg(self):
        level = (self.level / _MAX_LEVEL) * 30.
        level = random_negative(level)
        return level

    @staticmethod
    def _rotate_bbox(bbox, degree, img_h, img_w):
        radians = degree / 180 * np.pi
        x1 = int(bbox[0] - 0.5 * img_w)
        y1 = int(-(bbox[1] - 0.5 * img_h))
        x2 = int(bbox[2] - 0.5 * img_w)
        y2 = int(-(bbox[3] - 0.5 * img_h))
        coordinates = np.stack([[y1, x1], [y1, x2], [y2, x1], [y2, x2]])
        rotation_matrix = np.stack([[np.cos(radians),
                                     np.sin(radians)],
                                    [-np.sin(radians),
                                     np.cos(radians)]])
        new_coords = np.matmul(rotation_matrix, np.transpose(coordinates))
        y1 = -np.max(new_coords[0, :]) + img_h * 0.5
        x1 = np.min(new_coords[1, :]) + img_w * 0.5
        y2 = -np.min(new_coords[0, :]) + img_h * 0.5
        x2 = np.max(new_coords[1, :]) + img_w * 0.5

        y1 = y1.clip(min=0, max=img_h - 1)
        y2 = y2.clip(min=0, max=img_h - 1)
        x1 = x1.clip(min=0, max=img_w - 1)
        x2 = x2.clip(min=0, max=img_w - 1)

        x1, y1, x2, y2 = _check_bbox_area(x1, y1, x2, y2)

        return np.array([x1, y1, x2, y2])


class Color:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        factor = self.level_to_arg()
        img1 = Image.fromarray(img).convert('L')
        img1 = np.array(img1)
        img1 = np.tile(img1[..., np.newaxis], (1, 1, 3))
        return self.blind(img1, img, factor), bboxes

    def level_to_arg(self):
        return (self.level / _MAX_LEVEL) * 1.8 + 0.1

    @staticmethod
    def blind(img1, img2, factor):
        if factor == 0.0:
            return img1
        if factor == 1.0:
            return img2

        img1 = img1.astype(np.float)
        img2 = img2.astype(np.float)

        difference = img2 - img1
        scaled = factor * difference

        tmp = img1 + scaled

        if 0.0 < factor < 1.0:
            return tmp.astype(np.uint8)
        return tmp.clip(min=0.0, max=255.0).astype(np.uint8)


class Equalize:

    def __init__(self, level, prob):
        self.level = level
        self.prob = prob

    def __call__(self, img, bboxes):
        if np.random.rand() > self.prob:
            return img, bboxes
        img = Image.fromarray(img)
        img = ImageOps.equalize(img)
        return np.array(img), bboxes


def random_negative(level):
    if np.random.rand() < 0.5:
        return -level
    return level


def _check_bbox_area(x1, y1, x2, y2, delta=0.05):
    height = y2 - y1
    width = x2 - x1

    def _adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = max(max_coord, 0.0 + delta)
        min_coord = min(min_coord, 1.0 - delta)
        return min_coord, max_coord

    if height == 0:
        y1, y2 = _adjust_bbox_boundaries(y1, y2)

    if width == 0:
        x1, x2 = _adjust_bbox_boundaries(x1, x2)
    return x1, y1, x2, y2
