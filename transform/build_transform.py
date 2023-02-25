# Licensed under the MIT License

# Copyright (c) 2022 bbbdylan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import random
from PIL import ImageFilter, ImageOps
from torchvision import transforms

from .rand_augmentation import rand_augment_transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationMSiam(object):
    """
    Data Augmentation class for MSiam
    """

    def __init__(self, args):
        rgb_mean = (0.485, 0.456, 0.406)
        rgb_std = (0.229, 0.224, 0.225)
        ra_params = dict(
            translate_const=int(args.size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        flip_colorjitter_blur = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            rand_augment_transform('rand-n2-m10-mstd0.5',
                                   ra_params, use_cmc=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ])

        # transformation for the first augmented view
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.2, 1.)),
            flip_colorjitter_blur,
            normalize,
        ])
        # transformation for the second augmented view
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=(0.2, 1.)),
            flip_colorjitter_blur,
            # Solarization(0.2),          # could remove
            normalize,
        ])

        print('First View Transform: ', self.global_transfo1)
        print('Second View Transform: ', self.global_transfo2)

    def __call__(self, image):
        views = []
        views.append(self.global_transfo1(image))
        views.append(self.global_transfo2(image))

        return views
