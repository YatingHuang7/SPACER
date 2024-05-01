from PIL import ImageFilter
import random
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hed
from torchvision import transforms
from freq_transform import freq_transform_func

cmap = plt.get_cmap('jet')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            ])

augmentation_p = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        import random
        x1 = freq_transform_func(x, random.randint(1,3))
        q = self.base_transform(x1)
        k = self.base_transform(x)
        print(q.shape,k.shape)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
