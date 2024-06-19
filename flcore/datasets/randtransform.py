from torchvision import transforms
from .randaugment import RandAugmentMC


class RandTransform(object):
    def __init__(self, mean, std, crop_size, resize=None):
        trans = []
        trans.append(transforms.RandomHorizontalFlip())
        if resize is not None:
            trans.append(transforms.Resize(resize))
            trans.append(transforms.RandomCrop(size=crop_size, padding=0))
        else:
            trans.append(transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect'))

        self.weak = transforms.Compose(trans)
        self.strong = transforms.Compose([*trans, RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return [self.normalize(weak), self.normalize(strong)]