from .imageloader import ImageLoader
from .imageloader import MultiDomainImageLoader


datasets_map = {
    'caltech101': ImageLoader,
    'dtd': ImageLoader,
    'oxford_pets': ImageLoader,
    'eurosat': ImageLoader,
    'fgvc_aircraft': ImageLoader,
    'food101': ImageLoader,
    'oxford_flowers': ImageLoader,
    'ucf': ImageLoader,
    'sun397': ImageLoader,
    'stanford_cars': ImageLoader,
    'imagenet': ImageLoader,
    'imagenet_a': ImageLoader,
    'imagenet_r': ImageLoader,
    'imagenet_s': ImageLoader,
    'imagenetv2': ImageLoader,
    'tiny_imagenet': ImageLoader,
    'domain_net': MultiDomainImageLoader,

}
