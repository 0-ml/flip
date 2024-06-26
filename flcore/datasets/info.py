# TODO: get the moments of each dataset
INFO = {
    'caltech101': {
        'num_classes': 102,
        'shape': (3, 224, 224),
        'moments': [(0.5487, 0.5313, 0.5051), (0.2426, 0.2395, 0.2411)],
        'task': 'image',
        'img_folder': '101_ObjectCategories',
    },
    'oxford_pets': {
        'num_classes': 37,
        'shape': (3, 224, 224),
        'moments': [(0.4811, 0.4492, 0.3957), (0.2260, 0.2231, 0.2249)],
        'task': 'image',
        'img_folder': 'images',
    },
    'dtd': {
        'num_classes': 47,
        'shape': (3, 224, 224),
        'moments': [(0.5276, 0.4714, 0.4234), (0.1655, 0.1665, 0.1630)],
        'task': 'image',
        'img_folder': 'images',
    },
    'eurosat': {
        'num_classes': 10,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '2750',
    },
    'fgvc_aircraft': {
        'num_classes': 100,
        'shape': (3, 224, 224),
        'moments': [(0.4785, 0.5100, 0.5338), (0.1845, 0.1830, 0.2060)],
        'task': 'image',
        'img_folder': 'images',
    },
    'food101': {
        'num_classes': 101,
        'shape': (3, 224, 224),
        'moments': [(0.5458, 0.4444, 0.3443), (0.2295, 0.2406, 0.2391)],
        'task': 'image',
        'img_folder': 'images',
    },
    'oxford_flowers': {
        'num_classes': 102,
        'shape': (3, 224, 224),
        'moments': [(0.4355, 0.3777, 0.2880), (0.2621, 0.2086, 0.2158)],
        'task': 'image',
        'img_folder': 'jpg',
    },
    'ucf': {
        'num_classes': 101,
        'shape': (3, 224, 224),
        'moments': [(0.3952, 0.3792, 0.3493), (0.2409, 0.2340, 0.2295)],
        'task': 'image',
        'img_folder': 'UCF-101-midframes',
    },
    'sun397': {
        'num_classes': 397,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': 'SUN397',
    },
    'stanford_cars': {
        'num_classes': 196,
        'shape': (3, 224, 224),
        'moments': [(0.4708, 0.4602, 0.4550), (0.2594, 0.2585, 0.2635)],
        'task': 'image',
        'img_folder': 'cars_train',
    },
    'imagenet': {
        'num_classes': 1000,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
    'imagenet_a': {
        'num_classes': 200,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
    'imagenet_r': {
        'num_classes': 200,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
    'imagenet_s': {
        'num_classes': 1000,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
    'imagenetv2': {
        'num_classes': 1000,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
    'tiny_imagenet': {
        'num_classes': 200,
        'shape': (3, 224, 224),
        'moments': [([0.4805, 0.4483, 0.3978]), (0.2177, 0.2138, 0.2136)],
        'task': 'image',
        'img_folder': '',
    },
    'domain_net': {
        'num_classes': 345,
        'shape': (3, 224, 224),
        'moments': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'task': 'image',
        'img_folder': '',
    },
}