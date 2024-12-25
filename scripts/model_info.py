"""
This module contains information about available architectures and encoders for segmentation models.
"""

ARCHITECTURES = {
    'Unet': {
        'name': 'U-Net',
        'paper': 'https://arxiv.org/abs/1505.04597',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#unet',
    },
    'UnetPlusPlus': {
        'name': 'U-Net++',
        'paper': 'https://arxiv.org/abs/1807.10165',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#unetplusplus',
    },
    'MAnet': {
        'name': 'MA-Net',
        'paper': 'https://ieeexplore.ieee.org/abstract/document/9201310',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#manet',
    },
    'Linknet': {
        'name': 'LinkNet',
        'paper': 'https://arxiv.org/abs/1707.03718',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#linknet',
    },
    'FPN': {
        'name': 'FPN',
        'paper': 'https://arxiv.org/abs/1612.03144',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#fpn',
    },
    'PSPNet': {
        'name': 'PSPNet',
        'paper': 'https://arxiv.org/abs/1612.01105',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#pspnet',
    },
    'PAN': {
        'name': 'PAN',
        'paper': 'https://arxiv.org/abs/1805.10180',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#pan',
    },
    'DeepLabV3': {
        'name': 'DeepLabV3',
        'paper': 'https://arxiv.org/abs/1706.05587',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#deeplabv3',
    },
    'DeepLabV3Plus': {
        'name': 'DeepLabV3+',
        'paper': 'https://arxiv.org/abs/1802.02611',
        'docs': 'https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus',
    }
}

ENCODERS = {
    'ResNet': {
        'resnet18': {'weights': ['imagenet', 'ssl', 'swsl'], 'params': '11M'},
        'resnet34': {'weights': ['imagenet'], 'params': '21M'},
        'resnet50': {'weights': ['imagenet', 'ssl', 'swsl'], 'params': '23M'},
        'resnet101': {'weights': ['imagenet'], 'params': '42M'},
        'resnet152': {'weights': ['imagenet'], 'params': '58M'}
    },
    'ResNeXt': {
        'resnext50_32x4d': {'weights': ['imagenet', 'ssl', 'swsl'], 'params': '22M'},
        'resnext101_32x4d': {'weights': ['ssl', 'swsl'], 'params': '42M'},
        'resnext101_32x8d': {'weights': ['imagenet', 'instagram', 'ssl', 'swsl'], 'params': '86M'},
    },
    'EfficientNet': {
        'efficientnet-b0': {'weights': ['imagenet'], 'params': '4M'},
        'efficientnet-b1': {'weights': ['imagenet'], 'params': '6M'},
        'efficientnet-b2': {'weights': ['imagenet'], 'params': '7M'},
        'efficientnet-b3': {'weights': ['imagenet'], 'params': '10M'},
        'efficientnet-b4': {'weights': ['imagenet'], 'params': '17M'},
        'efficientnet-b5': {'weights': ['imagenet'], 'params': '28M'},
        'efficientnet-b6': {'weights': ['imagenet'], 'params': '40M'},
        'efficientnet-b7': {'weights': ['imagenet'], 'params': '63M'},
    },
    'SE-Net': {
        'se_resnet50': {'weights': ['imagenet'], 'params': '26M'},
        'se_resnet101': {'weights': ['imagenet'], 'params': '47M'},
        'se_resnet152': {'weights': ['imagenet'], 'params': '64M'},
        'se_resnext50_32x4d': {'weights': ['imagenet'], 'params': '25M'},
        'se_resnext101_32x4d': {'weights': ['imagenet'], 'params': '46M'},
    },
    'DenseNet': {
        'densenet121': {'weights': ['imagenet'], 'params': '6M'},
        'densenet169': {'weights': ['imagenet'], 'params': '12M'},
        'densenet201': {'weights': ['imagenet'], 'params': '18M'},
        'densenet161': {'weights': ['imagenet'], 'params': '26M'},
    },
    'MobileNet': {
        'mobilenet_v2': {'weights': ['imagenet'], 'params': '2M'},
        'timm-mobilenetv3_large_100': {'weights': ['imagenet'], 'params': '2.97M'},
        'timm-mobilenetv3_small_100': {'weights': ['imagenet'], 'params': '0.93M'},
    },
    'VGG': {
        'vgg11': {'weights': ['imagenet'], 'params': '9M'},
        'vgg13': {'weights': ['imagenet'], 'params': '9M'},
        'vgg16': {'weights': ['imagenet'], 'params': '14M'},
        'vgg19': {'weights': ['imagenet'], 'params': '20M'},
    }
}

def get_encoder_families():
    """Returns list of encoder families."""
    return list(ENCODERS.keys())

def get_encoders_for_family(family):
    """Returns dict of encoders for given family."""
    return ENCODERS.get(family, {})

def get_weights_for_encoder(family, encoder):
    """Returns available weights for given encoder."""
    encoders = get_encoders_for_family(family)
    return encoders.get(encoder, {}).get('weights', [])

def get_params_for_encoder(family, encoder):
    """Returns parameters count for given encoder."""
    encoders = get_encoders_for_family(family)
    return encoders.get(encoder, {}).get('params', 'N/A')

def get_architectures():
    """Returns dict of available architectures."""
    return ARCHITECTURES
