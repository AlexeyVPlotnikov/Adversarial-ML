import torch.nn as nn

import torchvision.models as models

class CustomCNNModel(nn.Module):
    pass


MODELS = {
    "resnet18": models.resnet18,
    "alexnet": models.alexnet,
    "vgg16": models.vgg16,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "mobilenet_v2": models.mobilenet_v2,
    "custom": CustomCNNModel
}


def get_model(model_name, pretrained=True):
    if model_name not in MODELS:
        raise NameError('Model is not presented. Valid model names are {}'.format(list(MODELS.keys())))
    return MODELS[model_name](pretrained=pretrained)
