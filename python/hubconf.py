dependencies = ['torch']

import torch

from main import SignalAsImage as _SignalAsImage
from main import Spectrogram as _Spectrogram
from main import CNNOneLayer as _CNNOneLayer
from main import CNNTwoLayers as _CNNTwoLayers
from main import lenet as _lenet
from main import alexnet as _alexnet
from main import vgg11 as _vgg11
from main import vgg13 as _vgg13
from main import vgg16 as _vgg16
from main import vgg19 as _vgg19
from main import resnet18 as _resnet18
from main import resnet34 as _resnet34
from main import resnet50 as _resnet50
from main import resnet101 as _resnet101
from main import resnet152 as _resnet152
from main import densenet121 as _densenet121
from main import densenet161 as _densenet161
from main import densenet169 as _densenet169
from main import densenet201 as _densenet201


def classification_model(model_name='resnet18', module_name='1D'):
    """
    model_name ('lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201'): Model name
    module_name ('1D', 'signal-as-image', 'spectogram', 'cnn-one-layer', 'cnn-two-layers'): Module name
    """
    num_classes = 5
    if model_name == 'lenet':
        base_model = _lenet
    elif model_name == 'alexnet':
        base_model = _alexnet
    elif model_name == 'vgg11':
        base_model = _vgg11
    elif model_name == 'vgg13':
        base_model = _vgg13
    elif model_name == 'vgg16':
        base_model = _vgg16
    elif model_name == 'vgg19':
        base_model = _vgg19
    elif model_name == 'resnet18':
        base_model = _resnet18
    elif model_name == 'resnet34':
        base_model = _resnet34
    elif model_name == 'resnet50':
        base_model = _resnet50
    elif model_name == 'resnet101':
        base_model = _resnet101
    elif model_name == 'resnet152':
        base_model = _resnet152
    elif model_name == 'densenet121':
        base_model = _densenet121
    elif model_name == 'densenet161':
        base_model = _densenet161
    elif model_name == 'densenet169':
        base_model = _densenet169
    elif model_name == 'densenet201':
        base_model = _densenet201
    if module_name == '1D':
        model = base_model(num_classes)
    elif module_name == 'signal-as-image':
        model = _SignalAsImage(num_classes, base_model(), combined_model_name, signals_all_max, signals_all_min)
    elif module_name == 'spectrogram':
        model = _Spectrogram(num_classes, base_model(), combined_model_name)
    elif module_name == 'cnn-one-layer':
        model = _CNNOneLayer(num_classes, base_model(), combined_model_name)
    elif module_name == 'cnn-two-layers':
        model = _CNNTwoLayers(num_classes, base_model(), combined_model_name)
    checkpoint = f'https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/releases/download/v1/{model_name}-{module_name}.pt'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, map_location='cpu'))
    return model
