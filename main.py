import collections
import os
from shutil import rmtree

import numpy as np
import onnx
import pandas as pd
import requests
import torch
from matplotlib import pyplot as plt
from onnx_tf.backend import prepare
from PIL import Image
from scipy.signal import spectrogram
from tensorflowjs.converters import tf_saved_model_conversion_v2
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models

tmpdir = os.getenv('TMPDIR')
full = os.getenv('FULL')


def save_tfjs(model, combined_model_name):
    combined_model_name_dir = f'{tmpdir}/tfjs-models/{combined_model_name}'
    os.makedirs(combined_model_name_dir, exist_ok=True)
    example_input = torch.randn(1, 1, 176, requires_grad=False)
    torch.onnx.export(model.cpu(), example_input, f'{combined_model_name_dir}/model.onnx', export_params=True, opset_version=11)
    onnx_model = onnx.load(f'{combined_model_name_dir}/model.onnx')
    tf_model = prepare(onnx_model)
    tf_model.export_graph(f'{combined_model_name_dir}/model')
    tf_saved_model_conversion_v2.convert_tf_saved_model(f'{combined_model_name_dir}/model', combined_model_name_dir, skip_op_check=True)
    rmtree(f'{combined_model_name_dir}/model')
    os.remove(f'{combined_model_name_dir}/model.onnx')


def save_signal(signals_all, signal_index, label_name):
    plt.figure()
    plt.plot(signals_all[signal_index].squeeze(), linewidth=4, color='k')
    plt.axis('off')
    plt.xlim([0, signals_all.shape[-1] - 1])
    plt.ylim([-1000, 1000])
    plt.savefig(f'{tmpdir}/signal-{label_name}.png')
    plt.close()


def save_signal_as_image(signals_all, signal_index, label_name):
    signals_all_min = -1000
    signals_all_max = 1000
    x = signals_all[signal_index] - signals_all_min
    x = signals_all.shape[-1] * x / (signals_all_max - signals_all_min)
    x = x.squeeze(0).floor().long()
    data = torch.zeros(signals_all.shape[-1], signals_all.shape[-1])
    for index in range(signals_all.shape[-1]):
        data[signals_all.shape[-1] - 1 - x[index], index] = 255
    plt.figure()
    plt.imsave(f'{tmpdir}/signal-as-image-{label_name}.png', data, cmap='gray')
    plt.close()


def save_spectrogram(signals_all, signal_index, label_name):
    _, _, Sxx = spectrogram(signals_all[signal_index], fs=signals_all.shape[-1], noverlap=4, nperseg=8, nfft=64, mode='magnitude')
    data = np.array(Image.fromarray(Sxx[0]).resize((signals_all.shape[-1], signals_all.shape[-1]), resample=1))
    plt.figure()
    plt.imsave(f'{tmpdir}/spectrogram-{label_name}.png', data, cmap='gray')
    plt.close()


def save_cnn(signals_all, signal_index, label_name, num_classes):
    model = CNNOneLayer(num_classes, models.alexnet(), 'alexnet-cnn-one-layer')
    model.load_state_dict(torch.load(f'{tmpdir}/alexnet-cnn-one-layer.pt'))
    signal = signals_all[signal_index].unsqueeze(0)
    outputs = []

    def hook(_, __, output):
        outputs.append(output)

    model.conv.register_forward_hook(hook)
    model(signal)
    data = outputs[0][0, 0].cpu().detach().numpy()
    data = np.array(Image.fromarray(data).resize((signals_all.shape[-1], signals_all.shape[-1]), resample=1))
    plt.figure()
    plt.imsave(f'{tmpdir}/cnn-{label_name}.png', data, cmap='gray')
    plt.close()


def replace_last_layer(base_model, combined_model_name, num_classes):
    if combined_model_name.startswith(('alexnet', 'vgg')):
        base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, num_classes)
    elif combined_model_name.startswith('resnet'):
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    elif combined_model_name.startswith('densenet'):
        base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
    return base_model


class SignalAsImage(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name, signals_all_max, signals_all_min):
        super().__init__()
        self.signals_all_max = signals_all_max
        self.signals_all_min = signals_all_min
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        x = x - self.signals_all_min
        x = x.shape[-1] * x / (self.signals_all_max - self.signals_all_min)
        x = x.floor().long()
        out = torch.zeros(x.shape[0], 1, x.shape[-1], x.shape[-1]).to(x.device)
        for index, _ in enumerate(x):
            out[index, 0, x.shape[-1] - 1 - x[index, 0, :], range(x.shape[-1])] = 255
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class Spectrogram(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name):
        super().__init__()
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, x.shape[-1], x.shape[-1]).to(x.device)
        for index, signal in enumerate(x):
            _, _, Sxx = spectrogram(signal.cpu(), fs=x.shape[-1], noverlap=4, nperseg=8, nfft=64, mode='magnitude')
            out[index, 0] = F.interpolate(torch.tensor(Sxx).unsqueeze(0), x.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class CNNTwoLayers(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=2)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=2)
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, 2)
        out = self.conv2(out)
        out.unsqueeze_(1)
        out = F.interpolate(out, x.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class CNNOneLayer(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 3, padding=2)
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out.unsqueeze_(1)
        out = F.interpolate(out, x.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class lenet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class alexnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4)
        x = self.classifier(x)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(2560, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes):
    model = VGG(make_layers(cfg['A']), num_classes)
    return model


def vgg13(num_classes):
    model = VGG(make_layers(cfg['B']), num_classes)
    return model


def vgg16(num_classes):
    model = VGG(make_layers(cfg['D']), num_classes)
    return model


def vgg19(num_classes):
    model = VGG(make_layers(cfg['E']), num_classes)
    return model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(num_classes):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def resnet34(num_classes):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model


def resnet50(num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def resnet101(num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def resnet152(num_classes):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super().forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, num_classes, num_init_features, growth_rate, block_config):
        super().__init__()
        bn_size = 4
        self.features = nn.Sequential(
            nn.Conv1d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet121(num_samples):
    model = DenseNet(num_samples, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))
    return model


def densenet169(num_samples):
    model = DenseNet(num_samples, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32))
    return model


def densenet201(num_samples):
    model = DenseNet(num_samples, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32))
    return model


def densenet161(num_samples):
    model = DenseNet(num_samples, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24))
    return model


class UCIEpilepsy(Dataset):
    def __init__(self, training_validation_test, num_samples):
        filename = f'{tmpdir}/data.csv'
        if not os.path.isfile(filename):
            with open(filename, 'wb') as file:
                response = requests.get('https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv')
                file.write(response.content)
        dataset = pd.read_csv(filename)
        dataset = dataset[:num_samples]
        signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
        labels_all = dataset['y']
        last_training_index = int(signals_all.shape[0] * 0.76)
        last_validation_index = int(signals_all.shape[0] * 0.88)
        if training_validation_test == 'training':
            self.data = torch.tensor(signals_all.values[:last_training_index, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[:last_training_index].values) - 1
        elif training_validation_test == 'validation':
            self.data = torch.tensor(signals_all.values[last_training_index:last_validation_index, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[last_training_index:last_validation_index].values) - 1
        elif training_validation_test == 'test':
            self.data = torch.tensor(signals_all.values[last_validation_index:, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[last_validation_index:].values) - 1
        self.data.unsqueeze_(1)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])

    def __len__(self):
        return self.labels.shape[0]


class LeNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(41 * 41 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def main():
    torch.hub.set_dir(tmpdir)
    num_samples = 11500
    num_epochs = 100
    if not full:
        num_samples = 10
        num_epochs = 1
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 5
    batch_size = 20
    signals_all_max = 2047
    signals_all_min = -1885
    training_dataset = UCIEpilepsy('training', num_samples)
    validation_dataset = UCIEpilepsy('validation', num_samples)
    test_dataset = UCIEpilepsy('test', num_samples)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    base_models_names = ['lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
    base_models_1D = [lenet, alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, densenet201]
    base_models_2D = [LeNet2D, models.alexnet, models.vgg11, models.vgg13, models.vgg16, models.vgg19, models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152, models.densenet121, models.densenet161, models.densenet169, models.densenet201]
    base_models_1D_names = [f'{model_name}-1D' for model_name in base_models_names]
    combined_models_signal_as_image_names = [f'{model_name}-signal-as-image' for model_name in base_models_names]
    combined_models_spectrogram_names = [f'{model_name}-spectrogram' for model_name in base_models_names]
    combined_models_cnn_one_layer_names = [f'{model_name}-cnn-one-layer' for model_name in base_models_names]
    combined_models_cnn_two_layers_names = [f'{model_name}-cnn-two-layers' for model_name in base_models_names]
    base_models = base_models_1D + base_models_2D + base_models_2D + base_models_2D + base_models_2D
    combined_models_names = base_models_1D_names + combined_models_signal_as_image_names + combined_models_spectrogram_names + combined_models_cnn_one_layer_names + combined_models_cnn_two_layers_names
    results = collections.defaultdict(dict)
    for combined_model_name in combined_models_names:
        results[combined_model_name]['training_loss'] = []
        results[combined_model_name]['validation_loss'] = []
        results[combined_model_name]['validation_accuracy'] = []
    for base_model, combined_model_name in zip(base_models, combined_models_names):
        if combined_model_name.endswith('-1D'):
            model = base_model(num_classes)
        elif combined_model_name.endswith('-signal-as-image'):
            model = SignalAsImage(num_classes, base_model(), combined_model_name, signals_all_max, signals_all_min)
        elif combined_model_name.endswith('-spectrogram'):
            model = Spectrogram(num_classes, base_model(), combined_model_name)
        elif combined_model_name.endswith('-cnn-one-layer'):
            model = CNNOneLayer(num_classes, base_model(), combined_model_name)
        elif combined_model_name.endswith('-cnn-two-layers'):
            model = CNNTwoLayers(num_classes, base_model(), combined_model_name)
        model = model.to(device)
        optimizer = Adam(model.parameters())
        best_validation_accuracy = -1
        for epoch in range(num_epochs):
            model.train()
            training_loss_sum = 0
            for signals, labels in training_dataloader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss_sum += loss.item()
            training_loss = training_loss_sum / (batch_size * len(training_dataloader))
            validation_loss_sum = 0
            corrects = 0
            model.eval()
            with torch.no_grad():
                for signals, labels in validation_dataloader:
                    signals = signals.to(device)
                    labels = labels.to(device)
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    corrects += sum(outputs.argmax(dim=1) == labels).item()
                    validation_loss_sum += loss.item()
            validation_accuracy = 100 * corrects / (batch_size * len(validation_dataloader))
            validation_loss = validation_loss_sum / (batch_size * len(validation_dataloader))
            print(f'Model: {combined_model_name}, Epoch: {epoch}, Loss: {validation_loss:.3f}, Accuracy: {validation_accuracy:.2f}%')
            results[combined_model_name]['training_loss'].append(training_loss)
            results[combined_model_name]['validation_loss'].append(validation_loss)
            results[combined_model_name]['validation_accuracy'].append(validation_accuracy)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model.state_dict(), f'{tmpdir}/{combined_model_name}.pt')
                print('Saved as best model')
        model.load_state_dict(torch.load(f'{tmpdir}/{combined_model_name}.pt'))
        model.eval()
        test_loss_sum = 0
        corrects = 0
        test_confussion_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for signals, labels in test_dataloader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                corrects += sum(outputs.argmax(dim=1) == labels).item()
                for t, p in zip(labels.view(-1), torch.argmax(outputs, 1).view(-1)):
                    test_confussion_matrix[t.long(), p.long()] += 1
                test_loss_sum += loss.item()
        test_accuracy = 100 * corrects / (batch_size * len(test_dataloader))
        test_loss = test_loss_sum / (batch_size * len(test_dataloader))
        results[combined_model_name]['test_confussion_matrix'] = test_confussion_matrix
        results[combined_model_name]['test_accuracy'] = test_accuracy
        print(f'Model: {combined_model_name}, Epoch: {epoch}, Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.2f}%')
        if combined_model_name in ['lenet-1D', 'alexnet-1D', 'resnet18-1D', 'resnet34-1D', 'resnet50-1D', 'resnet18-signal-as-image', 'resnet34-signal-as-image', 'resnet50-signal-as-image']:
            save_tfjs(model, combined_model_name)
            if (not full):
                rmtree(f'{tmpdir}/tfjs-models/{combined_model_name}')
        if (not full) and (combined_model_name != 'alexnet-cnn-one-layer'):
            os.remove(f'{tmpdir}/{combined_model_name}.pt')
    results_test_accuracy_for_paper = np.zeros((75,))
    for index, model in enumerate(results):
        results_test_accuracy_for_paper[index] = np.around(results[model]['test_accuracy'], 1)
    results_test_accuracy_for_paper = results_test_accuracy_for_paper.reshape(5, 15)
    df = pd.DataFrame(results_test_accuracy_for_paper, index=['1D', '2D, signal as image', '2D, spectrogram', '2D, one layer CNN', '2D, two layer CNN'])
    df.columns = base_models_names
    df.to_latex(f'{tmpdir}/results.tex', bold_rows=True, escape=False)
    dataset = pd.read_csv(f'{tmpdir}/data.csv')
    signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
    labels_all = dataset['y']
    signals_all = torch.tensor(signals_all.values, dtype=torch.float)
    labels_all = torch.tensor(labels_all.values) - 1
    labels_names = ['eyes-open', 'eyes-closed', 'healthy-area', 'tumor-area', 'epilepsy']
    for index, label_name in enumerate(labels_names):
        signal_index = (labels_all == index).nonzero()[-1]
        save_signal(signals_all, signal_index, label_name)
        save_signal_as_image(signals_all, signal_index, label_name)
        save_spectrogram(signals_all, signal_index, label_name)
        save_cnn(signals_all, signal_index, label_name, num_classes)


if __name__ == '__main__':
    main()
