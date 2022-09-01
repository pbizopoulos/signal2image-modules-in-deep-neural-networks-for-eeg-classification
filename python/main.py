from PIL import Image
from matplotlib import pyplot as plt
from onnx_tf.backend import prepare
from os import environ
from os.path import join
from scipy.signal import spectrogram
from shutil import move, rmtree
from tensorflowjs.converters import tf_saved_model_conversion_v2
from torch import nn
from torch.nn import functional
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import numpy as np
import onnx
import os
import pandas as pd
import requests
import torch


class Alexnet(nn.Module):

    def __init__(self, classes_num):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(256 * 4, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, classes_num)

    def forward(self, signal):
        out = self.conv1(signal)
        out = self.relu(out)
        out = self.maxpool1d(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool1d(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.maxpool1d(out)
        out = out.view(out.size(0), 256 * 4)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, downsample, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, signal):
        identity = signal
        out = self.conv1(signal)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(signal)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, downsample, inplanes, planes, stride=1):
        super().__init__()
        expansion = 4
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm1d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, signal):
        identity = signal
        out = self.conv1(signal)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(signal)
        out += identity
        out = self.relu(out)
        return out


class CNNOneLayer(nn.Module):

    def __init__(self, classes_num, model_base, model_file_name):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 3, padding=2)
        self.model_base = replace_last_layer(classes_num, model_base, model_file_name)

    def forward(self, signal):
        out = self.conv(signal)
        out.unsqueeze_(1)
        out = functional.interpolate(out, signal.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.model_base(out)
        return out


class CNNTwoLayers(nn.Module):

    def __init__(self, classes_num, model_base, model_file_name):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=2)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=2)
        self.model_base = replace_last_layer(classes_num, model_base, model_file_name)

    def forward(self, signal):
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool1d(out, 2)
        out = self.conv2(out)
        out.unsqueeze_(1)
        out = functional.interpolate(out, signal.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.model_base(out)
        return out


class DenseBlock(nn.Sequential):

    def __init__(self, bn_size, growth_rate, input_features_num, layers_num):
        super().__init__()
        for index in range(layers_num):
            layer = DenseLayer(bn_size, growth_rate, input_features_num + index * growth_rate)
            self.add_module('denselayer%d' % (index + 1), layer)


class DenseLayer(nn.Sequential):

    def __init__(self, bn_size, growth_rate, input_features_num):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm1d(input_features_num))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv1d(input_features_num, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, signal):
        new_features = super().forward(signal)
        return torch.cat([signal, new_features], 1)


class DenseNet(nn.Module):

    def __init__(self, block_config, classes_num, growth_rate, init_features_num):
        super().__init__()
        bn_size = 4
        self.features = nn.Sequential(nn.Conv1d(1, init_features_num, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm1d(init_features_num), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        features_num = init_features_num
        for (index, layers_num) in enumerate(block_config):
            block = DenseBlock(bn_size=bn_size, growth_rate=growth_rate, input_features_num=features_num, layers_num=layers_num)
            self.features.add_module('denseblock%d' % (index + 1), block)
            features_num = features_num + layers_num * growth_rate
            if index != len(block_config) - 1:
                trans = Transition(input_features_num=features_num, output_features_num=features_num // 2)
                self.features.add_module('transition%d' % (index + 1), trans)
                features_num = features_num // 2
        self.features.add_module('norm5', nn.BatchNorm1d(features_num))
        self.classifier = nn.Linear(features_num, classes_num)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, signal):
        features = self.features(signal)
        out = functional.relu(features, inplace=True)
        out = functional.adaptive_avg_pool1d(out, 1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class Hook:

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def __init__(self):
        self.outputs = []


class LeNet2D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(41 * 41 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, signal):
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool2d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Lenet(nn.Module):

    def __init__(self, classes_num):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)

    def forward(self, signal):
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool1d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, classes_num, expansion, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], expansion, 64)
        self.layer2 = self._make_layer(block, layers[1], expansion, 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], expansion, 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], expansion, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * expansion, classes_num)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, blocks, expansion, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * expansion, stride), nn.BatchNorm1d(planes * expansion))
        layers = []
        layers.append(block(downsample, self.inplanes, planes, stride))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(None, self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, signal):
        out = self.conv1(signal)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SignalAsImage(nn.Module):

    def __init__(self, classes_num, model_base, model_file_name, signals_all_max, signals_all_min):
        super().__init__()
        self.signals_all_max = signals_all_max
        self.signals_all_min = signals_all_min
        self.model_base = replace_last_layer(classes_num, model_base, model_file_name)

    def forward(self, signal):
        signal = signal - self.signals_all_min
        signal = signal.shape[-1] * signal / (self.signals_all_max - self.signals_all_min)
        signal = signal.floor().long()
        out = torch.zeros(signal.shape[0], 1, signal.shape[-1], signal.shape[-1]).to(signal.device)
        for (index, element) in enumerate(signal):
            out[index, 0, signal.shape[-1] - 1 - signal[index, 0, :], range(signal.shape[-1])] = 255
        out = torch.cat((out, out, out), 1)
        out = self.model_base(out)
        return out


class Spectrogram(nn.Module):

    def __init__(self, classes_num, model_base, model_file_name):
        super().__init__()
        self.model_base = replace_last_layer(classes_num, model_base, model_file_name)

    def forward(self, signal):
        out = torch.zeros(signal.shape[0], 1, signal.shape[-1], signal.shape[-1]).to(signal.device)
        for (index, signal) in enumerate(signal):
            (f_array, t_array, Sxx) = spectrogram(signal.cpu(), fs=signal.shape[-1], noverlap=4, nperseg=8, nfft=64, mode='magnitude')
            out[index, 0] = functional.interpolate(torch.tensor(Sxx).unsqueeze(0), signal.shape[-1], mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.model_base(out)
        return out


class Transition(nn.Sequential):

    def __init__(self, input_features_num, output_features_num):
        super().__init__()
        self.add_module('norm', nn.BatchNorm1d(input_features_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(input_features_num, output_features_num, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class UCIEpilepsy(Dataset):

    def __getitem__(self, index):
        return (self.data[index], self.target[index])

    def __init__(self, artifacts_dir, samples_num, training_validation_test):
        data_file_path = join(artifacts_dir, 'data.csv')
        if not os.path.isfile(data_file_path):
            with open(data_file_path, 'wb') as file:
                response = requests.get('https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv')
                file.write(response.content)
        dataset = pd.read_csv(data_file_path)
        dataset = dataset[:samples_num]
        signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
        classes_all = dataset['y']
        last_training_index = int(signals_all.shape[0] * 0.76)
        last_validation_index = int(signals_all.shape[0] * 0.88)
        if training_validation_test == 'training':
            self.data = torch.tensor(signals_all.values[:last_training_index, :], dtype=torch.float)
            self.target = torch.tensor(classes_all[:last_training_index].values) - 1
        elif training_validation_test == 'validation':
            self.data = torch.tensor(signals_all.values[last_training_index:last_validation_index, :], dtype=torch.float)
            self.target = torch.tensor(classes_all[last_training_index:last_validation_index].values) - 1
        elif training_validation_test == 'test':
            self.data = torch.tensor(signals_all.values[last_validation_index:, :], dtype=torch.float)
            self.target = torch.tensor(classes_all[last_validation_index:].values) - 1
        self.data.unsqueeze_(1)

    def __len__(self):
        return self.target.shape[0]


class VGG(nn.Module):

    def __init__(self, classes_num, features):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(2560, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, classes_num))
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, signal):
        out = self.features(signal)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def densenet121(classes_num):
    model = DenseNet(block_config=(6, 12, 24, 16), classes_num=classes_num, growth_rate=32, init_features_num=64)
    return model


def densenet161(classes_num):
    model = DenseNet(block_config=(6, 12, 36, 24), classes_num=classes_num, growth_rate=48, init_features_num=96)
    return model


def densenet169(classes_num):
    model = DenseNet(block_config=(6, 12, 32, 32), classes_num=classes_num, growth_rate=32, init_features_num=64)
    return model


def densenet201(classes_num):
    model = DenseNet(block_config=(6, 12, 48, 32), classes_num=classes_num, growth_rate=32, init_features_num=64)
    return model


def main():
    artifacts_dir = environ['ARTIFACTSDIR']
    full = environ['FULL']
    samples_num = 11500
    epochs_num = 100
    if not full:
        samples_num = 10
        epochs_num = 1
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes_num = 5
    batch_size = 20
    signals_all_max = 2047
    signals_all_min = -1885
    dataset_training = UCIEpilepsy(artifacts_dir, samples_num, 'training')
    dataset_validation = UCIEpilepsy(artifacts_dir, samples_num, 'validation')
    dataset_test = UCIEpilepsy(artifacts_dir, samples_num, 'test')
    dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=dataset_validation, batch_size=batch_size)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size)
    cross_entropy_loss = nn.CrossEntropyLoss()
    model_base_name_list = ['lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
    model_base_1d_list = [Lenet, Alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, densenet201]
    model_base_2d_list = [LeNet2D, models.alexnet, models.vgg11, models.vgg13, models.vgg16, models.vgg19, models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152, models.densenet121, models.densenet161, models.densenet169, models.densenet201]
    accuracy_test_array = np.zeros((5, 15))
    for (model_base_name_index, model_base_name) in enumerate(model_base_name_list):
        for (model_module_name_index, model_module_name) in enumerate(['1D', 'signal-as-image', 'spectrogram', 'cnn-one-layer', 'cnn-two-layers']):
            model_file_name = f'{model_base_name}-{model_module_name}'
            if model_module_name == '1D':
                model = model_base_1d_list[model_base_name_index](classes_num)
            elif model_module_name == 'signal-as-image':
                model = SignalAsImage(classes_num, model_base_2d_list[model_base_name_index](), model_file_name, signals_all_max, signals_all_min)
            elif model_module_name == 'spectrogram':
                model = Spectrogram(classes_num, model_base_2d_list[model_base_name_index](), model_file_name)
            elif model_module_name == 'cnn-one-layer':
                model = CNNOneLayer(classes_num, model_base_2d_list[model_base_name_index](), model_file_name)
            elif model_module_name == 'cnn-two-layers':
                model = CNNTwoLayers(classes_num, model_base_2d_list[model_base_name_index](), model_file_name)
            model = model.to(device)
            optimizer = Adam(model.parameters())
            accuracy_validation_best = -1
            for epoch in range(epochs_num):
                model.train()
                for (data, target) in dataloader_training:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = cross_entropy_loss(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_validation_sum = 0
                predictions_correct_num = 0
                predictions_num = 0
                model.eval()
                with torch.no_grad():
                    for (data, target) in validation_dataloader:
                        data = data.to(device)
                        target = target.to(device)
                        output = model(data)
                        prediction = output.argmax(dim=1)
                        predictions_correct_num += sum(prediction == target).item()
                        predictions_num += output.shape[0]
                        loss = cross_entropy_loss(output, target)
                        loss_validation_sum += loss.item()
                accuracy_validation = 100 * predictions_correct_num / predictions_num
                loss_validation = loss_validation_sum / predictions_num
                if accuracy_validation > accuracy_validation_best:
                    accuracy_validation_best = accuracy_validation
                    torch.save(model.state_dict(), join(artifacts_dir, f'{model_file_name}.pt'))
            model.load_state_dict(torch.load(join(artifacts_dir, f'{model_file_name}.pt')))
            loss_test_sum = 0
            predictions_correct_num = 0
            predictions_num = 0
            model.eval()
            with torch.no_grad():
                for (data, target) in dataloader_test:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    prediction = output.argmax(dim=1)
                    predictions_correct_num += sum(prediction == target).item()
                    predictions_num += output.shape[0]
                    loss = cross_entropy_loss(output, target)
                    loss_test_sum += loss.item()
            accuracy_test = 100 * predictions_correct_num / predictions_num
            loss_test = loss_test_sum / predictions_num
            accuracy_test_array[model_module_name_index, model_base_name_index] = accuracy_test
            if model_file_name == 'resnet34-1D':
                save_tfjs_from_torch(artifacts_dir, dataset_training[0][0].unsqueeze(0), model, model_file_name)
                if full:
                    rmtree(join('release', model_file_name))
                    move(join(artifacts_dir, model_file_name), join('release', model_file_name))
            if not full and model_file_name != 'alexnet-cnn-one-layer':
                os.remove(join(artifacts_dir, f'{model_file_name}.pt'))
    styler = pd.DataFrame(accuracy_test_array, index=['1D', '2D, signal as image', '2D, spectrogram', '2D, one layer CNN', '2D, two layer CNN'], columns=model_base_name_list).style
    styler.format(precision=1)
    styler.highlight_max(props='bfseries: ;')
    styler.to_latex(join(artifacts_dir, 'results.tex'), hrules=True)
    dataset = pd.read_csv(join(artifacts_dir, 'data.csv'))
    signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
    classes_all = dataset['y']
    signals_all = torch.tensor(signals_all.values, dtype=torch.float)
    classes_all = torch.tensor(classes_all.values) - 1
    class_name_list = ['eyes-open', 'eyes-closed', 'healthy-area', 'tumor-area', 'epilepsy']
    for (class_index, class_name) in enumerate(class_name_list):
        signal_index = (classes_all == class_index).nonzero()[-1]
        plt.figure()
        plt.plot(signals_all[signal_index].squeeze(), linewidth=4, color='k')
        plt.axis('off')
        plt.xlim([0, signals_all.shape[-1] - 1])
        plt.ylim([-1000, 1000])
        plt.savefig(join(artifacts_dir, f'signal-{class_name}.png'))
        plt.close()
        signals_all_min = -1000
        signals_all_max = 1000
        signal = signals_all[signal_index] - signals_all_min
        signal = signals_all.shape[-1] * signal / (signals_all_max - signals_all_min)
        signal = signal.squeeze(0).floor().long()
        data = torch.zeros(signals_all.shape[-1], signals_all.shape[-1])
        for index in range(signals_all.shape[-1]):
            data[signals_all.shape[-1] - 1 - signal[index], index] = 255
        plt.figure()
        plt.imsave(join(artifacts_dir, f'signal-as-image-{class_name}.png'), data, cmap='gray')
        plt.close()
        (f_array, t_array, Sxx) = spectrogram(signals_all[signal_index], fs=signals_all.shape[-1], noverlap=4, nperseg=8, nfft=64, mode='magnitude')
        data = np.array(Image.fromarray(Sxx[0]).resize((signals_all.shape[-1], signals_all.shape[-1]), resample=1))
        plt.figure()
        plt.imsave(join(artifacts_dir, f'spectrogram-{class_name}.png'), data, cmap='gray')
        plt.close()
        model = CNNOneLayer(classes_num, models.alexnet(), 'alexnet-cnn-one-layer')
        model.load_state_dict(torch.load(join(artifacts_dir, 'alexnet-cnn-one-layer.pt')))
        signal = signals_all[signal_index].unsqueeze(0)
        hook = Hook()
        model.conv.register_forward_hook(hook)
        model(signal)
        data = hook.outputs[0][0, 0].cpu().detach().numpy()
        data = np.array(Image.fromarray(data).resize((signals_all.shape[-1], signals_all.shape[-1]), resample=1))
        plt.figure()
        plt.imsave(join(artifacts_dir, f'cnn-{class_name}.png'), data, cmap='gray')
        plt.close()


def make_layers(cfg):
    layers = []
    in_channels = 1
    for cfg_element in cfg:
        if cfg_element == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, cfg_element, kernel_size=3, padding=1)
            layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = cfg_element
    return nn.Sequential(*layers)


def replace_last_layer(classes_num, model_base, model_file_name):
    if model_file_name.startswith(('alexnet', 'vgg')):
        model_base.classifier[-1] = nn.Linear(model_base.classifier[-1].in_features, classes_num)
    elif model_file_name.startswith('resnet'):
        model_base.fc = nn.Linear(model_base.fc.in_features, classes_num)
    elif model_file_name.startswith('densenet'):
        model_base.classifier = nn.Linear(model_base.classifier.in_features, classes_num)
    return model_base


def resnet101(classes_num):
    model = ResNet(Bottleneck, classes_num, expansion=4, layers=[3, 4, 23, 3])
    return model


def resnet152(classes_num):
    model = ResNet(Bottleneck, classes_num, expansion=4, layers=[3, 8, 36, 3])
    return model


def resnet18(classes_num):
    model = ResNet(BasicBlock, classes_num, expansion=1, layers=[2, 2, 2, 2])
    return model


def resnet34(classes_num):
    model = ResNet(BasicBlock, classes_num, expansion=1, layers=[3, 4, 6, 3])
    return model


def resnet50(classes_num):
    model = ResNet(Bottleneck, classes_num, expansion=4, layers=[3, 4, 6, 3])
    return model


def save_tfjs_from_torch(artifacts_dir, example_input, model, model_file_name):
    model_file_path = join(artifacts_dir, model_file_name)
    os.makedirs(model_file_path, exist_ok=True)
    torch.onnx.export(model.cpu(), example_input, join(model_file_path, 'model.onnx'), export_params=True, opset_version=11)
    model_onnx = onnx.load(join(model_file_path, 'model.onnx'))
    model_tf = prepare(model_onnx)
    model_tf.export_graph(join(model_file_path, 'model'))
    tf_saved_model_conversion_v2.convert_tf_saved_model(join(model_file_path, 'model'), model_file_path, skip_op_check=True)
    rmtree(join(model_file_path, 'model'))
    os.remove(join(model_file_path, 'model.onnx'))


def vgg11(classes_num):
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGG(classes_num, make_layers(cfg))
    return model


def vgg13(classes_num):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    model = VGG(classes_num, make_layers(cfg))
    return model


def vgg16(classes_num):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(classes_num, make_layers(cfg))
    return model


def vgg19(classes_num):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    model = VGG(classes_num, make_layers(cfg))
    return model


if __name__ == '__main__':
    main()