import torch
import torch.nn.functional as F

from scipy.signal import spectrogram
from torch import nn


def replace_last_layer(base_model, combined_model_name, num_classes):
    if combined_model_name.startswith(('alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19')):
        base_model.classifier[-1] = nn.Linear(base_model.classifier[-1].in_features, num_classes)
    elif combined_model_name.startswith(('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')):
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
    elif combined_model_name.startswith(('densenet121', 'densenet161', 'densenet169', 'densenet201')):
        base_model.classifier = nn.Linear(base_model.classifier.in_features, num_classes)
    return base_model

class SignalAsImage(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name, signals_all_max, signals_all_min, device):
        super(SignalAsImage, self).__init__()
        self.device = device
        self.signals_all_max = signals_all_max
        self.signals_all_min = signals_all_min
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        x = x - self.signals_all_min
        x = 178 * x / (self.signals_all_max - self.signals_all_min)
        x = x.floor().long()
        out = torch.zeros(x.shape[0], 1, 178, 178, device=self.device)
        for index, _ in enumerate(x):
            out[index, 0, 177 - x[index, 0, :], range(178)] = 255
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class Spectrogram(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name, device):
        super(Spectrogram, self).__init__()
        self.device = device
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, 178, 178, device=self.device)
        for index, signal in enumerate(x):
            _, _, Sxx = spectrogram(signal.cpu(), fs=178, noverlap=4, nperseg=8, nfft=64, mode='magnitude')
            out[index, 0, :, :] = F.interpolate(torch.tensor(Sxx[0, :, :]).unsqueeze(0).unsqueeze(0), 178, mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class CNN_two_layers(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name):
        super(CNN_two_layers, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=2)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=2)
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, 2)
        out = self.conv2(out)
        out.unsqueeze_(1)
        out = F.interpolate(out, 178, mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out


class CNN_one_layer(nn.Module):
    def __init__(self, num_classes, base_model, combined_model_name):
        super(CNN_one_layer, self).__init__()
        self.conv = nn.Conv1d(1, 8, 3, padding=2)
        self.base_model = replace_last_layer(base_model, combined_model_name, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out.unsqueeze_(1)
        out = F.interpolate(out, 178, mode='bilinear')
        out = torch.cat((out, out, out), 1)
        out = self.base_model(out)
        return out
