"""Signal2Image Modules in Deep Neural Networks for EEG Classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import PIL
import requests
import torch
from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from torch import nn
from torch.nn import functional
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class _Alexnet(nn.Module):
    def __init__(self: _Alexnet, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(256 * 4, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)

    def forward(self: _Alexnet, signal: torch.Tensor) -> torch.Tensor:
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
        output: torch.Tensor = self.linear3(out)
        return output


class _BasicBlock(nn.Module):
    def __init__(
        self: _BasicBlock,
        downsample: nn.Module | None,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self: _BasicBlock, signal: torch.Tensor) -> torch.Tensor:
        identity = signal
        out = self.conv1(signal)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(signal)
        out += identity
        output: torch.Tensor = self.relu(out)
        return output


class _Bottleneck(nn.Module):
    def __init__(
        self: _Bottleneck,
        downsample: nn.Module | None,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        expansion = 4
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = _conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm1d(planes * expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self: _Bottleneck, signal: torch.Tensor) -> torch.Tensor:
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
        output: torch.Tensor = self.relu(out)
        return output


class _CNNOneLayer(nn.Module):
    def __init__(
        self: _CNNOneLayer,
        num_classes: int,
        model_base: nn.Module,
        model_file_name: str,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 8, 3, padding=2)
        self.model_base = _replace_last_layer(num_classes, model_base, model_file_name)

    def forward(self: _CNNOneLayer, signal: torch.Tensor) -> torch.Tensor:
        out = self.conv(signal)
        out.unsqueeze_(1)
        out = functional.interpolate(out, signal.shape[-1], mode="bilinear")
        out = torch.cat((out, out, out), 1)
        output: torch.Tensor = self.model_base(out)
        return output


class _CNNTwoLayers(nn.Module):
    def __init__(
        self: _CNNTwoLayers,
        num_classes: int,
        model_base: nn.Module,
        model_file_name: str,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 3, padding=2)
        self.conv2 = nn.Conv1d(8, 16, 3, padding=2)
        self.model_base = _replace_last_layer(num_classes, model_base, model_file_name)

    def forward(self: _CNNTwoLayers, signal: torch.Tensor) -> torch.Tensor:
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool1d(out, 2)
        out = self.conv2(out)
        out.unsqueeze_(1)
        out = functional.interpolate(out, signal.shape[-1], mode="bilinear")
        out = torch.cat((out, out, out), 1)
        output: torch.Tensor = self.model_base(out)
        return output


class _DenseBlock(nn.Sequential):
    def __init__(
        self: _DenseBlock,
        bn_size: int,
        growth_rate: int,
        num_input_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        for index in range(num_layers):
            layer = _DenseLayer(
                bn_size,
                growth_rate,
                num_input_features + index * growth_rate,
            )
            self.add_module("denselayer%d" % (index + 1), layer)


class _DenseLayer(nn.Sequential):
    def __init__(
        self: _DenseLayer,
        bn_size: int,
        growth_rate: int,
        num_input_features: int,
    ) -> None:
        super().__init__()
        self.add_module("norm1", nn.BatchNorm1d(num_input_features))
        self.add_module("relu1", nn.ReLU())
        self.add_module(
            "conv1",
            nn.Conv1d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU())
        self.add_module(
            "conv2",
            nn.Conv1d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self: _DenseLayer, signal: torch.Tensor) -> torch.Tensor:
        new_features = super().forward(signal)  # type: ignore[no-untyped-call]
        return torch.cat([signal, new_features], 1)


class _DenseNet(nn.Module):
    def __init__(
        self: _DenseNet,
        block_config: tuple[int, ...],
        num_classes: int,
        growth_rate: int,
        num_init_features: int,
    ) -> None:
        super().__init__()
        bn_size = 4
        self.features = nn.Sequential(
            nn.Conv1d(
                1,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        for index, num_layers in enumerate(block_config):
            block = _DenseBlock(
                bn_size=bn_size,
                growth_rate=growth_rate,
                num_input_features=num_features,
                num_layers=num_layers,
            )
            self.features.add_module("denseblock%d" % (index + 1), block)
            num_features = num_features + num_layers * growth_rate
            if index != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (index + 1), trans)
                num_features = num_features // 2
        self.features.add_module("norm5", nn.BatchNorm1d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self: _DenseNet, signal: torch.Tensor) -> torch.Tensor:
        features = self.features(signal)
        out = functional.relu(features)
        out = functional.adaptive_avg_pool1d(out, 1).view(features.size(0), -1)
        output: torch.Tensor = self.classifier(out)
        return output


class _Hook:
    def __call__(
        self: _Hook,
        _: nn.Module,
        __: nn.Module,
        module_out: nn.Module,
    ) -> None:
        self.outputs.append(module_out)

    def __init__(self: _Hook) -> None:
        self.outputs: list[nn.Module] = []


class _LeNet2D(nn.Module):
    def __init__(self: _LeNet2D) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.conv2 = nn.Conv2d(3, 16, 5)
        self.fc1 = nn.Linear(41 * 41 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self: _LeNet2D, signal: torch.Tensor) -> torch.Tensor:
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool2d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        output: torch.Tensor = self.fc3(out)
        return output


class _Lenet(nn.Module):
    def __init__(self: _Lenet, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self: _Lenet, signal: torch.Tensor) -> torch.Tensor:
        out = functional.relu(self.conv1(signal))
        out = functional.max_pool1d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        output: torch.Tensor = self.fc3(out)
        return output


class _ResNet(nn.Module):
    def __init__(
        self: _ResNet,
        block: type[M],
        num_classes: int,
        expansion: int,
        layers: list[int],
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], expansion, 64)
        self.layer2 = self._make_layer(block, layers[1], expansion, 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], expansion, 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], expansion, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * expansion, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(  # noqa: PLR0913
        self: _ResNet,
        block: type[M],
        blocks: int,
        expansion: int,
        planes: int,
        stride: int = 1,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * expansion, stride),
                nn.BatchNorm1d(planes * expansion),
            )
        layers = [block(downsample, self.inplanes, planes, stride)]
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(None, self.inplanes, planes))  # noqa: PERF401
        return nn.Sequential(*layers)

    def forward(self: _ResNet, signal: torch.Tensor) -> torch.Tensor:
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
        output: torch.Tensor = self.fc(out)
        return output


class _SignalAsImage(nn.Module):
    def __init__(  # noqa: PLR0913
        self: _SignalAsImage,
        num_classes: int,
        model_base: nn.Module,
        model_file_name: str,
        signals_all_max: int,
        signals_all_min: int,
    ) -> None:
        super().__init__()
        self.signals_all_max = signals_all_max
        self.signals_all_min = signals_all_min
        self.model_base = _replace_last_layer(num_classes, model_base, model_file_name)

    def forward(self: _SignalAsImage, signal: torch.Tensor) -> torch.Tensor:
        signal = signal - self.signals_all_min
        signal = (
            signal.shape[-1] * signal / (self.signals_all_max - self.signals_all_min)
        )
        signal = signal.floor().long()
        out = torch.zeros(signal.shape[0], 1, signal.shape[-1], signal.shape[-1]).to(
            signal.device,
        )
        for index, _element in enumerate(signal):
            out[
                index,
                0,
                signal.shape[-1] - 1 - signal[index, 0, :],
                range(signal.shape[-1]),
            ] = 255
        out = torch.cat((out, out, out), 1)
        return self.model_base(out)  # type: ignore[no-any-return]


class _Spectrogram(nn.Module):
    def __init__(
        self: _Spectrogram,
        num_classes: int,
        model_base: nn.Module,
        model_file_name: str,
    ) -> None:
        super().__init__()
        self.model_base = _replace_last_layer(num_classes, model_base, model_file_name)

    def forward(self: _Spectrogram, signal: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(signal.shape[0], 1, signal.shape[-1], signal.shape[-1]).to(
            signal.device,
        )
        for index, signal_element in enumerate(signal):
            f_array, t_array, spectrogram_array = spectrogram(
                signal_element.cpu(),
                fs=signal_element.shape[-1],
                noverlap=4,
                nperseg=8,
                nfft=64,
                mode="magnitude",
            )
            out[index, 0] = functional.interpolate(
                torch.tensor(spectrogram_array).unsqueeze(0),
                signal_element.shape[-1],
                mode="bilinear",
            )
        out = torch.cat((out, out, out), 1)
        return self.model_base(out)  # type: ignore[no-any-return]


class _Transition(nn.Sequential):
    def __init__(
        self: _Transition,
        num_input_features: int,
        num_output_features: int,
    ) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(num_input_features))
        self.add_module("relu", nn.ReLU())
        self.add_module(
            "conv",
            nn.Conv1d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class _UCIEpilepsy(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __getitem__(
        self: _UCIEpilepsy,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.data[index], self.target[index])

    def __init__(
        self: _UCIEpilepsy,
        num_samples: int,
        train_validation_test: str,
    ) -> None:
        data_file_path = Path("tmp/data.csv")
        if not data_file_path.is_file():
            with data_file_path.open("wb") as file:
                response = requests.get(
                    "https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv",
                    timeout=60,
                )
                file.write(response.content)
        dataset = pd.read_csv(data_file_path.as_posix())
        dataset = dataset[:num_samples]
        signals_all = dataset.drop(columns=["Unnamed: 0", "y"])
        classes_all = dataset["y"]
        last_train_index = int(signals_all.shape[0] * 0.76)
        last_validation_index = int(signals_all.shape[0] * 0.88)
        if train_validation_test == "train":
            self.data = torch.tensor(
                signals_all.to_numpy()[:last_train_index, :],
                dtype=torch.float,
            )
            self.target = torch.tensor(classes_all[:last_train_index].to_numpy()) - 1
        elif train_validation_test == "validation":
            self.data = torch.tensor(
                signals_all.to_numpy()[last_train_index:last_validation_index, :],
                dtype=torch.float,
            )
            self.target = (
                torch.tensor(
                    classes_all[last_train_index:last_validation_index].to_numpy(),
                )
                - 1
            )
        elif train_validation_test == "test":
            self.data = torch.tensor(
                signals_all.to_numpy()[last_validation_index:, :],
                dtype=torch.float,
            )
            self.target = (
                torch.tensor(classes_all[last_validation_index:].to_numpy()) - 1
            )
        self.data.unsqueeze_(1)

    def __len__(self: _UCIEpilepsy) -> int:
        return self.target.shape[0]


class _VGG(nn.Module):
    def __init__(self: _VGG, num_classes: int, features: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(2560, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self: _VGG) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self: _VGG, signal: torch.Tensor) -> torch.Tensor:
        out = self.features(signal)
        out = out.view(out.size(0), -1)
        output: torch.Tensor = self.classifier(out)
        return output


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _densenet121(num_classes: int) -> nn.Module:
    return _DenseNet(
        block_config=(6, 12, 24, 16),
        num_classes=num_classes,
        growth_rate=32,
        num_init_features=64,
    )


def _densenet161(num_classes: int) -> nn.Module:
    return _DenseNet(
        block_config=(6, 12, 36, 24),
        num_classes=num_classes,
        growth_rate=48,
        num_init_features=96,
    )


def _densenet169(num_classes: int) -> nn.Module:
    return _DenseNet(
        block_config=(6, 12, 32, 32),
        num_classes=num_classes,
        growth_rate=32,
        num_init_features=64,
    )


def _densenet201(num_classes: int) -> nn.Module:
    return _DenseNet(
        block_config=(6, 12, 48, 32),
        num_classes=num_classes,
        growth_rate=32,
        num_init_features=64,
    )


def _main() -> None:  # noqa: C901,PLR0912,PLR0915
    if os.getenv("STAGE"):
        num_samples = 11500
        num_epochs = 100
    else:
        num_samples = 10
        num_epochs = 1
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    batch_size = 20
    signals_all_max = 2047
    signals_all_min = -1885
    uci_epilepsy_train = _UCIEpilepsy(num_samples, "train")
    uci_epilepsy_validation = _UCIEpilepsy(num_samples, "validation")
    uci_epilepsy_test = _UCIEpilepsy(num_samples, "test")
    dataloader_train = DataLoader(
        dataset=uci_epilepsy_train,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        dataset=uci_epilepsy_validation,
        batch_size=batch_size,
    )
    dataloader_test = DataLoader(dataset=uci_epilepsy_test, batch_size=batch_size)
    cross_entropy_loss = nn.CrossEntropyLoss()
    model_base_names = [
        "lenet",
        "alexnet",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
    ]
    models_base_1d = [
        _Lenet,
        _Alexnet,
        _vgg11,
        _vgg13,
        _vgg16,
        _vgg19,
        _resnet18,
        _resnet34,
        _resnet50,
        _resnet101,
        _resnet152,
        _densenet121,
        _densenet161,
        _densenet169,
        _densenet201,
    ]
    models_base_2d = [
        _LeNet2D,
        models.alexnet,
        models.vgg11,
        models.vgg13,
        models.vgg16,
        models.vgg19,
        models.resnet18,
        models.resnet34,
        models.resnet50,
        models.resnet101,
        models.resnet152,
        models.densenet121,
        models.densenet161,
        models.densenet169,
        models.densenet201,
    ]
    accuracy_test_array = np.zeros((5, 15))
    for model_base_name_index, model_base_name in enumerate(model_base_names):
        for model_module_name_index, model_module_name in enumerate(
            ["1D", "signal-as-image", "spectrogram", "cnn-one-layer", "cnn-two-layers"],
        ):
            model_file_name = f"{model_base_name}-{model_module_name}"
            if model_module_name == "1D":
                model = models_base_1d[model_base_name_index](num_classes)
            elif model_module_name == "signal-as-image":
                model = _SignalAsImage(
                    num_classes,
                    models_base_2d[model_base_name_index](),
                    model_file_name,
                    signals_all_max,
                    signals_all_min,
                )
            elif model_module_name == "spectrogram":
                model = _Spectrogram(
                    num_classes,
                    models_base_2d[model_base_name_index](),
                    model_file_name,
                )
            elif model_module_name == "cnn-one-layer":
                model = _CNNOneLayer(
                    num_classes,
                    models_base_2d[model_base_name_index](),
                    model_file_name,
                )
            elif model_module_name == "cnn-two-layers":
                model = _CNNTwoLayers(
                    num_classes,
                    models_base_2d[model_base_name_index](),
                    model_file_name,
                )
            model = model.to(device)
            optimizer = Adam(model.parameters())
            accuracy_validation_best = -1.0
            for _ in range(num_epochs):
                model.train()
                for data, target in dataloader_train:
                    data = data.to(device)  # noqa: PLW2901
                    target = target.to(device)  # noqa: PLW2901
                    output = model(data)
                    loss = cross_entropy_loss(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_validation_sum = 0
                num_predictions_correct = 0
                num_predictions = 0
                model.eval()
                with torch.no_grad():
                    for data, target in validation_dataloader:
                        data = data.to(device)  # noqa: PLW2901
                        target = target.to(device)  # noqa: PLW2901
                        output = model(data)
                        prediction = output.argmax(dim=1)
                        num_predictions_correct += sum(prediction == target).item()
                        num_predictions += output.shape[0]
                        loss = cross_entropy_loss(output, target)
                        loss_validation_sum += loss.item()
                accuracy_validation = 100 * num_predictions_correct / num_predictions
                if accuracy_validation > accuracy_validation_best:
                    accuracy_validation_best = accuracy_validation
                    model_file_path = Path(f"tmp/{model_file_name}.pt")
                    torch.save(model.state_dict(), model_file_path)
            model.load_state_dict(torch.load(model_file_path))
            loss_test_sum = 0
            num_predictions_correct = 0
            num_predictions = 0
            model.eval()
            with torch.no_grad():
                for data, target in dataloader_test:
                    data = data.to(device)  # noqa: PLW2901
                    target = target.to(device)  # noqa: PLW2901
                    output = model(data)
                    prediction = output.argmax(dim=1)
                    num_predictions_correct += sum(prediction == target).item()
                    num_predictions += output.shape[0]
                    loss = cross_entropy_loss(output, target)
                    loss_test_sum += loss.item()
            accuracy_test = 100 * num_predictions_correct / num_predictions
            accuracy_test_array[model_module_name_index, model_base_name_index] = (
                accuracy_test
            )
            if model_file_name == "resnet34-1D":
                example_input = (uci_epilepsy_train[0][0].unsqueeze(0),)
                torch.onnx.export(
                    model.cpu(),
                    example_input,
                    "tmp/model.onnx",
                    export_params=True,
                )
            if not os.getenv("STAGE") and model_file_name != "alexnet-cnn-one-layer":
                Path(f"tmp/{model_file_name}.pt").unlink()
    styler = pd.DataFrame(
        accuracy_test_array,
        index=[
            "1D",
            "2D, signal as image",
            "2D, spectrogram",
            "2D, one layer CNN",
            "2D, two layer CNN",
        ],
        columns=model_base_names,
    ).style
    styler.format(precision=1)
    styler.highlight_max(props="bfseries: ;")
    styler.to_latex("tmp/results.tex", hrules=True)
    dataset = pd.read_csv("tmp/data.csv")
    signals_all = dataset.drop(columns=["Unnamed: 0", "y"])
    classes_all = dataset["y"]
    signals_all = torch.tensor(signals_all.to_numpy(), dtype=torch.float)
    classes_all = torch.tensor(classes_all.to_numpy()) - 1
    class_names = ["eyes-open", "eyes-closed", "healthy-area", "tumor-area", "epilepsy"]
    for class_index, class_name in enumerate(class_names):
        signal_index = (classes_all == class_index).nonzero()[-1]
        plt.figure()
        plt.plot(signals_all[signal_index].squeeze(), linewidth=4, color="k")
        plt.axis("off")
        plt.xlim([0, signals_all.shape[-1] - 1])
        plt.ylim([-1000, 1000])
        plt.savefig(f"tmp/signal-{class_name}.png")
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
        plt.imsave(f"tmp/signal-as-image-{class_name}.png", data, cmap="gray")
        plt.close()
        f_array, t_array, spectrogram_array = spectrogram(
            signals_all[signal_index],
            fs=signals_all.shape[-1],
            noverlap=4,
            nperseg=8,
            nfft=64,
            mode="magnitude",
        )
        data = np.array(
            PIL.Image.fromarray(spectrogram_array[0]).resize(  # type: ignore[no-untyped-call]
                (signals_all.shape[-1], signals_all.shape[-1]),
                resample=1,
            ),
        )
        plt.figure()
        plt.imsave(Path(f"tmp/spectrogram-{class_name}.png"), data, cmap="gray")
        plt.close()
        model = _CNNOneLayer(num_classes, models.alexnet(), "alexnet-cnn-one-layer")
        model.load_state_dict(torch.load(Path("tmp/alexnet-cnn-one-layer.pt")))
        signal = signals_all[signal_index].unsqueeze(0)
        hook = _Hook()
        model.conv.register_forward_hook(hook)  # type: ignore[arg-type]
        model(signal)
        data = hook.outputs[0][0, 0].cpu().detach().numpy()  # type: ignore[index]
        data = np.array(
            PIL.Image.fromarray(data).resize(  # type: ignore[no-untyped-call]
                (signals_all.shape[-1], signals_all.shape[-1]),
                resample=1,
            ),
        )
        plt.figure()
        plt.imsave(Path(f"tmp/cnn-{class_name}.png"), data, cmap="gray")
        plt.close()


def _make_layers(cfg: list) -> nn.Module:  # type: ignore[type-arg]
    layers: list[nn.Module] = []
    in_channels = 1
    for cfg_element in cfg:
        if cfg_element == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, cfg_element, kernel_size=3, padding=1)
            layers += [conv1d, nn.ReLU()]
            in_channels = cfg_element
    return nn.Sequential(*layers)


def _replace_last_layer(
    num_classes: int,
    model_base: nn.Module,
    model_file_name: str,
) -> nn.Module:
    if model_file_name.startswith(("alexnet", "vgg")):
        model_base.classifier[-1] = nn.Linear(
            model_base.classifier[-1].in_features,
            num_classes,
        )
    elif model_file_name.startswith("resnet"):
        model_base.fc = nn.Linear(model_base.fc.in_features, num_classes)
    elif model_file_name.startswith("densenet"):
        model_base.classifier = nn.Linear(
            model_base.classifier.in_features,
            num_classes,
        )
    return model_base


def _resnet101(num_classes: int) -> nn.Module:
    return _ResNet(_Bottleneck, num_classes, expansion=4, layers=[3, 4, 23, 3])


def _resnet152(num_classes: int) -> nn.Module:
    return _ResNet(_Bottleneck, num_classes, expansion=4, layers=[3, 8, 36, 3])


def _resnet18(num_classes: int) -> nn.Module:
    return _ResNet(_BasicBlock, num_classes, expansion=1, layers=[2, 2, 2, 2])


def _resnet34(num_classes: int) -> nn.Module:
    return _ResNet(_BasicBlock, num_classes, expansion=1, layers=[3, 4, 6, 3])


def _resnet50(num_classes: int) -> nn.Module:
    return _ResNet(_Bottleneck, num_classes, expansion=4, layers=[3, 4, 6, 3])


def _vgg11(num_classes: int) -> nn.Module:
    cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    return _VGG(num_classes, _make_layers(cfg))


def _vgg13(num_classes: int) -> nn.Module:
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    return _VGG(num_classes, _make_layers(cfg))


def _vgg16(num_classes: int) -> nn.Module:
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]
    return _VGG(num_classes, _make_layers(cfg))


def _vgg19(num_classes: int) -> nn.Module:
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]
    return _VGG(num_classes, _make_layers(cfg))


M = TypeVar("M", _BasicBlock, _Bottleneck)


if __name__ == "__main__":
    _main()
