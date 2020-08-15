import collections
import numpy as np
import os
import pandas as pd
import argparse
import time
import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

from LeNet_2D import lenet_2D
from dataset import UCI_epilepsy
from models_1D import lenet, alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, densenet201
from signal2image_modules import SignalAsImage, Spectrogram, CNN_one_layer, CNN_two_layers
from utilities import save_signal, save_signal_as_image, save_spectrogram, save_cnn


if __name__ == '__main__':
    # Set random seeds.
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--cache-dir')
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    device = torch.device('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    if args.full:
        num_samples = 11500
        num_epochs = 100
    else:
        num_samples = 10
        num_epochs = 1
    num_classes = 5
    batch_size = 20
    signals_all_max = 2047
    signals_all_min = -1885
    training_dataset = UCI_epilepsy('training', num_samples, args.cache_dir)
    validation_dataset = UCI_epilepsy('validation', num_samples, args.cache_dir)
    test_dataset = UCI_epilepsy('test', num_samples, args.cache_dir)
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    base_models_names = ['lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
    base_models_1D = [lenet, alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, densenet201]
    base_models_2D = [lenet_2D, models.alexnet, models.vgg11, models.vgg13, models.vgg16, models.vgg19, models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152, models.densenet121, models.densenet161, models.densenet169, models.densenet201]
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
            model = SignalAsImage(num_classes, base_model(), combined_model_name, signals_all_max, signals_all_min, device)
        elif combined_model_name.endswith('-spectrogram'):
            model = Spectrogram(num_classes, base_model(), combined_model_name, device)
        elif combined_model_name.endswith('-cnn-one-layer'):
            model = CNN_one_layer(num_classes, base_model(), combined_model_name)
        elif combined_model_name.endswith('-cnn-two-layers'):
            model = CNN_two_layers(num_classes, base_model(), combined_model_name)
        model = model.to(device)
        optimizer = Adam(model.parameters())
        best_validation_accuracy = -1
        train_time = time.time()
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
            validation_accuracy = 100*corrects/(batch_size * len(validation_dataloader))
            validation_loss = validation_loss_sum / (batch_size * len(validation_dataloader))
            print(f'Model: {combined_model_name}, Epoch: {epoch}, Loss: {validation_loss:.3f}, Accuracy: {validation_accuracy:.2f}%')
            results[combined_model_name]['training_loss'].append(training_loss)
            results[combined_model_name]['validation_loss'].append(validation_loss)
            results[combined_model_name]['validation_accuracy'].append(validation_accuracy)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model, f'{args.results_dir}/{combined_model_name}.pt')
                print('saving as best model')
        train_time = time.time() - train_time
        results[combined_model_name]['train_time'] = train_time
    for combined_model_name in combined_models_names:
        model = torch.load(f'{args.results_dir}/{combined_model_name}.pt')
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
        test_accuracy = 100*corrects/(batch_size * len(test_dataloader))
        test_loss = test_loss_sum/(batch_size * len(test_dataloader))
        results[combined_model_name]['test_confussion_matrix'] = test_confussion_matrix
        results[combined_model_name]['test_accuracy'] = test_accuracy
        print(f'Model: {combined_model_name}, Epoch: {epoch}, Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.2f}%')

    results_test_accuracy_for_paper = np.zeros((75,))
    for index, model in enumerate(results):
        results_test_accuracy_for_paper[index] = np.around(results[model]['test_accuracy'], 1)
    results_test_accuracy_for_paper = results_test_accuracy_for_paper.reshape(5, 15)
    df = pd.DataFrame(results_test_accuracy_for_paper, index=['1D', '2D, signal as image', '2D, spectrogram', '2D, one layer CNN', '2D, two layer CNN'])
    df.columns = base_models_names
    with open(f'{args.results_dir}/results.tex', 'w') as f:
        df.to_latex(buf=f, bold_rows=True, escape=False, column_format='l|c|c|cccc|ccccc|cccc')

    dataset = pd.read_csv(f'{args.cache_dir}/data.csv')
    signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
    labels_all = dataset['y']
    signals_all = torch.tensor(signals_all.values, dtype=torch.float)
    labels_all = torch.tensor(labels_all.values) - 1
    labels_names = ['eyes-open', 'eyes-closed', 'healthy-area', 'tumor-area', 'epilepsy']
    for index, label_name in enumerate(labels_names):
        signal_index = (labels_all == index).nonzero()[-1]
        save_signal(signals_all, signal_index, label_name, args.results_dir)
        save_signal_as_image(signals_all, signal_index, label_name, args.results_dir)
        save_spectrogram(signals_all, signal_index, label_name, args.results_dir)
        save_cnn(signals_all, signal_index, label_name, args.results_dir)
