#!/usr/bin/python
import collections
import models_1D
import os
import pickle
import signal2image_modules
import time
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import models

from LeNet_2D import lenet_2D
from dataset import UCI_epilepsy


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    if not os.path.exists('selected_models'):
        os.mkdir('selected_models')
    num_classes = 5
    num_epochs = 100
    batch_size = 20
    signals_all_max = 2047
    signals_all_min = -1885
    training_dataset = UCI_epilepsy('training')
    validation_dataset = UCI_epilepsy('validation')
    test_dataset = UCI_epilepsy('test')
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    base_models_names = ['lenet', 'alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
    base_models_1D = [models_1D.lenet, models_1D.alexnet, models_1D.vgg11, models_1D.vgg13, models_1D.vgg16, models_1D.vgg19, models_1D.resnet18, models_1D.resnet34, models_1D.resnet50, models_1D.resnet101, models_1D.resnet152, models_1D.densenet121, models_1D.densenet161, models_1D.densenet169, models_1D.densenet201]
    base_models_2D = [lenet_2D, models.alexnet, models.vgg11, models.vgg13, models.vgg16, models.vgg19, models.resnet18, models.resnet34, models.resnet50, models.resnet101, models.resnet152, models.densenet121, models.densenet161, models.densenet169, models.densenet201]
    base_models_1D_names = [model_name + '_1D' for model_name in base_models_names]
    combined_models_signal_as_image_names = [model_name + '_signal_as_image' for model_name in base_models_names]
    combined_models_spectrogram_names = [model_name + '_spectrogram' for model_name in base_models_names]
    combined_models_cnn_one_layer_names = [model_name + '_cnn_one_layer' for model_name in base_models_names]
    combined_models_cnn_two_layers_names = [model_name + '_cnn_two_layers' for model_name in base_models_names]
    base_models = base_models_1D + base_models_2D + base_models_2D + base_models_2D + base_models_2D
    combined_models_names = base_models_1D_names + combined_models_signal_as_image_names + combined_models_spectrogram_names + combined_models_cnn_one_layer_names + combined_models_cnn_two_layers_names
    results = collections.defaultdict(dict)
    for _ in combined_models_names:
        results[_]['training_loss'] = []
        results[_]['validation_loss'] = []
        results[_]['validation_accuracy'] = []
    for base_model, combined_model_name in zip(base_models, combined_models_names):
        if combined_model_name.endswith('_1D'):
            model = base_model(num_classes).cuda()
        elif combined_model_name.endswith('_signal_as_image'):
            model = signal2image_modules.SignalAsImage(num_classes, base_model(), combined_model_name, signals_all_max, signals_all_min).cuda()
        elif combined_model_name.endswith('_spectrogram'):
            model = signal2image_modules.Spectrogram(num_classes, base_model(), combined_model_name).cuda()
        elif combined_model_name.endswith('_cnn_one_layer'):
            model = signal2image_modules.CNN_one_layer(num_classes, base_model(), combined_model_name).cuda()
        elif combined_model_name.endswith('_cnn_two_layers'):
            model = signal2image_modules.CNN_two_layers(num_classes, base_model(), combined_model_name).cuda()
        optimizer = Adam(model.parameters())
        best_validation_accuracy = 0
        train_time = time.time()
        for _ in range(num_epochs):
            model.train()
            training_loss_sum = 0
            for signals, labels in training_dataloader:
                signals = signals.cuda()
                labels = labels.cuda()
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
                    signals = signals.cuda()
                    labels = labels.cuda()
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
                    corrects += sum(outputs.argmax(dim=1) == labels).item()
                    validation_loss_sum += loss.item()
            validation_accuracy = 100*corrects/(batch_size * len(validation_dataloader))
            validation_loss = validation_loss_sum / (batch_size * len(validation_dataloader))
            print('Model: {}, Epoch: {}, Loss: {:.3f}, Accuracy: {:.2f}%'.format(combined_model_name, _, validation_loss, validation_accuracy))
            results[combined_model_name]['training_loss'].append(training_loss)
            results[combined_model_name]['validation_loss'].append(validation_loss)
            results[combined_model_name]['validation_accuracy'].append(validation_accuracy)
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(model, 'selected_models/{}.pt'.format(combined_model_name))
                print('saving as best model')
        train_time = time.time() - train_time
        results[combined_model_name]['train_time'] = train_time
    for combined_model_name in combined_models_names:
        model = torch.load('selected_models/{}.pt'.format(combined_model_name))
        model.eval()
        test_loss_sum = 0
        corrects = 0
        test_confussion_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for signals, labels in test_dataloader:
                signals = signals.cuda()
                labels = labels.cuda()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                corrects += sum(outputs.argmax(dim=1) == labels).item()
                for t, p in zip(labels.view(-1), torch.argmax(outputs, 1).view(-1)):
                    test_confussion_matrix[t.long(), p.long()] += 1
                test_loss_sum += loss.item()
        test_accuracy = 100*corrects/(batch_size * len(test_dataloader))
        test_loss = test_loss_sum / (batch_size * len(test_dataloader))
        results[combined_model_name]['test_confussion_matrix'] = test_confussion_matrix
        results[combined_model_name]['test_accuracy'] = test_accuracy
        print('Model: {}, Epoch: {}, Loss: {:.3f}, Accuracy: {:.2f}%'.format(combined_model_name, _, test_loss, test_accuracy))
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
