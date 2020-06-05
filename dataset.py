import os
import pandas as pd
import torch
import urllib

from torch.utils.data import Dataset


class UCI_epilepsy(Dataset):
    def __init__(self, training_validation_test, num_samples, path_cache):
        filename = f'{path_cache}/data.csv'
        if not os.path.isfile(filename):
            urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv", filename)
        dataset = pd.read_csv(filename)
        dataset = dataset[:num_samples]
        signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
        labels_all = dataset['y']
        last_training_index = int(signals_all.shape[0]*0.76)
        last_validation_index = int(signals_all.shape[0]*0.88)
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
