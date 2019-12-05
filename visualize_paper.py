#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image
from scipy.signal import spectrogram, resample


def save_signal(signal_index):
    data = signals_all[signal_index].squeeze().squeeze().cpu().detach().numpy()
    plt.figure()
    plt.plot(data, linewidth=4, color='k')
    plt.axis('off')
    plt.xlim([0, 177])
    plt.ylim([-1000, 1000])
    plt.savefig('images/signal_{}.png'.format(labels_names[labels_all[signal_index]]), bbox_tight='tight')

def save_signal_as_image(signal_index):
    signals_all_min = -1000
    signals_all_max = 1000
    x = signals_all[signal_index] - signals_all_min
    x = 178 * x / (signals_all_max - signals_all_min)
    x = x.squeeze(0).floor().long()
    data = torch.zeros(178, 178)
    for index in range(178):
        data[177 - x[index], index] = 255
    plt.figure()
    plt.imsave('images/signal_as_image_{}.png'.format(labels_names[labels_all[signal_index]]), data, cmap='gray')

def save_spectrogram(signal_index):
    _, _, Sxx = spectrogram(signals_all[signal_index].cpu(), fs=178, noverlap=4, nperseg=8, nfft=64, mode='magnitude')
    data = np.array(Image.fromarray(Sxx[0, :, :]).resize((178, 178), resample=1))
    plt.figure()
    plt.imsave('images/spectrogram_{}.png'.format(labels_names[labels_all[signal_index]]), data, cmap='gray')

def save_cnn(signal_index):
    signal = signals_all[signal_index].unsqueeze_(0).cuda()
    model = torch.load('selected_models/alexnet_cnn_one_layer.pt')
    outputs= []
    def hook(module, input, output):
        outputs.append(output)
    model.conv.register_forward_hook(hook)
    _ = model(signal)
    data = outputs[0][0, 0].cpu().detach().numpy()
    data = np.array(Image.fromarray(data).resize((178, 178), resample=1))
    plt.figure()
    plt.imsave('images/cnn_{}.png'.format(labels_names[labels_all[signal_index]]), data, cmap='gray')

def save_all(signal_index):
    save_signal(signal_index)
    save_signal_as_image(signal_index)
    save_spectrogram(signal_index)
    save_cnn(signal_index)

if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    if not os.path.exists('images'):
        os.mkdir('images')
    dataset = pd.read_csv('./data.csv')
    signals_all = dataset.drop(columns=['Unnamed: 0', 'y'])
    labels_all = dataset['y']
    signals_all = torch.tensor(signals_all.values, dtype=torch.float)
    labels_all = torch.tensor(labels_all.values) - 1
    labels_all_eyes_open = (labels_all == 0).nonzero()
    labels_all_eyes_closed = (labels_all == 1).nonzero()
    labels_all_healthy_area = (labels_all == 2).nonzero()
    labels_all_tumor_area = (labels_all == 3).nonzero()
    labels_all_epilepsy = (labels_all == 4).nonzero()
    labels_names = ['eyes_open', 'eyes_closed', 'healthy_area', 'tumor_area', 'epilepsy']
    save_all(labels_all_eyes_open[-1])
    save_all(labels_all_eyes_closed[-1])
    save_all(labels_all_healthy_area[-1])
    save_all(labels_all_tumor_area[-1])
    save_all(labels_all_epilepsy[-1])
