import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from scipy.signal import spectrogram, resample


def save_signal(signals_all, signal_index, label_name, path_results):
    data = signals_all[signal_index].squeeze().cpu().detach().numpy()
    plt.figure()
    plt.plot(data, linewidth=4, color='k')
    plt.axis('off')
    plt.xlim([0, 177])
    plt.ylim([-1000, 1000])
    plt.savefig(f'{path_results}/signal_{label_name}.png', bbox_tight='tight')
    plt.close()

def save_signal_as_image(signals_all, signal_index, label_name, path_results):
    signals_all_min = -1000
    signals_all_max = 1000
    x = signals_all[signal_index] - signals_all_min
    x = 178 * x / (signals_all_max - signals_all_min)
    x = x.squeeze(0).floor().long()
    data = torch.zeros(178, 178)
    for index in range(178):
        data[177 - x[index], index] = 255
    plt.figure()
    plt.imsave(f'{path_results}/signal_as_image_{label_name}.png', data, cmap='gray')
    plt.close()

def save_spectrogram(signals_all, signal_index, label_name, path_results):
    _, _, Sxx = spectrogram(signals_all[signal_index].cpu(), fs=178, noverlap=4, nperseg=8, nfft=64, mode='magnitude')
    data = np.array(Image.fromarray(Sxx[0, :, :]).resize((178, 178), resample=1))
    plt.figure()
    plt.imsave(f'{path_results}/spectrogram_{label_name}.png', data, cmap='gray')
    plt.close()

def save_cnn(signals_all, signal_index, label_name, path_models, path_results):
    model = torch.load(f'{path_models}/alexnet_cnn_one_layer.pt')
    device = next(model.parameters()).device
    signal = signals_all[signal_index].unsqueeze(0).to(device)
    outputs= []
    def hook(module, input, output):
        outputs.append(output)
    model.conv.register_forward_hook(hook)
    model(signal)
    data = outputs[0][0, 0].cpu().detach().numpy()
    data = np.array(Image.fromarray(data).resize((178, 178), resample=1))
    plt.figure()
    plt.imsave(f'{path_results}/cnn_{label_name}.png', data, cmap='gray')
    plt.close()
