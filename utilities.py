import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from scipy.signal import spectrogram, resample


def save_signal(signals_all, signal_index, label_name, results_dir):
    data = signals_all[signal_index].squeeze().cpu().detach().numpy()
    plt.figure()
    plt.plot(data, linewidth=4, color='k')
    plt.axis('off')
    plt.xlim([0, 177])
    plt.ylim([-1000, 1000])
    plt.savefig(f'{results_dir}/signal-{label_name}.png')
    plt.close()

def save_signal_as_image(signals_all, signal_index, label_name, results_dir):
    signals_all_min = -1000
    signals_all_max = 1000
    x = signals_all[signal_index] - signals_all_min
    x = 178 * x / (signals_all_max - signals_all_min)
    x = x.squeeze(0).floor().long()
    data = torch.zeros(178, 178)
    for index in range(178):
        data[177 - x[index], index] = 255
    plt.figure()
    plt.imsave(f'{results_dir}/signal-as-image-{label_name}.png', data, cmap='gray')
    plt.close()

def save_spectrogram(signals_all, signal_index, label_name, results_dir):
    _, _, Sxx = spectrogram(signals_all[signal_index].cpu(), fs=178, noverlap=4, nperseg=8, nfft=64, mode='magnitude')
    data = np.array(Image.fromarray(Sxx[0, :, :]).resize((178, 178), resample=1))
    plt.figure()
    plt.imsave(f'{results_dir}/spectrogram-{label_name}.png', data, cmap='gray')
    plt.close()

def save_cnn(signals_all, signal_index, label_name, results_dir, tmp_dir):
    model = torch.load(f'{tmp_dir}/alexnet-cnn-one-layer.pt')
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
    plt.imsave(f'{results_dir}/cnn-{label_name}.png', data, cmap='gray')
    plt.close()
