import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

BASE_DIR = Path().resolve().parent
MODELS_DIR = BASE_DIR / 'outputs' / 'models'
KERNELS_DIR = BASE_DIR / 'outputs' / 'figures' / 'kernels'


def plot_wavelet_kernels(model_class,
                         model_name,
                         num_kernels=9,
                         channel_idx=0):
    filename = model_name.lower().replace(' ', '_')
    model_path = MODELS_DIR / f'{filename}.pt'

    if not model_path.exists():
        raise FileNotFoundError(f'File not found: {model_path}')

    checkpoint = torch.load(model_path, map_location='cpu')
    model = model_class(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    wavkan_layer = model.wavkan

    with torch.no_grad():
        kernels = wavkan_layer.get_kernels().detach().cpu()
        kernels = kernels[:, channel_idx, :]
        scales = torch.abs(wavkan_layer.scale).detach().cpu().squeeze()

    sorted_indices = torch.argsort(scales)
    sorted_indices = sorted_indices[:num_kernels]
    kernels = kernels[sorted_indices]
    scales = scales[sorted_indices]

    selected_wavelets = kernels[:num_kernels].numpy()

    y_min = selected_wavelets.min()
    y_max = selected_wavelets.max()
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)
    y_ticks = np.linspace(y_min, -y_min, 7)

    cols = math.ceil(math.sqrt(num_kernels))
    rows = math.ceil(num_kernels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    axes_flat = axes.flat if num_kernels > 1 else [axes]

    for i in range(num_kernels):
        ax = axes_flat[i]
        wavelet = kernels[i].numpy()
        s = (
            scales[i].item()
            if scales.ndim > 0
            else scales.item()
        )

        ax.plot(wavelet, color='darkcyan', linewidth=1)
        ax.set_title(f'Wavelet {i+1} (s={s:.2f})', fontsize=10, fontweight='bold')

        ax.set_ylim(y_limits)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        ax.set_xticks([])

        ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(num_kernels, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(f'{model_name} - Learned Wavelets (Channel {channel_idx})',
                 fontsize=14,
                 fontweight='bold',
                 y=1.02)
    plt.tight_layout()

    out_path = KERNELS_DIR / f'{filename}_wavelet_kernels.png'
    plt.savefig(out_path, dpi=300)


def plot_fourier_kernels(model_class,
                         model_name,
                         num_kernels=9,
                         channel_idx=0):
    filename = model_name.lower().replace(' ', '_')
    model_path = MODELS_DIR / f'{filename}.pt'

    if not model_path.exists():
        raise FileNotFoundError(f'File not found: {model_path}')

    checkpoint = torch.load(model_path, map_location='cpu')
    model = model_class(**checkpoint['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    fkan_layer = model.fkan

    with torch.no_grad():
        kernels = fkan_layer.get_kernels().detach().cpu()
        kernels = kernels[:, channel_idx, :]

    energies = (kernels**2).sum(dim=1)
    sorted_indices = torch.argsort(energies, descending=True)[:num_kernels]

    kernels = kernels[sorted_indices]
    selected_kernels = kernels.numpy()

    y_min = selected_kernels.min()
    y_max = selected_kernels.max()
    y_margin = (y_max - y_min) * 0.1
    y_limits = (y_min - y_margin, y_max + y_margin)
    y_ticks = np.linspace(y_min, -y_min, 7)

    cols = math.ceil(math.sqrt(num_kernels))
    rows = math.ceil(num_kernels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
    axes_flat = axes.flat if num_kernels > 1 else [axes]

    for i in range(num_kernels):
        ax = axes_flat[i]
        kernel_signal = selected_kernels[i]
        orig_idx = sorted_indices[i].item() + 1
        amp = np.abs(kernel_signal).max()

        ax.plot(kernel_signal, color='crimson', linewidth=1)
        ax.set_title(
            f'Fourier Kernel {orig_idx} (A={amp:.2f})',
            fontsize=10,
            fontweight='bold',
        )

        ax.set_ylim(y_limits)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        ax.set_xticks([])

        ax.grid(True, linestyle='--', alpha=0.5)

    for i in range(num_kernels, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(
        f'{model_name} - Learned Fourier Kernels (Channel {channel_idx})',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()

    out_path = KERNELS_DIR / f'{filename}_fourier_kernels.png'
    plt.savefig(out_path, dpi=300)
    plt.show()