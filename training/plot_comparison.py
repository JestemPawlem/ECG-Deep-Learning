import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(histories,
                           metric_key,
                           output_path,
                           title,
                           y_bottom=50,
                           y_top=100):
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    label_suffix = 'ROC AUC' if metric_key == 'auroc' else 'F1-Score'

    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        values = np.array(history['val_' + metric_key]) * 100
        epochs_range = range(1, len(values) + 1)

        plt.plot(epochs_range,
                 values,
                 color=color,
                 linewidth=2,
                 label=f'{model_name}')

    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(y_bottom, y_top + 1, 5))
    plt.xticks(np.arange(len(epochs_range) // 10, len(epochs_range) + 1, len(epochs_range) // 10))

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel('Epoch')
    plt.ylabel(f'Validation {label_suffix} [%]')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_loss_comparison(histories,
                         output_path,
                         title,
                         y_bottom=0.0,
                         y_top=1.0):
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        values = np.array(history["val_loss"])
        epochs_range = range(1, len(values) + 1)

        plt.plot(epochs_range,
                 values,
                 color=color,
                 linewidth=2,
                 label=f'{model_name}')

    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(len(epochs_range) // 10, len(epochs_range) + 1, len(epochs_range) // 10))

    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)