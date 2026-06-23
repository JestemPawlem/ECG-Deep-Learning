import matplotlib.pyplot as plt
import numpy as np


def plot_curve(report,
                metric,
                output_path,
                title,
                y_bottom=None,
                y_top=None):
    plt.figure(figsize=(10, 6))

    is_loss = metric == 'loss'

    if is_loss:
        key_suffix = 'loss'
        label_suffix = 'Loss'
        y_label = 'Loss'
        legend_loc = 'upper right'

        scale = 1.0
        y_bottom = 0.0 if y_bottom is None else y_bottom
        y_top = 1.0 if y_top is None else y_top
        y_ticks = np.arange(0, 1 + 1e-6, 0.05)
    else:
        key_suffix = metric
        label_suffix = 'PR AUC' if metric == 'pr_auc' else 'ROC AUC'
        y_label = f'{label_suffix} [%]'
        legend_loc = 'lower right'

        scale = 100.0
        y_bottom = 50 if y_bottom is None else y_bottom
        y_top = 100 if y_top is None else y_top
        y_ticks = np.arange(y_bottom, y_top + 1e-6, 5)

    train_values = np.array(report['train_' + key_suffix]) * scale
    validation_values = np.array(report['val_' + key_suffix]) * scale
    epochs_range = range(1, len(validation_values) + 1)

    plt.plot(epochs_range,
             train_values,
             color='blue',
             linewidth=1,
             label=f'Train {label_suffix}')

    plt.plot(epochs_range,
             validation_values,
             color='green',
             linewidth=1,
             label=f'Validation {label_suffix}')

    plt.ylim(y_bottom, y_top)
    plt.yticks(y_ticks)
    plt.xticks(np.arange(len(epochs_range) // 10,
                         len(epochs_range) + 1,
                         len(epochs_range) // 10))

    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc=legend_loc)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)