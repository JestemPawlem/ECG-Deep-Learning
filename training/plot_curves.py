import matplotlib.pyplot as plt
import numpy as np

def plot_metric_curves(history,
                       metric_key,
                       output_path,
                       title,
                       y_bottom=50,
                       y_top=100):
    plt.figure(figsize=(10, 6))
    
    label_suffix = 'ROC AUC' if metric_key == 'auroc' else 'F1'
    train_values = np.array(history['train_' + metric_key]) * 100
    validation_values = np.array(history['val_' + metric_key]) * 100
    epochs_range = range(1, len(validation_values) + 1)
    
    plt.plot(epochs_range,
             train_values,
             color='blue',
             label=f'Train {label_suffix}')
    
    plt.plot(epochs_range,
             validation_values,
             color='green',
             label=f'Validation {label_suffix}')

    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(y_bottom, y_top + 1, 5))
    plt.xticks(np.arange(len(epochs_range) // 10, len(epochs_range) + 1, len(epochs_range) // 10))

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(f'{label_suffix} [%]')
    plt.grid()
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_loss_curves(history,
                     output_path,
                     title,
                     y_bottom=0,
                     y_top=1):
    plt.figure(figsize=(10, 6))

    train_values = np.array(history['train_loss'])
    validation_values = np.array(history['val_loss'])
    epochs_range = range(1, len(validation_values) + 1)

    plt.plot(epochs_range,
             train_values,
             color='blue',
             label='Train Loss')
    
    plt.plot(epochs_range,
             validation_values,
             color='green',
             label='Validation Loss')
    
    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(len(epochs_range) // 10, len(epochs_range) + 1, len(epochs_range) // 10))

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)