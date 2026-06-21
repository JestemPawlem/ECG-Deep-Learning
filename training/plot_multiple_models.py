import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_models(models_histories,
                            output_path,
                            title,
                            show_std=True,
                            y_bottom=50,
                            y_top=100):
    plt.figure(figsize=(10, 7))
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    
    first_model = list(models_histories.values())[0]
    epochs = len(first_model[0]['val_f1'])
    epochs_range = range(1, epochs + 1)
    
    for idx, (model_name, histories) in enumerate(models_histories.items()):
        color = colors[idx % len(colors)]
        
        val_f1_matrix = np.array([h['val_f1'] for h in histories]) * 100
        val_f1_mean = np.mean(val_f1_matrix, axis=0)
        
        plt.plot(epochs_range, val_f1_mean, label=f'{model_name} (Mean)', color=color, linewidth=2)
        
        if show_std:
            val_f1_std = np.std(val_f1_matrix, axis=0)
            plt.fill_between(epochs_range, 
                             val_f1_mean - val_f1_std, 
                             val_f1_mean + val_f1_std, 
                             color=color, alpha=0.1)

    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(y_bottom, y_top + 1, 5))
    
    plt.xticks(np.arange(1, epochs + 1, 1))

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1 [%]')
    plt.grid()
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)