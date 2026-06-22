from pathlib import Path

BASE_DIR = Path().resolve().parent
REPORTS_DIR = BASE_DIR / 'reports' / 'figures' / 'comparisons'
HISTORIES_DIR = BASE_DIR / 'reports' / 'histories'


import json

from training.plot_comparison import plot_metric_comparison, plot_loss_comparison


VALID_METRICS = ['f1', 'auroc', 'loss']


def run_comparison(model_names, metric):
    if metric not in VALID_METRICS:
        raise ValueError(f'Invalid metric: {metric}. Supported: {VALID_METRICS}')

    histories = {}
    model_filenames = []

    for name in model_names:
        filename = name.lower().replace(' ', '_')
        model_filenames.append(filename)
        json_path = HISTORIES_DIR / f'{filename}.json'

        with open(json_path, 'r') as f:
            histories[name] = json.load(f)

    suffix = "_vs_".join(model_filenames)

    if metric == 'loss':
        plot_loss_comparison(histories=histories,
                             output_path=REPORTS_DIR / f'comparison_loss_{suffix}.png',
                             title=f'Models Comparison - Validation Loss')
    else:
        label = 'F1-Score' if metric == 'f1' else 'AUROC'
        y_min = 55 if metric == 'f1' else 80
        y_max = 75 if metric == 'f1' else 100

        plot_metric_comparison(histories=histories,
                               metric_key=metric,
                               output_path=REPORTS_DIR / f'comparison_{metric}_{suffix}.png',
                               title=f'Models Comparison - Validation {label}',
                               y_bottom=y_min,
                               y_top=y_max)