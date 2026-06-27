from pathlib import Path

BASE_DIR = Path().resolve().parent
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures' / 'comparisons'
REPORTS_DIR = BASE_DIR / 'outputs' / 'reports'


import json

from training.plot_comparison import plot_comparison


VALID_METRICS = ['roc_auc', 'pr_auc', 'loss']


def run_comparison(model_names, metric):
    if metric not in VALID_METRICS:
        raise ValueError(f'Invalid metric: {metric}. Supported: {VALID_METRICS}')

    reports = {}
    model_filenames = []

    for name in model_names:
        filename = name.lower().replace(' ', '_')
        model_filenames.append(filename)
        json_path = REPORTS_DIR / f'{filename}.json'

        with open(json_path, 'r') as f:
            reports[name] = json.load(f)

    filename = '_vs_'.join(model_filenames) + '.png'

    if metric == 'loss':
        plot_comparison(reports=reports,
                        metric=metric,
                        output_path=FIGURES_DIR / metric / filename)
    else:
        y_min = 60 if metric == 'pr_auc' else 80
        y_max = 80 if metric == 'pr_auc' else 100

        plot_comparison(reports=reports,
                        metric=metric,
                        output_path=FIGURES_DIR / metric / filename,
                        y_bottom=y_min,
                        y_top=y_max)