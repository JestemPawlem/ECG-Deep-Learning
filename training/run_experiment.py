from pathlib import Path

BASE_DIR = Path().resolve().parent
CLEAN_DATA_DIR = BASE_DIR / 'datasets' / 'ptb-xl' / 'data_clean'
FIGURES_DIR = BASE_DIR / 'outputs' / 'figures' / 'learning_curves'
REPORTS_DIR = BASE_DIR / 'outputs' / 'reports'


import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from training.engine import train_model, evaluate_model
from training.plot_curve import plot_curve


def run_experiment(model_class, 
                   model_name, 
                   model_kwargs=None, 
                   epochs=20,
                   batch_size=64,
                   lr=0.001,
                   weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running experiment {model_name} on: {device}')
    
    X_train = torch.tensor(np.load(CLEAN_DATA_DIR / 'train_x.npy')).to(device)
    y_train = torch.tensor(np.load(CLEAN_DATA_DIR / 'train_y.npy')).to(device)
    X_val = torch.tensor(np.load(CLEAN_DATA_DIR / 'val_x.npy')).to(device)
    y_val = torch.tensor(np.load(CLEAN_DATA_DIR / 'val_y.npy')).to(device)
    X_test = torch.tensor(np.load(CLEAN_DATA_DIR / 'test_x.npy')).to(device)
    y_test = torch.tensor(np.load(CLEAN_DATA_DIR / 'test_y.npy')).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size,
                             shuffle=False)
    
    y_train_np = y_train.cpu().numpy()
    neg_counts = (y_train_np == 0).sum(axis=0)
    pos_counts = (y_train_np == 1).sum(axis=0)
    class_weights = np.sqrt(neg_counts / (pos_counts + 1e-5))
    pos_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs).to(device)
    print(f'Model parameters: {model.count_parameters():,}')
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=10,
                                                           factor=0.3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    train_report = train_model(model=model,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss_fn=loss_fn,
                               epochs=epochs)

    test_report = evaluate_model(model=model,
                                 test_loader=test_loader,
                                 thresholds=train_report['thresholds'])
    
    final_report = train_report | test_report

    filename = model_name.lower().replace(' ', '_')

    serializable_data = {}
    for key, val in final_report.items():
        if isinstance(val, list):
            serializable_data[key] = [float(x) for x in val]
        else:
            serializable_data[key] = float(val)

    with open(REPORTS_DIR / f'{filename}.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)
    
    plot_curve(final_report,
               'roc_auc',
               FIGURES_DIR / 'roc_auc' / f'{filename}.png',
               title=f'{model_name} - ROC AUC',
               y_bottom=80,
               y_top=100)
    
    plot_curve(final_report,
               'pr_auc',
               FIGURES_DIR / 'pr_auc' / f'{filename}.png',
               title=f'{model_name} - PR AUC',
               y_bottom=55,
               y_top=85)
    
    plot_curve(final_report,
               'loss',
               FIGURES_DIR / 'loss' / f'{filename}.png',
               title=f'{model_name} - Loss')

    print(f'Experiment {model_name} completed.')
    return final_report