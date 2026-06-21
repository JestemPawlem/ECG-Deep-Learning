import sys
from pathlib import Path

BASE_DIR = Path().resolve().parent
CLEAN_DATA_DIR = BASE_DIR / 'datasets' / 'ptb-xl' / 'data_clean'
REPORTS_DIR = BASE_DIR / 'reports' / 'figures'
HISTORIES_DIR = BASE_DIR / 'reports' / 'histories'


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from training.train_model import train_model
from training.save_history import save_history_to_json
from training.plot_curves import plot_learning_curves, plot_loss_curves


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
    
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val),
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
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=epochs
    )

    file_suffix = model_name.lower().replace(' ', '-')
    
    save_history_to_json(history, HISTORIES_DIR / f'{file_suffix}.json')
    
    plot_learning_curves(history,
                          'auroc',
                          REPORTS_DIR / f'curves_auroc_{file_suffix}.png',
                          title=f'{model_name} - AUROC',
                          y_bottom=80)
    plot_learning_curves(history,
                          'f1',
                          REPORTS_DIR / f'curves_f1_{file_suffix}.png',
                          title=f'{model_name} - F1-Score',
                          y_bottom=40,
                          y_top=80)
    plot_loss_curves(history,
                     REPORTS_DIR / f'curves_loss_{file_suffix}.png',
                     title=f'{model_name} - Loss')

    print(f'Experiment {model_name} completed.')
    return history