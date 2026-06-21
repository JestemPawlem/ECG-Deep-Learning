import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                loss_fn,
                epochs=20,
                print_epochs=True):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auroc': [],
        'val_auroc': [],
        'train_f1': [],
        'val_f1': []
    }

    for epoch in range(epochs):
        # --- TRENING ---
        model.train()
        train_loss, train_total = 0.0, 0
        train_probs, train_targets = [], []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)

            probs = torch.sigmoid(outputs)
            train_probs.append(probs.detach().cpu())
            train_targets.append(labels.cpu())
        
        train_probs = torch.cat(train_probs).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_preds = (train_probs >= 0.5).astype(np.float32)

        train_auroc = roc_auc_score(train_targets, train_probs, average='macro')
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        train_loss = train_loss / train_total

        # --- WALIDACJA ---
        model.eval()
        val_loss, val_total = 0.0, 0
        val_probs, val_targets = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)

                probs = torch.sigmoid(outputs)
                val_probs.append(probs.cpu())
                val_targets.append(labels.cpu())
            
        val_probs = torch.cat(val_probs).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_preds = (val_probs >= 0.4).astype(np.float32)

        val_auroc = roc_auc_score(val_targets, val_probs, average='macro')
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_loss = val_loss / val_total

        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if print_epochs:
            print(f'Epoch {epoch+1:02d}/{epochs}:')
            print(f'\tTrain Loss: {train_loss:.4f} | ROC AUC: {train_auroc*100:.2f}% | F1: {train_f1*100:.2f}%')
            print(f'\tVal Loss:   {val_loss:.4f} | ROC AUC: {val_auroc*100:.2f}% | F1: {val_f1*100:.2f}%')
    
    return history