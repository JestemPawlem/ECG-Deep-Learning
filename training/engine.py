import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score
)


def optimize_thresholds(targets, probs):
    num_classes = targets.shape[1]
    best_thresholds = np.zeros(num_classes)

    for c in range(num_classes):
        precisions, recalls, thresholds = precision_recall_curve(targets[:, c],
                                                                 probs[:, c])
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-6)
        best_idx = np.argmax(f1_scores)
        best_thresholds[c] = (thresholds[best_idx] if best_idx < len(thresholds)
                              else 0.5)
    
    return best_thresholds


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
        'train_roc_auc': [],
        'val_roc_auc': [],
        'train_pr_auc': [],
        'val_pr_auc': []
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

        train_loss = train_loss / train_total
        train_roc_auc = roc_auc_score(train_targets,
                                      train_probs,
                                      average='macro')
        train_pr_auc = average_precision_score(train_targets,
                                               train_probs,
                                               average='macro')

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

        val_loss = val_loss / val_total
        val_roc_auc = roc_auc_score(val_targets,
                                    val_probs,
                                    average='macro')
        val_pr_auc = average_precision_score(val_targets,
                                             val_probs,
                                             average='macro')
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_roc_auc'].append(train_roc_auc)
        history['val_roc_auc'].append(val_roc_auc)
        history['train_pr_auc'].append(train_pr_auc)
        history['val_pr_auc'].append(val_pr_auc)

        if print_epochs:
            print(f'Epoch {epoch+1:02d}/{epochs}:')
            print(f'\tTrain Loss: {train_loss:.4f} | ROC AUC: {train_roc_auc*100:.2f}% | PR AUC: {train_pr_auc*100:.2f}%')
            print(f'\tVal Loss:   {val_loss:.4f} | ROC AUC: {val_roc_auc*100:.2f}% | PR AUC: {val_pr_auc*100:.2f}%')
    
    print('\nOptimizing thresholds...')
    optimal_thresholds = optimize_thresholds(val_targets, val_probs)
    history['thresholds'] = optimal_thresholds.tolist()
    
    return history


def evaluate_model(model,
                   test_loader,
                   thresholds):
    model.eval()
    test_probs, test_targets = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            test_probs.append(probs.cpu())
            test_targets.append(labels.cpu())
    
    test_probs = torch.cat(test_probs).numpy()
    test_targets = torch.cat(test_targets).numpy()

    thresholds = np.array(thresholds)
    test_preds = (test_probs >= thresholds).astype(np.float32)

    test_roc_auc = roc_auc_score(test_targets,
                                 test_probs,
                                 average='macro')
    test_pr_auc = average_precision_score(test_targets,
                                          test_probs,
                                          average='macro')
    test_f1 = f1_score(test_targets, test_preds, average='macro')

    print(f'--- TEST EVALUATION ---')
    print(f'\tTest ROC AUC: {test_roc_auc*100:.2f}%')
    print(f'\tTest PR AUC:  {test_pr_auc*100:.2f}%')
    print(f'\tTest F1-Score: {test_f1*100:.2f}%\n')

    return {
        'test_roc_auc': test_roc_auc,
        'test_pr_auc': test_pr_auc,
        'test_f1': test_f1
    }