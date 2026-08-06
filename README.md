# ECG Classification using Deep Learning

A deep learning project focused on multi-label classification of cardiac abnormalities using the **PTB-XL** dataset. The pipeline includes ECG signal preprocessing and several neural network architectures optimized for 1-dimensional biomedical signals. Furthermore, this project incorporates novel KAN-based architectures (Fourier KAN and Wavelet KAN) to evaluate and compare their performance against classical deep learning models.

## Key Features

### Data Processing & Signal Pipeline
* **Signal Specification**: Processes 12-lead ECG signals sampled at 100 Hz (1,000 time steps per record).
* **Preprocessing**: Butterworth bandpass filtering (1-45 Hz) and per-channel Z-score normalization.
* **Spatial Context**: Utilizes all 12 leads (experiments with reduced lead subsets yielded inferior performance).
* **Rigorous Evaluation Split**: Strict adherence to the official PTB-XL recommended train, validation, and test split.

### Training Methodology & Metrics
* **Imbalance-aware Loss Function**: Uses `BCEWithLogitsLoss` with square-root class-frequency reweighting to mitigate multi-label class imbalance.
* **Validation-based Threshold Optimization**: Classification thresholds are fine-tuned per class on the validation set to maximize F1-score without test data leakage.
* **Multi-Metric Evaluation**: Comprehensive performance assessment using Macro ROC AUC, Macro PR AUC, and Macro F1-score.
* **Automated Experiment Tracking**: Automatically serializes training histories (`.json` reports) and renders learning curves (`figures/learning_curves/`).

### Workflow & Evaluation Notebooks
* **`testing_models.ipynb`**: Handles comprehensive model evaluation, automatic metric report generation, and plotting individual learning curves.
* **`comparison.ipynb`**: Allows selection of multiple trained models and evaluation metrics to generate cross-architecture comparison plots (saved to `figures/comparisons/`).

### Model Architectures
Evaluates classical deep learning baselines alongside novel Kolmogorov-Arnold Network variants:
* **Baselines**: CNN, ResNet, GRU, CNN-LSTM, CNN-GRU
* **KAN Variants**: Fourier KAN (FKAN), Wavelet KAN (WavKAN)

## Results & Visualizations
Detailed metric reports are saved in `outputs/reports/` as JSON files. Visualizations, including cross-model comparison curves and learned basis functions (Fourier and Wavelet kernels), can be rendered dynamically via the provided notebooks or viewed directly in `outputs/figures/`.