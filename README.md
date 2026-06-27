# ECG Classification using Deep Learning

A deep learning project focused on multi-label classification of cardiac abnormalities using the **PTB-XL** dataset. The pipeline includes ECG signal preprocessing and several neural network architectures optimized for 1-dimensional biomedical signals. Additionally, this project evaluates the impact of substituting traditional fully connected layers with a novel Fourier Kolmogorov-Arnold Network (FKAN) architecture across multiple models.

## Features
* **Signal Preprocessing**: Butterworth bandpass filtering (1-45 Hz) and independent channel Z-score normalization.
* **12-lead Processing**: Utilizes the complete spatial context of ECGs.
* **Rigorous Evaluation Split**: Strict adherence to the official train, calidation and test split.
* **Imbalance-aware Loss Function**: Uses a `BceWithLogitsLoss` weighted with class-specific penalties calculated from the training set distribution to counter severe multi-label class imbalance.
* **Validation-driven Threshold Tuning**: Optimal classification thresholds for each cardiac condition are calculated using the validation set, eliminating data leakage onto the final test report.
* **Automated Reproducibility Pipeline**: Automated workflow that serializes training histories (`.json` experiment reports) and automatically renders learning curves for training and validation set (Loss, ROC AUC, PR AUC).
* **Modern Architectures** Includes implementations of:
    * **CNN**
    * **CNN-FKAN**
    * **CNN-LSTM**
    * **CNN-LSTM-FKAN**
    * **CNN-GRU**
    * **CNN-GRU-FKAN**
    * **GRU**
    * **ResNet**

## Repository Structure
```
├── experiments/              # Jupyter notebooks for development and analysis
│   ├── preprocessing.ipynb
│   ├── testing_models.ipynb
│   └── comparison.ipynb
├── models/                   # Neural network architectures
│   ├── layers/               # Custom layers
│   ├── cnn.py
│   ├── resnet.py
│   ├── gru.py
│   └── ...                   # Hybrid variants (FKAN, LSTM, GRU)
├── outputs/                  # Pipeline artifacts
│   ├── figures/
│   │   ├── comparisons/      # Cross-model evaluation plots (loss, pr_auc, roc_auc)
│   │   └── learning_curves/  # Individual train vs val curves (loss, pr_auc, roc_auc)
│   └── reports/              # Serialized experiment metrics (.json)
└── training/                 # Core training and visualization logic
    ├── engine.py             # Train and evaluation loops
    ├── plot_curve.py         # Learning curve generator
    ├── plot_comparison.py    # Metric comparison generator
    ├── run_experiment.py     # Single model pipeline entrypoint
    └── run_comparison.py     # Multiple models comparison entrypoint
```