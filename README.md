# ECG Classification using Deep Learning

A deep learning project focused on multi-label classification of cardiac abnormalities using the **PTB-XL** dataset. The pipeline includes ECG signal preprocessing and several neural network architectures optimized for 1-dimensional biomedical signals.

## Features
* **Signal Preprocessing:** Butterworth bandpass filtering (1-45 Hz) and independent channel Z-score normalization
* **12-lead Processing:** Utilizes the complete spatial context of ECGs
* **Modern Architectures** Includes implementations of:
    * **CNN**
    * **CNN-LSTM**
    * **CNN-GRU**
    * **CNN-FKAN**
    * **GRU**
    * **ResNet**