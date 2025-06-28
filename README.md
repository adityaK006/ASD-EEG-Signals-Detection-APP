# 🧠 EEG-Based Autism Spectrum Disorder (ASD) Detection Using Deep Learning

## 📌 Project Overview

This project focuses on detecting Autism Spectrum Disorder (ASD) using EEG signals through deep learning models. It includes:

- EEG signal preprocessing and feature extraction
- Multiple deep learning model implementations (RNN, CNN-RNN, CNN-BiRNN)
- Model evaluation and comparison
- Deployment of the final model in an Android app using TensorFlow Lite
- And used python venv 3.12.7

---

## 📂 Project Structure
```
├── data/ # Raw and preprocessed EEG datasets
├── preprocessing/ # Scripts for loading, preprocessing EEG data
│ ├── load_data.py
│ ├── load_normal_data.py
│ ├── preprocess_normal_data.py
│ └── extract_features.py
├── models/ # Model definitions and training scripts
│ ├── rnn_model.py
│ ├── cnn_rnn_model.py
│ ├── cnn_birnn_model.py
│ ├── train_rnn.py
│ ├── train_cnn_rnn.py
│ └── train_cnn_birnn.py
├── android_app/ # Android Studio project files
│ ├── assets/
│ ├── MainActivity.kt
│ └── model.tflite
├── evaluation/ # Evaluation metrics, graphs, confusion matrices
├── results/ # Saved figures for Overleaf and reporting
├── README.md
└── requirements.txt
```

---

## 🧪 Dataset Description

### 1. Autism EEG Dataset
- Source: [P300-based BCI Training Dataset]
- Format: `.npy`
- Shape: `(105, 350, 1600, 8)`  
  - 105 EEG samples  
  - 350 epochs (time windows)  
  - 1600 time points per epoch  
  - 8 channels (electrodes)

### 2. Normal EEG Dataset
- Source: SPIS Resting-State EEG Dataset
- Raw `.mat` and `.csv` format
- Preprocessed to match the autism EEG shape

---

## ⚙️ Preprocessing & Feature Extraction

- All EEG data normalized for consistency.
- Feature extraction performed by calculating **mean and variance** across time points (1600) for each of the 8 channels.
- This results in a new feature vector of **16 features (8 means + 8 variances)** per epoch.

📌 Final shape after preprocessing:  
- Input shape: `(samples, 350, 16)`  
- Labels: `1 = Autism`, `0 = Normal`

---

## 🧠 Deep Learning Models

### 1. RNN (LSTM)
- Input: `(350, 16)`
- LSTM Layers + Dense Output
- Dropout: 0.5

### 2. CNN-RNN
- 1D Convolution → MaxPool → Global Pooling
- Dense output with softmax

### 3. CNN-BiRNN
- 1D Convolution → BiLSTM
- Dense layer + sigmoid output
- Best performing model (AUC: 0.94)

---

## 📊 Evaluation Metrics

| Model         | Accuracy | Loss   | ROC AUC |
|---------------|----------|--------|---------|
| RNN (LSTM)    | 88.2%    | 0.3174 | 0.90    |
| CNN-RNN       | 86.3%    | 0.3548 | 0.87    |
| CNN-BiRNN     | 91.4%    | 0.2785 | 0.94    |

---

## 📱 Android App Integration

### Built With
- Android Studio (Kotlin)
- TensorFlow Lite
- Material Design UI

### Features
- Load sample EEG signal from app assets
- Predict Autism/Normal using the embedded TFLite RNN model
- Display result on the app interface

### Input Format
- EEG signal in `.npy` or `.csv`
- Shape: `(350, 16)`

> 🚫 **Note**: Real-time EEG signal acquisition from wearable headsets is not yet implemented, but planned for future development.

---

## 🛠️ Installation & Setup

### 🧠 Model Training
```bash
pip install -r requirements.txt
python train_rnn.py
python train_cnn_rnn.py
python train_cnn_birnn.py
```


📱 Android App
Open android_app/ folder in Android Studio.

Ensure model.tflite is placed in assets/.

Build and run on an emulator or device (SDK >= 21).

📈 Results
Evaluation visuals such as accuracy curves, confusion matrices, and ROC curves are stored in the results/ folder. These can also be used in academic reports (e.g., Overleaf).

📚 Future Work
Support real-time EEG signal acquisition from headsets

Optimize model size and inference speed for mobile

Extend classification beyond binary (e.g., ADHD, depression)
