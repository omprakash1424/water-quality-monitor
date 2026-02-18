🌊 Water Quality Monitor - Anomaly Detection System

Production-Ready ML Engineering Project for Water Quality Monitoring

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


📋 Project Overview

This is my project that implements an intelligent anomaly detection system for water quality monitoring using advanced deep learning techniques.

🎯 Problem Statement

Real-time water quality monitoring is critical for public health and environmental protection. This system detects anomalies in water sensor data (pH, turbidity, temperature, dissolved oxygen, conductivity) to enable early warning of contamination or equipment malfunction.

🔑 Key Features

- Unsupervised Learning**: Works without labeled anomaly data
- LSTM Autoencoder: Deep learning model for time-series anomaly detection
- Baseline Comparison: Z-score, PCA, Isolation Forest, Robust Covariance
- Severity Classification: Categorizes anomalies as Normal, Moderate, or Severe
- Explainability: Feature-level explanation of detected anomalies
- Real-time Dashboard: Interactive Streamlit dashboard for monitoring

🏗️ System Architecture

Water Sensors → Data Ingestion → Preprocessing → Feature Engineering
                                                         ↓
                        Dashboard ← Explainability ← LSTM Autoencoder
                                                         ↓
                                              Anomaly Scoring → Severity Classification


📁 Project Structure

water-quality-ai/
│
├── data/
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Preprocessed data
│   └── simulated/              # Simulated data
│
├── src/
│   ├── data_loader.py          # Data loading and splitting
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── feature_engineering.py  # Feature creation
│   ├── baseline_models.py      # Baseline anomaly detectors
│   ├── lstm_autoencoder.py     # LSTM Autoencoder model
│   ├── anomaly_scoring.py      # Threshold and scoring
│   ├── severity_classifier.py  # Severity classification
│   ├── explainability.py       # Feature importance
│   └── utils.py                # Utility functions
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── models/
│   ├── lstm_autoencoder.h5     # Trained LSTM model
│   ├── isolation_forest.pkl    # Trained IF model
│   └── scaler.pkl              # Fitted scaler
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_lstm_autoencoder.ipynb
│
├── reports/
│   └── results_analysis.ipynb  # Final results
│
├── requirements.txt            # Dependencies
└── README.md                   # This file

📊 Model Comparison

| Model              | Precision | Recall | F1-Score | Detection Delay |
|--------------------|-----------|--------|----------|-----------------|
| Z-Score            | 0.65      | 0.58   | 0.61     | Low             |
| PCA                | 0.72      | 0.68   | 0.70     | Low             |
| Isolation Forest   | 0.78      | 0.75   | 0.76     | Low             |
| **LSTM Autoencoder** | **0.89**  | **0.86** | **0.87** | **Very Low**    |

LSTM Autoencoder achieves superior performance with minimal detection delay.


🧠 Model Details

LSTM Autoencoder Architecture

Input (10, 5)
    ↓
LSTM (64 units) + Dropout(0.2)
    ↓
LSTM (32 units) + Dropout(0.2)
    ↓
Dense (Bottleneck: 32)
    ↓
RepeatVector (10)
    ↓
LSTM (32 units) + Dropout(0.2)
    ↓
LSTM (64 units) + Dropout(0.2)
    ↓
TimeDistributed Dense (5)


Training Configuration

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)
- Early Stopping: Patience=10
- Training Data: Normal samples only
- Validation Split: 10%

Anomaly Detection Method

reconstruction_error = MSE(X_original, X_reconstructed)
anomaly = reconstruction_error > threshold
Threshold Calculation: 95th percentile of training reconstruction errors

📈 Feature Engineering

Created Features

1. Rolling Statistics
   - Rolling mean (windows: 6, 12, 24)
   - Rolling std (windows: 6, 12, 24)

2. Lag Features
   - Lag 1, 3, 6 time steps

3. Rate of Change
   - Period 1, 6

4. Interaction Features
   - pH × Temperature
   - DO × Temperature
   - Turbidity / Conductivity

5. Time Features (if timestamp available)
   - Hour, Day of Week, Month
   - Cyclical encoding (sin/cos)


🔍 Explainability

Feature-wise Reconstruction Error

For each detected anomaly, the system identifies:
- Which sensor caused the anomaly
- Contribution percentage of each feature
- Human-readable explanation

Example Output:

🚨 Anomaly detected at sample 1523
   Total reconstruction error: 0.045678
   
   Top contributing factors:
   1. turbidity: 45.2% contribution (error: 0.020634)
   2. pH: 28.7% contribution (error: 0.013109)
   3. conductivity: 16.1% contribution (error: 0.007354)

## 📊 Severity Classification

| Severity | Threshold | Action                              | Priority |
|----------|-----------|-------------------------------------|----------|
| Normal   | < T1      | No action required                  | Low      |
| Moderate | T1 - T2   | Monitor closely                     | Medium   |
| Severe   | > T2      | Immediate investigation required    | High     |

T1: 85th percentile, T2: 95th percentile

📝 Experimental Setup

Dataset
- Source: Simulated water quality sensor data
- Features: pH, Turbidity, Temperature, Dissolved Oxygen, Conductivity
- Samples: 10,000
- Anomaly Ratio: 5%
- Train/Test Split: 80/20

Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC
- Detection Delay
- Confusion Matrix

---

🛠️ Future Enhancements

- [ ] Real-time streaming data integration
- [ ] Multi-location monitoring
- [ ] Alert notification system (email/SMS)
- [ ] Model retraining pipeline
- [ ] Cloud deployment (AWS/Azure)
- [ ] Mobile app integration

👨‍💻 Author

OM PRAKASH SHARMA
- Email: omprakash829427@gmail.com
- LinkedIn: [Om Prakash Sharma](https://www.linkedin.com/in/om-prakash-sharma-42b9362a7/?isSelfProfile=true)
- GitHub: [@omprakash4124](https://github.com/omprakash4124)

📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

🙏 Acknowledgments

- Water quality sensor data standards
- TensorFlow/Keras documentation
- Scikit-learn anomaly detection methods
- Academic research on time-series anomaly detection

📚 References

1. Malhotra, P., et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"
2. Liu, F. T., et al. (2008). "Isolation Forest" IEEE ICDM
3. Chalapathy, R., et al. (2019). "Deep Learning for Anomaly Detection: A Survey"


