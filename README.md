# 🗣️ Language Identification from Short Audio Clips Using MFCC and ML

## Team Details
- **Course:** Predictive Analytics (AY 2025-26)
- **Project:** #39

## Problem Statement
Language Identification (LID) is the task of automatically detecting which language is
spoken in an audio recording. This project classifies 3–10 second audio clips into five
Indian languages - Malayalam, Tamil, Hindi, English, and Kannada - using classical ML
classifiers with handcrafted acoustic features. No deep learning is used.

The challenge is particularly interesting because three of the five languages (Malayalam,
Tamil, Kannada) belong to the Dravidian family and share significant phonological overlap,
making acoustic-level discrimination non-trivial.

## Dataset
| Language | Source | Samples |
|----------|--------|---------|
| Malayalam | OpenSLR SLR63 - Crowdsourced multi-speaker speech | ~1000 |
| Tamil | OpenSLR SLR65 - Crowdsourced multi-speaker speech | ~1000 |
| Hindi | OpenSLR SLR103 - Hindi ASR data | ~1000 |
| English | LibriSpeech SLR12 - test-clean | ~1000 |
| Kannada | OpenSLR SLR79 - Crowdsourced multi-speaker speech | ~1000 |

All clips are 3–10 seconds of real human speech, resampled to 16kHz mono.

## Methodology
All 10 stages of the Data Science Project Life Cycle are covered:
1. Problem Definition & Literature Review
2. Data Collection & Understanding
3. Data Preprocessing & Cleaning (VAD, noise reduction, pre-emphasis, normalization)
4. Exploratory Data Analysis (spectrograms, pitch, MFCC profiles, PCA)
5. Feature Engineering & Selection (MFCC+Δ+ΔΔ, prosodic, spectral, phonotactic)
6. Model Building & Training (SVM, Random Forest, Logistic Regression)
7. Model Evaluation & Comparison (confusion matrices, ROC curves, learning curves)
8. Model Interpretation & Explainability (permutation importance, SHAP, linguistic analysis)
9. Deployment (Streamlit web app)
10. Documentation (this README, PPT, requirements)

## Features Extracted
| Feature Group | Description | Dimensions |
|---|---|---|
| MFCC + Δ + ΔΔ | Cepstral coefficients (mean & std) | 78 |
| Prosodic | Pitch, energy, speaking rate | 6 |
| Spectral | Centroid, rolloff, bandwidth | 4 |
| Phonotactic | Zero crossing rate (voiced/unvoiced proxy) | 2 |
| **Total** | | **90** |

## Results
| Model | CV Macro F1 | Test Accuracy | Test Macro F1 |
|---|---|---|---|
| SVM (RBF) | - | - | - |
| Random Forest | - | - | - |
| Logistic Regression | - | - | - |

*(Results populated after running notebook)*

## Live App
🔗 [Streamlit Deployment Link](https://your-app.streamlit.app)

## How to Run Locally
```bash
git clone https://github.com/your-repo
cd your-repo
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── project1.ipynb          # Main notebook (all 10 stages)
├── app.py                  # Streamlit deployment app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/
│   ├── model_svm.pkl
│   ├── model_rf.pkl
│   └── model_lr.pkl
├── data/
│   ├── raw/
│   │   ├── Malayalam/
│   │   ├── Tamil/
│   │   ├── Hindi/
│   │   ├── English/
│   │   └── Kannada/
│   └── processed/
│       ├── X_features.npy
│       ├── y_labels.npy
│       └── dataset_manifest.csv
├── assets/                 # Saved plots and screenshots
└── individual_profiles/    # GitHub activity screenshots
```

## Acknowledgments
- OpenSLR for open-source speech datasets
- He, F. et al. (2020) for the crowdsourced speech corpora
- LibriSpeech for English speech data
# Lang-ID-Project1
