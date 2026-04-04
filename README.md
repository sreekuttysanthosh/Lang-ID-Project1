# 🗣️ Language Identification from Short Audio Clips Using MFCC and Classical ML

> **Project #39 — Predictive Analytics (AY 2025-26)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://language-project1.streamlit.app)

---

## 📌 Problem Statement

Language Identification (LID) is the task of automatically detecting which language is
spoken in an audio recording. This project classifies **3–10 second audio clips** into
**five Indian languages** — Malayalam, Tamil, Hindi, English, and Kannada — using
**classical machine learning** classifiers with **handcrafted acoustic features**.
No deep learning is used.

The challenge is particularly interesting because three of the five target languages
(Malayalam, Tamil, Kannada) belong to the **Dravidian language family** and share
significant phonological overlap, making acoustic-level discrimination non-trivial.

### Success Criterion

| Metric | Target | Rationale |
|---|---|---|
| **Macro F1-score (5-fold CV)** | ≥ 0.78 | Primary metric; handles class balance fairly across all 5 languages |

---

## 📊 Dataset

Audio data is sourced from well-established open-source speech corpora:

| Language | Source | Samples |
|---|---|---|
| Malayalam | OpenSLR SLR63 — Crowdsourced multi-speaker speech | ~1 000 |
| Tamil | OpenSLR SLR65 — Crowdsourced multi-speaker speech | ~1 000 |
| Hindi | OpenSLR SLR103 — Hindi ASR data | ~1 000 |
| English | LibriSpeech SLR12 — test-clean | ~1 000 |
| Kannada | OpenSLR SLR79 — Crowdsourced multi-speaker speech | ~1 000 |
| **Total** | | **~5 000** |

All clips are **3–10 seconds** of real human speech, resampled to **16 kHz mono**.
The dataset is balanced to ensure equitable representation across all languages.

---

## 🔬 Methodology — Full Data Science Lifecycle

All **10 stages** of the Data Science Project Life Cycle are implemented end-to-end:

| Stage | Description |
|---|---|
| 1. Problem Definition & Literature Review | Formalize LID as a multi-class classification task; survey relevant acoustic features |
| 2. Data Collection & Understanding | Acquire ~5 000 clips across 5 languages from OpenSLR/LibriSpeech |
| 3. Data Preprocessing & Cleaning | Voice Activity Detection (VAD), noise reduction, pre-emphasis, amplitude normalization |
| 4. Exploratory Data Analysis | Spectrograms, pitch contours, MFCC profiles, PCA projections |
| 5. Feature Engineering & Selection | Extract 90-dimensional feature vectors; variance-threshold + mutual-information selection |
| 6. Model Building & Training | SVM (RBF kernel), Random Forest, Logistic Regression — all classical ML |
| 7. Model Evaluation & Comparison | Confusion matrices, macro F1, ROC curves, learning curves |
| 8. Model Interpretation & Explainability | Permutation importance, SHAP values, linguistic error analysis |
| 9. Deployment | Streamlit web app with real-time audio upload & prediction |
| 10. Documentation | This README, project notebook, presentation |

---

## 🎤 Features Extracted

A total of **90 handcrafted acoustic features** are extracted from each audio clip:

| Feature Group | Description | Dimensions |
|---|---|---|
| MFCC + Δ + ΔΔ | Mel-Frequency Cepstral Coefficients (mean & std of 13 MFCCs and their first/second derivatives) | 78 |
| Prosodic | Pitch (F0 mean, std), energy (mean, std), speaking rate, duration | 6 |
| Spectral | Spectral centroid, spectral rolloff, spectral bandwidth, spectral contrast | 4 |
| Phonotactic | Zero-crossing rate mean & std (voiced/unvoiced speech proxy) | 2 |
| **Total** | | **90** |

After feature selection (variance thresholding + mutual information), the most discriminative subset is retained for model training.

---

## ✅ Results

### Model Comparison Summary

| Model | CV Macro F1 (5-fold) | Test Accuracy | Test Macro F1 | Macro AUC | Inference (ms/sample) |
|---|---|---|---|---|---|
| **SVM (RBF)** | **0.9301 ± 0.0073** | **0.921** | **0.9212** | **0.9943** | 0.18 |
| Random Forest | 0.8979 ± 0.0140 | 0.901 | 0.9011 | 0.9868 | 0.05 |
| Logistic Regression | 0.8562 ± 0.0197 | 0.860 | 0.8599 | 0.9764 | ~0.00 |

> **🏆 Best model: SVM (RBF)** — achieves a **Macro F1 of 0.92** on held-out test data,
> far exceeding the target threshold of **≥ 0.78**, with a Macro AUC of **0.99**.

### Per-Language Performance (Best Model — SVM RBF)

| Language | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Malayalam | 0.95 | 0.93 | 0.94 | 200 |
| Tamil | 0.97 | 0.95 | 0.96 | 200 |
| Hindi | 0.85 | 0.89 | 0.87 | 200 |
| English | 0.94 | 0.97 | 0.96 | 200 |
| Kannada | 0.89 | 0.86 | 0.88 | 200 |
| **Macro Avg** | **0.92** | **0.92** | **0.92** | **1 000** |

### Key Observations

- **Tamil** and **English** are the easiest to distinguish (F1 ≥ 0.96), likely due to their distinct phonological profiles.
- **Hindi** and **Kannada** are the most challenging classes (F1 ~ 0.87–0.88), with some mutual confusion — consistent with shared Indo-Aryan/Dravidian acoustic overlap.
- All three models comfortably surpass the target Macro F1 ≥ 0.78, validating the effectiveness of MFCC-based handcrafted features for LID, even without deep learning.

---

## 🌐 Live App

🔗 **[Try the deployed Streamlit app →](https://language-project1.streamlit.app)**

Upload a short audio clip (WAV format, 3–10 seconds) and get real-time language predictions with confidence scores.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/aaron43210/Lang-ID-Project1.git
cd Lang-ID-Project1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

> **Note:** The project is optimized for **Apple Silicon (ARM64)** with `n_jobs=-1` for parallel processing.
> It should work on any platform with Python 3.9+ and the listed dependencies.

---

## 📁 Project Structure

```
Lang-ID-Project1/
├── project1.ipynb              # Main notebook — full 10-stage DS lifecycle
├── app.py                      # Streamlit deployment app (real-time inference)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── PA_Project_Guidelines.docx.pdf  # Project guidelines reference
│
├── models/
│   ├── model_svm.pkl           # Trained SVM (RBF) classifier  ← best model
│   ├── model_rf.pkl            # Trained Random Forest classifier
│   └── model_lr.pkl            # Trained Logistic Regression classifier
│
├── data/
│   ├── raw/                    # Raw audio files (organized by language)
│   │   ├── Malayalam/
│   │   ├── Tamil/
│   │   ├── Hindi/
│   │   ├── English/
│   │   └── Kannada/
│   └── processed/
│       ├── X_features.npy          # Full extracted feature matrix
│       ├── X_selected.npy          # Feature-selected matrix
│       ├── y_labels.npy            # Encoded language labels
│       ├── scaler.pkl              # Fitted StandardScaler
│       ├── feature_selector.pkl    # Fitted feature selector
│       ├── feature_names.json      # All feature names
│       ├── selected_feature_names.json  # Selected feature names
│       ├── label_classes.json      # Label encoding classes
│       ├── dataset_manifest.csv    # Per-sample metadata
│       └── model_comparison.csv    # Model comparison results
│
├── assets/                     # Saved plots and screenshots
└── individual_profiles/        # Team member GitHub activity screenshots
```

---

## 🙏 Acknowledgments

- **[OpenSLR](https://openslr.org/)** for open-source speech datasets (SLR63, SLR65, SLR79, SLR103)
- **[LibriSpeech](https://www.openslr.org/12/)** for the English speech corpus
- He, F. et al. (2020) for the crowdsourced Indian-language speech corpora
- **[Streamlit](https://streamlit.io/)** for the deployment platform
- **librosa**, **scikit-learn**, **SHAP** — core libraries powering the pipeline

---

## 📜 License

This project was developed as part of the Predictive Analytics coursework (AY 2025-26).
