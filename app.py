"""
Advanced Language Identification App - Project #39
Predictive Analytics AY 2025-26
"""
import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import tempfile, os, json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import librosa.display
import warnings
import noisereduce as nr
from streamlit_mic_recorder import mic_recorder
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 16000
TARGET_DURATION = 5.0
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Load the actual label classes used during training (alphabetical order)
_label_classes_path = "data/processed/label_classes.json"
if os.path.exists(_label_classes_path):
    LANGUAGES = json.load(open(_label_classes_path))
else:
    # Fallback: must match the alphabetical order used by LabelEncoder during training
    LANGUAGES = ['English', 'Hindi', 'Kannada', 'Malayalam', 'Tamil']

LANGUAGE_COLORS = {'Malayalam':'#E74C3C','Tamil':'#3498DB','Hindi':'#2ECC71',
                   'English':'#F39C12','Kannada':'#9B59B6'}

st.set_page_config(page_title="LID Deep Analysis", page_icon="🗣️", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .predict-box { padding: 20px; border-radius: 15px; border-left: 10px solid #2ECC71; background-color: #e9f7ef; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all_models():
    models = {}
    for name, label in [('rf', 'Random Forest'), ('svm', 'SVM'), ('lr', 'Logistic Regression')]:
        path = f"models/model_{name}.pkl"
        if os.path.exists(path):
            models[label] = joblib.load(path)
    return models

@st.cache_resource
def load_assets():
    scaler = joblib.load("data/processed/scaler.pkl") if os.path.exists("data/processed/scaler.pkl") else None
    selector = joblib.load("data/processed/feature_selector.pkl") if os.path.exists("data/processed/feature_selector.pkl") else None
    if os.path.exists("data/processed/selected_feature_names.json"):
        feat_names = json.load(open("data/processed/selected_feature_names.json"))
    else:
        feat_names = []
    return scaler, selector, feat_names

def preprocess_audio(y, sr):
    target_len = int(SAMPLE_RATE * TARGET_DURATION)

    # Step 1: Load and resample to 16kHz mono (already handled partially via librosa.load, but just in case)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Step 2: VAD - trim silence (Relaxed for outsource audio to not cut off quiet speakers)
    y, _ = librosa.effects.trim(y, top_db=45, frame_length=2048, hop_length=512)
    intervals = librosa.effects.split(y, top_db=45)
    if len(intervals) > 0:
        y = np.concatenate([y[start:end] for start, end in intervals])

    if len(y) < int(0.5 * SAMPLE_RATE):
        # Instead of crashing on short outsource clips, pad them up to 0.5s
        pad_len = int(0.5 * SAMPLE_RATE) - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    # Step 3: Noise reduction (Make safer for very short clips)
    noise_sample = y[:int(0.5 * SAMPLE_RATE)] if len(y) > int(0.5 * SAMPLE_RATE) else y
    try:
        y = nr.reduce_noise(y=y, sr=SAMPLE_RATE, y_noise=noise_sample,
                            stationary=False, prop_decrease=0.75)
    except Exception:
        pass # Skip noise reduction if it fails on weird outsource audio

    # Step 4: Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Step 5: Fixed-length (center crop or symmetric pad)
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        pad = target_len - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode='constant')

    # Step 6: RMS normalization
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y / (rms + 1e-9) * 0.1

    return y.astype(np.float32)

def extract_features(y):
    """Extract acoustic feature vector — must match training pipeline exactly."""
    features = []

    # 1. MFCC with delta and delta-delta (78 features)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, mfcc_d, mfcc_d2])  # (39, T)
    features.extend(np.mean(mfcc_all, axis=1))  # 39 means
    features.extend(np.std(mfcc_all, axis=1))   # 39 stds

    # 2. Prosodic features (6 features)
    try:
        f0, voiced_flag, _ = librosa.pyin(y,
            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        f0_clean = f0[~np.isnan(f0)]
        pitch_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
        voiced_frac = float(np.sum(voiced_flag)) / max(len(voiced_flag), 1)
    except Exception:
        pitch_mean, pitch_std, voiced_frac = 0.0, 0.0, 0.0

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    onsets = librosa.onset.onset_detect(y=y, sr=SAMPLE_RATE, units='time',
                                         hop_length=HOP_LENGTH)
    speaking_rate = float(len(onsets) / TARGET_DURATION)

    features.extend([pitch_mean, pitch_std, voiced_frac,
                     energy_mean, energy_std, speaking_rate])

    # 3. Spectral features (4 features)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE,
                                                 hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE,
                                                hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE,
                                                    hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(centroid)), float(np.std(centroid)),
                     float(np.mean(rolloff)), float(np.mean(bandwidth))])

    # 4. Phonotactic - ZCR (2 features)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    features.extend([float(np.mean(zcr)), float(np.std(zcr))])

    return np.array(features, dtype=np.float32).reshape(1, -1)

# App UI
st.title("🗣️ Language Identification Expert System")
st.markdown("---")

models = load_all_models()
scaler, selector, selected_names = load_assets()

if not models:
    st.error("⚠️ No models found in `models/`. Please train them first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    selected_model_name = st.selectbox("Primary Classifier", list(models.keys()))
    show_comparison = st.checkbox("Show Multi-Model Comparison", value=True)
    st.markdown("---")
    st.info("This system uses 90 acoustic features to distinguish language patterns.")

audio_bytes = None

tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])

with tab_upload:
    uploaded = st.file_uploader("Upload Audio (WAV/MP3/FLAC)", type=["wav","mp3","flac"])
    if uploaded:
        # Use getvalue() instead of read() so we don't accidentally drain the Stream object pointer on a UI rerun
        audio_bytes = uploaded.getvalue()

with tab_record:
    st.info("Record a 3-10 second clip of you speaking.")
    audio_record = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", format="wav", key='recorder')
    if audio_record and len(audio_record.get('bytes', b'')) > 0:
        # Prioritize recording if currently captured
        audio_bytes = audio_record['bytes']

if audio_bytes:
    # Use proper suffix depending on what format the recorder actually returned
    suffix = ".wav" if not audio_record else f".{audio_record.get('format', 'wav')}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        y_proc = preprocess_audio(y, sr)
        raw_feats = extract_features(y_proc)
        
        # Processed features for the model
        proc_feats = scaler.transform(raw_feats) if scaler else raw_feats
        final_feats = selector.transform(proc_feats) if selector else proc_feats
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Waveform & Spectrogram")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            librosa.display.waveshow(y_proc, sr=SAMPLE_RATE, ax=ax[0], color='#3498DB')
            ax[0].set_title("Standardized Waveform")
            S = librosa.feature.melspectrogram(y=y_proc, sr=SAMPLE_RATE)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', ax=ax[1])
            ax[1].set_title("Mel Spectrogram")
            plt.tight_layout()
            st.pyplot(fig)
            st.audio(path)

        with col2:
            model = models[selected_model_name]
            probas = model.predict_proba(final_feats)[0]
            best_idx = np.argmax(probas)
            
            st.markdown(f"""
                <div class='predict-box'>
                    <h3>Primary Prediction: {LANGUAGES[best_idx]}</h3>
                    <p>Confidence: <b>{probas[best_idx]*100:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probabilities chart
            fig_prob = go.Figure(go.Bar(
                x=probas*100, y=LANGUAGES, orientation='h',
                marker_color=[LANGUAGE_COLORS[l] for l in LANGUAGES]
            ))
            fig_prob.update_layout(title="Class Probabilities (%)", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 Feature Analysis", "⚖️ Model Comparison"])
        
        with tab1:
            st.subheader("Acoustic Radar Analysis")
            radar_feats = ['Spectral Centroid', 'ZCR', 'RMS Energy', 'Speaking Rate', 'MFCC C2', 'Spectral Rolloff']
            radar_vals = [
                raw_feats[0, 84]/5000, raw_feats[0, 88]*10,
                raw_feats[0, 81]*5, raw_feats[0, 83]/10,
                raw_feats[0, 2]/100, raw_feats[0, 86]/8000
            ]
            fig_radar = go.Figure(data=go.Scatterpolar(r=radar_vals, theta=radar_feats, fill='toself', line_color='#E74C3C'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
            st.plotly_chart(fig_radar)
            
            with st.expander("Show All 90 Raw Features"):
                st.dataframe(pd.DataFrame(raw_feats, columns=[f"F_{i}" for i in range(90)]))

        with tab2:
            if show_comparison:
                st.subheader("Results Across Classifiers")
                comp_data = []
                for name, m in models.items():
                    p = m.predict_proba(final_feats)[0]
                    idx = np.argmax(p)
                    comp_data.append({"Model": name, "Prediction": LANGUAGES[idx], "Confidence": f"{p[idx]*100:.2f}%"})
                st.table(pd.DataFrame(comp_data))
    except Exception as e:
        import traceback
        st.error(f"Error processing audio: {e}")
        st.code(traceback.format_exc(), language="python")
    finally:
        if os.path.exists(path):
            os.unlink(path)
