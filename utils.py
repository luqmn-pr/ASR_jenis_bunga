import librosa
import librosa.display
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Konfigurasi Parameter (Harus sama persis dengan Training!)
SAMPLE_RATE = 16000
DURATION = 2.0  # detik
N_MFCC = 40

# ─────────────────────────────────────────────
# FUNGSI PREPROCESSING
# ─────────────────────────────────────────────

def apply_preemphasis(y, coeff=0.97):
    """Pre-emphasis: memperkuat frekuensi tinggi untuk memperjelas formant konsonan."""
    return np.append(y[0], y[1:] - coeff * y[:-1])

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=300.0, highcut=3400.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def apply_notch_filter(data, fs, freq=50.0, q=30.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return lfilter(b, a, data)

def preprocess_audio(file_path):
    """
    Pipeline preprocessing — identik dengan training.
    Urutan: Load → Pre-emphasis → Normalize → Trim → Bandpass → Notch → Pad/Trim
    """
    # 1. Load dan Resample ke 16kHz
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # 2. Pre-Emphasis (boost frekuensi tinggi, standar ASR)
    y = apply_preemphasis(y)

    # 3. Normalisasi Amplitudo
    y = librosa.util.normalize(y)

    # 4. Auto-Trim (buang keheningan di awal/akhir rekaman)
    y, _ = librosa.effects.trim(y, top_db=30)

    # 5. Bandpass Filter (300–3400 Hz, fokus frekuensi vokal)
    y = apply_bandpass_filter(y, sr)

    # 6. Notch Filter (hilangkan dengung 50Hz dari listrik)
    y = apply_notch_filter(y, sr)

    # 7. Padding / Trimming ke panjang tetap 2 detik
    max_len = int(SAMPLE_RATE * DURATION)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')

    return y, sr


# ─────────────────────────────────────────────
# FUNGSI EKSTRAKSI FITUR MFCC + CMVN
# ─────────────────────────────────────────────

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, return_sequence=True):
    """
    Ekstrak MFCC + Delta + Delta-Delta = 120 koefisien.
    - return_sequence=True  → CNN-1D & HMM: shape (TimeFrames, 120) + CMVN
    - return_sequence=False → MLP & SVM:    shape (240,) = mean + std
    """
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Gabung → (120, TimeFrames)
    combined = np.concatenate((mfcc, delta_mfcc, delta2_mfcc))

    if return_sequence:
        # Transpose → (TimeFrames, 120) untuk CNN-1D / HMM
        seq = combined.T
        # CMVN: normalisasi per-utterance (kurangi efek perbedaan speaker & mikrofon)
        seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
        return seq
    else:
        # Flatten mean+std → (240,) untuk MLP & SVM
        return np.concatenate((np.mean(combined, axis=1), np.std(combined, axis=1)))


# ─────────────────────────────────────────────
# FUNGSI PREDIKSI HMM
# ─────────────────────────────────────────────

def predict_with_hmm(hmm_models, sequence):
    """
    Prediksi kelas menggunakan HMM.
    Setiap model HMM (1 per kelas) memberikan log-likelihood.
    Kelas dengan skor tertinggi dipilih sebagai prediksi.
    Confidence dihitung menggunakan softmax dari skor.
    """
    scores = {}
    for label, model in hmm_models.items():
        try:
            scores[label] = model.score(sequence)
        except Exception:
            scores[label] = -np.inf

    predicted = max(scores, key=scores.get)

    # Softmax untuk confidence
    score_values = np.array(list(scores.values()), dtype=float)
    score_values -= score_values.max()  # stabilitas numerik
    exp_scores = np.exp(np.clip(score_values, -500, 0))
    confidence = float(exp_scores.max() / exp_scores.sum())

    return predicted, confidence


# ─────────────────────────────────────────────
# FUNGSI VISUALISASI MFCC
# ─────────────────────────────────────────────

def generate_mfcc_plot(y, sr, save_path):
    """Membuat dan menyimpan grafik spektrogram MFCC dengan tema gelap."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='magma', ax=ax)
    ax.set_title('Fitur MFCC (Mel-Frequency Cepstral Coefficients)', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    cb = fig.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(cb.ax.get_yticklabels(), color='white')

    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1e1e1e', transparent=False)
    plt.close()
