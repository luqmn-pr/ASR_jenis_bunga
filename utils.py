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

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=300.0, highcut=3400.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_notch_filter(data, fs, freq=50.0, q=30.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    y = lfilter(b, a, data)
    return y

def preprocess_audio(file_path):
    """
    Pipeline preprocessing yang identik dengan training.
    """
    # 1. Load dan Resample
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # 2. Normalisasi Amplitudo
    y = librosa.util.normalize(y)
    
    # 3. Filtering
    y = apply_bandpass_filter(y, sr)
    y = apply_notch_filter(y, sr)
    
    # 4. Padding / Trimming (Harus persis 2 detik)
    max_len = int(SAMPLE_RATE * DURATION)
    if len(y) > max_len:
        y = y[:max_len]
    else:
        pad_width = max_len - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
        
    return y, sr

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, return_sequence=True):
    """
    Ekstrak 120 koefisien (MFCC + Delta + Delta-Delta)
    Jika return_sequence=True -> untuk CNN-1D (TimeFrames, 120)
    Jika return_sequence=False -> untuk MLP dan SVM (240,)
    """
    # Ekstrak MFCC dasar
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Delta dan Delta-Delta
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Gabung menjadi (120, TimeFrames)
    combined = np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
    
    if return_sequence:
        # Transpose untuk CNN-1D: (TimeFrames, 120)
        return combined.T
    else:
        # Mean dan Std untuk MLP & SVM: (240,)
        mfcc_mean = np.mean(combined, axis=1)
        mfcc_std = np.std(combined, axis=1)
        return np.concatenate((mfcc_mean, mfcc_std))

def generate_mfcc_plot(y, sr, save_path):
    """
    Membuat grafik spektrogram MFCC dan menyimpannya.
    Desain plot disesuaikan agar elegan.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    plt.figure(figsize=(10, 4))
    # Menggunakan colormap magma/inferno yang cocok untuk tema gelap
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Fitur MFCC (Mel-Frequency Cepstral Coefficients)', color='white')
    
    # Styling untuk tema gelap
    plt.gca().set_facecolor('#1e1e1e')
    fig = plt.gcf()
    fig.patch.set_facecolor('#1e1e1e')
    
    ax = plt.gca()
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    cb = plt.colorbar()
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()
