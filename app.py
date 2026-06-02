import os
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import tensorflow as tf
import joblib
from utils import preprocess_audio, extract_mfcc, generate_mfcc_plot
import uuid
import time

app = Flask(__name__)

# Konfigurasi Path
MODELS_DIR = 'models'
TEMP_DIR = os.path.join('static', 'temp')

os.makedirs(TEMP_DIR, exist_ok=True)

# Muat Model, Scaler dan Label Encoder secara global
print("Loading models (CNN-1D, MLP, SVM)...")
model_cnn = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_cnn1d.h5'))
model_mlp = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'model_mlp.h5'))
model_svm = joblib.load(os.path.join(MODELS_DIR, 'model_svm.pkl'))

scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
print("Semua dependensi berhasil diload.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'Tidak ada file audio yang dikirim.'}), 400
        
    audio_file = request.files['audio']
    model_type = request.form.get('model_type', 'cnn')
    
    if audio_file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400
        
    unique_id = str(uuid.uuid4())
    audio_path = os.path.join(TEMP_DIR, f"{unique_id}.wav")
    plot_filename = f"mfcc_{unique_id}.png"
    plot_path = os.path.join(TEMP_DIR, plot_filename)
    
    try:
        audio_file.save(audio_path)
        y, sr = preprocess_audio(audio_path)
        
        if model_type == 'cnn':
            mfcc_features = extract_mfcc(y, sr, return_sequence=True)
            input_data = np.expand_dims(mfcc_features, axis=0)
            predictions = model_cnn.predict(input_data)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions) * 100)
            
        elif model_type == 'mlp':
            mfcc_features = extract_mfcc(y, sr, return_sequence=False)
            input_data = scaler.transform([mfcc_features])
            predictions = model_mlp.predict(input_data)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions) * 100)
            
        elif model_type == 'svm':
            mfcc_features = extract_mfcc(y, sr, return_sequence=False)
            input_data = scaler.transform([mfcc_features])
            predictions = model_svm.predict_proba(input_data)
            predicted_idx = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions) * 100)
        else:
            return jsonify({'error': 'Model type tidak valid.'}), 400
        
        # Konversi index kembali ke nama kelas
        predicted_class = le.inverse_transform([predicted_idx])[0]
        
        # 5. Buat visualisasi plot MFCC
        generate_mfcc_plot(y, sr, plot_path)
        
        # Tambahkan query string waktu agar gambar tidak ter-cache browser
        plot_url = f"{url_for('static', filename='temp/' + plot_filename)}?t={int(time.time())}"
        
        # Hapus file audio sementara (bisa juga dibiarkan jika butuh log, tp kita hapus agar rapi)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        # Return response sukses
        return jsonify({
            'success': True,
            'prediction': predicted_class.capitalize(),
            'confidence': f"{confidence:.2f}",
            'plot_url': plot_url
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction:\n{error_details}")
        # Pastikan file temporary dihapus walau error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Hapus file-file lama di folder temp setiap kali server menyala
    for f in os.listdir(TEMP_DIR):
        if f.endswith('.png') or f.endswith('.wav'):
            os.remove(os.path.join(TEMP_DIR, f))
            
    app.run(debug=True, port=5000)
