document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('recordBtn');
    const recordText = document.getElementById('recordText');
    const micWave = document.getElementById('micWave');
    const statusMessage = document.getElementById('statusMessage');
    
    const resultSection = document.getElementById('resultSection');
    const predText = document.getElementById('predText');
    const confProgress = document.getElementById('confProgress');
    const confValue = document.getElementById('confValue');
    const mfccImage = document.getElementById('mfccImage');
    const modelSelect = document.getElementById('modelSelect');
    
    // Update tulisan badge di header jika opsi model berubah
    const statusBadge = document.querySelector('.status-badge');
    modelSelect.addEventListener('change', () => {
        let modelName = modelSelect.options[modelSelect.selectedIndex].text.split(' (')[0];
        statusBadge.innerHTML = `<span class="pulse-dot"></span> Model Aktif: ${modelName}`;
    });

    let mediaRecorder;
    let audioChunks = [];
    const RECORD_DURATION = 2000; // 2 detik, menyesuaikan model
    
    // Inisialisasi MediaRecorder
    async function setupRecorder() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const rawBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = [];
                
                try {
                    // Konversi WebM/Ogg ke True PCM WAV agar librosa bisa membaca
                    const arrayBuffer = await rawBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = audioBufferToWav(audioBuffer);
                    sendData(wavBlob);
                } catch(e) {
                    console.error("Gagal convert WAV:", e);
                    sendData(rawBlob); // Fallback
                }
            };
            
        } catch (err) {
            console.error("Microphone access denied:", err);
            statusMessage.textContent = "Error: Mikrofon tidak terdeteksi atau diblokir.";
            statusMessage.style.color = "var(--danger)";
        }
    }
    
    setupRecorder();

    recordBtn.addEventListener('click', () => {
        if (!mediaRecorder) {
            alert("Harap izinkan akses mikrofon terlebih dahulu.");
            return;
        }

        // Mulai rekaman
        audioChunks = [];
        mediaRecorder.start();
        
        // Update UI state ke Recording
        recordBtn.classList.add('recording');
        micWave.classList.add('active');
        recordText.textContent = "Merekam...";
        statusMessage.textContent = "Sedang mendengarkan... (2 Detik)";
        
        // Sembunyikan hasil sebelumnya
        resultSection.style.display = 'none';
        confProgress.style.width = '0%';

        // Stop otomatis setelah 2 detik
        setTimeout(() => {
            if (mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                
                // Update UI state ke Loading (Backend Processing)
                recordBtn.classList.remove('recording');
                recordBtn.classList.add('loading');
                micWave.classList.remove('active');
                recordText.textContent = "Memproses...";
                statusMessage.textContent = "Menganalisis audio dengan AI...";
            }
        }, RECORD_DURATION);
    });

    // Kirim data ke backend Flask
    async function sendData(blob) {
        const formData = new FormData();
        // Flask mengharapkan nama field 'audio'
        formData.append('audio', blob, 'recording.wav');
        formData.append('model_type', modelSelect.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                // Tampilkan Hasil
                predText.textContent = result.prediction;
                
                // Set Bar & Value Confidence
                confValue.textContent = `${result.confidence}%`;
                setTimeout(() => {
                    confProgress.style.width = `${result.confidence}%`;
                }, 100);
                
                // Tampilkan gambar MFCC
                mfccImage.src = result.plot_url;
                
                resultSection.style.display = 'block';
                statusMessage.textContent = "Analisis selesai.";
            } else {
                statusMessage.textContent = `Error: ${result.error || 'Terjadi kesalahan'}`;
                statusMessage.style.color = "var(--danger)";
            }
        } catch (error) {
            console.error("Error submitting audio:", error);
            statusMessage.textContent = "Error komunikasi dengan server.";
            statusMessage.style.color = "var(--danger)";
        } finally {
            // Reset Button State
            recordBtn.classList.remove('loading');
            recordText.textContent = "Mulai Rekam (2 Detik)";
        }
    }
});

// Utility Function: Convert AudioBuffer to True PCM WAV Blob
function audioBufferToWav(buffer) {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);
    
    const setUint16 = (pos, data) => view.setUint16(pos, data, true);
    const setUint32 = (pos, data) => view.setUint32(pos, data, true);
    
    setUint32(0, 0x46464952); // "RIFF"
    setUint32(4, length - 8); // file length - 8
    setUint32(8, 0x45564157); // "WAVE"
    
    setUint32(12, 0x20746d66); // "fmt " chunk
    setUint32(16, 16); // length = 16
    setUint16(20, 1); // PCM (uncompressed)
    setUint16(22, numOfChan);
    setUint32(24, buffer.sampleRate);
    setUint32(28, buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(32, numOfChan * 2); // block-align
    setUint16(34, 16); // 16-bit
    
    setUint32(36, 0x61746164); // "data" - chunk
    setUint32(40, length - 44); // chunk length
    
    let offset = 44;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numOfChan; channel++) {
            let sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            view.setInt16(offset, sample, true);
            offset += 2;
        }
    }
    
    return new Blob([bufferArray], { type: 'audio/wav' });
}
