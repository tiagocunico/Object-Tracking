import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

# Configurações
VIDEO_PATH = os.path.join('Inputs', 'SoroLinearperto.mp4')
ROI_POSITION = (500, 400)  # (x,y) - Ajuste conforme a posição do gotejador no vídeo
ROI_SIZE = (400, 400)  # (width, height) - Área onde as gotas passam
FPS = 30                    # Taxa de quadros (ajuste conforme o vídeo)
DURATION = 30               # Duração máxima de análise em segundos

# Função para análise por zero-crossing
def zero_crossing_analysis(timestamps, signal):
    zero_crossings = []
    for i in range(1, len(signal)):
        if signal[i-1] > 0 and signal[i] <= 0:
            t_interp = timestamps[i-1] + (timestamps[i] - timestamps[i-1]) * (0 - signal[i-1]) / (signal[i] - signal[i-1])
            zero_crossings.append(t_interp)
    if len(zero_crossings) < 2:
        return 0, []
    periods = np.diff(zero_crossings)
    freq = 1 / np.mean(periods)
    return freq, zero_crossings

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir o vídeo")

# Variáveis para análise
time_series = []
timestamps = []
prev_roi = None
drop_count = 0
last_drop_time = 0

print("Iniciando análise... Pressione 'q' para encerrar")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / FPS
    
    if current_time > DURATION:
        break
    
    # Processamento da ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    
    x, y = ROI_POSITION
    w, h = ROI_SIZE
    roi = gray[y:y+h, x:x+w]
    
    # Detecção de movimento
    if prev_roi is not None:
        diff = cv2.absdiff(roi, prev_roi)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        movement = np.sum(thresh) / 255
        
        if movement > 50 and (current_time - last_drop_time) > 0.5:
            drop_count += 1
            last_drop_time = current_time
            cv2.putText(frame, "GOTA DETECTADA", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    prev_roi = roi.copy()
    time_series.append(np.mean(roi))
    timestamps.append(current_time)
    
    # Visualização
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"Gotas: {drop_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Analise de Gotejamento', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Análise de frequência (se houver dados suficientes)
if len(timestamps) > 10:
    print("\nProcessando dados...")
    
    # Pré-processamento
    time_series = np.array(time_series)
    timestamps = np.array(timestamps)
    detrended = signal.detrend(time_series)
    
    # Filtro IIR passa-banda (0.1 Hz a 2 Hz)
    nyquist = 0.5 * FPS
    b, a = signal.butter(2, [0.1/nyquist, 2.0/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, detrended)
    
    # Método 1: Contagem de picos
    peaks, _ = signal.find_peaks(filtered, height=0.5*np.max(filtered), distance=FPS/2)
    n_peaks = len(peaks)
    if n_peaks > 1:
        avg_period_peaks = np.mean(np.diff(timestamps[peaks]))
        freq_peaks = 1 / avg_period_peaks
    else:
        freq_peaks = 0
    
    # Método 2: Zero-Crossing Interpolation
    freq_zc, crossings = zero_crossing_analysis(timestamps, filtered)
    
    # Método 3: FFT
    n = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(n, 1/FPS)[:n//2]
    magnitude = np.abs(yf[0:n//2])
    magnitude[0] = 0  # Remove DC
    dominant_freq_fft = xf[np.argmax(magnitude)]
    
    # Resultados
    print("\n=== Resultados ===")
    print(f"1. Contagem de Picos:")
    print(f"   - Gotas detectadas: {drop_count}")
    print(f"   - Frequência: {freq_peaks:.3f} Hz")
    print(f"   - Período médio: {1/freq_peaks:.2f} s" if freq_peaks > 0 else "")
    
    print(f"\n2. Zero-Crossing Interpolation:")
    print(f"   - Cruzamentos detectados: {len(crossings)}")
    print(f"   - Frequência: {freq_zc:.3f} Hz")
    print(f"   - Período médio: {1/freq_zc:.2f} s" if freq_zc > 0 else "")
    
    print(f"\n3. Análise por FFT:")
    print(f"   - Frequência dominante: {dominant_freq_fft:.3f} Hz")
    
    # Plotagem comparativa
    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, filtered, label='Sinal filtrado')
    plt.plot(timestamps[peaks], filtered[peaks], "rx", label='Picos detectados')
    plt.title(f'Método 1: Contagem de Picos - {freq_peaks:.3f} Hz')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, filtered, label='Sinal filtrado')
    plt.plot(crossings, np.zeros(len(crossings)), 'go', label='Cruzamentos por zero')
    plt.title(f'Método 2: Zero-Crossing - {freq_zc:.3f} Hz')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(xf, magnitude)
    plt.title(f'Método 3: FFT - Frequência Dominante: {dominant_freq_fft:.3f} Hz')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 2)
    
    plt.tight_layout()
    plt.show()
else:
    print("Dados insuficientes para análise")