import numpy as np
import cv2
import os
import csv
import time

def inRange(cordinates, limits):
    x, y = cordinates
    X_Limit, Y_Limit = limits
    return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

def calculate_flow_magnitude(flow, margin_ratio, frame_shape):
    """Calcula a magnitude do maior vetor de flow na área central"""
    height, width = frame_shape
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    
    max_magnitude = 0
    
    for i in range(margin_y, height - margin_y, 5):
        for j in range(margin_x, width - margin_x, 5):
            if abs(flow[0][i,j]) > 0.1 or abs(flow[1][i,j]) > 0.1:
                magnitude = np.sqrt(flow[0][i,j]**2 + flow[1][i,j]**2)
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
    
    return max_magnitude

def optical_flow(old_frame, new_frame, window_size, min_quality=0.01, margin_ratio=0.1):
    height, width = old_frame.shape
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    
    mask = np.zeros_like(old_frame)
    mask[margin_y:-margin_y, margin_x:-margin_x] = 255
    masked_frame = cv2.bitwise_and(old_frame, mask)

    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(masked_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)
    old_frame = old_frame / 255
    new_frame = new_frame / 255

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)
    fy = cv2.filter2D(old_frame, -1, kernel_y)
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)

    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    if feature_list is not None:
        for feature in feature_list:
            j, i = feature.ravel()
            i, j = int(i), int(j)

            if i-w >= margin_y and i+w+1 <= old_frame.shape[0]-margin_y and j-w >= margin_x and j+w+1 <= old_frame.shape[1]-margin_x:
                I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

                if len(I_x) > 0 and len(I_y) > 0 and len(I_t) > 0:
                    b = np.reshape(I_t, (I_t.shape[0],1))
                    A = np.vstack((I_x, I_y)).T

                    if A.shape[0] >= 2:
                        try:
                            U = np.matmul(np.linalg.pinv(A), b)
                            u[i,j] = U[0][0]
                            v[i,j] = U[1][0]
                        except:
                            pass

    return (u, v)

def draw_flow(frame, flow, margin_ratio, timestamp=None, flow_value=None):
    height, width = frame.shape[:2]
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    
    max_magnitude = 0
    max_point = None
    
    # Fator de escala 2x maior para os vetores
    vector_scale = 30  # Antes era 15
    
    # Primeira passada: encontrar o vetor máximo
    for i in range(margin_y, height - margin_y, 8):
        for j in range(margin_x, width - margin_x, 8):
            if abs(flow[0][i,j]) > 0.1 or abs(flow[1][i,j]) > 0.1:
                magnitude = np.sqrt(flow[0][i,j]**2 + flow[1][i,j]**2)
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    max_point = (i, j)
    
    # Segunda passada: desenhar todos os vetores (2x maiores)
    for i in range(margin_y, height - margin_y, 8):
        for j in range(margin_x, width - margin_x, 8):
            if abs(flow[0][i,j]) > 0.1 or abs(flow[1][i,j]) > 0.1:
                start = (j, i)
                end = (int(j + vector_scale * flow[0][i,j]), int(i + vector_scale * flow[1][i,j]))
                if inRange(end, (width, height)):
                    if (i, j) == max_point:
                        cv2.arrowedLine(frame, start, end, (0, 0, 255), 2, tipLength=0.3)  # Vermelho
                    else:
                        cv2.arrowedLine(frame, start, end, (0, 255, 0), 1, tipLength=0.2)  # Verde
    
    # Desenhar retângulo da área ativa
    cv2.rectangle(frame, 
                 (margin_x, margin_y), 
                 (width - margin_x, height - margin_y),
                 (0, 255, 255), 2)
    
    # Adicionar informações textuais
    font = cv2.FONT_HERSHEY_SIMPLEX
    if flow_value is not None:
        cv2.putText(frame, f"Flow: {flow_value:.1f}", (width - 150, 30), font, 0.7, (255, 255, 255), 2)
    if timestamp is not None:
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (width - 150, 60), font, 0.7, (255, 255, 255), 2)
    
    return frame

def resize_if_large(frame, max_resolution=1000):
    height, width = frame.shape[:2]
    if height > max_resolution or width > max_resolution:
        return cv2.resize(frame, (int(width/2), int(height/2)))
    return frame

def select_peaks(potential_peaks, threshold_ratio, min_interval, avg_flow):
    """Seleciona picos válidos, ignorando extremos (8x acima da média)"""
    if not potential_peaks:
        return [], 0
    
    # 1. Filtrar picos extremos
    valid_peaks = [peak for peak in potential_peaks if peak[0] < avg_flow * 8]
    
    if not valid_peaks:
        return [], 0
    
    # 2. Encontrar o novo máximo válido (não necessariamente o mesmo do vídeo)
    valid_peaks_sorted = sorted(valid_peaks, key=lambda x: -x[0])
    max_valid_flow = valid_peaks_sorted[0][0]
    effective_threshold = max_valid_flow * threshold_ratio
    
    # 3. Selecionar picos que passam no threshold, respeitando o intervalo mínimo
    selected = []
    last_peak_time = -min_interval
    
    for peak in sorted(valid_peaks, key=lambda x: x[2]):  # Ordenar por tempo
        flow, frame_num, timestamp, frame = peak
        
        if flow >= effective_threshold and (timestamp - last_peak_time) >= min_interval:
            selected.append(peak)
            last_peak_time = timestamp
    
    return selected, max_valid_flow

def calculate_time_stats(timestamps):
    if len(timestamps) < 2:
        return 0, 0
    
    intervals = []
    for i in range(1, len(timestamps)):
        intervals.append(timestamps[i] - timestamps[i-1])
    
    avg_interval = np.mean(intervals)
    avg_frequency = 1.0 / avg_interval if avg_interval > 0 else 0
    
    return avg_interval, avg_frequency

def process_video(input_path, output_path, window_size=3, min_quality=0.01, 
                 peak_threshold_ratio=0.75, margin_ratio=0.1, min_peak_interval=1.0):
    
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diff_frames_dir = "./Results/MaiorDiferencaFlow"
    os.makedirs(diff_frames_dir, exist_ok=True)
    
    all_flows = []
    potential_peaks = []
    
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    need_resize = original_height > 1000 or original_width > 1000
    width = int(original_width/2) if need_resize else original_width
    height = int(original_height/2) if need_resize else original_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps/2, (width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Erro ao ler vídeo")
        return
    
    if need_resize:
        prev_frame = resize_if_large(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_number = 0
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        if need_resize:
            curr_frame = resize_if_large(curr_frame)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        
        # Calcular optical flow
        u, v = optical_flow(prev_gray, curr_gray, window_size, min_quality, margin_ratio)
        
        # Usar a magnitude do maior vetor como medida
        max_flow = calculate_flow_magnitude((u, v), margin_ratio, prev_gray.shape)
        all_flows.append((frame_number, timestamp, max_flow))
        
        # Desenhar fluxo e informações no frame (vetores 2x maiores)
        flow_frame = draw_flow(curr_frame.copy(), (u, v), margin_ratio, timestamp, max_flow)
        potential_peaks.append((max_flow, frame_number, timestamp, flow_frame.copy()))
        
        # Escrever frame no vídeo de saída
        out.write(flow_frame)
        out.write(flow_frame)  # Escreve duas vezes para reduzir velocidade
        
        prev_gray = curr_gray.copy()
        frame_number += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calcular média de flow (excluindo zeros)
    non_zero_flows = [f[2] for f in all_flows if f[2] > 0]
    avg_flow = np.mean(non_zero_flows) if non_zero_flows else 0
    
    # Selecionar picos válidos (ignorando extremos)
    selected_peaks, max_valid_flow = select_peaks(potential_peaks, 
                                                peak_threshold_ratio,
                                                min_peak_interval, 
                                                avg_flow)
    
    # Calcular estatísticas de tempo
    timestamps = [p[2] for p in selected_peaks]
    avg_interval, avg_frequency = calculate_time_stats(timestamps)
    
    # Salvar resultados
    save_results(selected_peaks, diff_frames_dir, time.time()-start_time,
                (original_width, original_height), (width, height),
                all_flows, peak_threshold_ratio * max_valid_flow, 
                max_valid_flow, avg_flow, margin_ratio, 
                peak_threshold_ratio, min_peak_interval,
                avg_interval, avg_frequency)

def save_results(selected_peaks, output_dir, processing_time, 
                original_res, processed_res, all_flows,
                threshold, max_flow, avg_flow, margin_ratio,
                peak_threshold_ratio, min_peak_interval,
                avg_interval, avg_frequency):
    
    # Salvar frames dos picos
    for i, peak in enumerate(selected_peaks, 1):
        flow, frame_num, timestamp, frame = peak
        annotated_frame = frame.copy()
        
        # Adicionar informações textuais
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, f"Pico {i}", (20, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Flow: {flow:.1f} ({flow/max_flow*100:.1f}%)", (20, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Time: {timestamp:.2f}s", (20, 90), font, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Média: {avg_flow:.1f}", (20, 120), font, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(f"{output_dir}/pico_{i}.png", annotated_frame)
    
    # Preparar metadados
    metadata = [
        f"=== Análise Baseada em Optical Flow ===",
        f"Tempo processamento: {processing_time:.2f}s",
        f"Resolução original: {original_res[0]}x{original_res[1]}",
        f"Resolução processada: {processed_res[0]}x{processed_res[1]}",
        f"Margem ignorada: {margin_ratio*100:.0f}% das bordas",
        "",
        f"Flow médio: {avg_flow:.1f}",
        f"Flow máximo válido: {max_flow:.1f} (excluindo extremos)",
        f"Threshold: {threshold:.1f} ({peak_threshold_ratio*100:.0f}% do máximo válido)",
        f"Intervalo mínimo entre picos: {min_peak_interval:.2f}s",
        f"Limite para picos extremos: {avg_flow*8:.1f} (8x a média)",
        "",
        f"Total picos detectados: {len(selected_peaks)}",
        f"Tempo médio entre picos: {avg_interval:.3f}s",
        f"Frequência média: {avg_frequency:.3f}Hz",
        "",
        "=== Picos Detectados ==="
    ]
    
    if selected_peaks:
        selected_peaks.sort(key=lambda x: x[2])
        for i, peak in enumerate(selected_peaks, 1):
            metadata.append(
                f"Pico {i}: Frame {peak[1]} | Tempo: {peak[2]:.3f}s | Flow: {peak[0]:.1f} ({peak[0]/max_flow*100:.1f}%)"
            )
    else:
        metadata.append("Nenhum pico significativo detectado")
    
    # Salvar metadados
    with open(f"{output_dir}/metadata.txt", 'w') as f:
        f.write("\n".join(metadata))
    
    # Salvar flows em CSV
    with open(f"{output_dir}/flows.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Tempo(s)', 'Flow'])
        writer.writerows(all_flows)

if __name__ == "__main__":
    process_video(
        input_path="./Inputs/soro.mp4",
        output_path="./Results/fluxosoro.mp4",
        margin_ratio=0.25,
        peak_threshold_ratio=0.85,
        min_peak_interval=3.0
    )