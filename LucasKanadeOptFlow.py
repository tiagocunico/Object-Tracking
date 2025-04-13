import numpy as np
import cv2
import os
import heapq
import csv
import time
from collections import deque

def inRange(cordinates, limits):
    x, y = cordinates
    X_Limit, Y_Limit = limits
    return 0 <= x and x < X_Limit and 0 <= y and y < Y_Limit

def calculate_frame_difference(prev_frame, curr_frame):
    """Calcula a diferença máxima entre dois frames"""
    diff = cv2.absdiff(prev_frame, curr_frame)
    return np.max(diff)

def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):
    max_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

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

            if i-w >= 0 and i+w+1 <= old_frame.shape[0] and j-w >= 0 and j+w+1 <= old_frame.shape[1]:
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

def resize_if_large(frame, max_resolution=1000):
    """Redimensiona o frame se for maior que max_resolution"""
    height, width = frame.shape[:2]
    if height > max_resolution or width > max_resolution:
        new_height = int(height / 2)
        new_width = int(width / 2)
        return cv2.resize(frame, (new_width, new_height))
    return frame

def process_video(input_path, output_path, window_size=3, min_quality=0.01):
    start_time = time.time()
    
    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    diff_frames_dir = "./Results/MaiorDiferencaSoro"
    os.makedirs(diff_frames_dir, exist_ok=True)
    
    # Data structures
    all_diffs = []
    potential_peaks = []
    min_peak_interval = 1  # Em segundos
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if we need to resize
    need_resize = original_height > 1000 or original_width > 1000
    if need_resize:
        width = int(original_width / 2)
        height = int(original_height / 2)
    else:
        width = original_width
        height = original_height
    
    # Reduce speed by half
    new_fps = original_fps / 2
    
    # Video writer for optical flow
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file")
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
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # in seconds
        
        # Calculate frame difference
        diff = calculate_frame_difference(prev_gray, curr_gray)
        
        # Store all differences
        all_diffs.append((frame_number, timestamp, diff))
        
        # Consider as potential peak
        potential_peaks.append((diff, frame_number, timestamp, curr_frame.copy()))
        
        # Calculate optical flow
        u, v = optical_flow(prev_gray, curr_gray, window_size, min_quality)
        
        # Draw optical flow on frame
        flow_frame = curr_frame.copy()
        for i in range(0, curr_gray.shape[0], 5):
            for j in range(0, curr_gray.shape[1], 5):
                if abs(u[i,j]) > 0.1 or abs(v[i,j]) > 0.1:
                    start_point = (j, i)
                    end_point = (int(j + 10 * u[i,j]), int(i + 10 * v[i,j]))
                    if inRange(end_point, (width, height)):
                        cv2.arrowedLine(flow_frame, start_point, end_point, 
                                       (0, 255, 0), 2, tipLength=0.3)
        
        # Write each frame twice to halve the speed
        out.write(flow_frame)
        out.write(flow_frame)
        
        # Update previous frame
        prev_gray = curr_gray.copy()
        frame_number += 1
    
    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Select the top 10 peaks with minimum interval
    selected_peaks = select_peaks(potential_peaks, min_peak_interval, top_n=10)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Save outputs
    save_selected_peaks(selected_peaks, diff_frames_dir, processing_time)
    save_all_differences(all_diffs, diff_frames_dir)

def select_peaks(potential_peaks, min_interval, top_n=10):
    """Seleciona os maiores picos com intervalo mínimo"""
    potential_peaks.sort(reverse=True, key=lambda x: x[0])  # Sort by difference
    selected = []
    
    for peak in potential_peaks:
        if len(selected) >= top_n:
            break
            
        # Check if this peak is far enough from selected peaks
        valid = True
        for selected_peak in selected:
            if abs(peak[2] - selected_peak[2]) < min_interval:
                valid = False
                break
                
        if valid:
            selected.append(peak)
    
    return selected

def calculate_frequencies(timestamps):
    """Calcula as frequências baseadas nos intervalos entre picos"""
    if len(timestamps) < 2:
        return []
    
    time_diffs = []
    for i in range(1, len(timestamps)):
        time_diffs.append(timestamps[i] - timestamps[i-1])
    
    frequencies = [1.0 / diff for diff in time_diffs]
    avg_frequency = np.mean(frequencies) if frequencies else 0
    
    return time_diffs, frequencies, avg_frequency

def save_selected_peaks(selected_peaks, output_dir, processing_time):
    """Salva os picos selecionados e calcula as frequências"""
    if not selected_peaks:
        print("Nenhum pico válido encontrado")
        return
    
    # Sort peaks by timestamp
    selected_peaks.sort(key=lambda x: x[2])
    
    # Calculate frequencies
    timestamps = [peak[2] for peak in selected_peaks]
    time_diffs, frequencies, avg_frequency = calculate_frequencies(timestamps)
    
    # Save peak frames
    for i, peak in enumerate(selected_peaks, 1):
        cv2.imwrite(os.path.join(output_dir, f"pico_{i}.png"), peak[3])
    
    # Create metadata
    metadata = [
        f"Processamento concluído em: {processing_time:.2f} segundos",
        f"Resolução original reduzida pela metade: {'Sim' if len(selected_peaks) > 0 and selected_peaks[0][3].shape[0] <= 500 else 'Não'}",
        "",
        "Top 10 picos com intervalo mínimo de 300ms:",
        "-----------------------------------------"
    ]
    
    for i, peak in enumerate(selected_peaks, 1):
        metadata.append(
            f"Pico {i}: Frame {peak[1]} | Timestamp: {peak[2]:.3f}s | Diferença: {peak[0]:.2f}"
        )
    
    metadata.extend([
        "",
        "Análise de Frequência:",
        "---------------------"
    ])
    
    for i in range(len(time_diffs)):
        metadata.append(
            f"Intervalo {i+1}: {time_diffs[i]:.3f}s | Frequência: {frequencies[i]:.3f}Hz"
        )
    
    metadata.extend([
        "",
        f"Frequência média: {avg_frequency:.3f}Hz",
        f"Frequência mínima: {min(frequencies):.3f}Hz" if frequencies else "N/A",
        f"Frequência máxima: {max(frequencies):.3f}Hz" if frequencies else "N/A"
    ])
    
    # Write metadata file
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write("\n".join(metadata))
    
    print(f"Picos salvos em: {output_dir}")
    print(f"Metadados salvos em: {metadata_path}")

def save_all_differences(all_diffs, output_dir):
    """Salva todas as diferenças em CSV"""
    csv_path = os.path.join(output_dir, "diferencas_maximas.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Frame', 'Timestamp (s)', 'Diferença Máxima'])
        for frame_num, timestamp, diff in all_diffs:
            writer.writerow([frame_num, f"{timestamp:.3f}", f"{diff:.2f}"])
    
    print(f"CSV com diferenças salvo em: {csv_path}")

# Process the video
input_video = "./Inputs/soro.mp4"
output_video = "./Results/fluxosoro.mp4"
process_video(input_video, output_video)