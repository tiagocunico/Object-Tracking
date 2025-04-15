import cv2
import numpy as np
import csv
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

def create_output_directory(video_name):
    """Cria um diretório de resultados com o nome do vídeo e retorna os caminhos completos"""
    # Cria o diretório principal 'results' se não existir
    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    
    # Cria o diretório específico para este vídeo
    video_dir = f'./Results/{video_name}'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # Define os caminhos completos para os arquivos de saída
    paths = {
        'input_video': f'./Inputs/{video_name}.mp4',
        'resized_video': f'{video_dir}/{video_name}_resized.mp4',
        'optical_flow_video': f'{video_dir}/{video_name}_optical_flow.mp4',
        'csv_output_path': f'{video_dir}/{video_name}_flow_analysis.csv',
        'output_txt': f'{video_dir}/{video_name}_flow_peaks_report.txt',
        'output_plot': f'{video_dir}/{video_name}_flow_plot.png'
    }
    
    return paths

def resize_video(input_path, output_path, new_width=1066, new_height=600):
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    
    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada")
        return False
    
    # Obtém as propriedades do vídeo de entrada
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define o codec e cria o objeto VideoWriter para o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Processa cada frame do vídeo
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Redimensiona o frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Escreve o frame redimensionado no vídeo de saída
        out.write(resized_frame)
    
    # Libera os recursos
    cap.release()
    out.release()
    
    return True

def calculate_optical_flow(input_video, output_video, csv_output_path=None):
    """
    Calcula o fluxo óptico usando SIFT para detecção de pontos e Lucas-Kanade para rastreamento.
    
    Args:
        input_video (str): Caminho do vídeo de entrada
        output_video (str): Caminho do vídeo de saída
        csv_output_path (str, optional): Caminho para salvar os dados em CSV
        
    Returns:
        bool: True se bem-sucedido, False caso contrário
    """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Configuração do arquivo CSV
    if csv_output_path:
        csv_file = open(csv_output_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'timestamp_ms', 'max_flow_magnitude', 'num_keypoints'])
    
    # Inicializa o SIFT
    sift = cv2.SIFT_create(
        nfeatures=0,           # Limita o número máximo de pontos
        contrastThreshold=0.04, # Aumentado para filtrar pontos de baixo contraste
        edgeThreshold=10,       # Aumentado para ignorar bordas muito próximas
        sigma=1.6              # Suavização mais forte
    )
    
    # Parâmetros LK (ajustados para gotas)
    lk_params = dict(
        winSize=(15, 15),      # Janela maior para objetos suaves
        maxLevel=3,            # Mais níveis piramidais
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    
    # Lê o primeiro frame
    ret, old_frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame")
        return False
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta keypoints com SIFT no primeiro frame
    keypoints = sift.detect(old_gray, None)
    p0 = cv2.KeyPoint_convert(keypoints).reshape(-1, 1, 2)
    
    mask = np.zeros_like(old_frame)
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        max_magnitude = 0
        num_keypoints = 0
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if p0 is not None and len(p0) > 0:
            # Calcula o fluxo óptico
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                num_keypoints = len(good_new)
                
                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    dx, dy = a - c, b - d
                    magnitude = np.hypot(dx, dy)
                    max_magnitude = max(max_magnitude, magnitude)
                    
                    mask = cv2.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            
            img = cv2.add(frame, mask)
            out.write(img)
        else:
            # Se não há pontos para rastrear, detecta novos keypoints com SIFT
            keypoints = sift.detect(frame_gray, None)
            p0 = cv2.KeyPoint_convert(keypoints).reshape(-1, 1, 2)
            mask = np.zeros_like(old_frame)
            out.write(frame)
        
        # Escreve dados no CSV
        if csv_output_path:
            csv_writer.writerow([frame_number, timestamp_ms, max_magnitude, num_keypoints])
    
    # Liberação de recursos
    cap.release()
    out.release()
    if csv_output_path:
        csv_file.close()
    cv2.destroyAllWindows()
    
    return True

def analyze_flow_peaks(csv_path, txt_output_path, plot_output_path=None, prominence=5, distance=10):
    """
    Analisa o CSV de fluxo óptico, gera relatório TXT e gráfico dos picos.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV de entrada
        txt_output_path (str): Caminho para salvar o relatório TXT
        plot_output_path (str, optional): Caminho para salvar o gráfico (se None, não gera)
        prominence (float): Prominência mínima para considerar um pico
        distance (int): Distância mínima entre picos (em frames)
        
    Returns:
        dict: Dicionário com os resultados da análise
    """
    # Carrega os dados do CSV
    frames = []
    timestamps_ms = []
    magnitudes = []
    
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            frames.append(int(row['frame_number']))
            timestamps_ms.append(float(row['timestamp_ms']))
            magnitudes.append(float(row['max_flow_magnitude']))
    
    # Converte para arrays numpy
    magnitudes = np.array(magnitudes)
    timestamps_s = np.array(timestamps_ms) / 1000  # Converte para segundos
    
    # Detecta picos
    peak_indices, _ = find_peaks(magnitudes, prominence=prominence, distance=distance)
    
    # Calcula os intervalos entre picos
    time_intervals = []
    for i in range(1, len(peak_indices)):
        interval = timestamps_s[peak_indices[i]] - timestamps_s[peak_indices[i-1]]
        time_intervals.append(interval)
    
    # Prepara os resultados
    avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 0
    results = {
        'total_peaks': len(peak_indices),
        'average_interval': avg_interval,
        'peaks_details': [],
        'parameters': {
            'prominence': prominence,
            'distance': distance
        },
        'timestamps': timestamps_s,
        'magnitudes': magnitudes,
        'peak_indices': peak_indices
    }
    
    for idx in peak_indices:
        results['peaks_details'].append({
            'frame': frames[idx],
            'timestamp': timestamps_s[idx],
            'magnitude': magnitudes[idx]
        })
    
    # Gera o arquivo TXT
    with open(txt_output_path, 'w') as txt_file:
        # Cabeçalho
        txt_file.write("=== RELATÓRIO DE ANÁLISE DE FLUXO ÓPTICO ===\n\n")
        txt_file.write(f"Arquivo analisado: {csv_path}\n")
        txt_file.write(f"Total de picos detectados: {results['total_peaks']}\n")
        txt_file.write(f"Tempo médio entre picos: {results['average_interval']:.2f} segundos\n\n")
        
        # Tabela de picos
        txt_file.write("Detalhes dos picos:\n")
        txt_file.write("Frame   | Tempo (s) | Magnitude\n")
        txt_file.write("--------|-----------|----------\n")
        for peak in results['peaks_details']:
            txt_file.write(f"{peak['frame']:6}  | {peak['timestamp']:7.2f}  | {peak['magnitude']:8.2f}\n")
        
        # Parâmetros
        txt_file.write("\nParâmetros utilizados:\n")
        txt_file.write(f"Prominência mínima: {prominence}\n")
        txt_file.write(f"Distância mínima entre picos: {distance} frames\n")
        
        # Estatísticas adicionais
        if len(time_intervals) > 0:
            txt_file.write("\nEstatísticas dos intervalos:\n")
            txt_file.write(f"Menor intervalo: {min(time_intervals):.2f} s\n")
            txt_file.write(f"Maior intervalo: {max(time_intervals):.2f} s\n")
            txt_file.write(f"Desvio padrão: {np.std(time_intervals):.2f} s\n")
    
    # Gera o gráfico se o caminho foi especificado
    if plot_output_path:
        plt.figure(figsize=(12, 6))
        
        # Plota a linha do fluxo máximo
        plt.plot(timestamps_s, magnitudes, 'b-', label='Fluxo Máximo', linewidth=1)
        
        # Destaca os picos
        plt.plot(timestamps_s[peak_indices], magnitudes[peak_indices], 'ro', label='Picos Detectados')
        
        # Linhas verticais nos picos
        for idx in peak_indices:
            plt.axvline(x=timestamps_s[idx], color='gray', linestyle='--', alpha=0.3)
        
        # Configurações do gráfico
        plt.title('Fluxo Óptico Máximo por Tempo')
        plt.xlabel('Tempo (segundos)')
        plt.ylabel('Magnitude do Fluxo')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Salva o gráfico
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

# Nome do vídeo (sem extensão)
video_name = "SoroLinearPerto"

# Cria a estrutura de diretórios e obtém os caminhos
paths = create_output_directory(video_name)

# Processamento do vídeo
success_resize = resize_video(paths['input_video'], paths['resized_video'])

if success_resize:
    print("Vídeo redimensionado com sucesso")
    success_flow = calculate_optical_flow(paths['resized_video'], paths['optical_flow_video'], paths['csv_output_path'])
    
    if success_flow:
        print(f"Fluxo óptico calculado e salvo em {paths['optical_flow_video']}")
        print(f"Dados de fluxo salvos em {paths['csv_output_path']}")
        
        analysis_results = analyze_flow_peaks(
            paths['csv_output_path'],
            paths['output_txt'],
            paths['output_plot'],
            prominence=5,
            distance=10
        )
        print(f"Análise concluída. Relatório salvo em: {paths['output_txt']}")
        print(f"Gráfico salvo em: {paths['output_plot']}")
        print(f"Total de picos detectados: {analysis_results['total_peaks']}")
        print(f"Tempo médio entre picos: {analysis_results['average_interval']:.2f} segundos")
        
    else:
        print("Falha ao calcular o fluxo óptico")
else:
    print("Falha ao redimensionar o vídeo")