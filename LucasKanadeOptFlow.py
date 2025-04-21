# Importa as bibliotecas necessárias
import cv2 # OpenCV para processamento de imagem e vídeo
import numpy as np # Numpy para operações numéricas, especialmente com arrays
import csv # Para ler e escrever arquivos CSV
from scipy.signal import find_peaks # Da SciPy, usada para encontrar picos em um sinal (a magnitude do fluxo)
import matplotlib.pyplot as plt # Para gerar gráficos
import os # Para lidar com operações do sistema de arquivos (criar diretórios, verificar existência)

# ==============================================================================
# FUNÇÃO PARA CRIAR DIRETÓRIOS E DEFINIR CAMINHOS
# ==============================================================================
def create_output_directory(video_name):
    """
    Cria um diretório de resultados com o nome do vídeo e retorna os caminhos completos para os arquivos de saída.
    Garante que a estrutura de pastas necessária exista antes de gerar os caminhos.
    """
    # Define o diretório base para todos os resultados
    results_base_dir = './Results'
    # Cria o diretório principal 'Results' se ele ainda não existir
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir) # Cria o diretório, incluindo quaisquer pais necessários

    # Define o diretório específico para os resultados deste vídeo
    video_dir = f'{results_base_dir}/{video_name}'
    # Cria o diretório específico do vídeo se ele ainda não existir
    if not os.path.exists(video_dir):
        os.makedirs(video_dir) # Cria o diretório do vídeo

    # Define um dicionário contendo os caminhos completos para os arquivos de entrada e saída
    paths = {
        'input_video': f'./Inputs/{video_name}.mp4', # Caminho esperado para o vídeo de entrada original
        'resized_video': f'{video_dir}/{video_name}_resized.mp4', # Caminho para o vídeo redimensionado de saída
        'optical_flow_video': f'{video_dir}/{video_name}_optical_flow.mp4', # Caminho para o vídeo de saída com visualização do fluxo
        'csv_output_path': f'{video_dir}/{video_name}_flow_analysis.csv', # Caminho para o arquivo CSV de dados de fluxo
        'output_txt': f'{video_dir}/{video_name}_flow_peaks_report.txt', # Caminho para o arquivo TXT do relatório de picos
        'output_plot': f'{video_dir}/{video_name}_flow_plot.png' # Caminho para o arquivo PNG do gráfico de fluxo
    }

    # Retorna o dicionário de caminhos
    return paths

# ==============================================================================
# FUNÇÃO PARA REDIMENSIONAR VÍDEO
# ==============================================================================
def resize_video(input_path, output_path, new_width=1066, new_height=600):
    """
    Redimensiona um vídeo para as novas dimensões especificadas.

    Args:
        input_path (str): Caminho para o vídeo de entrada.
        output_path (str): Caminho para salvar o vídeo redimensionado.
        new_width (int): Nova largura desejada.
        new_height (int): Nova altura desejada.

    Returns:
        bool: True se o redimensionamento for bem-sucedido, False caso contrário.
    """
    # Abre o vídeo de entrada para leitura
    cap = cv2.VideoCapture(input_path)

    # Verifica se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada")
        return False # Retorna False em caso de erro na abertura

    # Obtém as propriedades do vídeo de entrada (FPS e contagem de frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # A contagem de frames nem sempre é confiável, mas é útil para informação
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define o codec a ser usado para o arquivo de saída (.mp4 nesse caso)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Cria um objeto VideoWriter para escrever o vídeo redimensionado
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Verifica se o VideoWriter foi criado corretamente
    if not out.isOpened():
        print(f"Erro ao criar o arquivo de vídeo de saída: {output_path}")
        cap.release()
        return False # Retorna False em caso de erro

    # Loop para processar cada frame do vídeo
    print(f"Iniciando redimensionamento para {new_width}x{new_height}...")
    while True:
        # Lê o próximo frame do vídeo
        ret, frame = cap.read()

        # Se ret for False, significa que não há mais frames (fim do vídeo)
        if not ret:
            break # Sai do loop quando o vídeo terminar

        # Redimensiona o frame lido para as novas dimensões usando interpolação padrão (bilinear)
        # cv2.resize(frame, (new_width, new_height))
        # Usando INTER_AREA é geralmente melhor para diminuir o tamanho, previne moiré
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


        # Escreve o frame redimensionado no arquivo de vídeo de saída
        out.write(resized_frame)

    # Libera os recursos de leitura e escrita de vídeo
    cap.release()
    out.release()

    # Retorna True indicando sucesso
    print("Redimensionamento concluído.")
    return True

# ==============================================================================
# FUNÇÃO PARA CALCULAR O FLUXO ÓPTICO E SALVAR DADOS
# ==============================================================================
def calculate_optical_flow(input_video, output_video, csv_output_path=None,
                           # Parâmetros para goodFeaturesToTrack (Shi-Tomasi)
                           maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7,
                           # Parâmetros para calcOpticalFlowPyrLK (Lucas-Kanade Piramidal)
                           lk_winSize=(15, 15), lk_maxLevel=2, lk_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    """
    Calcula o fluxo óptico esparso usando o método Lucas-Kanade com detecção de features Shi-Tomasi.
    Visualiza o fluxo em um vídeo de saída e salva a magnitude máxima do fluxo por frame em um CSV.

    Args:
        input_video (str): Caminho do vídeo de entrada (geralmente o redimensionado).
        output_video (str): Caminho do vídeo de saída com a visualização do fluxo.
        csv_output_path (str, optional): Caminho para salvar os dados de magnitude em CSV.
        maxCorners (int): Máximo de features (cantos) a detectar.
        qualityLevel (float): Qualidade mínima aceitável para um canto.
        minDistance (int): Distância Euclidiana mínima entre cantos detectados.
        blockSize (int): Tamanho da vizinhança considerada para o cálculo do canto.
        lk_winSize (tuple): Tamanho da janela de busca para o rastreamento KLT.
        lk_maxLevel (int): Nível máximo da pirâmide de imagens.
        lk_criteria (tuple): Critério de término do algoritmo KLT.

    Returns:
        bool: True se bem-sucedido, False caso contrário.
    """
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_video)

    # Verifica se o vídeo foi aberto
    if not cap.isOpened():
        print("Erro ao abrir o vídeo de entrada para cálculo de fluxo óptico")
        return False # Retorna False em caso de erro

    # Obtém propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configura o VideoWriter para o vídeo de saída com visualização
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Verifica se o VideoWriter foi criado corretamente
    if not out.isOpened():
        print(f"Erro ao criar o arquivo de vídeo de saída para fluxo óptico: {output_video}")
        cap.release()
        return False # Retorna False em caso de erro

    # Configuração do arquivo CSV para salvar os dados de fluxo
    csv_file = None
    csv_writer = None
    if csv_output_path:
        try:
            csv_file = open(csv_output_path, 'w', newline='') # Abre o arquivo em modo escrita
            csv_writer = csv.writer(csv_file) # Cria um objeto writer
            csv_writer.writerow(['frame_number', 'timestamp_ms', 'max_flow_magnitude']) # Escreve o cabeçalho do CSV
            print(f"Arquivo CSV para dados de fluxo criado em: {csv_output_path}")
        except IOError as e:
            print(f"Erro ao abrir/criar o arquivo CSV {csv_output_path}: {e}")
            csv_writer = None # Garante que não tentará escrever se houver erro

    # --- Parâmetros para goodFeaturesToTrack (agora passados como argumentos) ---
    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=qualityLevel,
                          minDistance=minDistance,
                          blockSize=blockSize)

    # --- Parâmetros para calcOpticalFlowPyrLK (agora passados como argumentos) ---
    lk_params = dict(winSize=lk_winSize,
                     maxLevel=lk_maxLevel,
                     criteria=lk_criteria)

    # Lê o primeiro frame para inicializar o rastreamento
    ret, old_frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame do vídeo")
        cap.release() # Libera o recurso antes de sair
        if csv_file: csv_file.close()
        return False # Retorna False se não conseguiu ler o primeiro frame

    # Converte o primeiro frame para escala de cinza
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # Detecta os features (pontos) no primeiro frame usando Shi-Tomasi
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Cria uma máscara inicial para desenhar as linhas de fluxo óptico (setas)
    # Nota: No código original, essa máscara é resetada a cada frame. Para rastros persistentes,
    # ela deveria ser acumulada. Aqui ela é usada para desenhar o movimento *deste* frame.
    mask = np.zeros_like(old_frame)

    frame_number = 0 # Inicializa o contador de frames

    print("Iniciando cálculo e visualização do fluxo óptico...")
    # Loop principal para processar os frames restantes
    while True:
        # Lê o próximo frame
        ret, frame = cap.read()
        # Sai do loop se não houver mais frames
        if not ret:
            break

        frame_number += 1 # Incrementa o contador de frames
        # Obtém o timestamp do frame em milissegundos
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        max_magnitude = 0.0 # Inicializa a magnitude máxima do fluxo para este frame (usar float)

        # Converte o frame atual para escala de cinza
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Verifica se existem pontos para rastrear do frame anterior
        if p0 is not None and len(p0) > 0:
            # Calcula o fluxo óptico (rastreia p0 de old_gray para frame_gray)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Verifica se a função retornou pontos rastreados com sucesso
            if p1 is not None:
                # Seleciona apenas os pontos que foram encontrados no frame atual (status == 1)
                good_new = p1[st == 1] # Novos locais dos pontos
                good_old = p0[st == 1] # Locais antigos correspondentes

                # Redetecta features se o número cair abaixo de um limiar (ajustável)
                # Esta lógica pode ser movida para um intervalo fixo ou adaptada conforme necessário
                # No código original, a redetecção ocorria APENAS se todos os pontos fossem perdidos.
                # Vamos manter a lógica original de redetectar apenas se p1 is None ou p0 for None/empty.
                # Se good_new for muito pequeno, a lógica original não redetectaria até p1 ser completamente None.
                # Para o propósito do código original, focado em max_magnitude, isso pode ser aceitável.

                # Itera sobre os pares de pontos (novo e antigo)
                # Cria uma máscara temporária para este frame
                current_mask = np.zeros_like(frame) # Cria uma máscara em branco para este frame

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Extrai as coordenadas (desaninha o array 1x2)
                    a, b = new.ravel() # Coordenadas (x, y) do ponto no frame atual
                    c, d = old.ravel() # Coordenadas (x, y) do ponto no frame anterior

                    # Calcula o vetor de deslocamento (fluxo)
                    dx, dy = a - c, b - d
                    # Calcula a magnitude (comprimento) do vetor de fluxo
                    magnitude = np.hypot(dx, dy) # sqrt(dx^2 + dy^2)
                    # Atualiza a magnitude máxima encontrada neste frame
                    max_magnitude = max(max_magnitude, magnitude)

                    # Desenha uma seta na máscara *deste* frame para visualizar o fluxo (verde)
                    # Desenha de 'old' para 'new' (convertendo para int, pois coordenadas são float)
                    current_mask = cv2.arrowedLine(current_mask, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
                    # Desenha um círculo no novo local do ponto no frame original (vermelho)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                # Atualiza o frame anterior em escala de cinza para a próxima iteração
                old_gray = frame_gray.copy()
                # Atualiza os pontos a serem rastreados para a próxima iteração (os pontos encontrados no frame atual)
                p0 = good_new.reshape(-1, 1, 2) # Reshapes para o formato esperado (N, 1, 2)

                # Combina o frame original com a máscara que contém as setas de fluxo *deste* frame
                img = cv2.add(frame, current_mask)
                # Escreve o frame com a visualização do fluxo no vídeo de saída
                out.write(img)

            else: # Se nenhum ponto foi rastreado com sucesso (p1 is None)
                # Redetecta pontos no frame *atual* (frame_gray). Isso é mais lógico do que no frame anterior.
                # print(f"Frame {frame_number}: Nenhum ponto rastreado com sucesso. Redetectando features.") # Para debug
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                # A máscara já foi resetada implicitamente por não ser usada no loop acima com good_new
                # Escreve o frame original, pois não houve fluxo para visualizar
                out.write(frame)
                # old_gray ainda precisa ser atualizado para o próximo frame, mesmo que p0 tenha sido resetado/redetectado
                old_gray = frame_gray.copy()


        else: # Se não havia pontos para rastrear no início deste frame (p0 era None ou vazio)
            # Detecta pontos no frame *atual* (frame_gray) para começar a rastrear na próxima iteração.
            # print(f"Frame {frame_number}: Sem pontos para rastrear. Detectando features iniciais.") # Para debug
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            # A máscara já foi resetada implicitamente
            # Escreve o frame original, pois não houve fluxo para visualizar neste frame
            out.write(frame)
            # old_gray ainda precisa ser atualizado para o próximo frame
            old_gray = frame_gray.copy()


        # Escreve os dados do frame atual (número, timestamp, magnitude máxima) no arquivo CSV
        if csv_writer:
            # Garante que max_magnitude é escrito, mesmo que 0.0 se não houver pontos ou fluxo significativo
            csv_writer.writerow([frame_number, timestamp_ms, max_magnitude])

    # Liberação de recursos após o loop
    cap.release() # Libera o objeto de captura de vídeo
    out.release() # Libera o objeto de escrita de vídeo
    if csv_file:
        csv_file.close() # Fecha o arquivo CSV se ele foi aberto
        print(f"Dados de fluxo salvos com sucesso em {csv_output_path}")
    cv2.destroyAllWindows() # Fecha quaisquer janelas do OpenCV (se estivessem sendo mostradas com imshow)

    # Retorna True indicando sucesso
    print("Cálculo de fluxo óptico concluído.")
    return True

# ==============================================================================
# FUNÇÃO PARA ANALISAR EVENTOS DE FLUXO (PICOS)
# ==============================================================================
def analyze_flow_peaks(csv_path, txt_output_path, plot_output_path=None, prominence=5, distance=10):
    """
    Carrega os dados de magnitude máxima de fluxo de um CSV, detecta picos usando
    scipy.signal.find_peaks, gera um relatório TXT e um gráfico PNG.

    Args:
        csv_path (str): Caminho para o arquivo CSV contendo os dados de fluxo (gerado por calculate_optical_flow).
        txt_output_path (str): Caminho onde o relatório TXT será salvo.
        plot_output_path (str, optional): Caminho onde o gráfico PNG será salvo. Se None, o gráfico não é gerado.
        prominence (float): A proeminência mínima para ser considerada um pico. Ajuda a ignorar pequenas flutuações.
        distance (int): A distância mínima (em número de amostras/frames) entre picos consecutivos. Evita detectar o mesmo evento várias vezes.

    Returns:
        dict: Dicionário contendo os resultados da análise (total de picos, intervalo médio, etc.), ou None em caso de erro.
    """
    # Inicializa listas para armazenar os dados do CSV
    frames = []
    timestamps_ms = []
    magnitudes = []

    # Abre e lê o arquivo CSV
    try:
        with open(csv_path, 'r', newline='') as csv_file: # Adicionado newline='' por segurança
            reader = csv.DictReader(csv_file) # Usa DictReader para ler linhas como dicionários
            # Verifica se as colunas esperadas existem no cabeçalho
            if 'frame_number' not in reader.fieldnames or \
               'timestamp_ms' not in reader.fieldnames or \
               'max_flow_magnitude' not in reader.fieldnames:
                print(f"Erro: O arquivo CSV {csv_path} não contém as colunas esperadas.")
                return None

            for row in reader:
                # Converte os dados lidos para os tipos corretos e armazena nas listas
                # Usa .get() com valor padrão para evitar KeyError caso uma linha esteja malformada
                try:
                    frames.append(int(row.get('frame_number', 0)))
                    timestamps_ms.append(float(row.get('timestamp_ms', 0.0)))
                    magnitudes.append(float(row.get('max_flow_magnitude', 0.0)))
                except (ValueError, TypeError) as e:
                     print(f"Aviso: Pulando linha malformada no CSV: {row} - Erro: {e}")
                     continue # Pula para a próxima linha se houver erro na conversão


    except FileNotFoundError:
        print(f"Erro: Arquivo CSV não encontrado em {csv_path}")
        return None # Retorna None se o arquivo não existir
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV {csv_path}: {e}")
        return None # Retorna None para outros erros de leitura

    # Verifica se há dados para analisar
    if not magnitudes:
        print(f"Aviso: Nenhum dado de magnitude encontrado no CSV {csv_path}.")
        # Retorna resultados vazios mas válidos
        return {
            'total_peaks': 0,
            'average_interval': 0,
            'peaks_details': [],
            'parameters': {'prominence': prominence, 'distance': distance},
            'timestamps': np.array([]),
            'magnitudes': np.array([]),
            'peak_indices': np.array([], dtype=int),
            'time_intervals_s': []
        }


    # Converte as listas de dados em arrays NumPy para facilitar as operações
    magnitudes = np.array(magnitudes)
    timestamps_s = np.array(timestamps_ms) / 1000  # Converte timestamps de ms para segundos

    # --- Detecta picos na série de magnitude do fluxo ---
    # find_peaks retorna os índices onde os picos foram encontrados
    # prominence: A altura relativa que um pico deve ter para ser considerada. Ajuda a ignorar pequenas flutuações.
    # distance: O número mínimo de amostras (frames) entre dois picos. Evita detectar o mesmo evento várias vezes.
    peak_indices, _ = find_peaks(magnitudes, prominence=prominence, distance=distance)

    # --- Calcula os intervalos de tempo entre picos consecutivos ---
    time_intervals = []
    # Itera a partir do segundo pico
    for i in range(1, len(peak_indices)):
        # Calcula a diferença de tempo (em segundos) entre o pico atual e o anterior
        interval = timestamps_s[peak_indices[i]] - timestamps_s[peak_indices[i-1]]
        time_intervals.append(interval) # Adiciona o intervalo à lista

    # --- Calcula estatísticas e prepara os resultados ---
    # Calcula o intervalo médio, garantindo não dividir por zero se houver 0 ou 1 pico
    avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 0

    # Cria o dicionário de resultados
    results = {
        'total_peaks': len(peak_indices), # Número total de picos detectados
        'average_interval': avg_interval, # Intervalo médio entre picos
        'peaks_details': [], # Lista para armazenar os detalhes de cada pico
        'parameters': { # Parâmetros usados na detecção de picos
            'prominence': prominence,
            'distance': distance
        },
        # Inclui os dados originais e índices dos picos para o gráfico/relatório
        'timestamps': timestamps_s,
        'magnitudes': magnitudes,
        'peak_indices': peak_indices,
        'time_intervals_s': time_intervals # Adiciona os intervalos calculados
    }

    # Preenche a lista de detalhes dos picos
    for idx in peak_indices:
        results['peaks_details'].append({
            'frame': frames[idx], # Número do frame do pico
            'timestamp': timestamps_s[idx], # Tempo em segundos do pico
            'magnitude': magnitudes[idx] # Magnitude do fluxo no pico
        })

    # --- Gera o arquivo TXT de relatório ---
    try:
        with open(txt_output_path, 'w') as txt_file:
            # Escreve o cabeçalho do relatório
            txt_file.write("=== RELATÓRIO DE ANÁLISE DE FLUXO ÓPTICO ===\n\n")
            txt_file.write(f"Arquivo analisado: {os.path.basename(csv_path)}\n") # Nome base do arquivo no relatório
            txt_file.write(f"Total de picos detectados: {results['total_peaks']}\n")
            if results['total_peaks'] > 1:
                 txt_file.write(f"Tempo médio entre picos: {results['average_interval']:.2f} segundos\n\n")
            else:
                 txt_file.write("Tempo médio entre picos: N/A (menos de 2 picos)\n\n")


            # Escreve a tabela de detalhes dos picos
            if results['peaks_details']:
                txt_file.write("Detalhes dos picos:\n")
                txt_file.write("Frame   | Tempo (s) | Magnitude\n")
                txt_file.write("--------|-----------|----------\n")
                for peak in results['peaks_details']:
                    txt_file.write(f"{peak['frame']:<7} | {peak['timestamp']:<9.2f} | {peak['magnitude']:<8.2f}\n")
                txt_file.write("\n")
            else:
                 txt_file.write("Nenhum pico detectado.\n\n")


            # Escreve os parâmetros usados
            txt_file.write("\nParâmetros utilizados para detecção de picos:\n")
            txt_file.write(f"Prominência mínima: {prominence}\n")
            txt_file.write(f"Distância mínima entre picos: {distance} frames\n")

            # Escreve estatísticas adicionais dos intervalos (se houver mais de 1 pico)
            if len(time_intervals) > 0:
                txt_file.write("\nEstatísticas dos intervalos entre picos:\n")
                txt_file.write(f"Menor intervalo: {min(time_intervals):.2f} s\n")
                txt_file.write(f"Maior intervalo: {max(time_intervals):.2f} s\n")
                txt_file.write(f"Desvio padrão: {np.std(time_intervals):.2f} s\n")

        print(f"Relatório de análise salvo em: {txt_output_path}")

    except IOError as e:
        print(f"Erro ao escrever o arquivo de relatório TXT {txt_output_path}: {e}")

    # --- Gera o gráfico se o caminho foi especificado ---
    if plot_output_path:
        try:
            plt.figure(figsize=(12, 6)) # Cria uma nova figura para o gráfico

            if len(timestamps_s) > 0:
                # Plota a série temporal da magnitude máxima do fluxo
                plt.plot(timestamps_s, magnitudes, 'b-', label='Fluxo Máximo', linewidth=1)

                # Destaca os picos detectados no gráfico
                if len(peak_indices) > 0:
                     # Usa os timestamps e magnitudes nos índices onde os picos foram encontrados
                     plt.plot(timestamps_s[peak_indices], magnitudes[peak_indices], 'ro', label=f'Picos Detectados ({len(peak_indices)})')
                     # Adiciona linhas verticais nos tempos onde os picos foram detectados para melhor visualização
                     for idx in peak_indices:
                         plt.axvline(x=timestamps_s[idx], color='gray', linestyle='--', alpha=0.3)

                # Configurações visuais do gráfico
                plt.title(f'Fluxo Óptico Máximo por Tempo ({os.path.basename(csv_path).replace("_flow_analysis.csv", "")})') # Título do gráfico
                plt.xlabel('Tempo (segundos)') # Rótulo do eixo X
                plt.ylabel('Magnitude do Fluxo') # Rótulo do eixo Y
                plt.grid(True, linestyle='--', alpha=0.7) # Adiciona uma grade
                plt.legend() # Mostra a legenda
                plt.tight_layout() # Ajusta o layout para evitar cortar elementos

                # Salva a figura do gráfico em um arquivo PNG com alta resolução
                plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
                print(f"Gráfico salvo em: {plot_output_path}")
            else:
                 print("Aviso: Nenhum dado de tempo/magnitude para gerar o gráfico.")


            plt.close() # Fecha a figura para liberar memória

        except Exception as e:
            print(f"Erro ao gerar ou salvar o gráfico {plot_output_path}: {e}")


    # Retorna o dicionário de resultados da análise
    return results

# ==============================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL
# ==============================================================================
# Este bloco é executado apenas quando o script é rodado diretamente (não importado como módulo)
if __name__ == "__main__":

    # --- Configurações do Vídeo e Caminhos ---
    #SoroBase 2 
    #Aquario 3
    #Aquario2
    #SoroLinearperto 7
    #SoroLinearMedio 7
    #SoroNaoLinear 25
    #SoroMovendoCameraPerto
    #SoroMovendoCameraMedio
    #SoroBalancando 2
    #SoroLonge
    VIDEO_NAME = "Aquario2" # <<< NOME DO SEU ARQUIVO DE VÍDEO na pasta ./Inputs (sem a extensão .mp4)
    # Cria a estrutura de diretórios de saída e obtém todos os caminhos necessários
    paths = create_output_directory(VIDEO_NAME)

    print("--- Iniciando Análise de Fluxo Óptico ---")

    # Verifica se o arquivo de vídeo de entrada existe antes de continuar
    if not os.path.exists(paths['input_video']):
        print(f"Erro: Vídeo de entrada não encontrado em {paths['input_video']}")
        exit() # Sai do script se o vídeo não for encontrado

    # --- Parâmetros Ajustáveis ---
    # Estes parâmetros controlam o redimensionamento, cálculo do fluxo óptico e detecção de picos.
    # AJUSTE ESTES VALORES CONFORME NECESSÁRIO PARA O SEU VÍDEO!

    # --- Parâmetros de Redimensionamento do Vídeo (Etapa 1) ---
    RESIZE_WIDTH = 800 # Nova largura do vídeo redimensionado
    RESIZE_HEIGHT = 600 # Nova altura do vídeo redimensionado

    # --- Parâmetros para o Cálculo do Fluxo Óptico (Etapa 2) ---

    # Parâmetros para goodFeaturesToTrack (Shi-Tomasi) na detecção de Features
    # Usados para encontrar pontos de interesse para rastrear. Ajuste para capturar features nas gotas.
    FEAT_MAX_CORNERS = 0 # Número máximo de cantos a serem detectados. Aumente para mais pontos.
    FEAT_QUALITY_LEVEL = 0.009 # Limite mínimo de qualidade. Diminua para aceitar pontos mais fracos.
    FEAT_MIN_DISTANCE = 5 # Distância mínima entre pontos. Aumente para espalhar os pontos.
    FEAT_BLOCK_SIZE = 15 # Tamanho da vizinhança. Tamanhos maiores são menos sensíveis a ruído local.

    # Parâmetros para calcOpticalFlowPyrLK (Lucas-Kanade) no Rastreamento
    # Usados para rastrear os pontos detectados de um frame para o próximo.
    LK_WIN_SIZE = (21, 21)  # Tamanho da janela de busca. Maior = mais robusto a ruído/movimento, mas mais lento.
    LK_MAX_LEVEL = 3 # Nível máximo da pirâmide. Permite rastrear movimentos maiores.
    # Critério de término para o rastreamento: para quando atingir 10 iterações OU a mudança for menor que 0.03
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    # --- Parâmetros para a Análise de Picos no Fluxo (Etapa 3) ---

    # Parâmetros para find_peaks da SciPy
    # Usados para identificar os eventos de "gota" ou movimento significativo nos dados de magnitude do fluxo.
    PEAK_PROMINENCE = 3.0 # A altura relativa que um pico deve ter para ser considerado. Este é CRUCIAL para filtrar ruído e identificar gotas. Pode precisar ser ajustado significativamente (ex: de 1 a 20 ou mais).
    PEAK_DISTANCE = 15 # O número mínimo de frames entre dois picos detectados. Ajuda a não contar o mesmo evento várias vezes se ele gera vários picos próximos. Ajuste baseado na frequência das gotas.

    # --- Etapa 1: Redimensionar Vídeo ---
    print("\n--- Etapa 1: Redimensionando Vídeo ---")
    success_resize = resize_video(paths['input_video'], paths['resized_video'],
                                 new_width=RESIZE_WIDTH, new_height=RESIZE_HEIGHT)

    # --- Etapa 2: Calcular Fluxo Óptico ---
    if success_resize:
        print("\n--- Etapa 2: Calculando Fluxo Óptico ---")
        # Chama a função de cálculo de fluxo, passando os caminhos e todos os parâmetros definidos acima
        success_flow = calculate_optical_flow(paths['resized_video'], paths['optical_flow_video'], paths['csv_output_path'],
                                             maxCorners=FEAT_MAX_CORNERS,
                                             qualityLevel=FEAT_QUALITY_LEVEL,
                                             minDistance=FEAT_MIN_DISTANCE,
                                             blockSize=FEAT_BLOCK_SIZE,
                                             lk_winSize=LK_WIN_SIZE,
                                             lk_maxLevel=LK_MAX_LEVEL,
                                             lk_criteria=LK_CRITERIA)

        # --- Etapa 3: Analisar Picos no Fluxo ---
        if success_flow:
            print("\n--- Etapa 3: Analisando Picos no Fluxo Óptico ---")
            # Chama a função de análise de picos, passando os caminhos e os parâmetros de picos
            analysis_results = analyze_flow_peaks(
                paths['csv_output_path'],
                paths['output_txt'],
                paths['output_plot'],
                prominence=PEAK_PROMINENCE, # Usa a variável definida no início do bloco
                distance=PEAK_DISTANCE      # Usa a variável definida no início do bloco
            )

            # Imprime um resumo final da análise
            if analysis_results:
                print("\n--- Resumo da Análise de Picos ---")
                print(f"Total de picos detectados: {analysis_results['total_peaks']}")
                if analysis_results['total_peaks'] > 1:
                    print(f"Tempo médio entre picos: {analysis_results['average_interval']:.2f} segundos")
                else:
                    print("Apenas um ou nenhum pico detectado. Intervalo médio não calculado.")
                print(f"Relatório detalhado salvo em: {paths['output_txt']}")
                if paths['output_plot']:
                    print(f"Gráfico salvo em: {paths['output_plot']}")
            else:
                print("Falha na Etapa 3: Análise de picos não foi concluída.")

        else:
            print("\nFalha na Etapa 2: Cálculo do fluxo óptico não foi concluído.")
    else:
        print("\nFalha na Etapa 1: Redimensionamento do vídeo não foi concluído.")

    print("\n--- Script Finalizado ---")