import os
import cv2
import numpy as np
import threading
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
import logging
import queue
import time

# Carregar o modelo de Machine Learning
model = load_model('../models/teste4-3c.h5')

logging.basicConfig(level=logging.DEBUG)

# Classes mapeadas do modelo
label_categories = {
    0: "Jerry",
    1: "Tom",
    # 2: "None",
    3: "Both"
}

# Variáveis globais para armazenar o estado do vídeo e das threads
video_state = {}
frame_queue = queue.Queue(maxsize=10)  # Fila de quadros para processar
result_queue = queue.Queue(maxsize=10)  # Fila para resultados de inferência
processing_thread = None
stop_thread = False

# Página inicial que carrega o HTML com lista de vídeos
def home(request):
    media_dir = 'media/'
    videos = [f for f in os.listdir(media_dir) if f.endswith('.mp4')]
    return render(request, 'youtube.html', {'videos': videos})

# Função para processar um quadro de vídeo
def process_frame(frame):
    frame_resized = cv2.resize(frame, (256, 256))  # Tamanho de entrada do modelo
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    predictions = model.predict(frame_expanded)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_categories[predicted_class_index]

    return predicted_class

# Thread para captura de quadros
def frame_capture_thread(video_path):
    global stop_thread
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Erro ao abrir o vídeo.")
        return

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            time.sleep(0.1)  # Espera caso a fila esteja cheia
        else:
            frame_queue.put(frame)

    cap.release()
    logging.debug("Captura de quadros encerrada.")

# Thread para processar quadros e gerar predições
def frame_processing_thread():
    global stop_thread
    while not stop_thread or not frame_queue.empty():
        if frame_queue.empty():
            time.sleep(0.1)
            continue
        frame = frame_queue.get()
        predicted_class = process_frame(frame)
        result_queue.put(predicted_class)

    logging.debug("Processamento de quadros encerrado.")

# Função para iniciar o processamento do vídeo
@csrf_exempt
def start_video_processing(request):
    global stop_thread, processing_thread
    if request.method == 'POST':
        video_filename = request.POST.get('video_filename')
        logging.debug(f"Starting processing for video: {video_filename}")

        if not video_filename:
            return JsonResponse({'error': 'Arquivo de vídeo não fornecido'}, status=400)

        video_path = os.path.join('media', video_filename)
        if not os.path.exists(video_path):
            return JsonResponse({'error': 'Arquivo de vídeo não encontrado'}, status=404)

        # Reinicia o estado do processamento
        stop_thread = False
        frame_queue.queue.clear()
        result_queue.queue.clear()

        # Inicia as threads de captura e processamento
        capture_thread = threading.Thread(target=frame_capture_thread, args=(video_path,))
        processing_thread = threading.Thread(target=frame_processing_thread)
        capture_thread.start()
        processing_thread.start()
        
        # Armazena as threads no estado do vídeo
        video_state['capture_thread'] = capture_thread
        video_state['processing_thread'] = processing_thread

        return JsonResponse({'status': 'processing', 'video_filename': video_filename})

    return JsonResponse({'error': 'Método não suportado'}, status=405)

# Função para enviar a predição de personagem a cada quadro
@csrf_exempt
def get_next_prediction(request):
    if result_queue.empty():
        if not video_state['capture_thread'].is_alive() and result_queue.empty():
            stop_thread = True
            logging.debug("Fim do vídeo alcançado.")
            return JsonResponse({'status': 'complete', 'predicted_class': None})
        return JsonResponse({'status': 'waiting', 'predicted_class': None})
    
    predicted_class = result_queue.get()
    return JsonResponse({'status': 'processing', 'predicted_class': predicted_class})

# Finalizar processamento ao terminar a sessão
def stop_video_processing():
    global stop_thread
    stop_thread = True
    video_state.get('capture_thread').join()
    video_state.get('processing_thread').join()
    logging.debug("Processamento de vídeo encerrado.")
