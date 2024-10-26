import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.views.decorators.csrf import csrf_exempt
import logging

# Carregar o modelo de Machine Learning
model = load_model('../models/teste4-20epoch.h5')

logging.basicConfig(level=logging.DEBUG)

# Classes mapeadas do modelo
label_categories = {
    0: "Jerry",
    1: "Tom",
    2: "None",
    3: "Both"
}

# Variável global para armazenar o estado do vídeo (em produção use sessões)
video_state = {}

# Página inicial que carrega o HTML com lista de vídeos
def home(request):
    # Lista todos os arquivos de vídeo na pasta 'media'
    media_dir = 'media/'
    videos = [f for f in os.listdir(media_dir) if f.endswith('.mp4')]
    return render(request, 'youtube.html', {'videos': videos})

# processa um quadro de vídeo
def process_frame(frame):
    # Preprocessamento do quadro
    frame_resized = cv2.resize(frame, (256, 256))  # Tamanho de entrada do modelo
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Fazer previsão
    predictions = model.predict(frame_expanded)
    predicted_class_index = np.argmax(predictions)

    # Verificar se a classe detectada é válida 
    predicted_class = label_categories[predicted_class_index]

    return predicted_class

# Inicia o processamento do vídeo
@csrf_exempt
def start_video_processing(request):
    if request.method == 'POST':
        video_filename = request.POST.get('video_filename')
        logging.debug(f"Starting processing for video: {video_filename}")

        if not video_filename:
            return JsonResponse({'error': 'Arquivo de vídeo não fornecido'}, status=400)

        video_path = os.path.join('media', video_filename)

        if not os.path.exists(video_path):
            return JsonResponse({'error': 'Arquivo de vídeo não encontrado'}, status=404)

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return JsonResponse({'error': 'Erro ao abrir o vídeo'}, status=500)

            video_state['cap'] = cap 
            video_state['current_frame'] = 0
            return JsonResponse({'status': 'processing', 'video_filename': video_filename})
        
        except Exception as e:
            logging.error(f"Erro durante o processamento: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Método não suportado'}, status=405)

#  envia a predição de personagem a cada quadro
@csrf_exempt
def get_next_prediction(request):
    if 'cap' not in video_state:
        return JsonResponse({'error': 'Nenhum vídeo em processamento'}, status=400)

    cap = video_state['cap']
    
    try:
        ret, frame = cap.read()  # Lê o próximo quadro do vídeo
        if not ret:
            logging.debug("Fim do vídeo alcançado.")
            return JsonResponse({'status': 'complete', 'predicted_class': None})

        # Preve o personagem para o quadro atual
        predicted_class = process_frame(frame)

        # Atualiza o frame atual
        video_state['current_frame'] += 1

        return JsonResponse({'status': 'processing', 'predicted_class': predicted_class})
    
    except Exception as e:
        logging.error(f"Erro ao processar quadro: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
