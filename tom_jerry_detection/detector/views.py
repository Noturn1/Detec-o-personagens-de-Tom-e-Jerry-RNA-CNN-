import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
import yt_dlp
from django.views.decorators.csrf import csrf_exempt

# Carregar o modelo treinado
model = load_model('/home/vinicius/Documentos/Deteccao-personagens-de-Tom-e-Jerry/models/cnn_model2.h5')

def download_video(video_url):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': 'media/%(title)s.%(ext)s',  # Salva o vídeo na pasta 'media'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        video_path = ydl.prepare_filename(info_dict)
    
    return video_path

def process_frame(frame):
    # Preprocessamento do quadro
    frame_resized = cv2.resize(frame, (128, 72))  # img_width e img_height definidos anteriormente
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)

    # Fazer previsão
    predictions = model.predict(frame_expanded)
    predicted_class = np.argmax(predictions)

    return int(predicted_class)

@csrf_exempt
def detect_image(request):
    if request.method == 'POST':
        video_url = request.POST.get('video_url')
        
        # Baixar o vídeo
        video_path = download_video(video_url)

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        detected_characters = []  # Lista para armazenar previsões de cada quadro

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Processar quadro
            predicted_class = process_frame(frame)
            detected_characters.append(predicted_class)

            processed_frames += 1

            # A cada 10 quadros, enviar um progresso ao frontend
            if processed_frames % 10 == 0:
                progress = int((processed_frames / frame_count) * 100)
                return JsonResponse({
                    'status': 'processing',
                    'progress': progress,
                    'predicted_class': int(predicted_class)  # Certifique-se de que é int
                })

        cap.release()

        # Ao terminar, retorna o resultado completo
        video_filename = 'Tom & Jerry em Português ｜ Brasil ｜ A Gostosura ｜ WB Kids.mp4'  # Nome correto do arquivo
        return JsonResponse({
            'status': 'processing',
            'video_filename': video_filename,
            'predicted_class': 0  # Ou outro valor inicial
        })
    return render(request, 'youtube.html')
