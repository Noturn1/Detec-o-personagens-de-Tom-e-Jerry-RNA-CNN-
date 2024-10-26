import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Carrega as configurações do projeto
with open('config.json') as config_file:
    config = json.load(config_file)
    
img_height, img_width = config["img_size"]

# Carrega e prepara a imagem 
def carregar_imagem(caminho_imagem):
    img = image.load_img(caminho_imagem, target_size = (img_height, img_width))
    img_array = image.img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis = 0)
    
    return img_array

# Carrega o modelo treinado
modelo = tf.keras.models.load_model('models/cnn_model.h5')
modelo.compile()

# Faz a predição
def prever_personagem(caminho_imagem):
    img_array = carregar_imagem(caminho_imagem)
    print(f"Imagem carregada: {caminho_imagem}")
    print(f"Array da imagem: {img_array.shape}")
    
    predicao = modelo.predict(img_array)
    print(f"Predição bruta: {predicao}")
    
    classes = ['Nenhum', 'Tom', 'Jerry', 'Ambos']
    resultado = classes[np.argmax(predicao)]
    
    return resultado

if __name__ == "__main__":
    pasta_imagens = 'data/tom_jerry_1'
    for nome_arquivo in os.listdir(pasta_imagens):
        caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
        if os.path.isfile(caminho_imagem):
            resultado = prever_personagem(caminho_imagem)
            print(f"Personagem detectado na imagem {nome_arquivo}: {resultado}")
