import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

import json
import matplotlib as plt

# Carrega as configs do projeto
with open('/Users/arthurangelocencisilva/Programacao/detecta_tom_e_jerry/config.json') as config_file:
    config = json.load(config_file)
    
# Define as caminhos 
train_data_dir = config["train_data_dir"]
img_height, img_width = config["img_size"]
batch_size = 32

caminho_pasta = 'data/'
# Função para carregar imagens e rótulos

def carregar_imagens_e_rotulos(diretorio):
    imagens = []
    rotulos = []
    classes = os.listdir(diretorio)
    for classe in classes:
        caminho_classe = os.path.join(diretorio, classe)
        if os.path.isdir(caminho_classe):
            for nome_arquivo in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, nome_arquivo)
                if os.path.isfile(caminho_imagem):
                    img = load_img(caminho_imagem, target_size=(img_height, img_width))
                    img_array = img_to_array(img)
                    imagens.append(img_array)
                    rotulos.append(classe)
    return np.array(imagens), np.array(rotulos)

# Carregar imagens e rótulos
imagens, rotulos = carregar_imagens_e_rotulos(train_data_dir)

# Data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training' # Usar para treinamento
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation' # Usar para validação
)

# Verificar para garantir que as classes estejam corretas
print(train_generator.class_indices)

# Criação do modelo
def criar_modelo():
    model = Sequential()
    
    # Camada Convolucional 1
    model.add(Conv2D (32,(3,3), activation = 'relu', input_shape = (img_height, img_width, 3 )))
    model.add(MaxPooling2D (pool_size = (2,2)))
    
    # Camada Convolucional 2
    model.add(Conv2D (64,(3,3), activation = 'relu', input_shape = (img_height, img_width, 3 )))
    model.add(MaxPooling2D (pool_size = (2,2)))
    
    # Camada Convolucional 3
    model.add(Conv2D (128,(3,3), activation = 'relu', input_shape = (img_height, img_width, 3 )))
    model.add(MaxPooling2D (pool_size = (2,2)))
    
    # Camada de Flatten
    model.add(Flatten())
    
    # Camada totalmente conectada
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0,5))
    
    # Camada de Saída (4 classes: nenhum, ambos, tom, jerry)
    model.add(Dense(4, activation = 'softmax')) 
    
    # Compilando o modelo
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    model.save('models/cnn_model.h5')
    
    return model

# Criando o modelo
modelo = criar_modelo()

# Treinando o modelo
history = modelo.fit (
    train_generator,
    epochs = 5,
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size
)

# Salvando o histórico de treinamento
with open('models/training_history.json', 'w') as f:
    json.dump(history.history, f)

# Plotando a precisão e a perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(5)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treinamento')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treinamento')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda de Treinamento e Validação')
plt.show()