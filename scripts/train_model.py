import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Carregar as configs do projeto
with open('/Users/arthurangelocencisilva/Programacao/detecta_tom_e_jerry/config.json') as config_file:
    config = json.load(config_file)

train_data_dir = config["train_data_dir"]
img_height, img_width = config["img_size"]
batch_size = 32

# Carrega imagens e rotulos
def carregar_imagens_e_rotulos(diretorio):
    imagens = []
    rotulos = []
    classes = os.listdir(diretorio)
    class_indices = {classe: idx for idx, classe in enumerate(classes)}
    for classe in classes:
        caminho_classe = os.path.join(diretorio, classe)
        if os.path.isdir(caminho_classe):
            for nome_arquivo in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, nome_arquivo)
                if os.path.isfile(caminho_imagem):
                    img = load_img(caminho_imagem, target_size=(img_height, img_width))
                    img_array = img_to_array(img)
                    imagens.append(img_array)
                    rotulos.append(class_indices[classe])
    return np.array(imagens), np.array(rotulos)

# Caminho da pasta de dados
caminho_pasta = 'data/'

# Carregar imagens e rótulos
imagens, rotulos = carregar_imagens_e_rotulos(train_data_dir)

# Data augmentation
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Divide os dados entre treinamento e testes
X_train, X_test, y_train, y_test = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)

# Codificar os rótulos como categorias
y_train = to_categorical(y_train, num_classes=len(np.unique(rotulos)))
y_test = to_categorical(y_test, num_classes=len(np.unique(rotulos)))

# Criar geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# Função para criar o modelo
def criar_modelo():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(np.unique(rotulos)), activation='softmax')  # Número de classes
    ])
    
    # Compilando o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Criando o modelo
modelo = criar_modelo()

# Treinando o modelo
history = modelo.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
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
