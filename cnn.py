from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras


def mnist_cnn_model():#создание сверточной сети
    image_size = 28 
    num_channels = 1 #1 for grayscale images
    num_classes = 10  # количество классов, цифр для распознавания
    
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) #полносвязный слой
    model.add(Dense(128, activation='relu'))#выходной слой
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def mnist_cnn_train(model): #обучение сверточной сети
    (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()
    image_size = 28 #размер изображения
    num_channels = 1 # 1 для серых изображений

    train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels)) #изменение масштаба и формы данных изображений
    train_data = train_data.astype('float32') / 255.0 #кодируем классы - всего на выходе 10 классов
    #для 3 класса 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 10
    train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

    val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))#изменение масштаба и формы валидационных данных изображений
    val_data = val_data.astype('float32') / 255.0
    val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)#кодируем классы - всего на выходе 10 классов

    print("Обучаем сеть...")

    #t_start = time.time()# начинается обучение сети
    history = model.fit(train_data, train_labels_cat, epochs=25, batch_size=200, validation_data=(val_data, val_labels_cat))

    plt.figure(0) #создание графиков после обучения
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], '--')
    plt.title('Ошибки')
    plt.legend(['Ошибка', 'Валидация ошибки'])
    plt.xlabel('Эпохи (кол-во)')
    plt.ylabel('Процент ошибки (%)')
    plt.subplots_adjust(wspace=0.45, hspace=0.45)

    plt.subplot(2, 1, 2)
    plt.title('Точность')
    plt.legend(['Точность'])
    plt.plot(history.history['accuracy'])
    plt.xlabel('Эпохи (кол-во)')
    plt.ylabel('Процент точности (%)')
    plt.savefig('cnn_25.png', bbox_inches='tight')
    
    return model
 
model = mnist_cnn_model()#запись созданной и обученной модели в файл
mnist_cnn_train(model)
model.save('cnn_25.h5')

def cnn_digits_predict(model, image_file):#для распознавания приводим изображение к формату чб изображение 28x28 пикселей
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return result[0]

model = tf.keras.models.load_model('cnn.h5') #загрузка модели и изображений для распознавания третей обученной сетью
print(cnn_digits_predict(model, '0.png'))
print(cnn_digits_predict(model, '1.png'))
print(cnn_digits_predict(model, '2.png'))
print(cnn_digits_predict(model, '3.png'))
print(cnn_digits_predict(model, '4.png'))
print(cnn_digits_predict(model, '5.png'))
print(cnn_digits_predict(model, '6.png'))
print(cnn_digits_predict(model, '7.png'))
print(cnn_digits_predict(model, '8.png'))
print(cnn_digits_predict(model, '9.png'))