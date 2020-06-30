from tensorflow import keras #импортируем библиотеку keras, наш инструмент для создания нейронных сетей
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds  # pip install tensorflow-datasets
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.get_logger().setLevel(logging.ERROR)

def mnist_make_model(image_w: int, image_h: int): #первый вариант mlp сети
    
    model = Sequential()
    model.add(Dense(800, activation='relu', input_shape=(image_w*image_h,)))#входной слой, 28*28=784 размер изображения подаваемого на вход
    model.add(Dense(10, activation='softmax'))#выходной слой, 10 нейронов тк надо определять диапозон цифр от 0 до 9
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    
    return model

def mnist_make_model2(image_w: int, image_h: int): #второй вариант mlp сети
    
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(image_w*image_h,))) #нейронов также больше, чем в первом варианте
    model.add(Dropout(0.2))  #дополнительный слой в сравнении с первым вариантом  rate 0.2 - set 20% of inputs to zero
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model

def mnist_mlp_train(model): #обучение mlp сети
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()# x_train: 60000x28x28 массив, x_test: 10000x28x28 массив
  
    image_size = x_train.shape[1]
    train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
    test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255.0
    test_data /= 255.0
    num_classes = 10# кодируем классы - всего на выходе 10 классов
                    # для 3 класса -> [0 0 0 1 0 0 0 0 0 0], а для 5 -> [0 0 0 0 0 1 0 0 0 0]
    train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
    test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
    
    print("Обучение сети...")
    
    #t_start = time.time()
    history = model.fit(train_data, train_labels_cat, epochs=25, batch_size=200, verbose=1, validation_data=(test_data, test_labels_cat))
    # начинается обучение сети
    
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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'], '--')
    plt.legend(['Точность', 'Валидация точности'])
    plt.xlabel('Эпохи (кол-во)')
    plt.ylabel('Процент точности (%)')
    plt.savefig('mlp_1024_25.png', bbox_inches='tight')
    
    

model = mnist_make_model2(image_w=28, image_h=28)#запись созданной и обученной модели в файл
mnist_mlp_train(model)
model.save('mlp_1024_25.h5')

def mlp_digits_predict(model, image_file): #распознавания картинки из файла для mlp сети
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr = img_arr.reshape((1, image_size*image_size))
    result = model.predict_classes([img_arr])
    return result[0]

model = tf.keras.models.load_model('mlp_784.h5') #загрузка модели и изображений для распознавания первой обученной сетью
print(mlp_digits_predict(model, '0.png'))
print(mlp_digits_predict(model, '1.png'))
print(mlp_digits_predict(model, '2.png'))
print(mlp_digits_predict(model, '3.png'))
print(mlp_digits_predict(model, '4.png'))
print(mlp_digits_predict(model, '5.png'))
print(mlp_digits_predict(model, '6.png'))
print(mlp_digits_predict(model, '7.png'))
print(mlp_digits_predict(model, '8.png'))
print(mlp_digits_predict(model, '9.png'))