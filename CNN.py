import cv2
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf
import idx2numpy
import time
import matplotlib.pyplot as plt


def letters_extract_print(image_file: str, out_size=28):
    # ПОДГОТОВКА ФОТО

    # Первым шагом разобьем текст на отдельные буквы

    img_file = image_file
    # img_file = "123.JPG"
    # img_file = "img2.png"

    # открыли изображение
    img = cv2.imread(img_file)

    # перевод фото в ч\б

    # меняем цветовое пространство с BGR на GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img2 = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE) сразу и открывает и переводи в серый

    # threshold возвращает изображение, в котором все пиксели, которые темнее (меньше) 127 заменены на 0,
    # а все, которые ярче (больше) 127, — на 255.

    # Метод возвращает два выходных данных.
    # ret - это пороговое значение, которое было использовано (то есть 127 в моем случае),
    # а второй вывод - пороговое значение изображения.
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # адаптивный порог, который высчитывается из значения окружающих пикселей
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow("ThreshADAPTIV", th3)
    # cv2.waitKey(0)

    # увеличение изображения, cv2.erode () - увеличивает контуры, эрозия
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # определим контуры

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(hierarchy)

    output = img.copy()

    cv2.drawContours(output, contours, -1, (0, 255, 0), 3)

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print(idx)
        # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy [Next, Previous, First_Child, Parent]
        # только если родителем является нулевой контур - то есть рамка изображения!!
        if hierarchy[0][idx][3] == 0:
            print("Yes")
            # Пусть (x, y) - верхняя левая координата прямоугольника, а (w, h) - его ширина и высота.
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

            letter_crop = gray[y:y + h, x:x + w]
            # cv2.imshow("5", letter_crop)
            # print(letter_crop)
            print(letter_crop.shape)

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    # cv2.imshow("0", letters[0][2])
    # cv2.imshow("1", letters[1][2])
    # cv2.imshow("2", letters[2][2])
    # cv2.imshow("3", letters[3][2])
    # cv2.imshow("4", letters[4][2])

    # cv2.imshow("Input", img)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Erode", img_erode)
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# извлечение букв
def letters_extract(img_file: str, out_size=28):
    # img_file = "img.png"
    # img_file = "123.JPG"
    # img_file = "img2.png"

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]

            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


def emnist_model():
    model = Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def emnist_model2():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


# ЛУЧШАЯ МОДЕЛЬ!!
def emnist_model3():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model


# letters_extract_print("img.png")
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]


def emnist_train(model):
    t_start = time.time()

    emnist_path = 'datasets/'
    X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

    X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))

    # Test:
    # k = 100
    # X_train = X_train[:X_train.shape[0] // k]
    # y_train = y_train[:y_train.shape[0] // k]
    # X_test = X_test[:X_test.shape[0] // k]
    # y_test = y_test[:y_test.shape[0] // k]

    X_train = X_train[:X_train.shape[0]]
    y_train = y_train[:y_train.shape[0]]
    X_test = X_test[:X_test.shape[0]]
    y_test = y_test[:y_test.shape[0]]

    # Normalize
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,
                                                                factor=0.5,
                                                                min_lr=0.00001)

    # model = emnist_model2()
    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
              batch_size=64, epochs=30)
    print("Training done, dT:", time.time() - t_start)

    model.save('models/emnist_letters_number.h5')


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])


def img_to_str(model, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i + 1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1] / 4):
            s_out += ' '
    return s_out


if __name__ == "__main__":
    # model = emnist_model3()
    # emnist_train(model)
    # model.save('emnist_letters.h5')

    model = keras.models.load_model('models/emnist_letters3BIG.h5')
    s_out = img_to_str(model, "datasets/images/img3.png")
    print(s_out)
    # ОТВЕТ ВЫДАЕТ ВЕРНЫЙ

    # plt.plot(model.history['accuracy'],
    #          label='Доля верных ответов на обучающем наборе')
    # plt.plot(model.history['val_accuracy'],
    #          label='Доля верных ответов на проверочном наборе')
    # plt.xlabel('Эпоха обучения')
    # plt.ylabel('Доля верных ответов')
    # plt.legend()
    # plt.show()
