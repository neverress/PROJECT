import os
import random
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# ------------------------------------------
# Глобальные параметры
# ------------------------------------------

path = "myData"
label_file = 'labels.csv'
image_dimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2


# ------------------------------------------
# Исключения
# ------------------------------------------

class TrafficSignRecognitionError(Exception):
    """Базовый класс для исключений в этом модуле."""
    pass


class DataNotFoundError(TrafficSignRecognitionError):
    """Исключение, если данные не найдены."""
    pass


class InvalidInputError(TrafficSignRecognitionError):
    """Исключение для некорректного ввода."""
    pass


# ------------------------------------------
# Функции обработки данных
# ------------------------------------------

def preprocess(img):
    """
    Преобразование изображения в оттенки серого, выравнивание контраста и нормализация.

    :param img: 3D NumPy массив, представляющий RGB изображение.
    :type img: numpy.ndarray
    :raises DataNotFoundError: Если входное изображение равно None.
    :returns: Предварительно обработанное изображение, нормализованное и преобразованное в градации серого.
    :rtype: numpy.ndarray
    """
    if img is None:
        raise DataNotFoundError("Изображение не найдено или повреждено.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img


def load_data():
    """
    Загрузка изображений и меток из датасета.

    :raises DataNotFoundError: Если директория с набором данных отсутствует или недоступна.
    :returns: Кортеж, содержащий наборы данных для обучения, проверки и тестирования (изображения и метки).
    :rtype: tuple
    """
    if not os.path.exists(path):
        raise DataNotFoundError(f"Папка с данными '{path}' не найдена.")

    print("Загрузка данных...")
    print("Всего классов распознано:", len(os.listdir(path)))
    print("Импортируем классы.....")

    images, labels = [], []
    count = 0

    for class_id in range(len(os.listdir(path))):
        print(count, end=" ")
        count += 1

        class_path = os.path.join(path, str(class_id))
        if not os.path.exists(class_path):
            raise DataNotFoundError(f"Папка с классом '{class_id}' не найдена.")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise DataNotFoundError(f"Не удалось загрузить изображение '{img_path}'.")
            images.append(img)
            labels.append(class_id)

    print(" ")

    images = np.array(images)
    labels = np.array(labels)

    # Разделение данных на выборки
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_ratio)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_ratio)

    print("Распределение данных:")
    print("Train", end="");
    print(X_train.shape, y_train.shape)
    print("Validation", end="")
    print(X_validation.shape, y_validation.shape)
    print("Test", end="")
    print(X_test.shape, y_test.shape)

    # Предобработка изображений
    X_train = np.array([preprocess(img) for img in X_train]).reshape(-1, 32, 32, 1)
    X_validation = np.array([preprocess(img) for img in X_validation]).reshape(-1, 32, 32, 1)
    X_test = np.array([preprocess(img) for img in X_test]).reshape(-1, 32, 32, 1)

    # ------------------------------------------ Заимствованный код
    num_of_samples = []  # Количество изображений в каждом классе
    cols = 5  # Количество колонок для показа
    num_classes = len(np.unique(y_train))

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10 * cols, 15 * num_classes))
    fig.tight_layout()

    data = pd.read_csv(label_file)

    for j in range(num_classes):  # Перебор всех классов
        # Отбор изображений текущего класса
        mask = (y_train == j)
        x_selected = X_train[mask]  # Фильтрация по булевой маске

        if len(x_selected) > 0:  # Проверяем, что в классе есть изображения
            for i in range(min(cols, len(x_selected))):  # Отображаем не более `cols` изображений
                resized_img = cv2.resize(x_selected[random.randint(0, len(x_selected) - 1)],
                                         (1024, 1024))  # Увеличение изображения до 1024*1024
                axs[j, i].imshow(resized_img, cmap=plt.get_cmap("gray"))
                axs[j, i].axis("off")
                if i == 2:  # Устанавливаем заголовок только для одной колонки
                    axs[j, i].set_title(f"Class {j}")
            num_of_samples.append(len(x_selected))
        else:
            print(f"В классе {j} нет изображений!")
    plt.show()

    # Визуализация датасета
    plt.figure(figsize=(22, 5))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Распределение изображений каждого класса в датасете")
    plt.xlabel("Номер класса")
    plt.ylabel("Количество изображений")
    plt.show()
    # ------------------------------------------ Заимствованный код

    # One-hot encoding номеров классов (преобразование в бинарные массивы)
    num_classes = len(np.unique(labels))
    y_train = to_categorical(y_train, num_classes)
    y_validation = to_categorical(y_validation, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def create_model(num_classes):
    """
    Создание модели CNN.

    :param num_classes: Количество классов дорожных знаков.
    :type num_classes: int
    :raises InvalidInputError: Если количество классов недопустимо (например, меньше 1).
    :returns: Скомпилированная модель Keras, готовая к обучению.
    :rtype: tensorflow.keras.Model
    """
    if num_classes <= 0:
        raise InvalidInputError("Количество классов должно быть больше 0.")

    model = Sequential([
        Input(shape=(32, 32, 1)),
        Conv2D(60, (5, 5), activation='relu'),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(epochs, batch_size):
    """
    Обучение модели.

    :param epochs: Количество эпох обучения.
    :type epochs: int
    :param batch_size: Размер батча.
    :type batch_size: int
    :raises InvalidInputError: Если `epochs` или `batch_size` недопустимы.
    """
    if epochs <= 0 or batch_size <= 0:
        raise InvalidInputError("Эпохи и число батчей должны быть положительными числами.")

    X_train, X_validation, X_test, y_train, y_validation, y_test = load_data()
    num_classes = y_train.shape[1]

    data_gen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=10
    )
    # data_gen.fit(X_train)

    model = create_model(num_classes)
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_validation, y_validation),
        steps_per_epoch=len(X_train) // batch_size
    )

    model.save('traffic_sign_model.keras')
    print("Модель сохранена как 'traffic_sign_model.keras'")

    # Визуализация процесса обучения
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Потери')
    plt.xlabel('Эпоха')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.show()

    # Оценка модели
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Тестовая потеря: {score[0]:.4f}")
    print(f"Тестовая точность: {score[1]:.4f}")


# ------------------------------------------
# Дополнительные функции
# ------------------------------------------


def add_image_to_dataset(image_path, class_id):
    """Добавление нового изображения в датасет.

   Добавляет новое изображение дорожного знака в набор данных в указанную директорию класса.

   :param image_path: Путь к файлу изображения.
   :type image_path: str
   :param class_id: Идентификатор класса, соответствующий дорожному знаку.
   :type class_id: str
   :raises FileNotFoundError: Если файл изображения не существует.
   :raises ValueError: Если идентификатор класса недопустим (например, отрицательный).
   :returns: None

    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл изображения '{image_path}' не найден.")
    if not str(class_id).isdigit() or int(class_id) < 0:
        raise ValueError("Номер класса должен быть неотрицательным целым числом.")

    dest_folder = f"{path}/{class_id}"
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Изображение '{image_path}' не удалось загрузить. Проверьте файл.")

    try:
        img = preprocess(cv2.resize(img, (32, 32)))
        img_name = f"{len(os.listdir(dest_folder))}.jpg"
        save_path = os.path.join(dest_folder, img_name)

        # Убедимся, что изображение корректно сохраняется
        if not cv2.imwrite(save_path, img * 255):  # Масштабируем нормализованное изображение
            raise IOError(f"Ошибка при сохранении изображения в '{save_path}'.")
    except Exception as e:
        raise RuntimeError(f"Ошибка обработки или сохранения изображения: {e}")

    print(f"Изображение добавлено в папку: {dest_folder}")


'''

Исключения
----------

.. py:class:: DataNotFoundError(Exception)

   Вызывается, если необходимые данные или файл не найдены

.. py:class:: InvalidInputError(Exception)

   Вызывается, если функция получает недопустимые входные параметры

'''

