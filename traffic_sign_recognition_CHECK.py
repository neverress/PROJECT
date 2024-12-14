'''
TestCode
(файл для настройки входного изображения и
тестирования уже обученной модели)
'''

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уменьшение вывода логов TensorFlow


# ------------------------------------------
# Исключения
# ------------------------------------------
class TrafficSignRecognitionError(Exception):
    """Базовый класс для исключений."""
    pass


class FileNotFoundError(TrafficSignRecognitionError):
    """Файл не найден."""
    pass


class InvalidModelError(TrafficSignRecognitionError):
    """Некорректная модель."""
    pass


class InvalidImageError(TrafficSignRecognitionError):
    """Некорректное изображение."""
    pass


# ------------------------------------------
# Параметры
# ------------------------------------------
threshold = 0.6  # Порог уверенности модели для отображения результата
"""float: Порог уверенности модели для отображения результата.

Определяет минимальную вероятность (значение от 0 до 1), при которой модель
считается уверенной в своём предсказании. Если вероятность предсказания ниже
этого порога, результат не отображается.
"""
labels_path = "labels.csv"  # Путь к файлу с названиями классов
"""str: Путь к файлу с названиями классов.

Файл должен быть в формате CSV и содержать два столбца:
- `ClassId`: идентификатор класса.
- `Name`: название класса.
Этот файл используется для отображения названия класса, соответствующего предсказанию модели.
"""
font = cv2.FONT_HERSHEY_SIMPLEX  # Шрифт для отображения текста на изображении
"""int: Шрифт для отображения текста на изображении.

Шрифт предоставляется библиотекой OpenCV. Значение `cv2.FONT_HERSHEY_SIMPLEX`
указывает, что используется простой шрифт без засечек. Этот шрифт используется
для отображения текстовой информации на изображении, например, названия класса
и уверенности модели.
"""


# ------------------------------------------
# Предобработка изображения
# ------------------------------------------
def preprocess(img):
    """
    Выполняет предобработку изображения перед передачей в модель.

    :param img: Исходное изображение в формате NumPy массива.
    :type img: numpy.ndarray
    :raises InvalidImageError: Если входное изображение некорректное или отсутствует.
    :returns: Предобработанное изображение.
    :rtype: numpy.ndarray
    """
    if img is None:
        raise InvalidImageError("Не удалось обработать изображение. Проверьте файл.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Преобразование в градации серого
    img = cv2.equalizeHist(img)  # Выравнивание контраста
    img = img / 255.0  # Нормализация
    return img


# ------------------------------------------
# Загрузка названий классов
# ------------------------------------------
def load_class_names(labels_path):
    """
    Загружает названия классов из CSV файла.

    :param labels_path: Путь к CSV файлу с названиями классов.
    :type labels_path: str
    :raises FileNotFoundError: Если файл с названиями классов не найден.
    :raises TrafficSignRecognitionError: Если произошла ошибка при загрузке классов.
    :return: Словарь, где ключ - идентификатор класса, а значение - его название.
    :rtype: dict
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Файл с названиями классов '{labels_path}' не найден.")

    try:
        labels = pd.read_csv(labels_path)  # Чтение файла
        class_names = labels.set_index("ClassId")[
            "Name"].to_dict()  # Преобразование в словарь
        return class_names
    except Exception as e:
        raise TrafficSignRecognitionError(f"Ошибка при загрузке классов: {e}")


# ------------------------------------------
# Классификация изображения
# ------------------------------------------
def classify_image(model_path, image_path):
    """
    Выполняет классификацию изображения с использованием обученной модели.

    :param model_path: Путь к файлу модели.
    :type model_path: str
    :param image_path: Путь к файлу изображения.
    :type image_path: str
    :raises FileNotFoundError: Если модель или изображение не найдены.
    :raises InvalidModelError: Если произошла ошибка при загрузке модели.
    :raises InvalidImageError: Если произошла ошибка при обработке изображения.
    :raises TrafficSignRecognitionError: Если произошла ошибка при классификации.
    :return: None
    """
    # Проверка модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель '{model_path}' не найдена.")

    try:
        model = load_model(model_path)
    except Exception as e:
        raise InvalidModelError(f"Ошибка при загрузке модели: {e}")

    # Загрузка названий классов
    class_names = load_class_names(labels_path)

    # Загрузка изображения
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise FileNotFoundError(f"Изображение '{image_path}' не найдено.")

    # Предобработка изображения
    try:
        img = cv2.resize(img_original, (32, 32))  # Изменение размера
        img = preprocess(img)  # Предобработка
        img = img.reshape(1, 32, 32, 1)  # Преобразование для подачи в модель
    except Exception as e:
        raise InvalidImageError(f"Ошибка при предобработке изображения: {e}")

    # Классификация
    try:
        predictions = model.predict(img)  # Применение модели к изображению
        class_id = np.argmax(predictions)  # Индекс класса с наибольшей вероятностью
        probability = np.max(predictions)  # Максимальная вероятность
    except Exception as e:
        raise TrafficSignRecognitionError(f"Ошибка при классификации: {e}")

    # Получение названия класса
    class_name = class_names.get(class_id, "Неизвестный класс")

    # Вывод результата
    if probability > threshold:  # Если вероятность превышает порог
        print(f"Класс: {class_id} ({class_name}), Вероятность: {probability:.2f}")

        # Добавление текста на изображение
        cv2.putText(
            img_original,
            f"{class_id} = {class_name}",
            (100, 35),
            font,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img_original,
            f"{round(probability * 100, 2)}%",
            (160, 75),
            font,
            0.75,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        # Отображение изображения с результатами
        cv2.imshow("Result image", img_original)

        # Закрытие окна с помощью клавиши ESC
        if cv2.waitKey(0) == 27 or cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
    else:
        print("Низкая вероятность. Попробуйте другое изображение.")
