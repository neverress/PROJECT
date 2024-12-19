import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Скрытие сообщений TensorFlow для повышения читаемости

from traffic_sign_recognition_CHECK import classify_image  # Классификация изображения
from traffic_sign_recognition_MAIN import train_model, add_image_to_dataset  # Обучение и добавление данных


class ConsoleAppError(Exception):
    """
    Базовый класс для исключений в приложении.
    """
    pass


class InvalidInputError(ConsoleAppError):
    """
    Исключение, вызываемое при некорректном вводе пользователя.
    """
    pass


class FileNotFoundError(ConsoleAppError):
    """
    Исключение, вызываемое, если указанный файл не найден.
    """
    pass


def main():
    """
    Главная функция приложения.

    Реализует консольный интерфейс для взаимодействия с приложением, включая:
    - Обучение модели.
    - Классификацию изображения.
    - Добавление изображения в датасет.
    - Выход из приложения.

    Raises:
        InvalidInputError: Если пользователь ввёл некорректные данные.
        FileNotFoundError: Если указанный файл не найден.
        ValueError: Если введены данные, которые невозможно преобразовать в числовое значение.
        Exception: Для других непредвиденных ошибок.
    """
    while True:
        try:
            print("\nМеню:")
            print("1. Обучить модель")
            print("2. Классифицировать изображение")
            print("3. Добавить изображение в датасет")
            print("4. Выход")

            choice = input("Выберите действие: ").strip()
            if choice == '1':
                # Обучение модели
                epochs = int(input("Введите количество эпох: ").strip())
                if epochs <= 0:
                    raise InvalidInputError("Количество эпох должно быть положительным числом.")

                batch_size = int(input("Введите размер батча: ").strip())
                if batch_size <= 0:
                    raise InvalidInputError("Размер батча должен быть положительным числом.")

                print(f"Обучение модели на {epochs} эпохах с размером батча={batch_size}...")
                train_model(epochs, batch_size)
                print("Обучение завершено.")

            elif choice == '2':
                # Классификация изображения
                model_path = input("Введите путь к модели: ").strip()
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Файл модели '{model_path}' не найден.")

                image_path = input("Введите путь к изображению: ").strip()
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Файл изображения '{image_path}' не найден.")

                classify_image(model_path, image_path)
                print("Классификация завершена.")

            elif choice == '3':
                # Добавление изображения в датасет
                image_path = input("Введите путь к изображению: ").strip()
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Файл изображения '{image_path}' не найден.")

                class_id = input("Введите номер класса (или новый номер): ").strip()
                if not class_id.isdigit() or int(class_id) < 0:
                    raise InvalidInputError("Номер класса должен быть неотрицательным числом.")

                add_image_to_dataset(image_path, int(class_id))
                print("Изображение добавлено в датасет.")

            elif choice == '4':
                # Выход из программы
                print("Выход из программы.")
                break
            else:
                raise InvalidInputError("Неверный выбор. Попробуйте снова.")

        except InvalidInputError as e:
            print(f"Ошибка ввода: {e}")
        except FileNotFoundError as e:
            print(f"Ошибка: {e}")
        except ValueError as e:
            print(f"Ошибка: Ожидалось числовое значение. {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")


if __name__ == "__main__":
    main()
