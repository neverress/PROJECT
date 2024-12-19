import pytest
from unittest.mock import patch, MagicMock
from console_app import main


@pytest.fixture
def app_instance():
    """Создание мок-приложения для тестирования."""
    with patch('console_app') as mock_app:
        mock_app.do_train_model = MagicMock()
        mock_app.do_classify_image = MagicMock()
        mock_app.do_add_image_to_dataset = MagicMock()
        mock_app.do_menu_choice = MagicMock()
        yield mock_app


def test_train_model_positive(app_instance):
    """Тест успешного обучения модели."""
    app_instance.do_train_model.return_value = 0
    assert app_instance.do_train_model('train_model 10 32'.split()) == 0
    app_instance.do_train_model.assert_called_once_with(['train_model', '10', '32'])


def test_train_model_negative(app_instance):
    """Тест обучения модели с некорректными данными."""
    app_instance.do_train_model.return_value = -1
    assert app_instance.do_train_model('train_model -5 32'.split()) == -1
    assert app_instance.do_train_model('train_model 10 -1'.split()) == -1
    assert app_instance.do_train_model('train_model 10 0'.split()) == -1


def test_classify_image_positive(app_instance):
    """Тест успешной классификации изображения."""
    app_instance.do_classify_image.return_value = 0
    assert app_instance.do_classify_image('classify_image traffic_sign_model.keras test_image.jpg'.split()) == 0
    app_instance.do_classify_image.assert_called_once_with(
        ['classify_image', 'traffic_sign_model.keras', 'test_image.jpg'])


def test_classify_image_negative(app_instance):
    """Тест классификации изображения с некорректными данными."""
    app_instance.do_classify_image.return_value = -1
    assert app_instance.do_classify_image('classify_image invalid_model_path.keras test_image.jpg'.split()) == -1
    assert app_instance.do_classify_image('classify_image traffic_sign_model.keras invalid_image.jpg'.split()) == -1


def test_add_image_to_dataset_positive(app_instance):
    """Тест успешного добавления изображения в датасет."""
    app_instance.do_add_image_to_dataset.return_value = 0
    assert app_instance.do_add_image_to_dataset('add_image_to_dataset test_image.jpg 1'.split()) == 0
    app_instance.do_add_image_to_dataset.assert_called_once_with(['add_image_to_dataset', 'test_image.jpg', '1'])


def test_add_image_to_dataset_negative(app_instance):
    """Тест добавления изображения с некорректными данными."""
    app_instance.do_add_image_to_dataset.return_value = -1
    assert app_instance.do_add_image_to_dataset('add_image_to_dataset invalid_image.jpg 1'.split()) == -1
    assert app_instance.do_add_image_to_dataset('add_image_to_dataset test_image.jpg -1'.split()) == -1


def test_invalid_menu_choice(app_instance):
    """Тест некорректного выбора пункта меню."""
    app_instance.do_menu_choice.return_value = -1
    assert app_instance.do_menu_choice('invalid_choice'.split()) == -1


def test_menu_exit(app_instance):
    """Тест выхода из программы."""
    app_instance.do_menu_choice.return_value = 0
    assert app_instance.do_menu_choice('exit'.split()) == 0
