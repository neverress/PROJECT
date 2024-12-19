import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from traffic_sign_recognition_CHECK import preprocess, load_class_names, classify_image, FileNotFoundError, \
    InvalidModelError, InvalidImageError

# ------------------------------------------
# Частично заимствованный код 
# ------------------------------------------

# Тестовые данные
test_image_path = "test_image.jpg"
test_model_path = "test_model.keras"
test_labels_path = "test_labels.csv"
non_existent_path = "non_existent.jpg"

# ------------------------------------------
# Тесты для preprocess
# ------------------------------------------
def test_preprocess_valid_image():
    """Тест обработки корректного изображения."""
    test_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    processed_img = preprocess(test_img)
    assert processed_img.shape == (32, 32)
    assert (processed_img >= 0).all() and (processed_img <= 1).all()


def test_preprocess_invalid_image():
    """Тест обработки некорректного изображения."""
    with pytest.raises(InvalidImageError):
        preprocess(None)


# ------------------------------------------
# Тесты для load_class_names
# ------------------------------------------
def test_load_class_names_valid():
    """Тест загрузки корректного файла с классами."""
    mock_labels = "ClassId,Name\n0,Speed Limit 20\n1,Speed Limit 30\n"
    with patch("builtins.open", new=MagicMock()) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = mock_labels
        with patch("os.path.exists", return_value=True):
            class_names = load_class_names(test_labels_path)
            assert class_names == {0: "Speed Limit 20", 1: "Speed Limit 30"}


def test_load_class_names_missing_file():
    """Тест загрузки отсутствующего файла с классами."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_class_names(non_existent_path)


# ------------------------------------------
# Тесты для classify_image
# ------------------------------------------
def test_classify_image_valid():
    """Тест классификации изображения с корректной моделью и изображением."""
    with patch("os.path.exists", side_effect=lambda x: x in [test_image_path, test_model_path, test_labels_path]):
        with patch("cv2.imread", return_value=np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)):
            with patch("tensorflow.keras.models.load_model", return_value=MagicMock()) as mock_model:
                mock_model.return_value.predict.return_value = np.array([[0.1, 0.9]])
                with patch("pandas.read_csv",
                           return_value=pd.DataFrame({"ClassId": [0, 1], "Name": ["Stop", "Yield"]})):
                    classify_image(test_model_path, test_image_path)
                    mock_model.assert_called_once()


def test_classify_image_missing_model():
    """Тест классификации с отсутствующей моделью."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            classify_image(non_existent_path, test_image_path)


def test_classify_image_invalid_image():
    """Тест классификации с некорректным изображением."""
    with patch("os.path.exists", return_value=True):
        with patch("cv2.imread", return_value=None):
            with pytest.raises(FileNotFoundError):
                classify_image(test_model_path, test_image_path)


def test_classify_image_invalid_model():
    """Тест классификации с некорректной моделью."""
    with patch("os.path.exists", side_effect=lambda x: x in [test_image_path, test_model_path, test_labels_path]):
        with patch("tensorflow.keras.models.load_model", side_effect=InvalidModelError("Ошибка загрузки модели")):
            with pytest.raises(InvalidModelError):
                classify_image(test_model_path, test_image_path)
