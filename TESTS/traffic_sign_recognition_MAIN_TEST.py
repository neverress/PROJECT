import pytest
import numpy as np
import os
from unittest.mock import MagicMock, patch
from traffic_sign_recognition_MAIN import preprocess, load_data, create_model, add_image_to_dataset, DataNotFoundError, \
    InvalidInputError

# Путь к тестовым данным
test_image_path = "test_image.jpg"
test_class_id = "43"
non_existent_path = "non_existent.jpg"


# ------------------------------------------
# Тесты для preprocess
# ------------------------------------------
def test_preprocess_valid_image():
    # Мок изображения (32x32 пикселей, случайные значения)
    test_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    processed_img = preprocess(test_img)
    assert processed_img.shape == (32, 32)
    assert (processed_img >= 0).all() and (processed_img <= 1).all()


def test_preprocess_invalid_image():
    with pytest.raises(DataNotFoundError):
        preprocess(None)


# ------------------------------------------
# Тесты для load_data
# ------------------------------------------
def test_load_data_missing_path():
    with patch("os.path.exists", return_value=False):
        with pytest.raises(DataNotFoundError):
            load_data()


# Mock-тест для load_data с корректными данными
def test_load_data_mock():
    mock_images = np.random.randint(0, 256, (100, 32, 32, 3), dtype=np.uint8)
    mock_labels = np.random.randint(0, 43, 100, dtype=np.int32)

    with patch("os.listdir", return_value=[str(i) for i in range(43)]), \
            patch("cv2.imread", side_effect=lambda x: mock_images[np.random.randint(0, 100)]), \
            patch("os.path.exists", return_value=True):
        X_train, X_validation, X_test, y_train, y_validation, y_test = load_data()
        assert X_train.shape[1:] == (32, 32, 1)
        assert y_train.shape[1] == 43


# ------------------------------------------
# Тесты для create_model
# ------------------------------------------
def test_create_model():
    num_classes = 43
    model = create_model(num_classes)
    assert model.input_shape == (None, 32, 32, 1)
    assert model.output_shape == (None, num_classes)


def test_create_model_invalid_classes():
    with pytest.raises(InvalidInputError):
        create_model(-1)


# ------------------------------------------
# Тесты для add_image_to_dataset
# ------------------------------------------
def test_add_image_to_dataset_valid():
    with patch("os.path.exists", side_effect=lambda path: path == test_image_path):
        with patch("cv2.imread", return_value=np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)):
            with patch("cv2.imwrite", return_value=True) as mock_write:
                add_image_to_dataset(test_image_path, test_class_id)
                mock_write.assert_called_once()


def test_add_image_to_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        add_image_to_dataset(non_existent_path, test_class_id)


def test_add_image_to_dataset_invalid_class():
    with pytest.raises(ValueError):
        add_image_to_dataset(test_image_path, "-1")


def test_add_image_to_dataset_invalid_image():
    with patch("os.path.exists", return_value=True):
        with patch("cv2.imread", return_value=None):
            with pytest.raises(FileNotFoundError):
                add_image_to_dataset(test_image_path, test_class_id)
