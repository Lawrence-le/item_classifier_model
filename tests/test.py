import os
import sys
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from item_classifier import collect_multiple_images, train_model, predict, MODELS, CLASSES, IMG_SIZE, DATA_DIR


# Helper: Create dummy dataset
def create_dummy_dataset(data_dir, num_per_class=10):
    os.makedirs(data_dir, exist_ok=True)
    for cls in CLASSES:
        class_dir = os.path.join(data_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_per_class):
            img = Image.fromarray(np.random.randint(0, 255, IMG_SIZE + (3,), dtype=np.uint8))
            img.save(os.path.join(class_dir, f"{cls}_{i}.jpg"))


def test_collect_multiple_images(tmp_path):
    label = CLASSES[0]
    test_images = []

    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = tmp_path / f"img_{i}.jpg"
        img.save(img_path)
        test_images.append(str(img_path))

    # Ensure DATA_DIR/label exists
    target_dir = os.path.join(DATA_DIR, label)
    os.makedirs(target_dir, exist_ok=True)

    result = collect_multiple_images(test_images, label)

    assert "Saved 2 images" in result
    saved_files = os.listdir(target_dir)
    assert len(saved_files) >= 2


def test_train_model_mobilenet(tmp_path):
    # Set up dummy data
    test_data_dir = os.path.join(tmp_path, "item_data")
    create_dummy_dataset(test_data_dir)

    # Temporarily change global DATA_DIR and MODELS
    original_data_dir = DATA_DIR
    original_models = MODELS.copy()

    try:
        import item_classifier
        item_classifier.DATA_DIR = test_data_dir
        item_classifier.MODELS["MobileNetV2"] = os.path.join(tmp_path, "mobilenet_model.h5")

        result = train_model("MobileNetV2")
        assert "trained and saved" in result or "already exists" in result
        assert os.path.exists(item_classifier.MODELS["MobileNetV2"])
    finally:
        item_classifier.DATA_DIR = original_data_dir
        item_classifier.MODELS = original_models


def test_train_model_cnn(tmp_path):
    test_data_dir = os.path.join(tmp_path, "item_data")
    create_dummy_dataset(test_data_dir)

    original_data_dir = DATA_DIR
    original_models = MODELS.copy()

    try:
        import item_classifier
        item_classifier.DATA_DIR = test_data_dir
        item_classifier.MODELS["CNN"] = os.path.join(tmp_path, "cnn_model.h5")

        result = train_model("CNN")
        assert "trained and saved" in result or "already exists" in result
        assert os.path.exists(item_classifier.MODELS["CNN"])
    finally:
        item_classifier.DATA_DIR = original_data_dir
        item_classifier.MODELS = original_models


def test_train_model_no_data(tmp_path):
    test_data_dir = os.path.join(tmp_path, "item_data")
    os.makedirs(test_data_dir, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(test_data_dir, cls))

    model_path = os.path.join(tmp_path, "cnn_model.h5")
    if os.path.exists(model_path):
        os.remove(model_path)  # Make sure the model file does not exist

    original_data_dir = DATA_DIR
    original_models = MODELS.copy()

    try:
        import item_classifier
        item_classifier.DATA_DIR = test_data_dir
        item_classifier.MODELS["CNN"] = model_path
        result = train_model("CNN")
        assert "Need at least 2 classes" in result
    finally:
        item_classifier.DATA_DIR = original_data_dir
        item_classifier.MODELS = original_models


def test_predict_without_model(tmp_path):
    img = np.random.randint(0, 255, IMG_SIZE + (3,), dtype=np.uint8)
    missing_model_path = os.path.join(tmp_path, "non_existent_model.h5")

    original_models = MODELS.copy()

    try:
        import item_classifier
        item_classifier.MODELS["CNN"] = missing_model_path
        result = predict(img, "CNN")
        assert "model not found" in result
    finally:
        item_classifier.MODELS = original_models