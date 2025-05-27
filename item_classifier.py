import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
from sklearn.model_selection import train_test_split

# Globals
DATA_DIR = "item_data"
IMG_SIZE = (224, 224)
CLASSES = ["Cat", "Dog", "Other"]

MODELS = {
    "MobileNetV2": "item_classifier_mobilenetv2.h5",
    "CNN": "item_classifier_cnn.h5"
}

# Create class folders
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

# Save images
def collect_multiple_images(files, label):
    if not files:
        return "No images provided."
    for file in files:
        img = Image.open(file).convert("RGB").resize(IMG_SIZE)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        save_path = os.path.join(DATA_DIR, label, f"{timestamp}.jpg")
        img.save(save_path)
    return f"Saved {len(files)} images to {label}."

# Load dataset
def load_data():
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        for file in os.listdir(cls_dir):
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(cls_dir, file)).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(idx)
    X = np.array(images) / 255.0
    y = np.array(labels)
    return X, y

# Train selected model
def train_model(model_type):
    model_path = MODELS[model_type]
    if os.path.exists(model_path):
        return f"{model_type} model already exists."

    X, y = load_data()
    if len(np.unique(y)) < 2:
        return "Need at least 2 classes to train."

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    if model_type == "MobileNetV2": # using MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        base_model.trainable = False
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
    else: # using CNN 
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(X_train, y_train, batch_size=8), epochs=10, validation_data=(X_val, y_val))
    model.save(model_path)
    return f"{model_type} model trained and saved."

# Predict using selected model
def predict(image, model_type):
    model_path = MODELS[model_type]
    if not os.path.exists(model_path):
        return f"{model_type} model not found. Train it first."

    model = tf.keras.models.load_model(model_path)
    img = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr)[0]
    confidences = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    return confidences

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## Image Classifier App")

    gr.Markdown("---")
    gr.Markdown("### Convert Your Images into Training Data (Not Mandatory)")
    with gr.Row():
        img_files = gr.File(file_types=["image"], file_count="multiple", label="Upload Multiple Images")
        label_input = gr.Dropdown(choices=CLASSES, value="Cat", label="Label")
    save_btn = gr.Button("Save Images to Dataset")
    save_output = gr.Textbox(label="Save Status")
    save_btn.click(fn=collect_multiple_images, inputs=[img_files, label_input], outputs=save_output)

    gr.Markdown("---")
    gr.Markdown("### Train")

    with gr.Row():
        model_choice_train = gr.Dropdown(choices=["MobileNetV2", "CNN"], value="MobileNetV2", label="Select Model to Train")
        train_btn = gr.Button("Train Selected Model")
        train_output = gr.Textbox(label="Training Log")
        train_btn.click(fn=train_model, inputs=model_choice_train, outputs=train_output)

    gr.Markdown("---")
    gr.Markdown("### Predict")

    with gr.Row():
        model_choice_predict = gr.Dropdown(choices=["MobileNetV2", "CNN"], value="MobileNetV2", label="Select Model to Use")
        predict_input = gr.Image(sources=["upload"], label="Upload Image to Predict")
    predict_btn = gr.Button("Predict")
    predict_output = gr.Label(num_top_classes=3)
    predict_btn.click(fn=predict, inputs=[predict_input, model_choice_predict], outputs=predict_output)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)