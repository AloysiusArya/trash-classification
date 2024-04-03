from google.colab import drive
drive.mount('/content/drive')

import os, re, glob, cv2, numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

drive_path = '/content/drive/MyDrive/pengpol/FileCitraSampah'
print(os.listdir(drive_path))
training_path = os.path.join(drive_path, 'Training')
print(os.listdir(training_path))


import os
import tensorflow as tf

# Replace this with the path to your dataset directory
drive_path = '/content/drive/MyDrive/pengpol/FileCitraSampah'

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
NUM_CLASSES = 3

# Path to the 'Training' directory
training_path = os.path.join(drive_path, 'Training')
# Create the training dataset using image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='categorical'  # Use categorical labels
)


def prepare_image(img,target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size)
    img_array = image.img_to_array(img_resized)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def preprocess_batch(batch_images, batch_labels):
    preprocessed_images = []
    for img in batch_images:
        preprocessed_images.append(prepare_image(img.numpy()))  # Convert to numpy array
    return np.array(preprocessed_images), batch_labels

for batch_images, batch_labels in train_dataset:
    processed_images, labels = preprocess_batch(batch_images, batch_labels)



testing_path = os.path.join(drive_path, 'test_data')
testing_dataset = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='categorical'  # Use categorical labels
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from sklearn.metrics import classification_report, confusion_matrix


from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load a pre-trained model (e.g., MobileNet) with weights from ImageNet
base_model = MobileNet(include_top=False, weights='imagenet',  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Function to create a transfer learning model
def create_transfer_model(base_model):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

mobile_transfer = create_transfer_model(base_model)
mobile_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mobile_transfer.summary()

epochs = 10
mobile_history = mobile_transfer.fit(train_dataset, epochs=epochs, validation_data=testing_dataset)

# Evaluate the models
def evaluate_model(model, dataset, name):
    print(f"Evaluating {name} model:")
    results = model.evaluate(dataset)
    print(f"Loss: {results[0]}, Accuracy: {results[1]}")

evaluate_model(mobile_transfer, testing_dataset, 'MobileNet')

import matplotlib.pyplot as plt
# Plot training history
def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(mobile_history, 'MobileNet Transfer Learning')

# Predictions
def make_predictions(model, dataset):
    predictions = model.predict(dataset)
    return tf.argmax(predictions, axis=1)

# Collect true labels and predicted labels for classification report
true_labels = []
mobile_predictions = make_predictions(mobile_transfer, testing_dataset)

for images, labels in testing_dataset:
    true_labels.extend(tf.argmax(labels, axis=1))

# Generate classification reports
mobile_report = classification_report(true_labels, mobile_predictions.numpy())

# Print classification reports
print("MobileNet Transfer Learning Classification Report:")
print(mobile_report)

from sklearn.model_selection import StratifiedKFold


# Combine models and their names
models = [mobile_transfer]
model_names = ['MobileNet']

# Load dataset using image_dataset_from_directory
def load_dataset(path):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        labels='inferred',
        label_mode='categorical'
    )

# Function to evaluate a model on a dataset
def evaluate_model(model, dataset):
    predictions = model.predict(dataset)
    true_labels = []
    for _, labels in dataset:
        true_labels.extend(tf.argmax(labels, axis=1).numpy())
    predicted_labels = tf.argmax(predictions, axis=1).numpy()
    return classification_report(true_labels, predicted_labels, output_dict=True)

# K-fold cross-validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

for model, model_name in zip(models, model_names):
    print(f"Training and evaluating {model_name}...")
    fold_num = 1
    model_results = []  # List untuk menyimpan hasil evaluasi model tertentu
    for train_index, test_index in skf.split(range(len(true_labels)), true_labels):
        print(f"\nFold {fold_num}")
        fold_num += 1

        # Split dataset into train and test sets
        train_dataset = load_dataset(training_path)
        train_dataset = train_dataset.unbatch()
        train_dataset = train_dataset.shuffle(1000, seed=42)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dataset = load_dataset(training_path)
        test_dataset = test_dataset.unbatch()
        test_dataset = test_dataset.shuffle(1000, seed=42)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        train_dataset = train_dataset.take(len(train_index))
        test_dataset = test_dataset.take(len(test_index))

        # Train the model
        history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

        # Evaluate the model
        eval_result = evaluate_model(model, test_dataset)
        print(f"Evaluation result for fold {fold_num - 1}:\n{eval_result}")

        model_results.append(eval_result['accuracy'])  # Simpan hasil evaluasi ke dalam list model_results

    # Hitung dan tampilkan rata-rata akurasi untuk model tertentu
    average_accuracy = np.mean(model_results)
    print(f"Average accuracy across all folds ({model_name}): {average_accuracy}")