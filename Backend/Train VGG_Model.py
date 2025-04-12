import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# Plot training history
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axs[0].plot(model_history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(model_history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(model_history.history['loss'], label='Train Loss')
    axs[1].plot(model_history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.savefig('training_plot.png')
    plt.show()

# Paths
train_dir = 'data/train'
val_dir = 'data/test'

# Hyperparameters
batch_size = 64
num_epochs = 5
input_shape = (48, 48, 1)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# VGG16-based model
def create_model():
    # Input: grayscale image, replicate channels to match VGG16 input
    input_tensor = Input(shape=input_shape)
    x = Concatenate()([input_tensor, input_tensor, input_tensor])

    # Load VGG16 without top, with pretrained weights
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=x)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom classifier
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_tensor)
    return model

# Training pipeline
def train_model():
    model = create_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    model_info = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    plot_model_history(model_info)
    model.save('vgg_model.h5')
    print("Model trained and saved as 'vgg_model.h5'.")

if __name__ == "__main__":
    train_model()
