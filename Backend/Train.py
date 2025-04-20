import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
train_dir = '../../balanced_data/train'
val_dir = '../../balanced_data/test'

# Hyperparameters
batch_size = 32  # Reduced batch size
num_epochs = 35
input_shape = (48, 48, 1)
num_classes = 7  # Assuming 7 emotion classes

# Calculate class weights to handle imbalanced dataset
def calculate_class_weights(directory):
    class_counts = {}
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for class_name in classes:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(image_files)
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for i, class_name in enumerate(classes):
        if class_name in class_counts and class_counts[class_name] > 0:
            # Calculate balanced class weight
            class_weights[i] = total_samples / (len(class_counts) * class_counts[class_name])
    
    print("Class weights:", class_weights)
    return class_weights

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Fix: Set shuffle=True and correct steps_per_epoch calculation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Keep validation data in order for confusion matrix
)

# Calculate steps correctly

train_steps = math.ceil(train_generator.samples / batch_size)
val_steps = math.ceil(validation_generator.samples / batch_size)



# Print class distribution
print("Class distribution in training set:")
class_counts = np.bincount(train_generator.classes)
for class_name, count in zip(sorted(train_generator.class_indices.keys()), class_counts):
    print(f"{class_name}: {count} images")

def create_custom_cnn():
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Block 3
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Classifier
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def train_model():
    # Create model
    model = create_custom_cnn()
    
    # Calculate class weights to handle imbalance
    class_weights = calculate_class_weights(train_dir)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Callbacks for better training
    checkpoint = ModelCheckpoint(
        'best_model.keras',  # Use .keras extension to avoid warning
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks = [checkpoint, reduce_lr, early_stopping]
    
    # Train the model (fixed steps_per_epoch and validation_steps)
    history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
    )

    
    # Plot the training history
    plot_model_history(history)
    
    # Save model
    model.save('emotion_model_custom.keras')
    model.save_weights('model.h5')
    print("Model trained and saved as 'emotion_model_custom.keras'")
    
    # Evaluate model
    evaluation = model.evaluate(validation_generator, steps=val_steps)
    print(f"Final validation accuracy: {evaluation[1]*100:.2f}%")
    
   
   
    
    class_names = list(sorted(validation_generator.class_indices.keys()))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
   
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
   

if __name__ == "__main__":
    train_model()
