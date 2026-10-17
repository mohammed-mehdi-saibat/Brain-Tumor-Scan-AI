import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape=(224, 224, 3), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # BLOCK 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        
        # BLOCK 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        
        # BLOCK 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        
        # BLOCK 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPool2D((2, 2)),
        
        # CLASSIFICATION HEAD
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model