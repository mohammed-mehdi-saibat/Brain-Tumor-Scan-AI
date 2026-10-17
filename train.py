import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import (
    load_and_resize_images,
    shuffle_data,
    convert_to_numpy,
    calculate_class_weights,
    encode_labels,
    split_and_normalize_data
)
from src.model import build_cnn
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

def main():
    print("---- 1. Loading & Preprocessing Data ----")
    data_dir = "data/raw"
    
    # LOAD AND RESIZE IMAGES
    images, labels = load_and_resize_images(data_dir)
    
    # SHUFFLE DATASET
    images, labels = shuffle_data(images, labels)
    
    # CONVERT TO NUMPY ARRAYS
    images_np, labels_np = convert_to_numpy(images, labels)
    
    # ENCODE TEXT LABELS TO INTEGERS
    encoded_labels, label_encoder = encode_labels(labels_np)
    
    # COMPUTE CLASS WEIGHTS TO HANDLE IMBALANCE
    class_weights = calculate_class_weights(encoded_labels)
    print(f"Computed Class Weights: {class_weights}")
    
    # SPLIT DATA TO TRAIN AND TEST SETS AND NORMALIZE PIXELS TO [0, 1]
    X_train, X_test, y_train, y_test = split_and_normalize_data(images_np, encoded_labels)
    
    print("\n---- 2. Building CNN Model ---- ")
    model = build_cnn(input_shape=(224, 224, 3), num_classes=len(label_encoder.classes_))
    model.summary()
    
    # SAVE MODEL ARCHITECTURE PLOT (REQUIRES PYDOT & GRAPHVIZ)
    try:
        plot_model(model, to_file="cnn_architecture.png", show_shapes=True, show_layer_names=True)
        print("Model architecture diagram saved as 'cnn_architecture.png'.")
    except Exception as e:
        print(f"Note: Could not plot model architecture image (requires graphviz): {e}")

    print("\n---- 3. Setting Up Callbacks ----")
    checkpoint = ModelCheckpoint(
        "brain_tumor_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    callbacks_list = [checkpoint, early_stopping]
    
    print("\n---- 4. Training The Model (with timer) ----")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=6,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks_list
    )
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\nTraining complete in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes).")
    print("Best model saved as 'brain_tumor_model.keras'.")

    print("\n---- 5. Evaluating Model & Generating Graphs ----")
    # EVALUATE ON TEST SET
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # A. PLOT LEARNING CURVES (ACCURACY & LOSS)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    print("Learning curves saved as 'learning_curves.png'.")
    plt.show()

    # B. CONFUSION MATRIX & CLASSIFICATION REPORT
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'.")
    plt.show()

    # C. DISPLAY CORRECT AND INCORRECT PREDICTIONS
    correct_indices = np.where(y_pred == y_test)[0]
    incorrect_indices = np.where(y_pred != y_test)[0]

    plt.figure(figsize=(12, 6))
    
    # SHOW UP TO 3 CORRECT PREDICTIONS
    for i, idx in enumerate(correct_indices[:3]):
        plt.subplot(2, 3, i + 1)
        plt.imshow(X_test[idx])
        pred_label = label_encoder.classes_[y_pred[idx]]
        true_label = label_encoder.classes_[y_test[idx]]
        plt.title(f"Correct\nTrue: {true_label}\nPred: {pred_label}", color='green', fontsize=9)
        plt.axis('off')

    # SHOW UP TO 3 INCORRECT PREDICTIONS
    for i, idx in enumerate(incorrect_indices[:3]):
        plt.subplot(2, 3, i + 4)
        plt.imshow(X_test[idx])
        pred_label = label_encoder.classes_[y_pred[idx]]
        true_label = label_encoder.classes_[y_test[idx]]
        plt.title(f"Incorrect\nTrue: {true_label}\nPred: {pred_label}", color='red', fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("predictions_sample.png")
    print("Prediction samples saved as 'predictions_sample.png'.")
    plt.show()

if __name__ == "__main__":
    main()