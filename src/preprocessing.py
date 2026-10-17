import os
import cv2
import numpy as np
import random 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# DEFINE ALLOWED EXTENSIONS AND TARGET SIZE
VALID_EXTENSIONS = (".jpg", ".jpeg", ".bmp", ".png")
IMG_SIZE = (224, 224)


def load_and_resize_images(data_dir):
    images = []
    labels = []

    # EXPLORE CLASSES (FOLDERS NAMES REPRESENTING CLASSES)
    classes = sorted(os.listdir(data_dir))
    print(f"Classes found: {classes}")

    for label in classes:
        class_path = os.path.join(data_dir, label)

        if not os.path.isdir(class_path):
            continue

        print(f"Processing folder: {class_path}")

        for file_name in os.listdir(class_path):
            file_lower = file_name.lower()

            # VERIFY FILE EXTENSIONS
            if not file_lower.endswith(VALID_EXTENSIONS):
                continue

            file_path = os.path.join(class_path, file_name)

            # TRY EXCEPT TO HANDLE ERRORS
            try:
                img = cv2.imread(file_path)
                if img is None:
                    print(f"Warning: could not read {img}. Skipping.")
                    continue

                # RESIZE IMAGE TO A FIXED SIZE (224, 224)
                img_resized = cv2.resize(img, IMG_SIZE)

                # CONVERT IMAGE FROM BGR TO RGB (OpenCV LOADS AS BGR BY DEFAULT[])
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

                images.append(img_rgb)
                labels.append(label)

            except Exception as e:
                print(f"Error loading: {file_path}: {e}")
    return images, labels



# SHUFFLE ALL CLASSES IMAGES IN TWO LISTS KEEPING THE ALIGNMENT BETWEEN EACH IMAGE AND IT'S LABEL
def shuffle_data(images, labels):
    combined = list(zip(images, labels))
    random.shuffle(combined)
    shuffle_images, shuffle_labels = zip(*combined)
    
    return list(shuffle_images), list(shuffle_labels)

# CONVERT IMAGES AND LABELS LISTS TO NUMPY LISTS TO MAKE THEM EXPLOITABLE BY CNN
def convert_to_numpy(images, labels):
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=object) 


# RENDER GRAPHICALLY THE NUMBER OF IMAGES PER CLASS
def plot_class_distribution(labels):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=labels, order=sorted(np.unique(labels)), hue=labels, palette="viridis", legend=False)
    plt.title("Image per class distribution")
    plt.xlabel("Classes")
    plt.ylabel("Images number")
    plt.show()
    
# SHOW IMAGE SAMPLES PER CLASS
def show_sample_images(images, labels, num_samples=3):
    images = np.array(images)
    labels = np.array(labels)
    
    unique_classes = np.unique(labels)
    plt.figure(figsize=(12, len(unique_classes) * 2.5))
    
    for i, cls in enumerate(unique_classes):
        # FIND ALL INDICES THAT BELONGING TO THE CURRENT CLASS
        cls_indices = np.where(labels == cls)[0]
        # RANDOMLY CHOOSE A FEW SAMPLE INDICES 
        selected_indices = np.random.choice(cls_indices, size=min(num_samples, len(cls_indices)), replace=False)
        
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(unique_classes), num_samples, i * num_samples + j + 1)
            plt.imshow(images[idx])
            plt.title(f"Class: {cls}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    
# VERIFY CLASS BALANCE AND CALCULATE WEIGHTS FOR PENALIZING MINORITY CLASS ERRORS
def calculate_class_weights(encoded_labels):
    unique_classes = np.unique(encoded_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=encoded_labels
    )
    
    # CONVERT TO A DICTIONARY MAPPING CLASS INDEX TO IT'S WEIGHT
    weights_dic = dict(zip(unique_classes, weights))
    print(f"Computed class weights: {weights_dic}")
    return weights_dic

# ENCODE CLASS LABELS TO NUMERICAL VALUES
def encode_labels(labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    
    # LINK EACH TUMOR TO IT'S CORRESPONDING NUMBER
    class_mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    print(f"Label mapping: {class_mapping}")
    
    return encoded_labels, encoder

# TRAIN TEST SPLIT AND NORMALIZATION 
def split_and_normalize_data(images, encoded_labels, test_size=0.2, random_state=42):
    images_normalized = images.astype("float32") / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        images_normalized,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels
    )

    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing set shape: {X_test.shape}, Labels shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test