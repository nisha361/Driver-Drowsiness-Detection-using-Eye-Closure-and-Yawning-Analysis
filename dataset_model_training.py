!pip install tensorflow streamlit gdown

import os
import zipfile
import gdown
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Download dataset
file_id = "1Rj0EYgDy0sy7jc7-KXvQiHwpqA0Dzj__"
url = f"https://drive.google.com/uc?id={file_id}"
output = "dataset.zip"

gdown.download(url, output, quiet=False)

# Extract dataset
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("dataset")

dataset_path = "dataset/train"

# Create separated folders
os.makedirs("eye_dataset", exist_ok=True)
os.makedirs("mouth_dataset", exist_ok=True)

# Copy Eye Classes
shutil.copytree(f"{dataset_path}/Closed","eye_dataset/Closed",dirs_exist_ok=True)
shutil.copytree(f"{dataset_path}/Open","eye_dataset/Open",dirs_exist_ok=True)

# Copy Mouth Classes
shutil.copytree(f"{dataset_path}/yawn","mouth_dataset/yawn",dirs_exist_ok=True)
shutil.copytree(f"{dataset_path}/no_yawn","mouth_dataset/no_yawn",dirs_exist_ok=True)

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Eye generators
train_eye = datagen.flow_from_directory(
    "eye_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_eye = datagen.flow_from_directory(
    "eye_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Mouth generators
train_mouth = datagen.flow_from_directory(
    "mouth_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_mouth = datagen.flow_from_directory(
    "mouth_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model function
def build_model(num_classes):

    base = MobileNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes,activation='softmax')(x)

    model = Model(inputs=base.input,outputs=output)

    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Train Eye Model
eye_model = build_model(2)

eye_model.fit(
    train_eye,
    validation_data=val_eye,
    epochs=5
)

eye_model.save("eye_model.h5")

# Train Mouth Model
mouth_model = build_model(2)

mouth_model.fit(
    train_mouth,
    validation_data=val_mouth,
    epochs=5
)

mouth_model.save("mouth_model.h5")

print("Models saved successfully")
