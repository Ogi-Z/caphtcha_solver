import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Parametreler
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
IMAGE_CHANNELS = 1  # Gri seviye
DATA_DIR = '../data/train/'

# Veri Yükleme
def load_data():
    images = []
    labels = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.png'):
            label = filename.split('_')[0]
            path = os.path.join(DATA_DIR, filename)
            image = load_img(path, color_mode='grayscale', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
            image = img_to_array(image) / 255.0
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

print("📦 Veriler yükleniyor...")
X, y = load_data()
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Eğitim ve test seti
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Model Oluşturma
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Eğitim başlatılıyor...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Modeli kaydet
model.save('../model/captcha_model.h5')
print("✅ Model kaydedildi: captcha_model.h5")

# Label encoder'ı da kaydetmek istersen:
import pickle
with open('../model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Label encoder kaydedildi: label_encoder.pkl")