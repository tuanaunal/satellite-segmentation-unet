"""
problem tanimi: unet kullanarak uydu goruntuleri ile segmentasyon

unet
uydu goruntuleri: https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery
"""

# import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# veri hazirlama ve yukleme
def load_dataset(root, img_size=(128, 128)):
    images, masks = [], [] # goruntuler ve maskeler icin bos liste olustur
    for tile in sorted(os.listdir(root)): # her bir tile klasorunu sirasiyla dolas
        img_dir = os.path.join(root, tile, "images") # goruntulerin oldugu klasorler
        mask_dir = os.path.join(root, tile, "masks") # maskelerin oldugu klasor
        if not os.path.isdir(img_dir): continue # klasor yoksa atla
        for f in os.listdir(img_dir): # goruntu dosyalarini dolas
            if not f.lower().endswith(".jpg"): continue # sadece .jpg dosyalari ile ilgilen
            img_path = os.path.join(img_dir, f) # goruntu dosya yolunu elde ettik
            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png") # maskeye karsilik gelen dosya yolu
            if not os.path.exists(mask_path): continue # maske yoksa atla

            # goruntuyu oku ve rgbye cevir
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0 # gorutunyu yeniden boyutlandir ve normalize et

            # maskeyi gri tonlamada oku yeniden boyutlandir ve normalize et
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = np.expand_dims(mask, axis=-1) / 255.0 # (H,W) -> (H,W,1)

            images.append(img)
            masks.append(mask)

    return np.array(images, dtype="float32"), np.array(masks, dtype="float32") # numpy dizisine cevir

X, y = load_dataset("aerial_dataset", img_size=(128, 128))
print(f"Toplam ornek: {len(X)}") # toplam veri sayisi

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(f"Toplam train ornek: {len(X_train)}") # toplam veri sayisi
print(f"Toplam test ornek: {len(X_val)}") # toplam veri sayisi

# unet mimarisi tanimlama
def unet_model(input_size = (128, 128, 3)):

    inputs = keras.Input(input_size) # girdi katmani

    # encoder: feature extraction ve downsampling
    c1 = layers.Conv2D(16, 3, activation = "relu", padding = "same")(inputs) # 16 filtre 3x3 kernel
    c1 = layers.Conv2D(16, 3, activation = "relu", padding = "same")(c1)
    p1 = layers.MaxPooling2D()(c1) # downsampling 64x64

    c2 = layers.Conv2D(32, 3, activation = "relu", padding = "same")(p1)
    c2 = layers.Conv2D(32, 3, activation = "relu", padding = "same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation = "relu", padding = "same")(p2)
    c3 = layers.Conv2D(64, 3, activation = "relu", padding = "same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation = "relu", padding = "same")(p3)
    c4 = layers.Conv2D(128, 3, activation = "relu", padding = "same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck: en derin seviye
    c5 = layers.Conv2D(256, 3, activation = "relu", padding = "same")(p4)
    c5 = layers.Conv2D(256, 3, activation = "relu", padding = "same")(c5)

    # decoder: up sampling ve skip connection
    u6 = layers.Conv2DTranspose(128, 2, strides = (2,2), padding = "same")(c5) # up sample 8x8 -> 16x16
    u6 = layers.concatenate([u6, c4]) # concatenate duzeltildi
    c6 = layers.Conv2D(128, 3, activation = "relu", padding = "same")(u6)
    c6 = layers.Conv2D(128, 3, activation = "relu", padding = "same")(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides = (2,2), padding = "same")(c6) # 16 -> 32
    u7 = layers.concatenate([u7, c3]) # concatenate duzeltildi
    c7 = layers.Conv2D(64, 3, activation = "relu", padding = "same")(u7)
    c7 = layers.Conv2D(64, 3, activation = "relu", padding = "same")(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides = (2,2), padding = "same")(c7) # 32 -> 64
    u8 = layers.concatenate([u8, c2]) # concatenate duzeltildi
    c8 = layers.Conv2D(32, 3, activation = "relu", padding = "same")(u8)
    c8 = layers.Conv2D(32, 3, activation = "relu", padding = "same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides = (2,2), padding = "same")(c8)
    u9 = layers.concatenate([u9, c1]) # concatenate duzeltildi
    c9 = layers.Conv2D(16, 3, activation = "relu", padding = "same")(u9)
    c9 = layers.Conv2D(16, 3, activation = "relu", padding = "same")(c9)

    outputs = layers.Conv2D(1, 1, activation = "sigmoid")(c9)

    return keras.Model(inputs, outputs)

# egitim asamasi
unet_model = unet_model()
unet_model.compile(optimizer="adam", loss = "binary_crossentropy", metrics=["accuracy"])

# callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("model_best.h5", save_best_only = True), # en iyi modeli kaydet
    keras.callbacks.ReduceLROnPlateau(), # dogrulama kaybi dusmez ise learning rate i azalt
    keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) # 10 epoch boyunca iyilesmez ise durdur (s harfi eklendi)
]

history = unet_model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val), # dogrulama verisi
    epochs = 10,
    batch_size = 16,
    callbacks = callbacks
)

# sonuclarin degerlendirilmesi