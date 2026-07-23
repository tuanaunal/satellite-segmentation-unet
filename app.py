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

            # goruntuyu oku ve rgbye mevir
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size) / 255.0 # gorutunyu yeniden boyutlandir ve normalize et

            # maskeyi gri tonlamada oku yeniden boyutlandir ve normalize et
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = np.expand_dims(mask, axis=-1) / 255.0 # (H,W) -> (H,W,1)

            images.append(img)
            masks.append(mask)

    return np.array(images, dtype="float32"), np.array(masks, dtype="float32") # numpy dizisine mevir

X, y = load_dataset("aerial_dataset", img_size=(128, 128))
print(f"Toplam ornek: {len(X)}") # toplam veri sayisi

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(f"Toplam train ornek: {len(X_train)}") # toplam veri sayisi
print(f"Toplam test ornek: {len(X_val)}") # toplam veri sayisi

# unet mimarisi tanimlama


# egitim asamasi


# sonuclarin degerlendirilmesi