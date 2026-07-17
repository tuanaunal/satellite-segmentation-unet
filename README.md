# U-Net ile Uydu Görüntüleri Üzerinde Semantik Bölütleme

Bu proje, havadan çekilmiş uydu görüntüleri (aerial imagery) üzerinde derin öğrenme yöntemleri kullanarak nesne ve alan bölütleme (semantic segmentation) gerçekleştirmeyi amaçlamaktadır. Projede mimari olarak **U-Net** modeli tercih edilmiştir.

## Projenin Amacı

Uydu fotoğraflarındaki pikselleri analiz ederek; binalar, yollar, su yatakları ve bitki örtüsü gibi farklı sınıfları piksel hassasiyetinde tespit etmek.

## Veri Seti

Projede Kaggle üzerinde yer alan [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery) veri seti kullanılmıştır.

## Kullanılan Teknolojiler

- **Dil:** Python
- **Derin Öğrenme:** TensorFlow / Keras
- **Görüntü İşleme:** OpenCV, Gdal (isteğe bağlı)
- **Veri Analizi:** NumPy, Matplotlib, Scikit-learn

## Proje Adımları

1. [x] Proje altyapısının kurulması ve kütüphanelerin aktarılması
2. [ ] Veri setinin yüklenmesi ve ön işleme (Data Preprocessing)
3. [ ] U-Net model mimarisinin TensorFlow ile tanımlanması
4. [ ] Modelin eğitilmesi (Training)
5. [ ] IoU ve Dice katsayıları ile sonuçların değerlendirilmesi
