# U-Net ile Havadan Alınan Görüntülerde Semantik Segmentasyon (Aerial Image Segmentation)

Bu proje, yüksek çözünürlüklü aerial (havadan/uydu) görüntüler üzerindeki arazi, yol ve yapı gibi alanları **U-Net** derin öğrenme mimarisi kullanarak semantik olarak segmentte etmek amacıyla geliştirilmiştir.

---

## Proje Hakkında

Semantik segmentasyon (Semantic Segmentation), bir görseldeki her bir pikselin hangi sınıfa (örneğin yol, zemin, bina, arka plan) ait olduğunu sınıflandırma işlemidir. Bu çalışmada:

- Girdi olarak aerial uydu görüntüleri verilmiş,
- U-Net mimarisi ile piksel düzeyinde tahminler üretilmiş,
- Model çıktısı olan olasılık haritaları eşiklenerek/sınıflandırılarak **Ground Truth** (gerçek maske) ile karşılaştırılmıştır.

---

## Model Mimarisi (U-Net)

Projede encoder-decoder yapısına sahip **U-Net** mimarisi kullanılmıştır:

- **Encoder (Büzülme Yolu):** Görseldeki yüksek seviyeli özellikleri (feature extraction) yakalar.
- **Decoder (Genişleme Yolu):** Özellik haritalarını orijinal resim boyutuna geri yükseltir (upsampling).
- **Skip Connections:** Alt katmanlardaki mekânsal bilgiyi kaybetmemek adına encoder'dan decoder'a doğrudan bilgi aktarımı sağlar.

---

## Eğitim ve Değerlendirme

- **Eğitim/Doğrulama Kaybı (Loss Graphs):** Modelin `train_loss` ve `val_loss` değerleri epochs boyunca izlenmiş, aşırı öğrenme (overfitting) durumu kontrol edilmiştir.
- **Tahmin ve Görselleştirme:** Doğrulama veri seti (`X_val`) üzerindeki pikseller için üretilen ham olasılık haritaları (`pred_raw`) ve ikili/çoklu maskeler görselleştirilerek model performansı doğrulanmıştır.

### Örnek Tahmin Çıktısı

| Input Image | Ground Truth | U-Net Prediction |
| :---------: | :----------: | :--------------: |
| Girdi Resmi | Gerçek Maske |  Model Tahmini   |

> _Not: Modelin ürettiği olasılık haritaları `pred_raw` üzerinden alınmış ve maske sınırları görsel olarak karşılaştırılmıştır._

---

## Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın

```bash
git clone [https://github.com/KULLANICI_ADI/REPO_ADI.git](https://github.com/KULLANICI_ADI/REPO_ADI.git)
cd REPO_ADI

├── data/                  # Veri seti (Images & Masks)
├── notebooks/             # U-Net Eğitim ve Görselleştirme Kodları (.ipynb)
├── assets/                # README için görsel çıktı resimleri
├── README.md              # Proje dökümantasyonu
└── requirements.txt       # Proje bağımlılıkları
```
