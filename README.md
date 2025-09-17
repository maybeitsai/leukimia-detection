# 🩸 Deteksi Leukemia Limfoblastik Akut (ALL) dengan Deep Learning

Proyek ini mengembangkan model **Convolutional Neural Network (CNN)** untuk klasifikasi otomatis **Acute Lymphoblastic Leukemia (ALL)** menggunakan citra _Peripheral Blood Smear_ (PBS). Model ini mampu mengklasifikasikan sel darah menjadi 4 kategori: **Benign** (normal) dan 3 subtipe ALL (**Early Pre-B**, **Pre-B**, **Pro-B**).

## 📊 Hasil Utama

- **Akurasi Model**: 97.39%
- **F1-Score**: 0.974 (weighted average)
- **Parameter Model**: 127,300 (arsitektur efisien)
- **Training Time**: 19 epoch (early stopping)

## 🎯 Tujuan Proyek

1. **Diagnosis Otomatis**: Mengembangkan sistem AI untuk membantu diagnosis ALL
2. **Skrining Cepat**: Menyediakan alternatif non-invasif untuk skrining awal kanker
3. **Akurasi Tinggi**: Mencapai performa setara standar klinis
4. **Efisiensi**: Model ringan dengan parameter minimal untuk deployment praktis

## 📋 Daftar Isi

- [Dataset](#-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Metodologi](#-metodologi)
- [Hasil dan Evaluasi](#-hasil-dan-evaluasi)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)
- [Referensi](#-referensi)

## 📂 Dataset

### Sumber Data

Dataset dikembangkan oleh **Laboratorium Sumsum Tulang Rumah Sakit Taleqani (Tehran, Iran)**:

- **Total Citra**: 3,256 gambar PBS
- **Jumlah Pasien**: 89 pasien
- **Format**: JPG
- **Perbesaran**: 100x menggunakan mikroskop Zeiss
- **Validasi Label**: Flow cytometry oleh spesialis

### Distribusi Kelas

| Kelas      | Deskripsi                | Jumlah Training\* | Jumlah Test |
| ---------- | ------------------------ | ----------------- | ----------- |
| **Benign** | Sel normal (hematogones) | 3,000             | 101         |
| **Early**  | Early Pre-B ALL          | 3,000             | 197         |
| **Pre**    | Pre-B ALL                | 3,000             | 193         |
| **Pro**    | Pro-B ALL                | 3,000             | 161         |

\*Setelah data augmentation

### Struktur Dataset

```
data/
├── Original/          # Dataset asli
│   ├── Benign/
│   ├── Early/
│   ├── Pre/
│   └── Pro/
├── Segmented/         # Dataset tersegmentasi
│   ├── Benign/
│   ├── Early/
│   ├── Pre/
│   └── Pro/
└── Final/             # Dataset siap training
    ├── train/         # 12,000 citra (augmented)
    ├── val/           # 652 citra
    └── test/          # 652 citra
```

## 🏗️ Struktur Proyek

```
leukemia-detection/
├── main.ipynb         # Notebook utama dengan implementasi lengkap
├── data/              # Dataset dan preprocessing
├── model/             # Model tersimpan
│   └── cnn_all_model.h5
├── README.md          # Dokumentasi proyek
└── requirements.txt   # Dependencies
```

## 🔬 Metodologi

### 1. Data Preparation

- **Data Augmentation**: Meningkatkan dataset dari ~2,000 menjadi 12,000 citra training
- **Teknik Augmentasi**:
  - RandomFlip (horizontal/vertical)
  - RandomRotation (±10°)
  - RandomZoom (±10%)
  - RandomContrast (±10%)

### 2. Arsitektur Model

**CNN Custom Architecture:**

```python
Model Architecture:
├── Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D
├── Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D
├── Conv2D (128 filters, 3x3) + ReLU + MaxPooling2D
├── GlobalAveragePooling2D
├── Dense (128 units, Swish) + Dropout (30%)
├── Dense (64 units, Swish) + Dropout (30%)
└── Dense (4 units, Softmax)

Total Parameters: 127,300
```

### 3. Training Strategy

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Callbacks**:
  - Early Stopping (patience: 3)
  - ReduceLROnPlateau (patience: 2)
  - ModelCheckpoint (save best weights)
- **Batch Size**: 32 (train), 16 (val/test)

## 📈 Hasil dan Evaluasi

### Performa Model

| Metrik                   | Nilai  |
| ------------------------ | ------ |
| **Accuracy**             | 97.39% |
| **Precision (weighted)** | 97.4%  |
| **Recall (weighted)**    | 97.4%  |
| **F1-Score (weighted)**  | 97.4%  |

### Performa Per Kelas

| Kelas  | Precision | Recall | F1-Score | Support |
| ------ | --------- | ------ | -------- | ------- |
| Benign | 93.1%     | 94.1%  | 93.6%    | 101     |
| Early  | 98.0%     | 97.5%  | 97.7%    | 197     |
| Pre    | 99.5%     | 96.9%  | 98.2%    | 193     |
| Pro    | 97.0%     | 100%   | 98.5%    | 161     |

### Training Progress

- **Total Epochs**: 19 (early stopping dari max 30)
- **Best Epoch**: 16 (validation accuracy: 97.24%)
- **Final Training Loss**: 0.0855
- **Final Validation Loss**: 0.1089

## 💻 Instalasi

### Prerequisites

- Python 3.8+
- CUDA (untuk GPU acceleration)

### Install Dependencies

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
```

### Clone Repository

```bash
git clone https://github.com/maybeitsai/leukimia-detection.git
cd leukimia-detection
```

## 🚀 Penggunaan

### 1. Training Model

```python
# Jalankan notebook utama
jupyter notebook main.ipynb
```

### 2. Load Pre-trained Model

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model/cnn_all_model.h5')

# Prediksi single image
img = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(150, 150))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
class_names = ['Benign', 'Early', 'Pre', 'Pro']
predicted_class = class_names[np.argmax(predictions[0])]
```

### 3. Evaluasi Model

```python
# Evaluasi pada test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

## 📊 Visualisasi Hasil

Model menghasilkan:

- **Confusion Matrix**: Menunjukkan akurasi klasifikasi per kelas
- **Training/Validation Curves**: Tracking loss dan accuracy selama training
- **Classification Report**: Detailed metrics per kelas
- **Sample Predictions**: Visualisasi prediksi model pada test samples

## 🔍 Insight Penting

### Kelebihan Model:

1. **High Accuracy**: 97.39% accuracy pada data test
2. **Balanced Performance**: Performa konsisten di semua kelas
3. **Efficient Architecture**: Only 127K parameters
4. **Fast Training**: Konvergensi dalam 19 epoch
5. **Clinical-Grade**: Potensi aplikasi klinis nyata

### Implikasi Klinis:

- **Sensitivitas Tinggi**: Recall >96% untuk deteksi kanker
- **Spesifisitas Baik**: Precision >93% mengurangi false positive
- **Second Opinion**: Dapat membantu hematologist dalam diagnosis

## 🤝 Kontribusi

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 Lisensi

Proyek ini dilisensikan under MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- **Dataset**: Laboratorium Sumsum Tulang Rumah Sakit Taleqani, Tehran, Iran
- **Inspiration**: Paper "A Fast and Efficient CNN Model for B-ALL Diagnosis and its Subtypes Classification using Peripheral Blood Smear Images"
- **Framework**: TensorFlow/Keras team

## 📚 Referensi

1. **Paper Utama**: _A Fast and Efficient CNN Model for B-ALL Diagnosis and its Subtypes Classification using Peripheral Blood Smear Images_
2. **GitHub Repository**: [ALL-Subtype-Classification](https://github.com/MehradAria/ALL-Subtype-Classification)
3. **TensorFlow Documentation**: [tensorflow.org](https://tensorflow.org)
4. **Keras Documentation**: [keras.io](https://keras.io)

---

## 📞 Kontak

Jika ada pertanyaan atau saran, silakan buat issue di repository ini atau hubungi:

- **LinkedIn**: [Harry Mardika](https://www.linkedin.com/in/harry-mardika)
- **GitHub**: [@maybeitsai](https://github.com/maybeitsai)
