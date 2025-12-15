# ShelfSight

## 1. Persyaratan

ShelfSight dibuat dengan bahasa pemrograman Python, karena itu sistem harus sudah melakukan instalasi berikut:
* Python
* GIT

## 2. Clone Repository

```bash
git clone https://github.com/ChristabellaJC/ShelfSight.git
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install PyTorch

```bash
pip install torch torchvision
```

Untuk menentukan versi CUDA yang dibutuhkan agar dapat menjalankan webapp, silakan lihat tautan berikut:

```text
https://pytorch.org/get-started/locally/
```

## 5. Jalankan WebApp

```bash
streamlit run app.py
```

## 6. Akses WebApp

Setelah menjalankan webapp, browser akan terbuka secara otomatis. Jika tidak, akses melalui:

```text
http://localhost:8501
```

## 7. Menggunakan WebApp
- Pilih Model yang ingin digunakan, perincian mengenai model dapat dilihat di halaman Help
- Gunakan kamera perangkat untuk menangkap gambar atau meng-upload gambar yang sudah ada
- Klik tombol Start Detection
- Hasil deteksi akan disajikan di dua kolom, In Stock untuk produk yang telah terdeteksi dan Out Of Stock untuk produk yang tidak terdeteksi

Untuk pengarahan lebih rinci, dapat dibaca pada Manual Book

Developer: Christabella Jocelynne Chandra - 535220166
