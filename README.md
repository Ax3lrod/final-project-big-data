# 🎯 Final Project Big Data — Sistem Deteksi Spam/Bot Review di Steam

### Kelompok B13

* Fiorenza Adelia Nalle — 5027231053
* Harwinda — 5027231079
* Aryasatya Alaauddin — 5027231082

Proyek ini bertujuan membangun **prototipe sistem deteksi review game palsu** di Steam menggunakan arsitektur hybrid *(streaming + batch)*. Sistem dikembangkan untuk mengatasi **manipulasi ulasan** seperti **review bombing**, **spam bot**, dan **fake reviews** yang umum ditemukan di platform Steam.

## 📄 Laporan Lengkap

🔗 [Laporan Proyek Steam Review Detection](https://docs.google.com/document/d/1cxuOzms_iEBbs4OWGHUQYS4syrmBDehZl9OaO72FQg8/edit?usp=sharing)

## 📚 Latar Belakang

Steam adalah platform distribusi game terbesar dengan:

* 132+ juta pengguna aktif bulanan
* 143.000+ judul game (data 2025)

Manipulasi ulasan pengguna seperti spam, bot, dan review palsu dari developer menjadi tantangan serius. Untuk menjawabnya, prototipe ini dibangun dengan teknologi:

* **Kafka**
* **MinIO**
* **PySpark**
* **Hive Metastore**
* **Trino**

## 🏗️ Arsitektur Sistem

![arsitektur](https://github.com/user-attachments/assets/6e0aa796-7cd4-433c-9914-aeca8ebbe616)

### Alur Sistem:

1. **Ingestion & Streaming Layer**

   * Kafka menerima data ulasan real-time dari `Kafka Producer`
   * Disimpan ke MinIO (Raw Layer)

2. **Preprocessing Layer**

   * PySpark membersihkan data, menyimpannya kembali di MinIO (Clean Layer)

3. **Metadata & Query Layer**

   * Hive mencatat metadata
   * Trino digunakan untuk SQL query terhadap data di MinIO

4. **Application Layer**

   * **FastAPI** sebagai backend
   * **Streamlit** untuk dashboard visualisasi

## 🗃️ Dataset

Berisi >6.4 juta ulasan pengguna Steam dalam bahasa Inggris:

* `app_id`: ID game
* `app_name`: Nama game
* `review_text`: Isi ulasan
* `review_score`: Positif/Negatif
* `review_votes`: Jumlah voting

### 📌 Alasan Pemilihan Dataset:

* **Relevansi** tinggi dengan kasus nyata spam review
* **Volume besar** cocok untuk uji sistem big data
* **Fitur lengkap** untuk klasifikasi
* **Publik & bebas lisensi**

## ▶️ Cara Menjalankan Sistem

### 1. Jalankan Docker

```bash
docker-compose up -d
```

![Screenshot 2025-06-20 122648](https://github.com/user-attachments/assets/dded2e4c-f8de-4d93-9f2b-f2e39f9d12af)

### 2. Buat Topik Kafka

```bash
docker exec -it kafka bash

kafka-topics --create \
    --topic steam_reviews \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1

kafka-topics --create \
    --topic steam-posters-images \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1
```

![image](https://github.com/user-attachments/assets/12376470-268d-427d-a18f-9c6a47837d54)

### 3. Kafka Producer

```bash
docker exec -it producer bash

pip install kafka-python

python producer_csv.py
python producer_img.py
```

![Screenshot 2025-06-20 123812](https://github.com/user-attachments/assets/14ca9fd0-dd4e-445d-ad9a-623cf567ed0e)
![image](https://github.com/user-attachments/assets/37c2c6ef-9ccd-49ce-83e0-c3c0ba6bc7aa)
![image](https://github.com/user-attachments/assets/148c15e9-212e-4fc3-9d8f-38d67e5edb52)

### 4. Kafka Consumer

```bash
docker exec -it consumer bash

pip install kafka-python pandas minio

python consumer_csv.py
python consumer_img.py
```

![Screenshot 2025-06-20 123805](https://github.com/user-attachments/assets/65d5bac5-439d-4c83-ae58-f44650b8c87e)
![Screenshot 2025-06-20 124247](https://github.com/user-attachments/assets/a2f938ec-61a3-4609-8782-1a13cbdb3f7b)
![image](https://github.com/user-attachments/assets/1746ba58-7c83-4f65-9bca-2301fd8c9d26)

### 5. Cek Data di UI MinIO

![image](https://github.com/user-attachments/assets/f6359d97-8188-4bf9-89e7-678185ba256b)
![image](https://github.com/user-attachments/assets/972350ea-f25e-462d-9f66-3641905006dd)
![image](https://github.com/user-attachments/assets/149b519f-4a90-4fcc-8608-168d779b87ef)

### 6. Jalankan Preprocessing (PySpark)

```bash
docker exec -it train bash

pip install -r requirements.txt
spark-submit preprocess.py
```

![image](https://github.com/user-attachments/assets/81e0fb24-198a-49eb-8187-4b21d0eb4915)
![image](https://github.com/user-attachments/assets/b0ca0183-4865-4890-aaf4-c9fc4a481bf6)

### 7. Cek Clean Data di MinIO

![image](https://github.com/user-attachments/assets/3c57061b-95f9-4c9a-b27a-95aa90efc7a6)
![image](https://github.com/user-attachments/assets/74b246a6-1093-40e6-a01e-7d61ebec58d5)

### 8. Training Model di Kaggle

Untuk mengatasi ketiadaan label "spam" pada dataset, kami merancang sebuah pipeline *machine learning* dua tahap yang canggih untuk memastikan deteksi yang akurat dan andal.

**Fase 1: Pembuatan Label dengan Deteksi Anomali**

Kami tidak menggunakan aturan sederhana. Sebaliknya, kami membangun sebuah sistem untuk menemukan ulasan yang paling "mencurigakan" secara otomatis.
- **Ekstraksi Fitur Multi-Modal**: Kami mengubah setiap ulasan menjadi representasi data yang kaya dengan mengekstrak tiga jenis fitur: fitur linguistik (gaya penulisan), fitur TF-IDF (kata kunci penting), dan *sentence embeddings* (makna semantik).
- **Ensemble Anomaly Detection**: Kami menggunakan kombinasi model *unsupervised* seperti Isolation Forest dan K-Means clustering pada fitur-fitur ini. Tujuannya adalah untuk mengidentifikasi ulasan yang merupakan *outlier* atau anomali, yang kemungkinan besar adalah spam atau bot.
- **Pelabelan Cerdas**: Ulasan yang terdeteksi sebagai anomali dan juga cocok dengan pola spam umum (seperti sangat pendek atau penuh pengulangan) kami beri label sebagai **spam (1)**. Sisanya kami beri label sebagai **bukan spam (0)**.

**Fase 2: Training Model Klasifikasi Hybrid**

Setelah mendapatkan label yang andal, kami melatih `EnhancedBERTClassifier`, sebuah model deep learning yang dirancang khusus untuk tugas ini. Cara kerjanya:
1.  **Input Ganda**: Model menerima dua input untuk setiap ulasan: teks mentah dan fitur-fitur linguistik yang telah kami rekayasa.
2.  **Pemahaman Kontekstual**: Teks mentah diproses oleh **BERT** untuk mendapatkan pemahaman mendalam tentang makna dan sentimennya.
3.  **Analisis Gaya**: Fitur-fitur linguistik memberikan sinyal tentang gaya penulisan yang mencurigakan.
4.  **Prediksi Akhir**: Model kami menggabungkan pemahaman makna dari BERT dengan analisis gaya penulisan, lalu memasukkannya ke dalam jaringan saraf tiruan untuk menghasilkan **skor probabilitas spam** yang akurat antara 0 dan 1.

Pendekatan hybrid ini memungkinkan kami untuk mendeteksi spam dengan lebih efektif daripada hanya mengandalkan analisis teks saja.

📦 Model artifact disimpan di folder `/artifact` untuk tahap selanjutnya

![image](https://github.com/user-attachments/assets/ad6ad325-5795-4b8e-9bda-93eb72516104)

### 9. Enrichment (Inference)

#### Jalankan script:

```bash
docker exec -it train bash
python enrich_reviews.py
```

> Output akan diupload ke MinIO sebagai dataset enriched.

![Screenshot 2025-06-26 000704](https://github.com/user-attachments/assets/57635e9d-8b0e-4df6-a14a-6c2da2f6a999)

### 10. Query dengan Trino

#### Masuk Trino CLI:

```bash
docker exec -it trino trino
```

#### Buat Tabel:

```sql
CREATE SCHEMA IF NOT EXISTS hive.default;

CREATE TABLE hive.default.steam_reviews_validated (
    app_id VARCHAR,
    app_name VARCHAR,
    review_text VARCHAR,
    review_score VARCHAR,
    review_votes VARCHAR,
    pred_is_spam VARCHAR,
    pred_spam_score VARCHAR,
    pred_confidence VARCHAR,
    pred_explanation VARCHAR,
    pred_features VARCHAR
)
WITH (
    external_location = 's3a://lakehouse/clean/structured/steam_reviews_validated/',
    format = 'CSV',
    csv_escape = '"',
    csv_quote = '"',
    csv_separator = ',',
    skip_header_line_count = 1
);
```

![Screenshot 2025-06-26 002155](https://github.com/user-attachments/assets/906b87db-6757-4830-add7-8d2df3ab1857)

#### Lihat 10 Baris Pertama:

```sql
SELECT * FROM hive.default.steam_reviews_validated
LIMIT 10;
```

![Screenshot 2025-06-26 002441](https://github.com/user-attachments/assets/ef2bc99e-b1cb-4599-abbd-de4d3bdb2bf1)

### 11. Buat Endpoint FastAPI

Jalankan API:

```bash
docker exec -it train bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

![WhatsApp Image 2025-06-20 at 13 59 18\_38582ee3](https://github.com/user-attachments/assets/7ec3c996-cf24-4c48-9a26-1363ba843ab5)

> API siap digunakan untuk akses data dan model.

## Proses Otomasi 
Seluruh pipeline dapat dijalankan secara otomatis menggunakan script yang bernama run_all.sh di root folder project.
![WhatsApp Image 2025-06-27 at 18 29 01](https://github.com/user-attachments/assets/50ebf39b-b2f9-4691-ac21-df03a5d609ac)
![WhatsApp Image 2025-06-27 at 18 29 09](https://github.com/user-attachments/assets/302e69a5-a362-482b-9d3c-f9e5131fd756)
![WhatsApp Image 2025-06-27 at 18 50 28](https://github.com/user-attachments/assets/428306e0-73a7-405b-b5ec-7acdc5df1bcc)
![WhatsApp Image 2025-06-27 at 18 56 31](https://github.com/user-attachments/assets/8cb19e65-f5b5-4edd-ab7a-4cd4194548df)


## 📊 Tampilan UI Dashboard

Repository Front-end: https://github.com/Ax3lrod/fp-big-data-frontend

![WhatsApp Image 2025-06-27 at 00 33 51](https://github.com/user-attachments/assets/e6c7bce3-8fc4-4c52-a767-ded6e6202901)
