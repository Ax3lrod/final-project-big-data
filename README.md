#  Final Project Big Data - Sistem Deteksi Spam/Bot Review di Steam

Kelompok B13
- Fiorenza Adelia Nalle 5027231053
- Harwinda 5027231079
- Aryasatya Alaauddin 5027231082

Final project ini bertujuan membangun prototipe sistem deteksi review game palsu di Steam menggunakan arsitektur hybrid (streaming + batching). 
Sistem ini dibangun untuk menghadapi tantangan manipulasi ulasan pengguna seperti **review bombing**, **spam bot**, dan **fake reviews** yang umum ditemukan di platform distribusi digital terbesar, Steam.

## Laporan Lengkap
[Laporan Lengkap Proyek Steam Review Detection](https://docs.google.com/document/d/1cxuOzms_iEBbs4OWGHUQYS4syrmBDehZl9OaO72FQg8/edit?usp=sharing)

## Latar Belakang
Steam adalah platform distribusi game terbesar di dunia dengan lebih dari 132 juta pengguna aktif bulanan dan 143.000+ judul game hingga tahun 2025. Dengan jumlah review yang terus bertambah, review pengguna menjadi aspek krusial dalam proses pengambilan keputusan calon pembeli game. Namun, manipulasi review oleh bot, review palsu dari developer, hingga review bombing yang dilakukan secara masif masih menjadi tantangan besar.
Untuk menjawab tantangan ini, prototipe ini menggunakan pendekatan pemrosesan data besar berbasis hybrid architecture dengan menggabungkan streaming dan batch processing, serta memanfaatkan teknologi modern seperti Kafka, MinIO, PySpark, Hive Metastore, dan Trino.

## Arsitektur Sistem
![arsitektur](https://github.com/user-attachments/assets/6e0aa796-7cd4-433c-9914-aeca8ebbe616)
Alur:
1. Ingestion & Streaming Layer
  - Kafka berperan sebagai message broker yang menangani alur data secara real-time.
  - Kafka Producer membaca dataset review (dalam format CSV) dan mengirimkannya ke topik tertentu di Kafka.
  - Kafka Consumer mengambil data dari topik tersebut, lalu langsung menyimpannya ke MinIO (Raw Layer) dalam bentuk file (misalnya CSV atau JSON).
2. Preprocessing Layer
  - PySpark mengambil data dari MinIO Raw Layer, melakukan data cleaning dan transformasi (menghapus null, formatting, filtering).
  - Hasil dari preprocessing ditulis ulang ke MinIO Clean Layer sebagai data terstruktur yang siap digunakan.
3. Metadata & Query Layer
  - Hive Metastore mencatat metadata dari file di Clean Layer, termasuk schema, path, dan format.
  - Trino memungkinkan pengguna menjalankan query SQL langsung ke data yang tersimpan di MinIO menggunakan informasi dari Hive.
4. Application Layer
  - FastAPI berfungsi sebagai backend API layer untuk melayani permintaan dari aplikasi, termasuk mengambil data yang sudah diproses.
  - Streamlit digunakan untuk membangun dashboard interaktif bagi pengguna akhir, seperti visualisasi review game.

## Dataset
Dataset publik ini berisi lebih dari 6,4 juta ulasan berbahasa Inggris dari pengguna Steam, masing-masing mencakup:
- app_id: ID unik dari game
- app_name: Nama game
- review_text: Isi ulasan pengguna
- review_score: Penilaian (positif/negatif)
- review_votes: Jumlah voting terhadap ulasan tersebut

## Alasan Pemilihan Dataset
Dataset review Steam dipilih karena:

1. **Relevan dengan Masalah Nyata**  
   Manipulasi review seperti spam, bot, dan review palsu sering terjadi di Steam, sehingga dataset ini sangat cocok untuk studi deteksi review palsu.
2. **Volume Data Besar**  
   Dataset ini berisi jutaan review, sehingga ideal untuk menguji sistem big data dan arsitektur hybrid (streaming + batch).
3. **Fitur Lengkap**  
   Terdapat atribut penting seperti teks review, skor, dan jumlah voting, yang sangat mendukung analisis dan pembuatan model deteksi spam/bot.
4. **Data Publik**  
   Dataset ini tersedia secara publik, sehingga mudah diakses dan digunakan tanpa kendala lisensi.

Dengan alasan-alasan tersebut, dataset ini sangat sesuai untuk membangun dan menguji sistem deteksi review palsu berbasis big data.

## Langkah Menjalankan
1. Jalankan seluruh container
   ```
   docker-compose up -d
   ```
   ![Screenshot 2025-06-20 122648](https://github.com/user-attachments/assets/dded2e4c-f8de-4d93-9f2b-f2e39f9d12af)
2. Buat topik kafka
   ```
   docker exec -it kafka bash

   kafka-topics --create \
        --topic steam_reviews \
        --bootstrap-server localhost:9092 \
        --partitions 1 \
        --replication-factor 1
   ```
   ![Screenshot 2025-06-20 123739](https://github.com/user-attachments/assets/4bd078f6-1b4b-4de6-97d7-681454b69925)
3. Jalankan Kafka Producer
   ```
   docker exec -it producer bash

   pip install kafka-python
  
   python kafka_producer.py
   ```
   ![Screenshot 2025-06-20 123812](https://github.com/user-attachments/assets/14ca9fd0-dd4e-445d-ad9a-623cf567ed0e)
   ![Screenshot 2025-06-20 124228](https://github.com/user-attachments/assets/260b9ed5-3e44-44c8-9822-8b8be32b7f85)
3. Jalankan Kafka Consumer
   ```
   docker exec -it consumer bash

   pip install kafka-python pandas minio
  
   python kafka_consumer.py
   ```
   ![Screenshot 2025-06-20 123805](https://github.com/user-attachments/assets/65d5bac5-439d-4c83-ae58-f44650b8c87e)
   ![Screenshot 2025-06-20 124247](https://github.com/user-attachments/assets/a2f938ec-61a3-4609-8782-1a13cbdb3f7b)
4. Upload data unstructured ke MinIO
   ```
   python upload_unstructured.py
   ```
   ![image](https://github.com/user-attachments/assets/9b6d4da1-96d2-4fe8-b048-11880ca98163)
5. Lihat di UI MinIO untuk mengecek apakah raw data sudah tersimpan
   ![image](https://github.com/user-attachments/assets/f6359d97-8188-4bf9-89e7-678185ba256b)
   ![image](https://github.com/user-attachments/assets/972350ea-f25e-462d-9f66-3641905006dd)
   ![image](https://github.com/user-attachments/assets/b642e89d-9c27-4323-b706-882207e1381a)
6. Jalankan PySpark untuk preprocessing
   ```
   docker exec -it train bash

   pip install minio
  
   spark-submit preprocess.py
   ```
   ![image](https://github.com/user-attachments/assets/81e0fb24-198a-49eb-8187-4b21d0eb4915)
   ![image](https://github.com/user-attachments/assets/b0ca0183-4865-4890-aaf4-c9fc4a481bf6)
7. Lihat di UI MinIO untuk mengecek apakah clean data sudah tersimpan
   ![image](https://github.com/user-attachments/assets/3c57061b-95f9-4c9a-b27a-95aa90efc7a6)
   ![image](https://github.com/user-attachments/assets/74b246a6-1093-40e6-a01e-7d61ebec58d5)


8. Masuk ke trino
   ```
   docker exec -it trino trino
   ```
   Buat schema dan tabel untuk steam_review

   ```sql
   CREATE SCHEMA IF NOT EXISTS hive.default;

   CREATE TABLE hive.default.steam_review (
       app_id VARCHAR,
       app_name VARCHAR,
       review_text VARCHAR,
       review_score VARCHAR,
       review_votes VARCHAR
   )
   WITH (
       external_location = 's3a://lakehouse/clean/ structured/',
       format = 'CSV',
       skip_header_line_count = 1
   );
   ```
   ![image](https://github.com/user-attachments/assets/cc094a52-5811-4788-a840-0790235f4343)

   Tampilkan 10 baris pertama dari steam_review
   ```sql
   SELECT * FROM hive.default.steam_review
   LIMIT 10;
   ```
   ![image](https://github.com/user-attachments/assets/4220fe9c-8533-4c8a-9557-64f5d0216bf6)

9. Memulai training model

   Masuk ke container train
   ```
   docker exec -it train bash
   ```
   Install semua dependencies yang diperlukan
   ```
   pip install -r requirements.txt
   ```
   ![WhatsApp Image 2025-06-20 at 14 00 08_06dc6227](https://github.com/user-attachments/assets/88bbc4a0-78a9-4d26-9bb9-ac3332df06aa)

   Mulai training.
   ```
   spark-submit train.py
   ```
   ![WhatsApp Image 2025-06-20 at 14 01 03_ddef6bc0](https://github.com/user-attachments/assets/e196c4ed-d65b-42b7-a92e-a78e48c8476d)

10. Membuat endpoint Fast API

    Di dalam container train. Jalankan command:
    ```
    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    ![WhatsApp Image 2025-06-20 at 13 59 18_38582ee3](https://github.com/user-attachments/assets/7ec3c996-cf24-4c48-9a26-1363ba843ab5)

11. Membuat UI Streamlit

    Masuk ke container frontend
    ```
    docker exec -it frontend bash
    ```
    Install semua dependencies yang diperlukan
    ```
    pip install -r requirements.txt
    ```
    Nyalakan Streamlit
    ```
    streamlit run app.py --server.port=7000 --server.address=0.0.0.0
    ```
    ![WhatsApp Image 2025-06-20 at 14 38 00_411fbe96](https://github.com/user-attachments/assets/ff5ee111-0bfa-4a78-ad7a-93fbd3c8489e)

