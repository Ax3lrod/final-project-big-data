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

## Dataset
Dataset publik ini berisi lebih dari 6,4 juta ulasan berbahasa Inggris dari pengguna Steam, masing-masing mencakup:
- app_id: ID unik dari game
- app_name: Nama game
- review_text: Isi ulasan pengguna
- review_score: Penilaian (positif/negatif)
- review_votes: Jumlah voting terhadap ulasan tersebut
