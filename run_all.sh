#!/bin/bash

# --- Pengaturan Warna untuk Log ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# --- 1. Setup Docker Environment ---
log "Langkah 1: Reset dan memulai ulang container..."
docker-compose down -v --remove-orphans
docker-compose up -d

log "Menunggu service stabil (30 detik)..."
sleep 30

# --- 2. Kafka Topics Creation ---
log "Langkah 2: Membuat topik Kafka..."

docker exec kafka kafka-topics --create \
    --topic steam_reviews \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1 --if-not-exists

docker exec kafka kafka-topics --create \
    --topic steam-posters-images \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1 --if-not-exists

if [ $? -ne 0 ]; then
    log_error "Gagal membuat topik Kafka."
    exit 1
else
    log "Topik Kafka siap."
fi

# --- 3. Jalankan Producer & Consumer untuk CSV ---
log "Langkah 3: Ingestion data structured..."

docker exec -d producer sh -c "pip install -q kafka-python && python producer_csv.py"

log "Menjalankan Consumer CSV..."
docker exec consumer sh -c "pip install -q kafka-python pandas minio && python consumer_csv.py"

# --- 4. Jalankan Producer & Consumer untuk Gambar ---
log "Langkah 4: Ingestion data unstructured (gambar)..."

docker exec -d producer sh -c "python producer_img.py"

log "Menjalankan Consumer Gambar..."
docker exec consumer sh -c "python consumer_img.py" >/dev/null 2>&1 &

# --- 5. Preprocessing ---
log "Langkah 5: Preprocessing dengan PySpark..."

docker exec train sh -c "pip install -r requirements.txt"
docker exec train sh -c "spark-submit preprocess.py"

if [ $? -ne 0 ]; then
    log_error "Gagal preprocessing."
    exit 1
else
    log "Preprocessing selesai."
fi

# --- 6. Enrichment Data ---
log "Langkah 6: Enrichment data dengan model spam detection..."
docker exec -u 0 train sh -c "python enrich_reviews.py"

if [ $? -ne 0 ]; then
    log_error "Gagal enrichment data."
    exit 1
else
    log "Enrichment selesai."
fi

# --- 7. Pembuatan Tabel di Trino ---
log "Langkah 7: Menyiapkan tabel di Trino (steam_reviews_validated)..."

docker exec trino trino --execute "
CREATE TABLE IF NOT EXISTS hive.default.steam_reviews_validated (
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
    csv_escape = '\"',
    csv_quote = '\"',
    csv_separator = ',',
    skip_header_line_count = 1
);"

if [ $? -ne 0 ]; then
    log_error "Gagal membuat tabel Trino."
    exit 1
else
    log "Tabel Trino berhasil dibuat."
fi

# --- 8. Menjalankan Inference API ---
log "Langkah 8: Menjalankan API inference..."

docker exec train sh -c "pip install trino"

docker exec train sh -c "python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload"

log "Pipeline selesai dijalankan!"