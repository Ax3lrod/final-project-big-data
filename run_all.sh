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

if [ $? -ne 0 ]; then
    log_error "Gagal ingestion data CSV."
    exit 1
else
    log "Data CSV berhasil diupload ke MinIO."
fi

# --- 4. Jalankan Producer & Consumer untuk Gambar ---
log "Langkah 4: Ingestion data unstructured (gambar)..."

docker exec -d producer sh -c "python producer_img.py"

log "Menjalankan Consumer Gambar..."
docker exec consumer sh -c "python consumer_img.py"

if [ $? -ne 0 ]; then
    log_error "Gagal ingestion data gambar."
    exit 1
else
    log "Data gambar berhasil diupload ke MinIO."
fi
