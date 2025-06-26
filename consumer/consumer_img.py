from kafka import KafkaConsumer
from minio import Minio
from minio.error import S3Error
import time
import os

# Setup folder lokal
local_folder = "steam_images"
os.makedirs(local_folder, exist_ok=True)

# Setup MinIO
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "lakehouse"
object_prefix = "raw/unstructured/steam_images/"

if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# Kafka Consumer
consumer = KafkaConsumer(
    "steam-posters-images",
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=False
)

# Proses hanya 51 gambar
max_images = 51
count = 0

for msg in consumer:
    filename = msg.key.decode('utf-8') if msg.key else f"image_{count}.jpg"
    local_path = os.path.join(local_folder, filename)

    # Simpan lokal
    with open(local_path, "wb") as f:
        f.write(msg.value)
    print(f"Disimpan lokal: {filename}")

    # Upload ke MinIO
    minio_path = object_prefix + filename
    minio_client.fput_object(bucket_name, minio_path, local_path)
    print(f"Upload ke MinIO: {minio_path}")
    time.sleep(0.2)

    count += 1
    if count >= max_images:
        print("Consumer telah selesai memproses gambar.")
        break
