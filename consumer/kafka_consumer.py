from kafka import KafkaConsumer
import pandas as pd
from datetime import datetime, timedelta
import json
from minio import Minio
from minio.error import S3Error

# MinIO setup
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "lakehouse"
object_path = "raw/structured/steam_reviews.csv"

# Pastikan bucket MinIO ada
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)

# Kafka Consumer setup
topic = 'steam-reviews'

consumer = KafkaConsumer(
    topic,
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Buffer dan timer 10 menit
buffer = []
interval = timedelta(minutes=10)
start_time = datetime.now()

for msg in consumer:
    data = msg.value
    print(f"Terima data: {data}")
    buffer.append(data)

    if datetime.now() - start_time >= interval:
        filename = "steam_reviews.csv"

        # Simpan ke lokal
        df = pd.DataFrame(buffer)
        df.to_csv(filename, index=False)
        print(f"Saved locally: {filename}")

        # Upload ke MinIO
        minio_client.fput_object(bucket_name, object_path, filename)
        print(f"Uploaded to MinIO as {object_path}")
        break
