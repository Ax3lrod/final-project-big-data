import csv
import json
import time
from kafka import KafkaProducer

# Kafka producer setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

start_time = time.time()  # Catat waktu mulai

with open('data/dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        current_time = time.time()
        elapsed = current_time - start_time

        if elapsed > 300:  # 300 detik = 5 menit
            print("Sudah 5 menit, berhenti mengirim data.")
            break

        producer.send('steam-reviews', value=row)
        print("Sent:", row)
        time.sleep(0.5)  # Simulasi streaming
