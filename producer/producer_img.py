import os
from kafka import KafkaProducer
import time

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: v 
)

folder_path = 'top_50_popular_posters_archive'

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'rb') as img_file:
            img_bytes = img_file.read()
            producer.send('steam-posters-images', value=img_bytes, key=filename.encode('utf-8'))
            print(f"Sent: {filename}")
            time.sleep(0.2)

producer.flush()
