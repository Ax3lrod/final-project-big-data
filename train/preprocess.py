from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim, length
from minio import Minio
import io

# Spark session
spark = SparkSession.builder \
    .appName("SteamReviewCleaning") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Baca dari MinIO
df = spark.read.csv("s3a://lakehouse/raw/structured/steam_reviews.csv", header=True, inferSchema=True)

# Preprocessing kolom review_text
df_clean = df.dropna(subset=["review_text"]) \
    .withColumn("review_text", lower(col("review_text"))) \
    .withColumn("review_text", regexp_replace(col("review_text"), r"[^a-z0-9\s]", "")) \
    .withColumn("review_text", regexp_replace(col("review_text"), r"\s+", " ")) \
    .withColumn("review_text", trim(col("review_text"))) \
    .filter(length(col("review_text")) > 10) \
    .dropna()

# Simpan ke MinIO (sementara)
temp_path = "s3a://lakehouse/clean/structured/temp_csv_output"
df_clean.coalesce(1).write.mode("overwrite") \
    .option("header", True) \
    .csv(temp_path)

# Koneksi MinIO
minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Copy dari part-*.csv ke steam_reviews.csv
bucket = "lakehouse"
temp_prefix = "clean/structured/temp_csv_output/"
final_object = "clean/structured/steam_reviews.csv"

# Cari part-0000*.csv
for obj in minio_client.list_objects(bucket, prefix=temp_prefix, recursive=True):
    if obj.object_name.endswith(".csv"):
        response = minio_client.get_object(bucket, obj.object_name)
        content = response.read()
        minio_client.put_object(
            bucket,
            final_object,
            data=io.BytesIO(content),
            length=len(content),
            content_type="text/csv"
        )
        print(f"Saved single CSV to: {final_object}")
        break

# (Opsional) Bersihkan folder temp
for obj in minio_client.list_objects(bucket, prefix=temp_prefix, recursive=True):
    minio_client.remove_object(bucket, obj.object_name)
    print(f"Deleted: {obj.object_name}")

print("Data cleaning completed and uploaded to MinIO.")
