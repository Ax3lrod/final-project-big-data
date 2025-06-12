from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType
from delta import configure_spark_with_delta_pip

# Define Kafka and Delta parameters
KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "steam-reviews"
DELTA_PATH = "delta_lake/reviews"

# Schema dari Kafka message
schema = StructType() \
    .add("app_id", StringType()) \
    .add("app_name", StringType()) \
    .add("review_text", StringType()) \
    .add("review_score", StringType()) \
    .add("review_votes", StringType())

# Spark session setup
builder = SparkSession.builder \
    .appName("SteamReviewsDeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Streaming dari Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Decode value JSON
parsed = df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# Tulis ke Delta
query = parsed.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "delta_lake/checkpoints/") \
    .start(DELTA_PATH)

query.awaitTermination()
