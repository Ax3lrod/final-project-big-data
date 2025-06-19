from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, when, udf
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --- 1. Konfigurasi Spark Session & MinIO ---
spark = SparkSession.builder \
    .appName("SpamReviewModelTrainingFinal") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# --- 2. Tentukan Path Input dan Output ---
CLEAN_DATA_PATH = "s3a://lakehouse/clean/structured/steam_reviews.csv"
LOCAL_MODEL_OUTPUT_PATH = "spam_detection_pipeline_final" 

# --- 3. Fungsi Utama untuk Training ---
def main():
    print("--- Memulai Proses Training Model ---")

    # A. Load Data Bersih dari MinIO
    try:
        print(f"Membaca data bersih dari: {CLEAN_DATA_PATH}")
        df = spark.read.csv(CLEAN_DATA_PATH, header=True, inferSchema=True)
        # Pastikan semua kolom yang dibutuhkan ada dan tidak null
        df = df.dropna(subset=['review_text', 'review_votes', 'review_score'])
    except Exception as e:
        print(f"❌ ERROR: Gagal membaca data dari MinIO. Error: {e}")
        return

    # B. Heuristic Labeling yang Disempurnakan
    print("Membuat kolom 'label' berdasarkan aturan heuristik yang lebih cerdas...")
    # [PENYEMPURNAAN] Menambahkan kondisi `review_score == 1` ke dalam aturan.
    df_labeled = df.withColumn("label", 
        when(
            (length(col("review_text")) < 30) & 
            (col("review_votes") == 0) &
            (col("review_score") == 1), 
            1.0  # Dianggap Spam/Bot
        ).otherwise(0.0) # Dianggap Ulasan Asli
    )
    
    print("Distribusi Label:")
    df_labeled.groupBy("label").count().show()
    
    # C. Definisikan Tahapan Pipeline ML (Tidak ada perubahan di sini)
    print("Mendefinisikan tahapan pipeline ML...")
    tokenizer = Tokenizer(inputCol="review_text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
    idf = IDF(inputCol="raw_features", outputCol="weighted_features")
    assembler = VectorAssembler(inputCols=["weighted_features"], outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    # D. Bangun Pipeline
    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, assembler, lr])

    # E. Split Data dan Latih Model
    print("Membagi data menjadi set training (80%) dan testing (20%)...")
    (trainingData, testData) = df_labeled.randomSplit([0.8, 0.2], seed=42)
    print("Memulai training pipeline model...")
    model = pipeline.fit(trainingData)
    print("✅ Training selesai.")

    # F. Evaluasi Model dan Ekstrak Probabilitas
    print("Mengevaluasi model pada data tes...")
    predictions = model.transform(testData)
    
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"📈 Performa Model - Area Under ROC (AUC) pada data tes: {auc:.4f}")

    # G. Simpan Artifact Model ke Filesystem Lokal
    print(f"Menyimpan artifact model ke path LOKAL: {LOCAL_MODEL_OUTPUT_PATH}")
    try:
        model.write().overwrite().save(LOCAL_MODEL_OUTPUT_PATH)
        print(f"✅ Model berhasil disimpan secara lokal. Anda bisa mengambilnya dari container.")
    except Exception as e:
        print(f"❌ ERROR: Gagal menyimpan model secara lokal. Error: {e}")

    print("--- Proses Training Selesai ---")

# --- Jalankan Fungsi Utama ---
if __name__ == "__main__":
    main()
    spark.stop()