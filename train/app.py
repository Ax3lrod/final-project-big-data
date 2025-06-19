import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# --- 1. Konfigurasi dan Variabel Global ---

# Atur variabel lingkungan untuk PySpark
# Ini penting agar PySpark bisa menemukan driver yang benar
os.environ['PYSPARK_PYTHON'] = os.sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = os.sys.executable

# Path ke artifact model yang disimpan secara lokal
# Sesuaikan nama folder ini jika berbeda
LOCAL_MODEL_PATH = "spam_detection_pipeline_final"

# Variabel global untuk menampung Spark session dan model yang sudah di-load
# Ini akan diinisialisasi saat aplikasi startup
spark: SparkSession = None
model: PipelineModel = None

# --- 2. Pydantic Models untuk Validasi Request & Response ---

# Model untuk data yang masuk dari request API
class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=1, example="this game is absolutely amazing, highly recommended!")

# Model untuk data yang akan dikirim sebagai response
class ReviewResponse(BaseModel):
    review_text: str
    spam_score: float = Field(..., ge=0, le=1, description="Skor kemungkinan review adalah spam (0 = bukan spam, 1 = spam).")
    is_spam: bool = Field(..., description="Prediksi akhir apakah review dianggap spam (jika skor > 0.5).")

# --- 3. Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="Steam Review Spam Detection API",
    description="API untuk memprediksi skor spam dari ulasan game Steam menggunakan model PySpark ML.",
    version="1.0.0"
)

# --- 4. Fungsi Startup dan Shutdown ---

@app.on_event("startup")
def startup_event():
    """
    Fungsi yang dijalankan SEKALI saat aplikasi dimulai.
    Menginisialisasi Spark Session dan me-load model dari disk.
    Ini jauh lebih efisien daripada me-load model untuk setiap request.
    """
    global spark, model
    
    print("🚀 Aplikasi memulai... Menginisialisasi Spark Session...")
    # Buat Spark Session lokal yang ringan
    spark = SparkSession.builder \
        .appName("SpamDetectionInference") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    print(f"🔄 Me-load model dari path: {LOCAL_MODEL_PATH}")
    try:
        model = PipelineModel.load(LOCAL_MODEL_PATH)
        print("✅ Model berhasil di-load dan siap menerima request.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Gagal me-load model. Aplikasi tidak akan berfungsi. Error: {e}")
        # Dalam produksi, Anda mungkin ingin menghentikan aplikasi jika model gagal dimuat.
        
@app.on_event("shutdown")
def shutdown_event():
    """
    Fungsi yang dijalankan SEKALI saat aplikasi dimatikan.
    Menghentikan Spark Session untuk melepaskan resource.
    """
    global spark
    if spark:
        print("🔌 Aplikasi mematikan... Menghentikan Spark Session...")
        spark.stop()

# --- 5. Endpoint API untuk Prediksi ---

@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
def predict_spam(request: ReviewRequest):
    """
    Menerima teks ulasan dan mengembalikan skor kemungkinan spam.
    """
    if not model or not spark:
        raise HTTPException(status_code=503, detail="Model atau Spark Session tidak tersedia. Aplikasi mungkin sedang dalam proses startup atau mengalami error.")

    try:
        # 1. Buat DataFrame Spark kecil dari input request
        # Nama kolom ('review_text') HARUS sama dengan inputCol pada tahap pertama pipeline (Tokenizer)
        schema = ["review_text"]
        data = [(request.review_text,)]
        review_df = spark.createDataFrame(data, schema)

        # 2. Lakukan transformasi dan prediksi menggunakan pipeline model
        prediction_df = model.transform(review_df)

        # 3. Ekstrak hasil probabilitas
        # Kolom 'probability' adalah vektor [prob_bukan_spam, prob_spam]
        result = prediction_df.select("probability").first()
        
        if not result:
            raise HTTPException(status_code=500, detail="Model tidak menghasilkan prediksi.")

        # Ambil skor probabilitas untuk kelas "spam" (indeks 1)
        spam_probability = float(result.probability[1])
        
        # Tentukan label boolean berdasarkan threshold 0.5
        is_spam_prediction = spam_probability > 0.5

        return ReviewResponse(
            review_text=request.review_text,
            spam_score=spam_probability,
            is_spam=is_spam_prediction
        )

    except Exception as e:
        # Menangkap error tak terduga selama proses prediksi
        raise HTTPException(status_code=500, detail=f"Terjadi error internal saat prediksi: {str(e)}")

# Endpoint root untuk pengecekan kesehatan
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running"}