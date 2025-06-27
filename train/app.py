import os
os.environ["HF_HOME"] = "/app/.cache/huggingface"
import json
import warnings
from datetime import datetime
from typing import List, Optional, Dict
from collections import Counter
import math
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import string
import uvicorn

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import textstat
from trino.dbapi import connect
import minio
from minio import Minio
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
# Buat direktori jika belum ada
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Beritahu NLTK untuk mencari data di path custom ini
nltk.data.path.append(nltk_data_path)

# Download NLTK data ke path yang sudah ditentukan
print("Downloading NLTK data to local directory...")
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
print("NLTK data download complete.")

warnings.filterwarnings('ignore')

# --- 1. Definisi Kelas dari Notebook Training ---
# Kelas-kelas ini direplikasi persis seperti di notebook untuk memastikan
# arsitektur model dan logika ekstraksi fitur konsisten.

origins = [
    "http://localhost",
    "http://localhost:3000", # Origin dari Next.js development server
    # "https://your-production-frontend.com", # Tambahkan domain frontend produksi Anda di sini
]

class AdvancedFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        self.extreme_sentiment_words = ['amazing', 'awesome', 'perfect', 'terrible', 'fantastic', 'incredible', 'outstanding', 'brilliant', 'excellent', 'awful', 'horrible', 'pathetic', 'worst', 'best', 'epic', 'legendary', 'masterpiece', 'trash', 'garbage']
        self.generic_phrases = ['good game', 'bad game', 'very good', 'very bad', 'recommended', 'not recommended', 'waste of money', 'worth it', 'best game', 'worst game', 'must buy', 'dont buy', 'amazing graphics', 'terrible graphics']
        self.suspicious_patterns = [r'(.)\1{3,}', r'\b(\w+)(\s+\1){2,}\b', r'[!]{2,}', r'[A-Z]{6,}', r'[0-9]{8,}', r'(.{1,3})\1{4,}']
        self.bot_indicators = [r'^\d+/10$', r'^\w+\s*$', r'^(yes|no|good|bad|ok|fine)\s*[.!]*$']

    def _clean_text(self, text):
        if pd.isna(text) or text == '': return ''
        text = str(text).strip()
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)
        return text

    def extract_advanced_features(self, text):
        if pd.isna(text) or text == '': return self._get_default_features()
        text_clean = self._clean_text(text)
        text_lower, tokens, sentences = text_clean.lower(), word_tokenize(text_clean.lower()), sent_tokenize(text_clean)
        features = {}
        features['text_length'], features['word_count'], features['sentence_count'] = len(text_clean), len(tokens), max(len(sentences), 1)
        features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        unique_words = set(tokens)
        features['unique_word_ratio'] = len(unique_words) / len(tokens) if tokens else 0
        features['stopword_ratio'] = sum(1 for w in tokens if w in self.stop_words) / len(tokens) if tokens else 0
        features['type_token_ratio'] = len(unique_words) / len(tokens) if tokens else 0
        features['hapax_legomena_ratio'] = sum(1 for w in Counter(tokens).values() if w == 1) / len(tokens) if tokens else 0
        features['extreme_sentiment_count'] = sum(1 for w in tokens if w in self.extreme_sentiment_words)
        features['extreme_sentiment_ratio'] = features['extreme_sentiment_count'] / len(tokens) if tokens else 0
        features['generic_phrase_count'] = sum(1 for p in self.generic_phrases if p in text_lower)
        features['suspicious_pattern_count'] = sum(len(re.findall(p, text_clean)) for p in self.suspicious_patterns)
        features['bot_indicator_count'] = sum(len(re.findall(p, text_clean, re.IGNORECASE)) for p in self.bot_indicators)
        features['exclamation_count'], features['question_count'] = text_clean.count('!'), text_clean.count('?')
        features['caps_ratio'] = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if text_clean else 0
        features['digit_ratio'] = sum(1 for c in text_clean if c.isdigit()) / len(text_clean) if text_clean else 0
        sentiment_scores = self.sia.polarity_scores(text_clean)
        for k, v in sentiment_scores.items(): features[f'sentiment_{k}'] = v
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text_clean)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text_clean)
            features['automated_readability_index'] = textstat.automated_readability_index(text_clean)
        except: features['flesch_reading_ease'], features['flesch_kincaid_grade'], features['automated_readability_index'] = 50.0, 8.0, 8.0
        features['is_very_short'], features['is_very_long'] = (1 if len(tokens) < 3 else 0), (1 if len(tokens) > 500 else 0)
        features['has_repeated_chars'], features['all_caps_words'] = (1 if re.search(r'(.)\1{3,}', text_clean) else 0), sum(1 for w in tokens if len(w) > 2 and w.isupper())
        return features

    def _get_default_features(self):
        # ... (sama seperti di skrip training)
        return {key: 0 for key in self.get_feature_names()}
    
    def get_feature_names(self): # Helper untuk memastikan urutan konsisten
        return list(self.extract_advanced_features("sample text").keys())


class EnhancedBERTClassifier(nn.Module):
    def __init__(self, model_name, num_features, dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + num_features, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(768, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.3),
            nn.Linear(32, 1)
        )
    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        features_norm = self.feature_norm(features)
        combined = torch.cat([cls_output, features_norm], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

# --- 2. Konfigurasi dan Variabel Global ---
ARTIFACTS_PATH = "artifacts"
CONFIG_PATH = os.path.join(ARTIFACTS_PATH, "config.json")
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_PATH, "spam_model_weights.pth")
TOKENIZER_PATH = os.path.join(ARTIFACTS_PATH, "spam_tokenizer")
SCALER_PARAMS_PATH = os.path.join(ARTIFACTS_PATH, "scaler_params.npz")
TRINO_HOST, TRINO_PORT, TRINO_USER = "trino", 8080, "user"
MINIO_PUBLIC_URL = "http://minio:9000"
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "lakehouse"
VALIDATED_CSV_PATH = "clean/structured/steam_reviews_validated/steam_reviews_validated.csv"
artifacts = {}

# --- 3. Pydantic Models ---
class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=1, example="good game")

class SpamPrediction(BaseModel):
    review_text: str
    is_spam: bool
    spam_score: float
    confidence: str
    explanation: List[str]
    features: Dict[str, float]

class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]

class BatchSummary(BaseModel):
    total_reviews_processed: int; spam_count: int; legit_count: int
    spam_percentage: float; average_spam_score: float

class BatchPredictionResponse(BaseModel):
    predictions: List[SpamPrediction]
    summary: BatchSummary

# Pydantic Model Baru untuk Halaman Utama
class GameInfo(BaseModel):
    app_id: str; app_name: str; poster_url: str; review_count: int
class PaginationInfo(BaseModel):
    total_items: int; total_pages: int; current_page: int; limit: int
class PaginatedGamesResponse(BaseModel):
    pagination: PaginationInfo; data: List[GameInfo]
# Pydantic Model Baru untuk Detail Game
class ReviewInfo(BaseModel):
    review_text: str; review_score: str; is_spam: bool; spam_score: float
class PaginatedReviewsResponse(BaseModel):
    pagination: PaginationInfo; data: List[ReviewInfo]

class NewReviewRequest(BaseModel):
    app_id: str = Field(..., example="730")
    app_name: str = Field(..., example="Counter-Strike: Global Offensive")
    review_text: str = Field(..., min_length=1)
    review_score: str = Field(..., example="1")
    review_votes: str = Field(..., example="0")

class NewReviewResponse(BaseModel):
    message: str
    data_written: Dict

minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

# --- 4. Aplikasi FastAPI & Fungsi Helper ---
app = FastAPI(title="Advanced Spam Detection API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Izinkan origin yang ada di daftar
    allow_credentials=True, # Izinkan cookies (jika diperlukan)
    allow_methods=["*"],    # Izinkan semua metode (GET, POST, dll.)
    allow_headers=["*"],    # Izinkan semua header
)

def append_to_csv_in_minio(bucket: str, object_path: str, new_data_df: pd.DataFrame):
    """
    Mengunduh CSV, menambahkan data, dan meng-upload kembali.
    Ini adalah operasi atomik yang 'mahal' jika file besar.
    """
    try:
        response = minio_client.get_object(bucket, object_path)
        existing_data = response.read()
        existing_df = pd.read_csv(io.BytesIO(existing_data))
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    except Exception:
        print(f"File '{object_path}' tidak ditemukan. Membuat file baru.")
        combined_df = new_data_df
    finally:
        if 'response' in locals(): response.close(); response.release_conn()

    csv_bytes = combined_df.to_csv(index=False).encode('utf-8')
    csv_buffer = io.BytesIO(csv_bytes)

    minio_client.put_object(
        bucket, object_path, data=csv_buffer,
        length=len(csv_bytes), content_type='application/csv'
    )
    print(f"Berhasil meng-update '{object_path}' dengan total {len(combined_df)} baris.")

def get_confidence(score: float) -> str:
    distance = abs(score - 0.5)
    if distance > 0.4: return "Very High"
    if distance > 0.25: return "High"
    if distance > 0.1: return "Medium"
    return "Low"

def get_spam_explanation(features: dict, score: float) -> List[str]:
    if score < 0.5: return ["Review appears to be legitimate."]
    reasons = []
    if features.get('bot_indicator_count', 0) > 0: reasons.append("Matches common bot-like patterns (e.g., single-word review).")
    if features.get('is_very_short'): reasons.append("Review is extremely short.")
    if features.get('suspicious_pattern_count', 0) > 1: reasons.append("Contains suspicious text patterns (e.g., excessive repetition).")
    if features.get('caps_ratio', 0) > 0.5: reasons.append("Excessive use of capital letters.")
    if features.get('extreme_sentiment_ratio', 0) > 0.5: reasons.append("Overly reliant on extreme sentiment words.")
    return reasons if reasons else ["Detected as spam based on a combination of text and feature patterns."]

@app.on_event("startup")
def load_artifacts():
    global artifacts
    print("🚀 Loading advanced model artifacts...")
    for pkg in ['punkt', 'stopwords', 'vader_lexicon']:
        try: nltk.data.find(f'tokenizers/{pkg}')
        except LookupError: nltk.download(pkg, quiet=True)
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(CONFIG_PATH, 'r') as f: config = json.load(f)
        
        feature_extractor = AdvancedFeatureExtractor()
        scaler = StandardScaler()
        scaler_params = np.load(SCALER_PARAMS_PATH)
        scaler.mean_, scaler.scale_ = scaler_params['mean'], scaler_params['scale']
        
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        
        model = EnhancedBERTClassifier(model_name=config['model_name'], num_features=len(config['feature_names']))
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device); model.eval()
        
        artifacts = {"model": model, "tokenizer": tokenizer, "scaler": scaler, "feature_extractor": feature_extractor, "config": config, "device": device}
        print(f"✅ Artifacts loaded successfully on device: {device}")
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        raise RuntimeError(f"Could not load model artifacts: {e}") from e

# --- 5. Endpoints ---
def perform_prediction(review_text: str) -> dict:
    with torch.no_grad():
        features_dict = artifacts['feature_extractor'].extract_advanced_features(review_text)
        features_ordered = np.array([features_dict[name] for name in artifacts['config']['feature_names']]).reshape(1, -1)
        scaled_features = artifacts['scaler'].transform(features_ordered)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float).to(artifacts['device'])
        
        encoding = artifacts['tokenizer'](review_text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
        input_ids = encoding['input_ids'].to(artifacts['device'])
        attention_mask = encoding['attention_mask'].to(artifacts['device'])
        
        score = artifacts['model'](input_ids, attention_mask, features_tensor).item()
    return {"score": score, "features": features_dict}

@app.post("/predict", response_model=SpamPrediction, tags=["Prediction"])
def predict_single(request: ReviewRequest):
    result = perform_prediction(request.review_text)
    score = result['score']
    return SpamPrediction(
        review_text=request.review_text,
        is_spam=score > artifacts['config'].get('prediction_threshold', 0.5),
        spam_score=score,
        confidence=get_confidence(score),
        explanation=get_spam_explanation(result['features'], score),
        features=result['features']
    )

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchReviewRequest):
    if len(request.reviews) > 100: raise HTTPException(413, "Batch size cannot exceed 100.")
    results, spam_count, total_score = [], 0, 0.0
    for review in request.reviews:
        if not review.review_text.strip(): continue
        result = perform_prediction(review.review_text)
        score = result['score']
        is_spam = score > artifacts['config'].get('prediction_threshold', 0.5)
        if is_spam: spam_count += 1
        total_score += score
        results.append(SpamPrediction(review_text=review.review_text, is_spam=is_spam, spam_score=score, confidence=get_confidence(score), explanation=get_spam_explanation(result['features'], score), features=result['features']))
    total = len(results)
    summary = BatchSummary(total_reviews_processed=total, spam_count=spam_count, legit_count=total-spam_count, spam_percentage=(spam_count/total*100 if total>0 else 0), average_spam_score=(total_score/total if total>0 else 0))
    return BatchPredictionResponse(predictions=results, summary=summary)

@app.get("/images/posters/{app_id}.jpg",
         tags=["Image Proxy"],
         responses={
             200: {"content": {"image/jpeg": {}}},
             404: {"description": "Image not found"}
         })
def get_game_poster(app_id: str):
    """
    Bertindak sebagai proxy aman untuk mengambil gambar poster dari MinIO.
    Ini memungkinkan bucket MinIO tetap privat.
    """
    object_name = f"raw/unstructured/steam_images/{app_id}.jpg"
    
    try:
        # Gunakan klien MinIO internal untuk mengambil objek
        response = minio_client.get_object(BUCKET_NAME, object_name)
        
        # Alirkan konten gambar langsung ke klien
        # Ini lebih efisien memori daripada memuat seluruh file
        return StreamingResponse(response, media_type="image/jpeg")

    except Exception as e:
        # Tangani jika file tidak ditemukan atau error lain
        print(f"Error fetching image {object_name} from MinIO: {e}")
        # Kembalikan gambar placeholder atau error 404
        # Untuk kesederhanaan, kita kembalikan 404
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/games", response_model=PaginatedGamesResponse, tags=["Frontend Data"])
def get_all_games(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    """
    Mengambil daftar game unik dengan pagination untuk ditampilkan di halaman utama.
    """
    conn = None
    try:
        conn = connect(host=TRINO_HOST, port=TRINO_PORT, user=TRINO_USER)
        cur = conn.cursor()
        
        # Query 1: Hitung total game unik untuk metadata pagination
        cur.execute("SELECT COUNT(DISTINCT app_id) FROM hive.default.steam_reviews_validated")
        total_items = cur.fetchone()[0]
        total_pages = math.ceil(total_items / limit)
        offset = (page - 1) * limit
        
        # Query 2: Ambil data game untuk halaman saat ini
        query = f"""
            SELECT
                app_id,
                any_value(app_name) as app_name,
                COUNT(*) as review_count
            FROM
                hive.default.steam_reviews_validated
            GROUP BY
                app_id
            ORDER BY
                review_count DESC
            OFFSET {offset}
            LIMIT {limit}
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        game_data = [
            GameInfo(
                app_id=row[0],
                app_name=row[1],
                poster_url=f"/images/posters/{row[0]}.jpg",
                review_count=row[2]
            ) for row in rows
        ]

        pagination_info = PaginationInfo(
            total_items=total_items, total_pages=total_pages,
            current_page=page, limit=limit
        )
        
        return PaginatedGamesResponse(pagination=pagination_info, data=game_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trino query failed: {e}")
    finally:
        if conn: conn.close()


@app.get("/games/{app_id}/reviews", response_model=PaginatedReviewsResponse, tags=["Frontend Data"])
def get_reviews_for_game(app_id: str, page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=50)):
    """
    Mengambil semua ulasan untuk satu game spesifik, dengan pagination.
    """
    conn = None
    try:
        conn = connect(host=TRINO_HOST, port=TRINO_PORT, user=TRINO_USER)
        cur = conn.cursor()
        
        # Query 1: Hitung total review untuk game ini
        cur.execute(f"SELECT COUNT(*) FROM hive.default.steam_reviews_validated WHERE app_id = '{app_id}'")
        total_items = cur.fetchone()[0]
        if total_items == 0:
            raise HTTPException(status_code=404, detail=f"Game with app_id '{app_id}' not found.")
        
        total_pages = math.ceil(total_items / limit)
        offset = (page - 1) * limit
        
        # Query 2: Ambil data review untuk halaman saat ini
        query = f"""
            SELECT
                review_text,
                review_score,
                CAST(pred_is_spam AS BOOLEAN) AS is_spam,
                CAST(pred_spam_score AS DOUBLE) AS spam_score
            FROM
                hive.default.steam_reviews_validated
            WHERE
                app_id = '{app_id}'
            ORDER BY
                pred_spam_score DESC
            OFFSET {offset}
            LIMIT {limit}
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        review_data = [
            ReviewInfo(
                review_text=row[0], review_score=row[1],
                is_spam=row[2], spam_score=row[3]
            ) for row in rows
        ]
        
        pagination_info = PaginationInfo(
            total_items=total_items, total_pages=total_pages,
            current_page=page, limit=limit
        )
        
        return PaginatedReviewsResponse(pagination=pagination_info, data=review_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trino query failed: {e}")
    finally:
        if conn: conn.close()

@app.post("/reviews", response_model=NewReviewResponse, status_code=201, tags=["Real-time Ingestion"])
def post_and_save_review(request: NewReviewRequest, background_tasks: BackgroundTasks):
    """
    Menerima ulasan baru, melakukan prediksi, dan langsung menambahkan
    hasilnya ke file CSV di MinIO yang dibaca oleh Trino.
    """
    if not artifacts:
        raise HTTPException(status_code=503, detail="Model ML tidak siap.")

    # 1. Lakukan prediksi
    try:
        prediction_result = perform_prediction(request.review_text)
        score = prediction_result['score']
        features = prediction_result['features']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {e}")

    # 2. Siapkan data lengkap yang akan ditambahkan
    enriched_data = {
        "app_id": request.app_id,
        "app_name": request.app_name,
        "review_text": request.review_text,
        "review_score": request.review_score,
        "review_votes": request.review_votes,
        "pred_is_spam": score > artifacts['config'].get('prediction_threshold', 0.5),
        "pred_spam_score": score,
        "pred_confidence": get_confidence(score),
        "pred_explanation": get_spam_explanation(features, score),
        "pred_features": json.dumps(features)
    }
    
    # 3. Buat DataFrame dari data baru
    new_data_df = pd.DataFrame([enriched_data])

    # 4. Jalankan operasi I/O yang lambat di background
    # Ini membuat endpoint merespons dengan cepat kepada pengguna
    # sementara proses penulisan ke MinIO terjadi di belakang layar.
    background_tasks.add_task(
        append_to_csv_in_minio,
        BUCKET_NAME,
        VALIDATED_CSV_PATH,
        new_data_df
    )
    
    return NewReviewResponse(
        message="Review received and scheduled for saving.",
        data_written=enriched_data
    )

# --- Jalankan server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)