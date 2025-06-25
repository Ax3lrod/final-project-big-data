import os
import json
import warnings
from typing import Dict, List
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import string
import io

from minio import Minio
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import textstat

warnings.filterwarnings('ignore')

# --- 1. Definisi Kelas (HARUS SAMA PERSIS DENGAN SAAT TRAINING) ---

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

    def extract_advanced_features(self, text: str) -> Dict[str, float]:
        if pd.isna(text) or text == '': return self._get_default_features()
        text_clean = self._clean_text(text)
        text_lower = text_clean.lower()
        try:
            tokens = word_tokenize(text_lower)
            sentences = sent_tokenize(text_clean)
        except:
            tokens, sentences = text_lower.split(), text_clean.split('.')
        features = {}
        features['text_length'] = float(len(text_clean))
        features['word_count'] = float(len(tokens))
        features['sentence_count'] = float(max(len(sentences), 1))
        features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0.0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0.0
        unique_words = set(tokens)
        features['unique_word_ratio'] = len(unique_words) / len(tokens) if tokens else 0.0
        features['stopword_ratio'] = sum(1 for w in tokens if w in self.stop_words) / len(tokens) if tokens else 0.0
        features['type_token_ratio'] = len(unique_words) / len(tokens) if tokens else 0.0
        features['hapax_legomena_ratio'] = sum(1 for w in Counter(tokens).values() if w == 1) / len(tokens) if tokens else 0.0
        features['extreme_sentiment_count'] = float(sum(1 for w in tokens if w in self.extreme_sentiment_words))
        features['extreme_sentiment_ratio'] = features['extreme_sentiment_count'] / len(tokens) if tokens else 0.0
        features['generic_phrase_count'] = float(sum(1 for p in self.generic_phrases if p in text_lower))
        features['suspicious_pattern_count'] = float(sum(len(re.findall(p, text_clean)) for p in self.suspicious_patterns))
        features['bot_indicator_count'] = float(sum(len(re.findall(p, text_clean, re.IGNORECASE)) for p in self.bot_indicators))
        features['exclamation_count'], features['question_count'] = float(text_clean.count('!')), float(text_clean.count('?'))
        features['caps_ratio'] = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if text_clean else 0.0
        features['digit_ratio'] = sum(1 for c in text_clean if c.isdigit()) / len(text_clean) if text_clean else 0.0
        sentiment_scores = self.sia.polarity_scores(text_clean)
        for k, v in sentiment_scores.items(): features[f'sentiment_{k}'] = v
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text_clean)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text_clean)
            features['automated_readability_index'] = textstat.automated_readability_index(text_clean)
        except: features['flesch_reading_ease'], features['flesch_kincaid_grade'], features['automated_readability_index'] = 50.0, 8.0, 8.0
        features['is_very_short'] = 1.0 if len(tokens) < 3 else 0.0
        features['is_very_long'] = 1.0 if len(tokens) > 500 else 0.0
        features['has_repeated_chars'] = 1.0 if re.search(r'(.)\1{3,}', text_clean) else 0.0
        features['all_caps_words'] = float(sum(1 for w in tokens if len(w) > 2 and w.isupper()))
        return features

    def get_feature_names(self) -> List[str]:
        return list(self.extract_advanced_features("sample text").keys())

    def _get_default_features(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.get_feature_names()}


class EnhancedBERTClassifier(nn.Module):
    def __init__(self, model_name, num_features, dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.classifier = nn.Sequential(nn.Linear(bert_dim + num_features, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(dropout), nn.Linear(768, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(dropout * 0.7), nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout * 0.3), nn.Linear(32, 1))
    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        features_norm = self.feature_norm(features)
        combined = torch.cat([cls_output, features_norm], dim=1)
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

# --- 2. Konfigurasi ---
MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY = "minio:9000", "minioadmin", "minioadmin"
BUCKET_NAME, INPUT_OBJECT_PATH, OUTPUT_OBJECT_PATH = "lakehouse", "clean/structured/steam_reviews.csv", "clean/structured/steam_reviews_validated.csv"
ARTIFACTS_PATH, CONFIG_PATH = "artifacts", os.path.join("artifacts", "config.json")
MODEL_WEIGHTS_PATH, TOKENIZER_PATH, SCALER_PARAMS_PATH = os.path.join(ARTIFACTS_PATH, "spam_model_weights.pth"), os.path.join(ARTIFACTS_PATH, "spam_tokenizer"), os.path.join(ARTIFACTS_PATH, "scaler_params.npz")

# --- 3. Fungsi Helper ---
def get_confidence(score: float) -> str:
    distance = abs(score - 0.5);
    if distance > 0.4: return "Very High"
    if distance > 0.25: return "High"
    if distance > 0.1: return "Medium"
    return "Low"
def get_spam_explanation(features: dict, score: float) -> str:
    if score < 0.5: return "Review appears to be legitimate."
    reasons = []
    if features.get('bot_indicator_count', 0) > 0: reasons.append("Matches bot-like patterns")
    if features.get('is_very_short'): reasons.append("Extremely short review")
    if features.get('suspicious_pattern_count', 0) > 1: reasons.append("Contains suspicious text patterns")
    if features.get('caps_ratio', 0) > 0.5: reasons.append("Excessive use of capital letters")
    if features.get('extreme_sentiment_ratio', 0) > 0.5: reasons.append("Overly reliant on extreme sentiment words")
    return ", ".join(reasons) if reasons else "Detected as spam based on a combination of patterns."

# --- 4. Fungsi Utama ---
def main():
    print("--- Memulai Proses Enrichment Lokal (Advanced Model) ---")
    
    # A. Muat artifacts
    print("1. Me-load artifacts...")
    artifacts = {}
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(CONFIG_PATH, 'r') as f: config = json.load(f)
        feature_extractor = AdvancedFeatureExtractor()
        scaler = StandardScaler(); scaler_params = np.load(SCALER_PARAMS_PATH)
        scaler.mean_, scaler.scale_ = scaler_params['mean'], scaler_params['scale']
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = EnhancedBERTClassifier(model_name=config['model_name'], num_features=len(config['feature_names']))
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device); model.eval()
        artifacts = {"model": model, "tokenizer": tokenizer, "scaler": scaler, "feature_extractor": feature_extractor, "config": config, "device": device}
        print(f"✅ Artifacts berhasil di-load. Menggunakan device: {device}")
    except Exception as e: print(f"❌ GAGAL memuat artifacts. Error: {e}"); return

    # B. Unduh dataset dari MinIO
    print("\n2. Mengunduh dataset dari MinIO...")
    try:
        minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        response = minio_client.get_object(BUCKET_NAME, INPUT_OBJECT_PATH)
        df = pd.read_csv(io.BytesIO(response.read()))
    finally: response.close(); response.release_conn()
    print(f"Berhasil memuat {len(df)} review.")
    
    # C. Lakukan Prediksi
    print("\n3. Melakukan inferensi untuk setiap review...")
    all_results = []
    with torch.no_grad():
        for text in tqdm(df['review_text'].fillna(''), desc="Enriching Reviews"):
            features_dict = artifacts['feature_extractor'].extract_advanced_features(text)
            features_ordered = np.array([features_dict.get(name, 0.0) for name in artifacts['config']['feature_names']]).reshape(1, -1)
            scaled_features = artifacts['scaler'].transform(features_ordered)
            features_tensor = torch.tensor(scaled_features, dtype=torch.float).to(artifacts['device'])
            encoding = artifacts['tokenizer'](text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
            input_ids, attention_mask = encoding['input_ids'].to(artifacts['device']), encoding['attention_mask'].to(artifacts['device'])
            score = artifacts['model'](input_ids, attention_mask, features_tensor).item()
            all_results.append({
                "pred_is_spam": score > artifacts['config'].get('prediction_threshold', 0.5),
                "pred_spam_score": score,
                "pred_confidence": get_confidence(score),
                "pred_explanation": get_spam_explanation(features_dict, score),
                "pred_features": json.dumps(features_dict)
            })

    # D. Gabungkan hasil
    print("\n4. Menggabungkan hasil prediksi...")
    pred_df = pd.DataFrame(all_results)
    df_enriched = pd.concat([df, pred_df], axis=1)

    # E. Simpan dan upload ke MinIO
    local_filename = "steam_reviews_validated.csv"
    print(f"\n5. Menyimpan dan meng-upload '{local_filename}' ke MinIO...")
    df_enriched.to_csv(local_filename, index=False)
    try:
        minio_client.fput_object(BUCKET_NAME, OUTPUT_OBJECT_PATH, local_filename)
        print(f"✅ Upload berhasil ke '{BUCKET_NAME}/{OUTPUT_OBJECT_PATH}'")
    except Exception as e: print(f"❌ Error saat meng-upload ke MinIO: {e}")
    os.remove(local_filename)
    print("--- Proses Enrichment Lokal Selesai ---")

if __name__ == "__main__":
    for pkg in ['punkt', 'stopwords', 'vader_lexicon']:
        try: nltk.data.find(f'tokenizers/{pkg}')
        except LookupError: nltk.download(pkg, quiet=True)
    main()