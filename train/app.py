import os
import json
import warnings
from datetime import datetime
from typing import List, Optional, Dict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import string

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import textstat

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

# --- 4. Aplikasi FastAPI & Fungsi Helper ---
app = FastAPI(title="Advanced Spam Detection API", version="5.0.0")

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

# --- Jalankan server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)