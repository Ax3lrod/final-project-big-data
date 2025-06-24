import pandas as pd
import numpy as np
import re
import string
import warnings
from datetime import datetime
import os
import json
from collections import Counter
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.optim as optim
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
# pip install textstat
import textstat
from sentence_transformers import SentenceTransformer
import random
from scipy import stats
from sklearn.metrics import silhouette_score

# Enhanced Configuration
warnings.filterwarnings('ignore')
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with bias-resistant features and better spam indicators
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        # Enhanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000, 
            ngram_range=(1, 4), 
            stop_words='english',
            min_df=3, 
            max_df=0.7, 
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens, min 2 chars
        )
        
        # Better sentence transformer model
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # More sophisticated spam indicators
        self.extreme_sentiment_words = [
            'amazing', 'awesome', 'perfect', 'terrible', 'fantastic', 'incredible',
            'outstanding', 'brilliant', 'excellent', 'awful', 'horrible', 'pathetic',
            'worst', 'best', 'epic', 'legendary', 'masterpiece', 'trash', 'garbage'
        ]
        
        self.generic_phrases = [
            'good game', 'bad game', 'very good', 'very bad', 'recommended',
            'not recommended', 'waste of money', 'worth it', 'best game', 'worst game',
            'must buy', 'dont buy', 'amazing graphics', 'terrible graphics'
        ]
        
        # Enhanced suspicious patterns
        self.suspicious_patterns = [
            r'(.)\1{3,}',  # Repeated characters (4+)
            r'\b(\w+)(\s+\1){2,}\b',  # Word repeated 3+ times
            r'[!]{2,}',  # Multiple exclamations (2+)
            r'[A-Z]{6,}',  # All caps words (6+ chars)
            r'[0-9]{8,}',  # Long number sequences
            r'(.{1,3})\1{4,}',  # Repeated short patterns
        ]
        
        # Bot behavior patterns
        self.bot_indicators = [
            r'^\d+/10$',  # Simple rating format
            r'^\w+\s*$',  # Single word reviews
            r'^(yes|no|good|bad|ok|fine)\s*[.!]*$',  # One-word responses
        ]
        
        self.fitted = False

    def fit(self, texts):
        """Fit the feature extractor on the text data"""
        cleaned_texts = [self._clean_text(text) for text in texts]
        self.tfidf_vectorizer.fit(cleaned_texts)
        self.fitted = True
        return self

    def _clean_text(self, text):
        """Advanced text cleaning"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).strip()
        # Remove URLs, mentions, and excessive whitespace
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive punctuation but keep some
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)
        
        return text

    def extract_advanced_features(self, text):
        """Extract comprehensive features for spam detection"""
        if pd.isna(text) or text == '':
            return self._get_default_features()
        
        text_clean = self._clean_text(text)
        text_lower = text_clean.lower()
        
        try:
            tokens = word_tokenize(text_lower)
            sentences = sent_tokenize(text_clean)
        except:
            tokens = text_lower.split()
            sentences = text_clean.split('.')

        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text_clean)
        features['word_count'] = len(tokens)
        features['sentence_count'] = max(len(sentences), 1)
        features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        
        # Lexical diversity and complexity
        unique_words = set(tokens)
        features['unique_word_ratio'] = len(unique_words) / len(tokens) if tokens else 0
        features['stopword_ratio'] = sum(1 for w in tokens if w in self.stop_words) / len(tokens) if tokens else 0
        
        # Advanced linguistic features
        features['type_token_ratio'] = len(unique_words) / len(tokens) if tokens else 0
        features['hapax_legomena_ratio'] = sum(1 for w in Counter(tokens).values() if w == 1) / len(tokens) if tokens else 0
        
        # Spam indicators
        features['extreme_sentiment_count'] = sum(1 for w in tokens if w in self.extreme_sentiment_words)
        features['extreme_sentiment_ratio'] = features['extreme_sentiment_count'] / len(tokens) if tokens else 0
        features['generic_phrase_count'] = sum(1 for p in self.generic_phrases if p in text_lower)
        
        # Suspicious patterns
        features['suspicious_pattern_count'] = sum(len(re.findall(p, text_clean)) for p in self.suspicious_patterns)
        features['bot_indicator_count'] = sum(len(re.findall(p, text_clean, re.IGNORECASE)) for p in self.bot_indicators)
        
        # Punctuation and formatting
        features['exclamation_count'] = text_clean.count('!')
        features['question_count'] = text_clean.count('?')
        features['caps_ratio'] = sum(1 for c in text_clean if c.isupper()) / len(text_clean) if text_clean else 0
        features['digit_ratio'] = sum(1 for c in text_clean if c.isdigit()) / len(text_clean) if text_clean else 0
        
        # Sentiment analysis
        sentiment_scores = self.sia.polarity_scores(text_clean)
        for k, v in sentiment_scores.items():
            features[f'sentiment_{k}'] = v
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text_clean)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text_clean)
            features['automated_readability_index'] = textstat.automated_readability_index(text_clean)
        except:
            features['flesch_reading_ease'] = 50.0
            features['flesch_kincaid_grade'] = 8.0
            features['automated_readability_index'] = 8.0
        
        # Additional spam indicators
        features['is_very_short'] = 1 if len(tokens) < 3 else 0
        features['is_very_long'] = 1 if len(tokens) > 500 else 0
        features['has_repeated_chars'] = 1 if re.search(r'(.)\1{3,}', text_clean) else 0
        features['all_caps_words'] = sum(1 for w in tokens if len(w) > 2 and w.isupper())
        
        return list(features.values())

    def get_feature_names(self):
        """Get ordered feature names"""
        dummy_text = "This is a sample text for feature extraction."
        text_clean = self._clean_text(dummy_text)
        text_lower = text_clean.lower()
        
        try:
            tokens = word_tokenize(text_lower)
            sentences = sent_tokenize(text_clean)
        except:
            tokens = text_lower.split()
            sentences = text_clean.split('.')

        features = {}
        features['text_length'] = 0
        features['word_count'] = 0
        features['sentence_count'] = 0
        features['avg_word_length'] = 0
        features['avg_sentence_length'] = 0
        features['unique_word_ratio'] = 0
        features['stopword_ratio'] = 0
        features['type_token_ratio'] = 0
        features['hapax_legomena_ratio'] = 0
        features['extreme_sentiment_count'] = 0
        features['extreme_sentiment_ratio'] = 0
        features['generic_phrase_count'] = 0
        features['suspicious_pattern_count'] = 0
        features['bot_indicator_count'] = 0
        features['exclamation_count'] = 0
        features['question_count'] = 0
        features['caps_ratio'] = 0
        features['digit_ratio'] = 0
        
        for k in ['neg', 'neu', 'pos', 'compound']:
            features[f'sentiment_{k}'] = 0
            
        features['flesch_reading_ease'] = 0
        features['flesch_kincaid_grade'] = 0
        features['automated_readability_index'] = 0
        features['is_very_short'] = 0
        features['is_very_long'] = 0
        features['has_repeated_chars'] = 0
        features['all_caps_words'] = 0
        
        return list(features.keys())

    def get_tfidf_features(self, texts):
        """Get TF-IDF features"""
        if not self.fitted:
            raise ValueError("Must fit the extractor first!")
        return self.tfidf_vectorizer.transform([self._clean_text(text) for text in texts])

    def get_sentence_embeddings(self, texts):
        """Get sentence embeddings"""
        cleaned_texts = [self._clean_text(text) for text in texts]
        return self.sentence_model.encode(cleaned_texts, show_progress_bar=True)

    def _get_default_features(self):
        """Return default features for empty/null text"""
        return [0] * len(self.get_feature_names())

class MultiModalSpamDetector:
    """
    Multi-modal spam detector combining multiple approaches for better accuracy
    """
    def __init__(self, contamination=0.08):
        self.contamination = contamination
        self.feature_extractor = AdvancedFeatureExtractor()
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.98, random_state=42)
        
        # Multiple anomaly detectors
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_estimators=300, 
            max_samples=0.7,
            max_features=0.8,
            n_jobs=-1
        )
        
        self.fitted_features_ = None
        self.embedding_clusters_ = None

    def fit(self, texts):
        """Fit the multi-modal detector"""
        print("Fitting advanced feature extractor...")
        self.feature_extractor.fit(texts)
        
        print("Extracting engineered features...")
        engineered_features = np.array([
            self.feature_extractor.extract_advanced_features(text) for text in texts
        ])
        
        print("Extracting TF-IDF features...")
        tfidf_features = self.feature_extractor.get_tfidf_features(texts).toarray()
        
        print("Extracting sentence embeddings...")
        sentence_embeddings = self.feature_extractor.get_sentence_embeddings(texts)
        
        print("Performing embedding clustering for anomaly detection...")
        # Use embeddings to find clusters and identify outliers
        kmeans = KMeans(n_clusters=min(20, len(texts)//50), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sentence_embeddings)
        
        # Calculate distances to cluster centers
        cluster_distances = []
        for i, (embedding, label) in enumerate(zip(sentence_embeddings, cluster_labels)):
            center = kmeans.cluster_centers_[label]
            distance = np.linalg.norm(embedding - center)
            cluster_distances.append(distance)
        
        cluster_distances = np.array(cluster_distances)
        self.embedding_clusters_ = {
            'kmeans': kmeans,
            'distances': cluster_distances
        }
        
        print("Combining and scaling features...")
        # Combine all features
        combined_features = np.hstack([
            engineered_features, 
            tfidf_features, 
            sentence_embeddings,
            cluster_distances.reshape(-1, 1)
        ])
        
        # Scale and reduce dimensionality
        scaled_features = self.scaler.fit_transform(combined_features)        
        self.fitted_features_ = self.pca.fit_transform(scaled_features)
        
        print(f"Final feature shape: {self.fitted_features_.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        print("Fitting anomaly detection models...")
        self.isolation_forest.fit(self.fitted_features_)
        
        return self

    def predict_anomalies(self):
        """Predict anomalies using ensemble approach"""
        # Isolation Forest predictions
        if_scores = self.isolation_forest.decision_function(self.fitted_features_)
        if_labels = self.isolation_forest.predict(self.fitted_features_)
        
        # Embedding cluster-based anomaly detection
        cluster_distances = self.embedding_clusters_['distances']
        cluster_threshold = np.percentile(cluster_distances, 95)
        cluster_anomalies = (cluster_distances > cluster_threshold).astype(int)
        
        # Combine predictions (both methods must agree for high confidence)
        ensemble_labels = ((if_labels == -1) | (cluster_anomalies == 1)).astype(int)
        ensemble_scores = if_scores + (cluster_distances - cluster_distances.mean()) / cluster_distances.std()
        
        return ensemble_labels, ensemble_scores

class SmartLabelingStrategy:
    """
    Smart labeling strategy using multiple heuristics to reduce bias
    """
    def __init__(self):
        self.spam_heuristics = [
            self._is_too_short,
            self._has_excessive_repetition,
            self._is_nonsensical,
            self._has_bot_patterns,
            self._has_extreme_sentiment_only
        ]
    
    def _is_too_short(self, text):
        """Check if text is suspiciously short"""
        words = text.split()
        return len(words) <= 2 and len(text.strip()) < 10
    
    def _has_excessive_repetition(self, text):
        """Check for excessive character or word repetition"""
        # Character repetition
        if re.search(r'(.)\1{4,}', text):
            return True
        # Word repetition
        words = text.lower().split()
        if len(words) > 2:
            word_counts = Counter(words)
            max_count = max(word_counts.values())
            if max_count / len(words) > 0.5:  # More than 50% repetition
                return True
        return False
    
    def _is_nonsensical(self, text):
        """Check if text appears nonsensical"""
        # Very high ratio of non-alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:
            return True
        
        # Random character sequences
        if re.search(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{6,}', text):
            return True
        
        return False
    
    def _has_bot_patterns(self, text):
        """Check for common bot patterns"""
        text_clean = text.strip().lower()
        
        # Simple rating patterns
        if re.match(r'^\d+[\/\-]\d+$', text_clean):
            return True
        
        # Single word responses
        if re.match(r'^(good|bad|ok|yes|no|fine|nice|cool|great)[\.\!]*$', text_clean):
            return True
        
        return False
    
    def _has_extreme_sentiment_only(self, text):
        """Check if text has only extreme sentiment words"""
        words = text.lower().split()
        if len(words) <= 5:
            extreme_words = ['amazing', 'terrible', 'perfect', 'awful', 'best', 'worst']
            extreme_count = sum(1 for w in words if w in extreme_words)
            if extreme_count / len(words) > 0.6:
                return True
        return False
    
    def label_text(self, text, anomaly_prediction, anomaly_score):
        """Smart labeling using multiple heuristics"""
        if pd.isna(text) or text.strip() == '':
            return 1  # Empty text is spam
        
        # If anomaly detector says it's normal and it passes basic checks, likely legitimate
        if anomaly_prediction == 0 and anomaly_score > -0.1:
            return 0
        
        # Apply heuristics
        spam_indicators = sum(heuristic(text) for heuristic in self.spam_heuristics)
        
        # Strong spam indicators
        if spam_indicators >= 2:
            return 1
        
        # Single strong indicator + anomaly detection
        if spam_indicators >= 1 and anomaly_prediction == 1:
            return 1
        
        # Very strong anomaly score
        if anomaly_score < -0.5:
            return 1
        
        # Default to not spam for borderline cases (reduce false positives)
        return 0

class EnhancedBERTClassifier(nn.Module):
    """
    Enhanced BERT classifier with better architecture and regularization
    """
    def __init__(self, model_name, num_features, dropout=0.4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        
        # Freeze embedding layers to prevent overfitting
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first few transformer layers
        for layer in self.bert.encoder.layer[:6]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.feature_norm = nn.BatchNorm1d(num_features)
        
        # More sophisticated classifier
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + num_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, features):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(bert_output.last_hidden_state[:, 0, :])
        
        # Feature normalization
        features_norm = self.feature_norm(features)
        
        # Combine features
        combined = torch.cat([cls_output, features_norm], dim=1)
        
        # Classification
        logits = self.classifier(combined)
        return torch.sigmoid(logits)

class SpamDataset(Dataset):
    """Enhanced dataset class"""
    def __init__(self, texts, features, labels, tokenizer, max_length=256):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def advanced_train_loop(model, train_loader, val_loader, device, epochs=5, lr=2e-5):
    """Advanced training loop with better optimization"""
    # Separate learning rates for BERT and classifier
    bert_params = list(model.bert.parameters())
    classifier_params = list(model.classifier.parameters()) + list(model.feature_norm.parameters())
    
    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': lr},
        {'params': classifier_params, 'lr': lr * 5}
    ], weight_decay=0.01)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) * epochs * 0.1,
        num_training_steps=len(train_loader) * epochs
    )
    
    # Focal loss to handle class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(alpha=2, gamma=2)  # Focus on hard examples
    
    best_val_f1 = 0
    patience, max_patience = 0, 3
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['features'].to(device)
            ).squeeze()
            
            loss = criterion(outputs, batch['labels'].to(device))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_preds.extend((outputs.cpu() > 0.5).numpy())
            train_labels.extend(batch['labels'].cpu().numpy())
        
        # Validation phase
        model.eval()
        val_preds, val_labels, val_probs = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['features'].to(device)
                ).squeeze()
                
                val_probs.extend(outputs.cpu().numpy())
                val_preds.extend((outputs.cpu() > 0.5).numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        train_f1 = precision_recall_fscore_support(train_labels, train_preds, average='binary', zero_division=0)[2]
        val_f1 = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)[2]
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {total_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}')
        print(f'  Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        
        # Early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_state_dict.pth')
            print(f'  ✅ New best model saved! (F1: {val_f1:.4f})')
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f'  🛑 Early stopping after {epoch+1} epochs.')
                break
    
    return model

def save_artifacts(tokenizer, scaler, config, path='artifacts'):
    os.makedirs(path, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(path, "spam_tokenizer"))
    np.savez(os.path.join(path, "scaler_params.npz"), mean=scaler.mean_, scale=scaler.scale_)
    with open(os.path.join(path, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

def finalize_training(df, labels, texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Extract engineered features
    print("\nExtracting engineered features for final model...")
    feature_extractor = AdvancedFeatureExtractor()
    feature_extractor.fit(texts)
    features = np.array([feature_extractor.extract_advanced_features(t) for t in texts])

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Train/Val split
    X_train, X_val, feat_train, feat_val, y_train, y_val = train_test_split(
        texts, scaled_features, labels, test_size=0.15, stratify=labels, random_state=42
    )

    # Dataset and Dataloader
    train_dataset = SpamDataset(X_train, feat_train, y_train, tokenizer)
    val_dataset = SpamDataset(X_val, feat_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=16, sampler=SequentialSampler(val_dataset))

    # Model
    model = EnhancedBERTClassifier(model_name=model_name, num_features=features.shape[1])
    model = model.to(device)

    print("\nStarting final model training...")
    model = advanced_train_loop(model, train_loader, val_loader, device, epochs=5)

    print("\nSaving model artifacts...")
    ARTIFACTS_DIR = 'artifacts'
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), "artifacts/spam_model_weights.pth")
    save_artifacts(tokenizer, scaler, config={
        "model_name": model_name,
        "feature_names": feature_extractor.get_feature_names(),
        "version": "1.0"
    })
    print("✅ All artifacts saved in 'artifacts/' folder.")

if __name__ == '__main__':
    df = pd.read_csv('/kaggle/input/steam-reviews-cleaned/steam_reviews.csv')
    df['review_text'] = df['review_text'].fillna('').astype(str)
    df = df[df['review_text'].str.strip().str.len() > 0].reset_index(drop=True)
    df = df[df['review_text'].str.len() < 5000]

    # Anomaly detection + smart labeling
    detector = MultiModalSpamDetector(contamination=0.08)
    detector.fit(df['review_text'].values)
    anomaly_labels, anomaly_scores = detector.predict_anomalies()

    smart_labeler = SmartLabelingStrategy()
    labels = np.array([
        smart_labeler.label_text(text, pred, score)
        for text, pred, score in zip(df['review_text'].values, anomaly_labels, anomaly_scores)
    ])

    finalize_training(df, labels, df['review_text'].tolist())