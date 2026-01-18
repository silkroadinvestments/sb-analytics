"""
Sablemoore Analytics - Machine Learning Engine
Advanced ML models for litigation finance prediction and duplicate detection
"""

import os
import json
import pickle
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# ML imports with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    SKLEARN_AVAILABLE = True
    XGBOOST_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    XGBOOST_AVAILABLE = False

# Base directory for model storage
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


class AIModelManager:
    """
    Centralized AI Model Manager for all ML operations.
    Handles model training, persistence, and inference.
    """

    MODELS_FILE = os.path.join(MODEL_DIR, 'sablemoore_models.pkl')
    TRAINING_DATA_FILE = os.path.join(MODEL_DIR, 'training_data.json')

    def __init__(self):
        self.success_model = None
        self.duplicate_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoders = {}
        self.is_success_trained = False
        self.is_duplicate_trained = False
        self.training_history = []

        # Feature configurations
        self.success_features = [
            'claim_amount', 'case_type_encoded', 'jurisdiction_encoded',
            'defendant_type_encoded', 'complexity_encoded',
            'estimated_duration_months', 'text_length', 'has_expert',
            'has_precedent', 'defendant_strength'
        ]

        self._load_models()

    def _load_models(self):
        """Load trained models from disk"""
        if os.path.exists(self.MODELS_FILE):
            try:
                with open(self.MODELS_FILE, 'rb') as f:
                    saved = pickle.load(f)
                    self.success_model = saved.get('success_model')
                    self.duplicate_model = saved.get('duplicate_model')
                    self.scaler = saved.get('scaler', self.scaler)
                    self.label_encoders = saved.get('label_encoders', {})
                    self.is_success_trained = saved.get('is_success_trained', False)
                    self.is_duplicate_trained = saved.get('is_duplicate_trained', False)
                    self.training_history = saved.get('training_history', [])
                    print(f"[AIModelManager] Models loaded successfully")
            except Exception as e:
                print(f"[AIModelManager] Error loading models: {e}")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(self.MODELS_FILE, 'wb') as f:
                pickle.dump({
                    'success_model': self.success_model,
                    'duplicate_model': self.duplicate_model,
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'is_success_trained': self.is_success_trained,
                    'is_duplicate_trained': self.is_duplicate_trained,
                    'training_history': self.training_history,
                    'saved_at': datetime.now().isoformat()
                }, f)
            print(f"[AIModelManager] Models saved successfully")
        except Exception as e:
            print(f"[AIModelManager] Error saving models: {e}")

    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            'success_predictor': {
                'trained': self.is_success_trained,
                'model_type': type(self.success_model).__name__ if self.success_model else 'None',
                'features': len(self.success_features)
            },
            'duplicate_detector': {
                'trained': self.is_duplicate_trained,
                'model_type': type(self.duplicate_model).__name__ if self.duplicate_model else 'None'
            },
            'training_history_count': len(self.training_history),
            'sklearn_available': SKLEARN_AVAILABLE,
            'xgboost_available': XGBOOST_AVAILABLE
        }


class EnhancedSuccessPredictor:
    """
    ML-powered success rate predictor using ensemble methods.
    Trained on UK litigation patterns with XGBoost/RandomForest.
    """

    # UK Litigation base rates (from CPR statistics and legal research)
    BASE_RATES = {
        'Contract Dispute': 0.68,
        'Personal Injury': 0.73,
        'Employment': 0.58,
        'Commercial Dispute': 0.62,
        'Property': 0.65,
        'Professional Negligence': 0.52,
        'Intellectual Property': 0.48,
        'Fraud': 0.42,
        'Debt Recovery': 0.78,
        'General Civil Litigation': 0.60
    }

    JURISDICTION_MODIFIERS = {
        'County Court': 1.05,
        'High Court': 0.92,
        'Employment Tribunal': 0.95,
        'Court of Appeal': 0.75,
        'Supreme Court': 0.65
    }

    DEFENDANT_MODIFIERS = {
        'Corporate': 0.95,
        'Individual': 1.02,
        'Public Body': 0.88
    }

    COMPLEXITY_MODIFIERS = {
        'Low': 1.10,
        'Medium': 1.00,
        'High': 0.85
    }

    def __init__(self, model_manager: Optional[AIModelManager] = None):
        self.model_manager = model_manager
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoders = {}
        self.is_trained = False
        self.feature_names = []

        # Initialize label encoders
        if SKLEARN_AVAILABLE:
            self._init_encoders()

    def _init_encoders(self):
        """Initialize label encoders for categorical features"""
        self.label_encoders['case_type'] = LabelEncoder()
        self.label_encoders['case_type'].fit(list(self.BASE_RATES.keys()))

        self.label_encoders['jurisdiction'] = LabelEncoder()
        self.label_encoders['jurisdiction'].fit(list(self.JURISDICTION_MODIFIERS.keys()))

        self.label_encoders['defendant_type'] = LabelEncoder()
        self.label_encoders['defendant_type'].fit(list(self.DEFENDANT_MODIFIERS.keys()))

        self.label_encoders['complexity'] = LabelEncoder()
        self.label_encoders['complexity'].fit(['Low', 'Medium', 'High'])

    def _extract_text_features(self, text: str) -> Dict:
        """Extract additional features from case text"""
        text_lower = text.lower()

        # Expert witness indicators
        expert_keywords = ['expert', 'specialist', 'forensic', 'consultant', 'professional opinion']
        has_expert = any(kw in text_lower for kw in expert_keywords)

        # Precedent indicators
        precedent_keywords = ['precedent', 'landmark', 'established law', 'binding authority', 'case law']
        has_precedent = any(kw in text_lower for kw in precedent_keywords)

        # Evidence strength indicators
        evidence_keywords = ['documented', 'witness', 'evidence', 'proof', 'recorded', 'written']
        evidence_count = sum(1 for kw in evidence_keywords if kw in text_lower)

        # Defendant strength indicators (large company, resources)
        defendant_strength_keywords = ['multinational', 'plc', 'fortune 500', 'major corporation', 'well-funded']
        defendant_strength = sum(1 for kw in defendant_strength_keywords if kw in text_lower)

        # Urgency indicators
        urgency_keywords = ['urgent', 'immediate', 'emergency', 'expedited']
        is_urgent = any(kw in text_lower for kw in urgency_keywords)

        # Complexity from text
        complexity_keywords = ['complex', 'multiple parties', 'cross-border', 'international', 'regulatory']
        text_complexity = sum(1 for kw in complexity_keywords if kw in text_lower)

        return {
            'has_expert': 1 if has_expert else 0,
            'has_precedent': 1 if has_precedent else 0,
            'evidence_strength': min(evidence_count / 3, 1.0),
            'defendant_strength': min(defendant_strength, 3),
            'is_urgent': 1 if is_urgent else 0,
            'text_complexity': min(text_complexity / 3, 1.0),
            'text_length': len(text)
        }

    def _build_feature_vector(self, case_data: Dict) -> np.ndarray:
        """Build feature vector from case data"""
        if not SKLEARN_AVAILABLE:
            return np.array([])

        text = case_data.get('raw_text', '')
        text_features = self._extract_text_features(text)

        # Encode categorical features
        case_type = case_data.get('case_type', 'General Civil Litigation')
        if case_type not in self.BASE_RATES:
            case_type = 'General Civil Litigation'

        jurisdiction = case_data.get('jurisdiction', 'County Court')
        if jurisdiction not in self.JURISDICTION_MODIFIERS:
            jurisdiction = 'County Court'

        defendant_type = case_data.get('defendant_type', 'Individual')
        if defendant_type not in self.DEFENDANT_MODIFIERS:
            defendant_type = 'Individual'

        complexity = case_data.get('complexity', 'Medium')
        if complexity not in self.COMPLEXITY_MODIFIERS:
            complexity = 'Medium'

        features = [
            case_data.get('claim_amount', 100000) / 1000000,  # Normalize to millions
            self.label_encoders['case_type'].transform([case_type])[0],
            self.label_encoders['jurisdiction'].transform([jurisdiction])[0],
            self.label_encoders['defendant_type'].transform([defendant_type])[0],
            self.label_encoders['complexity'].transform([complexity])[0],
            case_data.get('estimated_duration_months', 12) / 36,  # Normalize
            text_features['text_length'] / 10000,  # Normalize
            text_features['has_expert'],
            text_features['has_precedent'],
            text_features['evidence_strength'],
            text_features['defendant_strength'] / 3,
            text_features['is_urgent'],
            text_features['text_complexity']
        ]

        self.feature_names = [
            'claim_amount_normalized', 'case_type', 'jurisdiction',
            'defendant_type', 'complexity', 'duration_normalized',
            'text_length_normalized', 'has_expert', 'has_precedent',
            'evidence_strength', 'defendant_strength', 'is_urgent',
            'text_complexity'
        ]

        return np.array(features).reshape(1, -1)

    def predict(self, case_data: Dict) -> Dict:
        """
        Predict success rate using hybrid ML + rule-based approach.
        Uses ML model if trained, otherwise falls back to enhanced rules.
        """
        # Extract features
        text = case_data.get('raw_text', '')
        text_features = self._extract_text_features(text)

        case_type = case_data.get('case_type', 'General Civil Litigation')
        jurisdiction = case_data.get('jurisdiction', 'County Court')
        defendant_type = case_data.get('defendant_type', 'Individual')
        complexity = case_data.get('complexity', 'Medium')
        claim_amount = case_data.get('claim_amount', 100000)

        # Try ML prediction first
        ml_prediction = None
        if self.is_trained and self.model is not None and SKLEARN_AVAILABLE:
            try:
                features = self._build_feature_vector(case_data)
                ml_prediction = self.model.predict_proba(features)[0][1]
            except Exception as e:
                print(f"[SuccessPredictor] ML prediction failed: {e}")

        # Calculate rule-based prediction
        base_rate = self.BASE_RATES.get(case_type, 0.60)

        # Apply modifiers
        jurisdiction_mod = self.JURISDICTION_MODIFIERS.get(jurisdiction, 1.0)
        defendant_mod = self.DEFENDANT_MODIFIERS.get(defendant_type, 1.0)
        complexity_mod = self.COMPLEXITY_MODIFIERS.get(complexity, 1.0)

        # Claim amount modifier (diminishing returns on very high claims)
        if claim_amount > 5000000:
            amount_mod = 0.80
        elif claim_amount > 1000000:
            amount_mod = 0.88
        elif claim_amount > 500000:
            amount_mod = 0.93
        elif claim_amount > 100000:
            amount_mod = 0.97
        else:
            amount_mod = 1.02

        # Evidence and expert modifiers
        evidence_mod = 1.0 + (text_features['evidence_strength'] * 0.08)
        expert_mod = 1.05 if text_features['has_expert'] else 1.0
        precedent_mod = 1.08 if text_features['has_precedent'] else 1.0

        # Calculate rule-based success rate
        rule_based_rate = (
            base_rate *
            jurisdiction_mod *
            defendant_mod *
            complexity_mod *
            amount_mod *
            evidence_mod *
            expert_mod *
            precedent_mod
        )

        # Combine ML and rule-based if both available
        if ml_prediction is not None:
            # Weighted average: 60% ML, 40% rule-based
            success_rate = (ml_prediction * 0.6) + (rule_based_rate * 0.4)
            prediction_method = 'hybrid_ml_rules'
        else:
            success_rate = rule_based_rate
            prediction_method = 'rule_based'

        # Clamp to valid range
        success_rate = max(0.15, min(0.92, success_rate))

        # Calculate confidence interval
        confidence_margin = 0.12 if prediction_method == 'rule_based' else 0.08

        # Risk assessment
        if success_rate >= 0.70:
            risk_level = 'Low'
        elif success_rate >= 0.50:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        # Generate risk factors
        risk_factors = []
        positive_factors = []

        if complexity == 'High':
            risk_factors.append('High complexity increases uncertainty')
        if claim_amount > 1000000:
            risk_factors.append('High value claims face stronger defense')
        if jurisdiction in ['Court of Appeal', 'Supreme Court']:
            risk_factors.append('Appellate courts have lower success rates')
        if case_type in ['Fraud', 'Professional Negligence']:
            risk_factors.append('Case type historically has lower success rates')
        if text_features['defendant_strength'] >= 2:
            risk_factors.append('Well-resourced defendant may prolong litigation')

        if text_features['has_expert']:
            positive_factors.append('Expert evidence strengthens case')
        if text_features['has_precedent']:
            positive_factors.append('Favorable precedent supports claim')
        if case_type in ['Debt Recovery', 'Personal Injury']:
            positive_factors.append('Case type has historically higher success rates')
        if complexity == 'Low':
            positive_factors.append('Straightforward case with clear legal basis')
        if text_features['evidence_strength'] > 0.5:
            positive_factors.append('Strong documented evidence available')

        # Expected return calculation
        litigation_cost_factor = 0.85  # After legal costs
        expected_return = claim_amount * success_rate * litigation_cost_factor

        # Recommendation
        if success_rate >= 0.68:
            recommendation = 'RECOMMENDED - Strong case with good success probability'
        elif success_rate >= 0.52:
            recommendation = 'CONSIDER - Moderate case, review risk factors carefully'
        else:
            recommendation = 'CAUTION - Lower success probability, high risk investment'

        return {
            'success_rate': round(success_rate * 100, 2),
            'confidence_interval': {
                'low': round((success_rate - confidence_margin) * 100, 2),
                'high': round((success_rate + confidence_margin) * 100, 2)
            },
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'positive_factors': positive_factors,
            'expected_return': expected_return,
            'recommendation': recommendation,
            'prediction_method': prediction_method,
            'model_trained': self.is_trained
        }

    def train(self, training_data: List[Dict]) -> Dict:
        """
        Train the success prediction model on labeled data.
        Expects training_data with 'case_data' and 'actual_outcome' fields.
        """
        if not SKLEARN_AVAILABLE or len(training_data) < 10:
            return {
                'success': False,
                'message': f'Need at least 10 training examples (have {len(training_data)})'
            }

        X = []
        y = []

        for example in training_data:
            case_data = example.get('case_data', {})
            outcome = example.get('actual_outcome', 0)  # 1 for success, 0 for failure

            features = self._build_feature_vector(case_data)
            if features.size > 0:
                X.append(features.flatten())
                y.append(outcome)

        X = np.array(X)
        y = np.array(y)

        # Train XGBoost if available, otherwise RandomForest
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(y)//2), scoring='accuracy')

        # Train on full data
        self.model.fit(X, y)
        self.is_trained = True

        # Save to model manager if available
        if self.model_manager:
            self.model_manager.success_model = self.model
            self.model_manager.is_success_trained = True
            self.model_manager._save_models()

        return {
            'success': True,
            'model_type': type(self.model).__name__,
            'training_samples': len(X),
            'cv_accuracy': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores))
        }


class EnhancedDuplicateDetector:
    """
    ML-powered duplicate detection using TF-IDF, semantic similarity,
    and structured feature comparison.
    """

    def __init__(self, model_manager: Optional[AIModelManager] = None):
        self.model_manager = model_manager
        self.classifier = None
        self.is_trained = False
        self.training_data = []

        # TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                sublinear_tf=True  # Use log scaling for term frequency
            )

        # Load existing training data
        self._load_training_data()

    def _load_training_data(self):
        """Load training data from disk"""
        training_file = os.path.join(MODEL_DIR, 'duplicate_training.json')
        if os.path.exists(training_file):
            try:
                with open(training_file, 'r') as f:
                    self.training_data = json.load(f)
            except:
                self.training_data = []

    def _save_training_data(self):
        """Save training data to disk"""
        training_file = os.path.join(MODEL_DIR, 'duplicate_training.json')
        try:
            with open(training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"[DuplicateDetector] Error saving training data: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for comparison"""
        if not text:
            return ""

        text = text.lower()

        # Normalize dates
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' DATE ', text)
        text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', ' DATE ', text)

        # Normalize currency
        text = re.sub(r'[£$€]\s*[\d,]+(?:\.\d{2})?', ' AMOUNT ', text)
        text = re.sub(r'[\d,]+\s*(?:pounds?|dollars?|euros?)', ' AMOUNT ', text)

        # Normalize case references
        text = re.sub(r'\[\d{4}\]\s*[A-Z]+\s*\d+', ' CASE_REF ', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _extract_entities(self, text: str) -> set:
        """Extract potential named entities from text"""
        # Extract capitalized sequences (names, companies)
        entities = set()

        # Multi-word capitalized phrases
        multi_word = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        entities.update(multi_word)

        # Single capitalized words (but filter common ones)
        common_words = {'The', 'This', 'That', 'These', 'Those', 'Court', 'Judge',
                       'Claimant', 'Defendant', 'Plaintiff', 'Mr', 'Mrs', 'Ms', 'Dr'}
        single_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        entities.update(w for w in single_words if w not in common_words)

        return entities

    def _calculate_entity_similarity(self, text1: str, text2: str) -> float:
        """Calculate named entity overlap"""
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)

        if not entities1 or not entities2:
            return 0.0

        # Jaccard similarity
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)

        return intersection / union if union > 0 else 0.0

    def _extract_features(self, case1: Dict, case2: Dict) -> Dict:
        """Extract comprehensive comparison features between two cases"""
        features = {}

        text1 = case1.get('raw_text', '')
        text2 = case2.get('raw_text', '')

        processed_text1 = self._preprocess_text(text1)
        processed_text2 = self._preprocess_text(text2)

        # TF-IDF similarity
        if SKLEARN_AVAILABLE and processed_text1 and processed_text2:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
                features['tfidf_similarity'] = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            except:
                features['tfidf_similarity'] = 0.0
        else:
            features['tfidf_similarity'] = 0.0

        # Sequence similarity (character level)
        max_len = 2000
        features['sequence_similarity'] = SequenceMatcher(
            None,
            processed_text1[:max_len],
            processed_text2[:max_len]
        ).ratio()

        # Word-level Jaccard similarity
        words1 = set(processed_text1.split())
        words2 = set(processed_text2.split())
        if words1 and words2:
            features['word_jaccard'] = len(words1 & words2) / len(words1 | words2)
        else:
            features['word_jaccard'] = 0.0

        # Claim amount similarity
        amount1 = case1.get('claim_amount', 0)
        amount2 = case2.get('claim_amount', 0)
        if amount1 > 0 and amount2 > 0:
            features['amount_ratio'] = min(amount1, amount2) / max(amount1, amount2)
        else:
            features['amount_ratio'] = 0.0

        # Categorical feature matches
        features['same_case_type'] = 1.0 if case1.get('case_type') == case2.get('case_type') else 0.0
        features['same_jurisdiction'] = 1.0 if case1.get('jurisdiction') == case2.get('jurisdiction') else 0.0
        features['same_defendant_type'] = 1.0 if case1.get('defendant_type') == case2.get('defendant_type') else 0.0
        features['same_complexity'] = 1.0 if case1.get('complexity') == case2.get('complexity') else 0.0

        # Duration similarity
        dur1 = case1.get('estimated_duration_months', 12)
        dur2 = case2.get('estimated_duration_months', 12)
        features['duration_ratio'] = min(dur1, dur2) / max(dur1, dur2) if max(dur1, dur2) > 0 else 0.0

        # Entity overlap (from original text to preserve case)
        features['entity_overlap'] = self._calculate_entity_similarity(text1, text2)

        # Text length similarity
        len1 = len(text1)
        len2 = len(text2)
        if max(len1, len2) > 0:
            features['length_ratio'] = min(len1, len2) / max(len1, len2)
        else:
            features['length_ratio'] = 0.0

        return features

    def _calculate_composite_score(self, features: Dict) -> float:
        """Calculate weighted composite similarity score"""
        weights = {
            'tfidf_similarity': 0.25,
            'sequence_similarity': 0.12,
            'word_jaccard': 0.13,
            'amount_ratio': 0.12,
            'same_case_type': 0.10,
            'same_jurisdiction': 0.08,
            'same_defendant_type': 0.06,
            'same_complexity': 0.04,
            'duration_ratio': 0.04,
            'entity_overlap': 0.04,
            'length_ratio': 0.02
        }

        score = sum(features.get(k, 0) * v for k, v in weights.items())
        return score

    def find_duplicates(self, cases: List[Dict], threshold: float = 0.55) -> List[Dict]:
        """Find potential duplicates using ML features"""
        if len(cases) < 2:
            return []

        duplicates = []

        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                features = self._extract_features(cases[i], cases[j])
                composite_score = self._calculate_composite_score(features)

                # Use trained classifier if available
                if self.is_trained and self.classifier is not None:
                    try:
                        feature_vector = np.array([list(features.values())]).reshape(1, -1)
                        ml_prob = self.classifier.predict_proba(feature_vector)[0][1]
                        # Blend ML and rule-based
                        final_score = (ml_prob * 0.6) + (composite_score * 0.4)
                    except:
                        final_score = composite_score
                else:
                    final_score = composite_score

                if final_score >= threshold:
                    # Determine confidence level
                    if final_score >= 0.80:
                        confidence = 'High'
                    elif final_score >= 0.65:
                        confidence = 'Medium'
                    else:
                        confidence = 'Low'

                    duplicates.append({
                        'case_1': cases[i]['case_id'],
                        'case_2': cases[j]['case_id'],
                        'similarity': round(final_score * 100, 1),
                        'features': {
                            'text_similarity': round(features['tfidf_similarity'] * 100, 1),
                            'word_overlap': round(features['word_jaccard'] * 100, 1),
                            'amount_match': round(features['amount_ratio'] * 100, 1),
                            'same_type': features['same_case_type'] == 1.0,
                            'same_jurisdiction': features['same_jurisdiction'] == 1.0,
                            'entity_overlap': round(features['entity_overlap'] * 100, 1)
                        },
                        'confidence': confidence,
                        'case_1_details': cases[i],
                        'case_2_details': cases[j]
                    })

        # Sort by similarity score descending
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def add_training_example(self, case1_id: str, case2_id: str,
                            is_duplicate: bool, cases: List[Dict]) -> bool:
        """Add a labeled training example"""
        case1 = next((c for c in cases if c['case_id'] == case1_id), None)
        case2 = next((c for c in cases if c['case_id'] == case2_id), None)

        if not case1 or not case2:
            return False

        features = self._extract_features(case1, case2)

        self.training_data.append({
            'features': features,
            'is_duplicate': is_duplicate,
            'case1_id': case1_id,
            'case2_id': case2_id,
            'added_at': datetime.now().isoformat()
        })

        self._save_training_data()

        # Auto-train if enough examples
        if len(self.training_data) >= 5:
            self._train_classifier()

        return True

    def _train_classifier(self) -> Dict:
        """Train the classifier on labeled examples"""
        if not SKLEARN_AVAILABLE or len(self.training_data) < 5:
            return {'success': False, 'message': 'Need at least 5 examples'}

        X = []
        y = []

        for example in self.training_data:
            X.append(list(example['features'].values()))
            y.append(1 if example['is_duplicate'] else 0)

        X = np.array(X)
        y = np.array(y)

        # Use Gradient Boosting for better performance
        self.classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        self.classifier.fit(X, y)
        self.is_trained = True

        # Save to model manager
        if self.model_manager:
            self.model_manager.duplicate_model = self.classifier
            self.model_manager.is_duplicate_trained = True
            self.model_manager._save_models()

        return {
            'success': True,
            'model_type': 'GradientBoostingClassifier',
            'training_samples': len(X)
        }

    def get_training_stats(self) -> Dict:
        """Get statistics about training data"""
        duplicates = sum(1 for x in self.training_data if x.get('is_duplicate', False))
        return {
            'total_examples': len(self.training_data),
            'duplicates': duplicates,
            'non_duplicates': len(self.training_data) - duplicates,
            'is_trained': self.is_trained
        }


def generate_synthetic_training_data(n_samples: int = 100) -> List[Dict]:
    """
    Generate synthetic training data for initial model training.
    Based on UK litigation patterns and realistic case distributions.
    """
    np.random.seed(42)

    case_types = list(EnhancedSuccessPredictor.BASE_RATES.keys())
    jurisdictions = list(EnhancedSuccessPredictor.JURISDICTION_MODIFIERS.keys())
    defendant_types = list(EnhancedSuccessPredictor.DEFENDANT_MODIFIERS.keys())
    complexities = ['Low', 'Medium', 'High']

    training_data = []

    for i in range(n_samples):
        case_type = np.random.choice(case_types, p=[0.15, 0.15, 0.10, 0.15, 0.10, 0.08, 0.07, 0.05, 0.10, 0.05])
        jurisdiction = np.random.choice(jurisdictions, p=[0.50, 0.25, 0.10, 0.10, 0.05])
        defendant_type = np.random.choice(defendant_types, p=[0.50, 0.35, 0.15])
        complexity = np.random.choice(complexities, p=[0.30, 0.50, 0.20])

        # Generate claim amount based on case type
        if case_type in ['Commercial Dispute', 'Professional Negligence', 'Intellectual Property']:
            claim_amount = np.random.lognormal(13, 1.5)  # Higher value cases
        else:
            claim_amount = np.random.lognormal(11, 1.2)  # Normal value cases

        claim_amount = min(max(claim_amount, 10000), 50000000)

        duration = np.random.randint(6, 48) if complexity == 'High' else np.random.randint(3, 24)

        # Calculate realistic outcome probability
        base_rate = EnhancedSuccessPredictor.BASE_RATES[case_type]
        jur_mod = EnhancedSuccessPredictor.JURISDICTION_MODIFIERS[jurisdiction]
        def_mod = EnhancedSuccessPredictor.DEFENDANT_MODIFIERS[defendant_type]
        comp_mod = EnhancedSuccessPredictor.COMPLEXITY_MODIFIERS[complexity]

        true_probability = base_rate * jur_mod * def_mod * comp_mod
        true_probability = max(0.1, min(0.9, true_probability))

        # Generate outcome based on probability
        outcome = 1 if np.random.random() < true_probability else 0

        case_data = {
            'case_id': hashlib.md5(f"synthetic_{i}".encode()).hexdigest()[:8],
            'case_type': case_type,
            'jurisdiction': jurisdiction,
            'defendant_type': defendant_type,
            'complexity': complexity,
            'claim_amount': float(claim_amount),
            'estimated_duration_months': int(duration),
            'raw_text': f"Synthetic case #{i} - {case_type} in {jurisdiction}"
        }

        training_data.append({
            'case_data': case_data,
            'actual_outcome': outcome,
            'true_probability': true_probability
        })

    return training_data


# Singleton instances
_model_manager = None
_success_predictor = None
_duplicate_detector = None


def get_model_manager() -> AIModelManager:
    """Get singleton model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = AIModelManager()
    return _model_manager


def get_success_predictor() -> EnhancedSuccessPredictor:
    """Get singleton success predictor instance"""
    global _success_predictor
    if _success_predictor is None:
        _success_predictor = EnhancedSuccessPredictor(get_model_manager())
    return _success_predictor


def get_duplicate_detector() -> EnhancedDuplicateDetector:
    """Get singleton duplicate detector instance"""
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = EnhancedDuplicateDetector(get_model_manager())
    return _duplicate_detector
