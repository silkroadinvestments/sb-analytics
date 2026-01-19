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
from typing import Dict, List, Tuple, Optional, Set
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# FUZZY NAME MATCHING UTILITIES
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    This measures the minimum number of single-character edits needed
    to transform one string into the other.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def soundex(name: str) -> str:
    """
    Generate Soundex code for a name - phonetic algorithm for indexing names by sound.
    Names that sound similar will have the same Soundex code.
    e.g., "Robert" and "Rupert" both -> R163
          "John" and "Jon" both -> J500
    """
    if not name:
        return "0000"

    name = name.upper()
    # Keep first letter
    soundex_code = name[0]

    # Soundex coding map
    coding = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    # Process remaining characters
    prev_code = coding.get(name[0], '0')
    for char in name[1:]:
        code = coding.get(char, '0')
        if code != '0' and code != prev_code:
            soundex_code += code
        prev_code = code
        if len(soundex_code) == 4:
            break

    # Pad with zeros if needed
    soundex_code = soundex_code.ljust(4, '0')
    return soundex_code[:4]


def metaphone(name: str) -> str:
    """
    Simple Metaphone algorithm - another phonetic algorithm that's often
    more accurate than Soundex for English names.
    """
    if not name:
        return ""

    name = name.upper()
    result = []

    # Simple transformation rules
    replacements = [
        ('GH', ''), ('GN', 'N'), ('KN', 'N'), ('PN', 'N'), ('WR', 'R'),
        ('PS', 'S'), ('PH', 'F'), ('CK', 'K'), ('SCH', 'SK'), ('GHT', 'T'),
        ('DG', 'J'), ('TCH', 'CH'), ('SH', 'X'), ('CH', 'X'), ('TH', '0'),
        ('X', 'KS'), ('QU', 'KW'), ('Q', 'K'), ('Z', 'S'), ('V', 'F'),
        ('W', ''), ('Y', ''),
    ]

    for old, new in replacements:
        name = name.replace(old, new)

    # Remove duplicate adjacent letters
    prev = ''
    for char in name:
        if char != prev:
            result.append(char)
        prev = char

    # Remove vowels except at start
    if result:
        first = result[0]
        result = [first] + [c for c in result[1:] if c not in 'AEIOU']

    return ''.join(result)


class NameMatcher:
    """
    Advanced name matching class that handles:
    - Fuzzy matching for misspellings (John vs Jon)
    - Phonetic matching for similar-sounding names
    - First name + surname separate comparison
    - Common nickname/variant handling
    """

    # Common name variants/nicknames
    NAME_VARIANTS = {
        'william': ['will', 'bill', 'billy', 'willy', 'liam'],
        'robert': ['rob', 'bob', 'bobby', 'robbie', 'bert'],
        'richard': ['rich', 'rick', 'dick', 'ricky'],
        'james': ['jim', 'jimmy', 'jamie'],
        'john': ['jon', 'johnny', 'jack'],
        'jonathan': ['jon', 'john', 'johnny', 'nathan'],
        'michael': ['mike', 'mick', 'mikey'],
        'christopher': ['chris', 'kit'],
        'nicholas': ['nick', 'nicky', 'nicolas'],
        'steven': ['steve', 'stephen', 'stefan'],
        'stephen': ['steve', 'steven', 'stefan'],
        'thomas': ['tom', 'tommy', 'thom'],
        'anthony': ['tony', 'ant'],
        'andrew': ['andy', 'drew'],
        'matthew': ['matt', 'matty'],
        'daniel': ['dan', 'danny'],
        'david': ['dave', 'davey'],
        'joseph': ['joe', 'joey', 'jo'],
        'edward': ['ed', 'eddie', 'ted', 'teddy', 'ned'],
        'charles': ['charlie', 'chuck', 'chas'],
        'benjamin': ['ben', 'benny', 'benji'],
        'alexander': ['alex', 'al', 'sandy', 'xander'],
        'elizabeth': ['liz', 'lizzy', 'beth', 'betty', 'eliza', 'libby'],
        'margaret': ['maggie', 'meg', 'marge', 'peggy', 'rita'],
        'katherine': ['kate', 'kathy', 'katie', 'cathy', 'kit'],
        'catherine': ['kate', 'kathy', 'katie', 'cathy', 'cat'],
        'jennifer': ['jen', 'jenny'],
        'patricia': ['pat', 'patty', 'tricia'],
        'barbara': ['barb', 'babs'],
        'susan': ['sue', 'susie', 'suzy'],
        'rebecca': ['becky', 'becca', 'reba'],
        'samantha': ['sam', 'sammy'],
        'victoria': ['vicky', 'vic', 'tori'],
        'deborah': ['deb', 'debbie'],
        'dorothy': ['dot', 'dotty', 'dora'],
        'gerald': ['gerry', 'jerry'],
        'geoffrey': ['geoff', 'jeff'],
        'jeffrey': ['jeff', 'geoff'],
        'peter': ['pete'],
        'philip': ['phil'],
        'lawrence': ['larry', 'laurie'],
        'raymond': ['ray'],
        'ronald': ['ron', 'ronnie'],
        'timothy': ['tim', 'timmy'],
        'gregory': ['greg'],
        'patrick': ['pat', 'paddy'],
        'henry': ['harry', 'hank'],
        'frederick': ['fred', 'freddy', 'fritz'],
        'albert': ['al', 'bert', 'bertie'],
        'arthur': ['art', 'artie'],
        'harold': ['harry', 'hal'],
        'samuel': ['sam', 'sammy'],
        'nathan': ['nate', 'nat'],
        'francis': ['frank', 'fran'],
        'frank': ['francis', 'frankie'],
    }

    def __init__(self):
        # Build reverse lookup for variants
        self.variant_lookup = defaultdict(set)
        for canonical, variants in self.NAME_VARIANTS.items():
            self.variant_lookup[canonical].add(canonical)
            for v in variants:
                self.variant_lookup[canonical].add(v)
                self.variant_lookup[v].add(canonical)
                for v2 in variants:
                    self.variant_lookup[v].add(v2)

    def normalize_name(self, name: str) -> str:
        """Normalize a name for comparison"""
        if not name:
            return ""
        # Remove titles, punctuation, extra spaces
        name = re.sub(r'\b(mr|mrs|ms|miss|dr|prof|sir|dame|lord|lady)\.?\b', '', name.lower())
        name = re.sub(r'[^\w\s]', '', name)
        name = ' '.join(name.split())
        return name.strip()

    def split_name(self, full_name: str) -> Tuple[str, str]:
        """Split full name into first name and surname"""
        parts = self.normalize_name(full_name).split()
        if len(parts) == 0:
            return ('', '')
        elif len(parts) == 1:
            return (parts[0], '')
        else:
            # First name is first part, surname is last part
            # (handles middle names by ignoring them)
            return (parts[0], parts[-1])

    def are_name_variants(self, name1: str, name2: str) -> bool:
        """Check if two names are known variants of each other"""
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        if n1 == n2:
            return True

        # Check direct variant lookup
        if n2 in self.variant_lookup.get(n1, set()):
            return True
        if n1 in self.variant_lookup.get(n2, set()):
            return True

        return False

    def name_similarity(self, name1: str, name2: str, max_distance: int = 2) -> float:
        """
        Calculate similarity between two names.
        Returns a score from 0.0 (no match) to 1.0 (exact/variant match).

        Considers:
        - Exact match
        - Known variants (John/Jon, William/Bill)
        - Phonetic similarity (Soundex/Metaphone)
        - Edit distance (for misspellings within max_distance)
        """
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)

        if not n1 or not n2:
            return 0.0

        # Exact match
        if n1 == n2:
            return 1.0

        # Known variant
        if self.are_name_variants(n1, n2):
            return 0.95

        # Phonetic match (Soundex)
        if soundex(n1) == soundex(n2):
            return 0.85

        # Phonetic match (Metaphone)
        if metaphone(n1) == metaphone(n2):
            return 0.80

        # Edit distance check
        distance = levenshtein_distance(n1, n2)
        max_len = max(len(n1), len(n2))

        if distance <= max_distance:
            # Within allowed edit distance - likely misspelling
            # Score decreases with distance
            return max(0.70, 1.0 - (distance / max_len))

        # Partial match (one name contains the other)
        if n1 in n2 or n2 in n1:
            shorter = min(len(n1), len(n2))
            longer = max(len(n1), len(n2))
            return 0.60 * (shorter / longer)

        # No strong match
        return 0.0

    def full_name_similarity(self, full_name1: str, full_name2: str) -> Tuple[float, Dict]:
        """
        Compare two full names, handling first name and surname separately.

        Returns:
        - Overall similarity score (0.0 to 1.0)
        - Dictionary with detailed breakdown
        """
        first1, last1 = self.split_name(full_name1)
        first2, last2 = self.split_name(full_name2)

        details = {
            'first_name_1': first1,
            'first_name_2': first2,
            'surname_1': last1,
            'surname_2': last2,
            'first_name_similarity': 0.0,
            'surname_similarity': 0.0,
        }

        # Calculate individual similarities
        if first1 and first2:
            details['first_name_similarity'] = self.name_similarity(first1, first2)

        if last1 and last2:
            details['surname_similarity'] = self.name_similarity(last1, last2)

        # Overall score: surname is more important for matching
        # Weights: surname 60%, first name 40%
        if last1 and last2 and first1 and first2:
            overall = (details['surname_similarity'] * 0.6) + (details['first_name_similarity'] * 0.4)
        elif last1 and last2:
            overall = details['surname_similarity']
        elif first1 and first2:
            overall = details['first_name_similarity'] * 0.7  # Less confident with only first name
        else:
            overall = 0.0

        details['overall_similarity'] = overall
        return overall, details

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
    and structured feature comparison with enhanced name matching.
    """

    def __init__(self, model_manager: Optional[AIModelManager] = None):
        self.model_manager = model_manager
        self.classifier = None
        self.is_trained = False
        self.training_data = []

        # Name matcher for fuzzy name comparison
        self.name_matcher = NameMatcher()

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

    def _extract_party_names(self, text: str) -> Dict[str, List[str]]:
        """
        Extract defendant and claimant names from case text.
        Uses multiple patterns to identify party names in legal documents.

        Returns dict with 'defendants' and 'claimants' lists.
        """
        defendants = []
        claimants = []

        if not text:
            return {'defendants': defendants, 'claimants': claimants}

        # Pattern 1: "Claimant: Name" or "Claimant - Name" format
        claimant_patterns = [
            r'claimant[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'claimants?[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'plaintiff[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'plaintiffs?[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+(?:brings|brought|claims|claimed|sues|sued)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+(?:v\.?|vs\.?|versus)\s+',
            r'the claimant,?\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+\(claimant\)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+\(the claimant\)',
        ]

        # Pattern 2: "Defendant: Name" or "Defendant - Name" format
        defendant_patterns = [
            r'defendant[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'defendants?[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'respondent[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'respondents?[:\s\-]+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'(?:v\.?|vs\.?|versus)\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'against\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'the defendant,?\s+([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+\(defendant\)',
            r'([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)*)\s+\(the defendant\)',
        ]

        # Common words to exclude (not names)
        exclude_words = {
            'the', 'court', 'judge', 'claim', 'case', 'action', 'proceedings',
            'claimant', 'defendant', 'plaintiff', 'respondent', 'this', 'that',
            'high', 'county', 'supreme', 'appeal', 'tribunal', 'employment',
            'contract', 'dispute', 'injury', 'personal', 'negligence', 'fraud',
            'commercial', 'property', 'recovery', 'debt', 'intellectual',
            'professional', 'civil', 'litigation', 'matter', 'application',
            'order', 'judgment', 'decision', 'hearing', 'trial', 'evidence',
            'witness', 'expert', 'damages', 'costs', 'claim', 'amount'
        }

        # Extract claimants
        for pattern in claimant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip()
                # Filter out non-names
                if name and len(name) > 2:
                    name_words = name.lower().split()
                    if not any(w in exclude_words for w in name_words):
                        claimants.append(name)

        # Extract defendants
        for pattern in defendant_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip()
                # Filter out non-names
                if name and len(name) > 2:
                    name_words = name.lower().split()
                    if not any(w in exclude_words for w in name_words):
                        defendants.append(name)

        # Also try to extract from structured data format (if present)
        # Pattern: "Name: John Smith" after Claimant/Defendant header
        structured_pattern = r'(?:claimant|plaintiff)\s*(?:name)?[:\s]*([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)+)'
        matches = re.findall(structured_pattern, text, re.IGNORECASE)
        claimants.extend([m.strip() for m in matches if m.strip() and len(m.strip()) > 2])

        structured_pattern = r'(?:defendant|respondent)\s*(?:name)?[:\s]*([A-Z][a-zA-Z\'\-]+(?:\s+[A-Z][a-zA-Z\'\-]+)+)'
        matches = re.findall(structured_pattern, text, re.IGNORECASE)
        defendants.extend([m.strip() for m in matches if m.strip() and len(m.strip()) > 2])

        # Deduplicate while preserving order
        seen = set()
        unique_claimants = []
        for c in claimants:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique_claimants.append(c)

        seen = set()
        unique_defendants = []
        for d in defendants:
            d_lower = d.lower()
            if d_lower not in seen:
                seen.add(d_lower)
                unique_defendants.append(d)

        return {
            'defendants': unique_defendants,
            'claimants': unique_claimants
        }

    def _calculate_party_name_similarity(self, case1: Dict, case2: Dict) -> Dict[str, float]:
        """
        Calculate similarity between party names (defendants and claimants) in two cases.
        Uses fuzzy matching to handle misspellings and name variants.

        Returns dict with defendant_similarity, claimant_similarity, and best matches.
        """
        # First check if names are provided in case metadata
        def1_names = case1.get('defendant_names', [])
        def2_names = case2.get('defendant_names', [])
        claim1_names = case1.get('claimant_names', [])
        claim2_names = case2.get('claimant_names', [])

        # If not in metadata, extract from text
        if not def1_names or not claim1_names:
            extracted1 = self._extract_party_names(case1.get('raw_text', ''))
            if not def1_names:
                def1_names = extracted1['defendants']
            if not claim1_names:
                claim1_names = extracted1['claimants']

        if not def2_names or not claim2_names:
            extracted2 = self._extract_party_names(case2.get('raw_text', ''))
            if not def2_names:
                def2_names = extracted2['defendants']
            if not claim2_names:
                claim2_names = extracted2['claimants']

        result = {
            'defendant_similarity': 0.0,
            'claimant_similarity': 0.0,
            'defendant_matches': [],
            'claimant_matches': [],
            'case1_defendants': def1_names,
            'case2_defendants': def2_names,
            'case1_claimants': claim1_names,
            'case2_claimants': claim2_names
        }

        # Calculate defendant similarity
        if def1_names and def2_names:
            max_similarities = []
            for name1 in def1_names:
                best_sim = 0.0
                best_match = None
                for name2 in def2_names:
                    sim, details = self.name_matcher.full_name_similarity(name1, name2)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = (name1, name2, sim, details)
                max_similarities.append(best_sim)
                if best_match and best_sim >= 0.6:
                    result['defendant_matches'].append(best_match)

            # Average of best matches (each name finds its best match)
            result['defendant_similarity'] = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0

        # Calculate claimant similarity
        if claim1_names and claim2_names:
            max_similarities = []
            for name1 in claim1_names:
                best_sim = 0.0
                best_match = None
                for name2 in claim2_names:
                    sim, details = self.name_matcher.full_name_similarity(name1, name2)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = (name1, name2, sim, details)
                max_similarities.append(best_sim)
                if best_match and best_sim >= 0.6:
                    result['claimant_matches'].append(best_match)

            result['claimant_similarity'] = sum(max_similarities) / len(max_similarities) if max_similarities else 0.0

        return result

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

        # *** NEW: Party name similarity with fuzzy matching ***
        party_sim = self._calculate_party_name_similarity(case1, case2)
        features['defendant_name_similarity'] = party_sim['defendant_similarity']
        features['claimant_name_similarity'] = party_sim['claimant_similarity']

        # Store match details for reporting (not used in scoring)
        features['_defendant_matches'] = party_sim.get('defendant_matches', [])
        features['_claimant_matches'] = party_sim.get('claimant_matches', [])
        features['_case1_defendants'] = party_sim.get('case1_defendants', [])
        features['_case2_defendants'] = party_sim.get('case2_defendants', [])
        features['_case1_claimants'] = party_sim.get('case1_claimants', [])
        features['_case2_claimants'] = party_sim.get('case2_claimants', [])

        return features

    def _calculate_composite_score(self, features: Dict) -> float:
        """
        Calculate weighted composite similarity score.

        Updated weights to prioritize name matching:
        - Defendant/claimant name similarity now has HIGH weight (0.20 + 0.15 = 0.35 total)
        - If names match strongly, this is a strong signal of duplicate
        """
        weights = {
            # Party name matching (NEW - high priority)
            'defendant_name_similarity': 0.20,  # Same defendant is strong signal
            'claimant_name_similarity': 0.15,   # Same claimant also important

            # Text similarity
            'tfidf_similarity': 0.15,           # Reduced from 0.25
            'sequence_similarity': 0.08,        # Reduced from 0.12
            'word_jaccard': 0.10,               # Reduced from 0.13

            # Case characteristics
            'amount_ratio': 0.08,               # Reduced from 0.12
            'same_case_type': 0.08,             # Reduced from 0.10
            'same_jurisdiction': 0.06,          # Reduced from 0.08
            'same_defendant_type': 0.04,        # Reduced from 0.06
            'same_complexity': 0.02,            # Reduced from 0.04
            'duration_ratio': 0.02,             # Reduced from 0.04
            'entity_overlap': 0.02,             # Reduced from 0.04
            'length_ratio': 0.00                # Removed - not useful
        }

        score = sum(features.get(k, 0) * v for k, v in weights.items())

        # Boost score if strong name match found
        # If both defendant AND claimant match well (>0.7), boost significantly
        def_sim = features.get('defendant_name_similarity', 0)
        claim_sim = features.get('claimant_name_similarity', 0)

        if def_sim >= 0.7 and claim_sim >= 0.7:
            # Both parties match - very likely duplicate
            score = min(1.0, score * 1.3)
        elif def_sim >= 0.8 or claim_sim >= 0.8:
            # At least one party has excellent match
            score = min(1.0, score * 1.15)

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
                            'entity_overlap': round(features['entity_overlap'] * 100, 1),
                            # NEW: Name matching features
                            'defendant_name_match': round(features.get('defendant_name_similarity', 0) * 100, 1),
                            'claimant_name_match': round(features.get('claimant_name_similarity', 0) * 100, 1),
                        },
                        # NEW: Detailed name matching info
                        'name_matches': {
                            'defendant_matches': features.get('_defendant_matches', []),
                            'claimant_matches': features.get('_claimant_matches', []),
                            'case1_defendants': features.get('_case1_defendants', []),
                            'case2_defendants': features.get('_case2_defendants', []),
                            'case1_claimants': features.get('_case1_claimants', []),
                            'case2_claimants': features.get('_case2_claimants', []),
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
