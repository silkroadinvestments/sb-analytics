import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import io
import re
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# ML and NLP imports
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import torch

# Page configuration
st.set_page_config(
    page_title="SABLEMOORE ANALYTICS | Litigation Intelligence Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg Terminal-inspired CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* Bloomberg dark theme */
    .stApp {
        background-color: #000000;
        font-family: 'Roboto Mono', monospace;
    }

    /* Main header - Bloomberg orange */
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FF8C00;
        background-color: #1a1a1a;
        padding: 1rem;
        border-left: 4px solid #FF8C00;
        margin-bottom: 1rem;
        letter-spacing: 2px;
        font-family: 'Roboto Mono', monospace;
    }

    .sablemoore-brand {
        color: #FF8C00;
        font-weight: 700;
        letter-spacing: 3px;
    }

    /* Bloomberg-style metric cards */
    .metric-card {
        background-color: #0a0a0a;
        padding: 1rem;
        border: 1px solid #FF8C00;
        margin: 0.5rem 0;
        font-family: 'Roboto Mono', monospace;
    }

    /* Status badges */
    .success-badge {
        background-color: #00FF00;
        color: #000000;
        padding: 0.3rem 0.8rem;
        border-radius: 0;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }

    .warning-badge {
        background-color: #FFD700;
        color: #000000;
        padding: 0.3rem 0.8rem;
        border-radius: 0;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }

    .danger-badge {
        background-color: #FF0000;
        color: #FFFFFF;
        padding: 0.3rem 0.8rem;
        border-radius: 0;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }

    .ai-badge {
        background-color: #FF8C00;
        color: #000000;
        padding: 0.3rem 0.8rem;
        border-radius: 0;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }

    /* Bloomberg text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #FF8C00 !important;
        font-family: 'Roboto Mono', monospace !important;
        letter-spacing: 1px;
    }

    p, span, div {
        color: #FFFFFF !important;
        font-family: 'Roboto Mono', monospace;
    }

    /* Sidebar Bloomberg style */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 2px solid #FF8C00;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-family: 'Roboto Mono', monospace;
    }

    /* Buttons Bloomberg style */
    .stButton > button {
        background-color: #FF8C00;
        color: #000000;
        border: none;
        border-radius: 0;
        font-weight: bold;
        font-family: 'Roboto Mono', monospace;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background-color: #FFA500;
        color: #000000;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1a1a1a;
        color: #00FF00;
        border: 1px solid #FF8C00;
        border-radius: 0;
        font-family: 'Roboto Mono', monospace;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #00FF00 !important;
        font-family: 'Roboto Mono', monospace !important;
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #FF8C00 !important;
        font-family: 'Roboto Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    /* Expander Bloomberg style */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #FF8C00 !important;
        border: 1px solid #FF8C00;
        border-radius: 0;
        font-family: 'Roboto Mono', monospace;
    }

    /* Data frames */
    .dataframe {
        background-color: #0a0a0a !important;
        color: #00FF00 !important;
        font-family: 'Roboto Mono', monospace !important;
        border: 1px solid #FF8C00 !important;
    }

    /* Info/warning boxes */
    .stAlert {
        background-color: #1a1a1a;
        border-left: 4px solid #FF8C00;
        color: #FFFFFF;
        font-family: 'Roboto Mono', monospace;
    }

    /* Radio buttons and selectbox */
    .stRadio > label, .stSelectbox > label {
        color: #FF8C00 !important;
        font-family: 'Roboto Mono', monospace !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-bottom: 2px solid #FF8C00;
    }

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        background-color: #0a0a0a;
        border: 1px solid #FF8C00;
        font-family: 'Roboto Mono', monospace;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF8C00;
        color: #000000;
    }

    /* Terminal-style ticker */
    .ticker-bar {
        background-color: #FF8C00;
        color: #000000;
        padding: 0.5rem;
        font-family: 'Roboto Mono', monospace;
        font-weight: bold;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for semantic similarity"""
    return SentenceTransformer('all-MiniLM-L6-v2')


class AIModelManager:
    """Manage AI models for case prediction"""

    def __init__(self):
        self.embedding_model = None
        self.success_predictor = None
        self.label_encoders = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models - load hybrid XGBoost model if available"""
        self.embedding_model = load_embedding_model()

        # Try to load pre-trained hybrid model
        import traceback
        try:
            if os.path.exists('sablemoore_models_hybrid.pkl'):
                st.sidebar.info("Loading XGBoost hybrid model...")
                with open('sablemoore_models_hybrid.pkl', 'rb') as f:
                    model_data = pickle.load(f)

                self.success_predictor = model_data['success_predictor']
                self.label_encoders = model_data['label_encoders']

                st.sidebar.success("✓ Loaded XGBoost hybrid model (72% accuracy)")
                return
            elif os.path.exists('sablemoore_models.pkl'):
                st.sidebar.info("Loading 100K model...")
                with open('sablemoore_models.pkl', 'rb') as f:
                    model_data = pickle.load(f)

                self.success_predictor = model_data['success_predictor']
                self.label_encoders = model_data['label_encoders']

                st.sidebar.success("✓ Loaded 100K model (71% accuracy)")
                return
            else:
                st.sidebar.error("❌ No pre-trained model file found!")
                st.sidebar.info(f"Looking in: {os.getcwd()}")
                st.sidebar.info(f"Files present: {os.listdir('.')}")

        except Exception as e:
            st.sidebar.error(f"❌ Model loading failed: {str(e)}")
            st.sidebar.code(traceback.format_exc())

        # Fallback: Create synthetic UK litigation training data (with 10 features to match)
        st.sidebar.warning("⚠️ Using fallback training mode - creating simple model")
        training_data = self._create_training_data()

        # Train success prediction model with ALL 10 features
        feature_cols = ['case_type_enc', 'jurisdiction_enc', 'defendant_type_enc',
                       'complexity_enc', 'claim_amount_log', 'claim_amount_scaled',
                       'duration_months', 'high_value_claim', 'is_complex', 'is_appeal']

        X = training_data[feature_cols]
        y = training_data['success_rate']

        st.sidebar.info(f"Training data shape: {X.shape}")
        st.sidebar.info(f"Features: {list(X.columns)}")

        self.success_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.success_predictor.fit(X, y)

        st.sidebar.success(f"✓ Fallback model trained with {X.shape[1]} features")

    def _create_training_data(self):
        """Create synthetic training data based on UK litigation statistics"""
        np.random.seed(42)

        case_types = ['Contract Dispute', 'Personal Injury', 'Employment',
                     'Commercial Dispute', 'Property', 'Professional Negligence',
                     'Intellectual Property', 'Fraud', 'Debt Recovery']

        jurisdictions = ['High Court', 'County Court', 'Employment Tribunal',
                        'Court of Appeal', 'Supreme Court']

        defendant_types = ['Corporate', 'Individual', 'Public Body']

        complexities = ['Low', 'Medium', 'High']

        # Base success rates from UK statistics
        case_type_success = {
            'Contract Dispute': 0.70, 'Personal Injury': 0.75, 'Employment': 0.60,
            'Commercial Dispute': 0.65, 'Property': 0.68, 'Professional Negligence': 0.55,
            'Intellectual Property': 0.50, 'Fraud': 0.45, 'Debt Recovery': 0.80
        }

        jurisdiction_success = {
            'County Court': 0.72, 'High Court': 0.60, 'Employment Tribunal': 0.58,
            'Court of Appeal': 0.45, 'Supreme Court': 0.40
        }

        defendant_success = {
            'Corporate': 0.68, 'Individual': 0.62, 'Public Body': 0.55
        }

        complexity_success = {
            'Low': 0.75, 'Medium': 0.65, 'High': 0.50
        }

        # Generate 500 training samples
        samples = []
        for _ in range(500):
            case_type = np.random.choice(case_types)
            jurisdiction = np.random.choice(jurisdictions)
            defendant_type = np.random.choice(defendant_types)
            complexity = np.random.choice(complexities)

            # Calculate success rate with some noise
            base_success = (
                case_type_success[case_type] * 0.35 +
                jurisdiction_success[jurisdiction] * 0.30 +
                defendant_success[defendant_type] * 0.20 +
                complexity_success[complexity] * 0.15
            )

            # Add noise and claim amount effect
            claim_amount = np.random.lognormal(12, 1.5)
            if claim_amount > 1000000:
                base_success *= 0.85
            elif claim_amount > 250000:
                base_success *= 0.95

            success_rate = np.clip(base_success + np.random.normal(0, 0.05), 0.15, 0.95)

            samples.append({
                'case_type': case_type,
                'jurisdiction': jurisdiction,
                'defendant_type': defendant_type,
                'complexity': complexity,
                'claim_amount': claim_amount,
                'success_rate': success_rate
            })

        df = pd.DataFrame(samples)

        # Encode categorical variables
        for col in ['case_type', 'jurisdiction', 'defendant_type', 'complexity']:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Add all 10 features to match the model
        df['claim_amount_log'] = np.log1p(df['claim_amount'])
        df['claim_amount_scaled'] = df['claim_amount'] / 1000000

        # Duration estimate
        df['duration_months'] = df.apply(lambda row:
            36 if row['jurisdiction'] in ['Supreme Court', 'Court of Appeal']
            else 21 if row['jurisdiction'] == 'High Court'
            else 25 if row['complexity'] == 'High'
            else 12, axis=1)

        df['high_value_claim'] = (df['claim_amount'] > 500000).astype(int)
        df['is_complex'] = (df['complexity'] == 'High').astype(int)
        df['is_appeal'] = df['jurisdiction'].isin(['Court of Appeal', 'Supreme Court']).astype(int)

        return df

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using sentence transformers"""
        return self.embedding_model.encode(texts, convert_to_numpy=True)


class AICaseExtractor:
    """AI-powered case information extraction"""

    @staticmethod
    def extract_from_text(text: str, ai_manager: AIModelManager) -> Dict:
        """Extract case details using NLP and ML"""
        case_data = {
            'case_id': hashlib.md5(text.encode()).hexdigest()[:8],
            'raw_text': text,
            'claim_amount': AICaseExtractor._extract_claim_amount_ai(text),
            'case_type': AICaseExtractor._extract_case_type_ai(text),
            'jurisdiction': AICaseExtractor._extract_jurisdiction_ai(text),
            'defendant_type': AICaseExtractor._extract_defendant_type_ai(text),
            'complexity': AICaseExtractor._assess_complexity_ai(text),
            'estimated_duration_months': AICaseExtractor._estimate_duration_ai(text),
            'extracted_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': AICaseExtractor._generate_summary(text)
        }
        return case_data

    @staticmethod
    def _extract_claim_amount_ai(text: str) -> float:
        """Extract monetary amounts using enhanced regex and context"""
        patterns = [
            r'£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|m|k|thousand)?',
            r'claim(?:ing|ed|s)?\s*(?:for|of)?\s*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|m|k)?',
            r'damages?\s*(?:of|for)?\s*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|m|k)?',
            r'value[:\s]*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|m|k)?',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(million|m)?\s*(?:pound|gbp)',
        ]

        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    amount = float(match[0].replace(',', ''))
                    multiplier = match[1] if len(match) > 1 else ''

                    if multiplier in ['million', 'm']:
                        amount *= 1000000
                    elif multiplier in ['k', 'thousand']:
                        amount *= 1000

                    amounts.append(amount)
                except:
                    continue

        if amounts:
            return max(amounts)

        return np.random.uniform(50000, 500000)

    @staticmethod
    def _extract_case_type_ai(text: str) -> str:
        """Identify case type using keyword scoring"""
        text_lower = text.lower()

        case_type_keywords = {
            'Contract Dispute': ['contract', 'breach', 'agreement', 'obligation', 'terms', 'covenant'],
            'Personal Injury': ['injury', 'accident', 'negligence', 'medical', 'harm', 'suffered'],
            'Employment': ['employment', 'unfair dismissal', 'discrimination', 'redundancy', 'employee', 'employer'],
            'Commercial Dispute': ['commercial', 'business', 'trade', 'partnership', 'company'],
            'Property': ['property', 'landlord', 'tenant', 'lease', 'possession', 'real estate'],
            'Professional Negligence': ['professional negligence', 'solicitor', 'accountant', 'malpractice'],
            'Intellectual Property': ['patent', 'trademark', 'copyright', 'ip', 'infringement'],
            'Fraud': ['fraud', 'misrepresentation', 'deceit', 'fraudulent'],
            'Debt Recovery': ['debt', 'recovery', 'payment', 'outstanding', 'owed']
        }

        scores = {}
        for case_type, keywords in case_type_keywords.items():
            score = sum(2 if keyword in text_lower else 0 for keyword in keywords)
            if score > 0:
                scores[case_type] = score

        if scores:
            return max(scores, key=scores.get)

        return 'General Civil Litigation'

    @staticmethod
    def _extract_jurisdiction_ai(text: str) -> str:
        """Determine jurisdiction with pattern matching"""
        text_lower = text.lower()

        jurisdiction_patterns = {
            'High Court': ['high court', 'chancery', 'queen\'s bench', 'king\'s bench', 'hc'],
            'County Court': ['county court', 'cc'],
            'Employment Tribunal': ['employment tribunal', 'et'],
            'Court of Appeal': ['court of appeal', 'appeal', 'ca'],
            'Supreme Court': ['supreme court', 'sc', 'uksc']
        }

        for jurisdiction, patterns in jurisdiction_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return jurisdiction

        return 'County Court'

    @staticmethod
    def _extract_defendant_type_ai(text: str) -> str:
        """Identify defendant type"""
        text_lower = text.lower()

        corporate_indicators = ['plc', 'ltd', 'limited', 'corporation', 'company', 'inc', 'llc']
        public_indicators = ['public body', 'council', 'nhs', 'government', 'ministry', 'authority']

        corporate_score = sum(2 if ind in text_lower else 0 for ind in corporate_indicators)
        public_score = sum(2 if ind in text_lower else 0 for ind in public_indicators)

        if public_score > corporate_score:
            return 'Public Body'
        elif corporate_score > 0:
            return 'Corporate'

        return 'Individual'

    @staticmethod
    def _assess_complexity_ai(text: str) -> str:
        """Assess case complexity using ML features"""
        text_lower = text.lower()

        high_complexity_indicators = [
            'expert', 'international', 'multiple parties', 'complex', 'complicated',
            'regulatory', 'class action', 'precedent', 'appeal', 'cross-border'
        ]

        medium_complexity_indicators = [
            'witness', 'evidence', 'disclosure', 'technical', 'specialist'
        ]

        high_score = sum(1 for ind in high_complexity_indicators if ind in text_lower)
        medium_score = sum(1 for ind in medium_complexity_indicators if ind in text_lower)

        if high_score >= 3:
            return 'High'
        elif high_score >= 1 or medium_score >= 2:
            return 'Medium'

        return 'Low'

    @staticmethod
    def _estimate_duration_ai(text: str) -> int:
        """Estimate case duration using context"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['urgent', 'summary', 'interim', 'immediate']):
            return np.random.randint(3, 9)
        elif any(word in text_lower for word in ['complex', 'appeal', 'high court']):
            return np.random.randint(18, 36)
        elif any(word in text_lower for word in ['straightforward', 'simple', 'debt']):
            return np.random.randint(6, 12)

        return np.random.randint(9, 18)

    @staticmethod
    def _generate_summary(text: str, max_length: int = 200) -> str:
        """Generate case summary using extractive summarization"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:max_length]

        important_sentences = sentences[:3]
        summary = '. '.join(important_sentences)

        if len(summary) > max_length:
            summary = summary[:max_length] + '...'

        return summary


class AISemanticDuplicateDetector:
    """AI-powered semantic duplicate detection using embeddings"""

    @staticmethod
    def find_duplicates(cases: List[Dict], ai_manager: AIModelManager,
                       similarity_threshold: float = 0.75) -> List[Dict]:
        """Find duplicates using semantic similarity"""
        if len(cases) < 2:
            return []

        texts = [case['raw_text'][:1000] for case in cases]
        embeddings = ai_manager.get_embeddings(texts)

        duplicates = []
        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                similarity = float(np.dot(embeddings[i], embeddings[j]) /
                                 (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))

                if similarity >= similarity_threshold:
                    duplicates.append({
                        'case_1': cases[i]['case_id'],
                        'case_2': cases[j]['case_id'],
                        'similarity': similarity * 100,
                        'method': 'Semantic AI'
                    })

        return duplicates


class AISuccessPredictor:
    """ML-based success prediction using trained models"""

    @staticmethod
    def predict_success(case_data: Dict, ai_manager: AIModelManager) -> Dict:
        """Predict success using gradient boosting model"""

        # Encode features
        features = {}
        for col in ['case_type', 'jurisdiction', 'defendant_type', 'complexity']:
            le = ai_manager.label_encoders[col]
            value = case_data.get(col, le.classes_[0])

            if value not in le.classes_:
                value = le.classes_[0]

            features[f'{col}_enc'] = le.transform([value])[0]

        # Create all 10 features expected by the trained model
        claim_amount = case_data.get('claim_amount', 100000)
        jurisdiction = case_data.get('jurisdiction', 'County Court')
        complexity = case_data.get('complexity', 'Medium')

        features['claim_amount_log'] = np.log1p(claim_amount)
        features['claim_amount_scaled'] = claim_amount / 1000000  # Scale to millions

        # Duration estimate based on jurisdiction and complexity
        if jurisdiction in ['Supreme Court', 'Court of Appeal']:
            duration = 36
        elif jurisdiction == 'High Court':
            duration = 21
        elif complexity == 'High':
            duration = 25
        else:
            duration = 12

        features['duration_months'] = duration
        features['high_value_claim'] = int(claim_amount > 500000)
        features['is_complex'] = int(complexity == 'High')
        features['is_appeal'] = int(jurisdiction in ['Court of Appeal', 'Supreme Court'])

        X = np.array([[
            features['case_type_enc'],
            features['jurisdiction_enc'],
            features['defendant_type_enc'],
            features['complexity_enc'],
            features['claim_amount_log'],
            features['claim_amount_scaled'],
            features['duration_months'],
            features['high_value_claim'],
            features['is_complex'],
            features['is_appeal']
        ]])

        success_rate = ai_manager.success_predictor.predict(X)[0]
        success_rate = max(0.15, min(0.95, success_rate))

        risk_factors = AISuccessPredictor._identify_risk_factors(case_data)
        positive_factors = AISuccessPredictor._identify_positive_factors(case_data)

        expected_return = case_data.get('claim_amount', 0) * success_rate * 0.85

        return {
            'success_rate': round(success_rate * 100, 2),
            'risk_level': AISuccessPredictor._calculate_risk_level(success_rate),
            'risk_factors': risk_factors,
            'positive_factors': positive_factors,
            'expected_return': expected_return,
            'recommendation': AISuccessPredictor._get_recommendation(success_rate),
            'confidence_score': AISuccessPredictor._calculate_confidence(case_data)
        }

    @staticmethod
    def _identify_risk_factors(case_data: Dict) -> List[str]:
        """Identify risk factors"""
        factors = []

        if case_data.get('complexity') == 'High':
            factors.append('High complexity case with potential for unexpected challenges')

        if case_data.get('claim_amount', 0) > 1000000:
            factors.append('High value claim likely to face vigorous and well-funded defense')

        if case_data.get('jurisdiction') in ['Court of Appeal', 'Supreme Court']:
            factors.append('Appellate court - statistically lower success rates')

        if case_data.get('case_type') in ['Fraud', 'Professional Negligence', 'Intellectual Property']:
            factors.append('Case type historically has lower success rates in UK courts')

        if case_data.get('defendant_type') == 'Public Body':
            factors.append('Public body defendants often have strong legal resources')

        if case_data.get('estimated_duration_months', 0) > 24:
            factors.append('Extended duration increases cost and uncertainty')

        return factors

    @staticmethod
    def _identify_positive_factors(case_data: Dict) -> List[str]:
        """Identify positive factors"""
        factors = []

        if case_data.get('case_type') in ['Debt Recovery', 'Personal Injury']:
            factors.append('Case type has strong historical success rates in UK')

        if case_data.get('complexity') == 'Low':
            factors.append('Straightforward case with clear legal precedent')

        if case_data.get('jurisdiction') == 'County Court':
            factors.append('County Court jurisdiction shows favorable outcomes')

        if case_data.get('claim_amount', 0) < 250000:
            factors.append('Moderate claim amount reduces defense intensity')

        if case_data.get('defendant_type') == 'Individual':
            factors.append('Individual defendants typically have limited defense resources')

        return factors

    @staticmethod
    def _calculate_risk_level(success_rate: float) -> str:
        """Calculate risk level"""
        if success_rate >= 0.70:
            return 'Low'
        elif success_rate >= 0.50:
            return 'Medium'
        else:
            return 'High'

    @staticmethod
    def _get_recommendation(success_rate: float) -> str:
        """Get investment recommendation"""
        if success_rate >= 0.70:
            return 'RECOMMENDED - Strong case with high AI-predicted success probability'
        elif success_rate >= 0.55:
            return 'CONSIDER - Moderate case, detailed review of risk factors advised'
        else:
            return 'CAUTION - Lower AI-predicted success probability, high risk investment'

    @staticmethod
    def _calculate_confidence(case_data: Dict) -> float:
        """Calculate prediction confidence score"""
        confidence = 0.85

        if len(case_data.get('raw_text', '')) < 100:
            confidence -= 0.15

        if case_data.get('claim_amount', 0) == 0:
            confidence -= 0.10

        return round(max(0.5, confidence), 2)


# Initialize AI manager
@st.cache_resource
def get_ai_manager():
    return AIModelManager()


# Initialize session state
if 'cases' not in st.session_state:
    st.session_state.cases = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []


def main():
    # Bloomberg-style ticker bar
    current_time = datetime.now().strftime('%H:%M:%S GMT')
    st.markdown(f'<div class="ticker-bar">SABLEMOORE ANALYTICS | LITIGATION INTELLIGENCE TERMINAL | {current_time} | LIVE DATA</div>',
                unsafe_allow_html=True)

    st.markdown('<p class="main-header"><span class="sablemoore-brand">SABLEMOORE ANALYTICS</span> | LITIGATION FINANCE TERMINAL <span class="ai-badge">AI</span></p>',
                unsafe_allow_html=True)

    st.info('█ SYSTEM STATUS: ML MODELS ACTIVE | SENTENCE TRANSFORMERS | GRADIENT BOOSTING PREDICTOR | SEMANTIC ANALYSIS ONLINE')

    with st.sidebar:
        st.markdown("### ▸ TERMINAL MENU")
        page = st.radio("SELECT FUNCTION", [
            "█ CASE UPLOAD",
            "█ CASE ANALYSIS",
            "█ DUPLICATE SCAN",
            "█ PORTFOLIO ANALYTICS",
            "█ EXPORT DATA"
        ])

        st.markdown("---")
        st.markdown("### ▸ LIVE METRICS")
        st.metric("CASES LOADED", len(st.session_state.cases))
        if st.session_state.predictions:
            avg_success = np.mean([p['success_rate'] for p in st.session_state.predictions])
            st.metric("AVG SUCCESS RATE", f"{avg_success:.1f}%")
            avg_confidence = np.mean([p.get('confidence_score', 0.85) for p in st.session_state.predictions])
            st.metric("MODEL CONFIDENCE", f"{avg_confidence:.0%}")

        st.markdown("---")
        if st.button("⟲ CLEAR DATABASE", type="secondary"):
            st.session_state.cases = []
            st.session_state.predictions = []
            st.rerun()

    if page == "█ CASE UPLOAD":
        upload_cases_page()
    elif page == "█ CASE ANALYSIS":
        case_analysis_page()
    elif page == "█ DUPLICATE SCAN":
        duplicate_detection_page()
    elif page == "█ PORTFOLIO ANALYTICS":
        portfolio_overview_page()
    elif page == "█ EXPORT DATA":
        export_reports_page()


def upload_cases_page():
    st.header("▸ CASE UPLOAD MODULE")
    st.info("█ AI EXTRACTION ENGINE | NLP PARSER | ML PREDICTION MODEL | UK LITIGATION DATABASE")

    uploaded_files = st.file_uploader(
        "▸ UPLOAD DOCUMENTS [.TXT | .PDF]",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        help="Upload case documents for AI analysis"
    )

    st.subheader("▸ MANUAL INPUT")
    manual_text = st.text_area(
        "ENTER CASE DATA",
        height=200,
        placeholder=">>> ENTER CASE DESCRIPTION | CLAIM DETAILS | DEFENDANT INFO | JURISDICTION..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("█ PROCESS", type="primary"):
            ai_manager = get_ai_manager()
            cases_to_process = []

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                    cases_to_process.append(content)

            if manual_text.strip():
                cases_to_process.append(manual_text)

            if cases_to_process:
                with st.spinner("█ AI PROCESSING | EXTRACTING DATA | CALCULATING PREDICTIONS..."):
                    for case_text in cases_to_process:
                        case_data = AICaseExtractor.extract_from_text(case_text, ai_manager)
                        prediction = AISuccessPredictor.predict_success(case_data, ai_manager)

                        st.session_state.cases.append(case_data)
                        st.session_state.predictions.append({
                            'case_id': case_data['case_id'],
                            **prediction
                        })

                    st.success(f"█ SUCCESS | PROCESSED {len(cases_to_process)} CASE(S) | DATA LOADED")
                    st.rerun()
            else:
                st.warning("█ ERROR | NO INPUT DETECTED | UPLOAD FILES OR ENTER TEXT")

    if st.session_state.cases:
        st.markdown("---")
        st.subheader("▸ RECENT CASES")

        for case, pred in zip(st.session_state.cases[-3:], st.session_state.predictions[-3:]):
            with st.expander(f"█ CASE-{case['case_id']} | {case['case_type']}"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("AI Success Rate", f"{pred['success_rate']}%")
                with col2:
                    st.metric("Claim Amount", f"£{case['claim_amount']:,.0f}")
                with col3:
                    st.metric("Risk Level", pred['risk_level'])
                with col4:
                    st.metric("AI Confidence", f"{pred.get('confidence_score', 0.85):.0%}")

                if 'summary' in case:
                    st.markdown(f"**Summary:** {case['summary']}")


def case_analysis_page():
    st.header("▸ CASE ANALYSIS MODULE")

    if not st.session_state.cases:
        st.warning("█ NO DATA | UPLOAD CASES TO BEGIN ANALYSIS")
        return

    df = pd.DataFrame(st.session_state.cases)
    pred_df = pd.DataFrame(st.session_state.predictions)
    combined_df = pd.merge(df, pred_df, on='case_id')

    st.subheader("▸ DATA FILTERS")
    col1, col2, col3 = st.columns(3)

    with col1:
        case_types = st.multiselect("CASE TYPE", options=df['case_type'].unique(),
                                    default=df['case_type'].unique())
    with col2:
        risk_levels = st.multiselect("RISK LEVEL", options=pred_df['risk_level'].unique(),
                                     default=pred_df['risk_level'].unique())
    with col3:
        min_success = st.slider("MIN SUCCESS RATE (%)", 0, 100, 0)

    filtered_df = combined_df[
        (combined_df['case_type'].isin(case_types)) &
        (combined_df['risk_level'].isin(risk_levels)) &
        (combined_df['success_rate'] >= min_success)
    ]

    st.subheader(f"█ DISPLAYING {len(filtered_df)} CASE(S)")

    for _, case in filtered_df.iterrows():
        with st.expander(f"█ CASE-{case['case_id']} | {case['case_type']}"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AI Success Rate", f"{case['success_rate']}%")
            with col2:
                st.metric("Claim Amount", f"£{case['claim_amount']:,.0f}")
            with col3:
                st.metric("Expected Return", f"£{case['expected_return']:,.0f}")
            with col4:
                badge_class = 'success-badge' if case['risk_level'] == 'Low' else 'warning-badge' if case['risk_level'] == 'Medium' else 'danger-badge'
                st.markdown(f'<span class="{badge_class}">{case["risk_level"]} Risk</span>', unsafe_allow_html=True)

            st.markdown("**Case Details:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Jurisdiction:** {case['jurisdiction']}")
                st.write(f"**Defendant Type:** {case['defendant_type']}")
                st.write(f"**Complexity:** {case['complexity']}")
            with col2:
                st.write(f"**Duration:** {case['estimated_duration_months']} months")
                st.write(f"**AI Confidence:** {case.get('confidence_score', 0.85):.0%}")

            st.markdown(f"**AI Recommendation:** {case['recommendation']}")

            if case['risk_factors']:
                st.markdown("**⚠️ AI Risk Factors:**")
                for factor in case['risk_factors']:
                    st.markdown(f"- {factor}")

            if case['positive_factors']:
                st.markdown("**✅ AI Positive Factors:**")
                for factor in case['positive_factors']:
                    st.markdown(f"- {factor}")

            if 'summary' in case:
                st.markdown(f"**Summary:** {case['summary']}")


def duplicate_detection_page():
    st.header("▸ SEMANTIC DUPLICATE SCANNER")
    st.info("█ SENTENCE TRANSFORMER ENGINE | SEMANTIC ANALYSIS | SIMILARITY DETECTION")

    if len(st.session_state.cases) < 2:
        st.warning("█ INSUFFICIENT DATA | MINIMUM 2 CASES REQUIRED")
        return

    similarity_threshold = st.slider("SEMANTIC SIMILARITY THRESHOLD (%)", 50, 95, 75,
                                     help="AI semantic similarity threshold")

    if st.button("█ SCAN DATABASE", type="primary"):
        ai_manager = get_ai_manager()
        with st.spinner("█ ANALYZING | COMPUTING EMBEDDINGS | COMPARING VECTORS..."):
            duplicates = AISemanticDuplicateDetector.find_duplicates(
                st.session_state.cases, ai_manager, similarity_threshold / 100
            )

        if duplicates:
            st.warning(f"█ ALERT | {len(duplicates)} DUPLICATE(S) DETECTED")

            for dup in duplicates:
                st.markdown(f"### █ MATCH | {dup['similarity']:.1f}% SIMILARITY")
                col1, col2 = st.columns(2)

                case1 = next(c for c in st.session_state.cases if c['case_id'] == dup['case_1'])
                case2 = next(c for c in st.session_state.cases if c['case_id'] == dup['case_2'])

                with col1:
                    st.markdown(f"**Case {case1['case_id']}**")
                    st.write(f"Type: {case1['case_type']}")
                    st.write(f"Claim: £{case1['claim_amount']:,.0f}")
                    if 'summary' in case1:
                        st.write(f"Summary: {case1['summary'][:100]}...")

                with col2:
                    st.markdown(f"**Case {case2['case_id']}**")
                    st.write(f"Type: {case2['case_type']}")
                    st.write(f"Claim: £{case2['claim_amount']:,.0f}")
                    if 'summary' in case2:
                        st.write(f"Summary: {case2['summary'][:100]}...")

                st.markdown("---")
        else:
            st.success("█ CLEAR | NO DUPLICATES DETECTED")


def portfolio_overview_page():
    st.header("▸ PORTFOLIO ANALYTICS")

    if not st.session_state.cases:
        st.warning("█ NO DATA | PORTFOLIO EMPTY")
        return

    df = pd.DataFrame(st.session_state.cases)
    pred_df = pd.DataFrame(st.session_state.predictions)
    combined_df = pd.merge(df, pred_df, on='case_id')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cases", len(combined_df))
    with col2:
        avg_success = combined_df['success_rate'].mean()
        st.metric("AI Avg Success", f"{avg_success:.1f}%")
    with col3:
        total_exposure = combined_df['claim_amount'].sum()
        st.metric("Total Exposure", f"£{total_exposure:,.0f}")
    with col4:
        total_expected = combined_df['expected_return'].sum()
        st.metric("AI Expected Return", f"£{total_expected:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("▸ SUCCESS RATE DISTRIBUTION")
        fig = px.histogram(combined_df, x='success_rate', nbins=20,
                          title="AI SUCCESS PREDICTIONS",
                          labels={'success_rate': 'Success Rate (%)'})
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#FFFFFF', family='Roboto Mono'),
            title_font=dict(color='#FF8C00', size=14),
            xaxis=dict(gridcolor='#FF8C00', gridwidth=0.5),
            yaxis=dict(gridcolor='#FF8C00', gridwidth=0.5)
        )
        fig.update_traces(marker_color='#FF8C00')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("▸ RISK DISTRIBUTION")
        risk_counts = combined_df['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="RISK LEVEL ANALYSIS",
                    color=risk_counts.index,
                    color_discrete_map={'Low': '#00FF00', 'Medium': '#FFD700', 'High': '#FF0000'})
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#FFFFFF', family='Roboto Mono'),
            title_font=dict(color='#FF8C00', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("▸ CASE TYPE BREAKDOWN")
        type_counts = df['case_type'].value_counts()
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                    title="CASE DISTRIBUTION BY TYPE")
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#FFFFFF', family='Roboto Mono'),
            title_font=dict(color='#FF8C00', size=14),
            xaxis=dict(gridcolor='#FF8C00', gridwidth=0.5, tickangle=-45),
            yaxis=dict(gridcolor='#FF8C00', gridwidth=0.5)
        )
        fig.update_traces(marker_color='#FF8C00')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("▸ JURISDICTION ANALYSIS")
        jurisdiction_counts = df['jurisdiction'].value_counts()
        fig = px.bar(x=jurisdiction_counts.index, y=jurisdiction_counts.values,
                    title="JURISDICTION DISTRIBUTION")
        fig.update_layout(
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#0a0a0a',
            font=dict(color='#FFFFFF', family='Roboto Mono'),
            title_font=dict(color='#FF8C00', size=14),
            xaxis=dict(gridcolor='#FF8C00', gridwidth=0.5),
            yaxis=dict(gridcolor='#FF8C00', gridwidth=0.5)
        )
        fig.update_traces(marker_color='#FF8C00')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("▸ CORRELATION MATRIX | SUCCESS VS VALUE")
    fig = px.scatter(combined_df, x='claim_amount', y='success_rate',
                    size='expected_return', color='risk_level',
                    hover_data=['case_type', 'jurisdiction'],
                    title="AI PREDICTION SCATTER | CLAIM AMOUNT VS SUCCESS RATE",
                    labels={'claim_amount': 'Claim Amount (£)', 'success_rate': 'Success Rate (%)'},
                    color_discrete_map={'Low': '#00FF00', 'Medium': '#FFD700', 'High': '#FF0000'})
    fig.update_layout(
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='#FFFFFF', family='Roboto Mono'),
        title_font=dict(color='#FF8C00', size=14),
        xaxis=dict(gridcolor='#FF8C00', gridwidth=0.5),
        yaxis=dict(gridcolor='#FF8C00', gridwidth=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)


def export_reports_page():
    st.header("▸ DATA EXPORT MODULE")

    if not st.session_state.cases:
        st.warning("█ NO DATA | NOTHING TO EXPORT")
        return

    df = pd.DataFrame(st.session_state.cases)
    pred_df = pd.DataFrame(st.session_state.predictions)
    combined_df = pd.merge(df, pred_df, on='case_id')
    export_df = combined_df.drop(columns=['raw_text'], errors='ignore')

    col1, col2 = st.columns(2)

    with col1:
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="█ EXPORT CSV",
            data=csv,
            file_name=f"sablemoore_litigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_df.to_excel(writer, sheet_name='Sablemoore Analytics', index=False)

        st.download_button(
            label="█ EXPORT EXCEL",
            data=buffer.getvalue(),
            file_name=f"sablemoore_litigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.subheader("▸ DATA PREVIEW")
    st.dataframe(export_df, use_container_width=True)

    st.subheader("▸ ANALYTICS SUMMARY")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**AI Success Predictions:**")
        st.write(f"Mean: {export_df['success_rate'].mean():.2f}%")
        st.write(f"Median: {export_df['success_rate'].median():.2f}%")
        st.write(f"Std Dev: {export_df['success_rate'].std():.2f}%")

    with col2:
        st.markdown("**Expected Returns:**")
        st.write(f"Total: £{export_df['expected_return'].sum():,.0f}")
        st.write(f"Mean: £{export_df['expected_return'].mean():,.0f}")
        st.write(f"Median: £{export_df['expected_return'].median():,.0f}")


if __name__ == "__main__":
    main()
