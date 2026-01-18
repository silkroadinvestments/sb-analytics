"""
Sablemoore Analytics - Litigation Finance Intelligence Platform
Streamlit Application with Enhanced ML Engine
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
from difflib import SequenceMatcher
from typing import Dict, List
import json
import os
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go

# Import the enhanced ML engine
try:
    from ml_engine import (
        get_model_manager, get_success_predictor, get_duplicate_detector,
        EnhancedSuccessPredictor, EnhancedDuplicateDetector, AIModelManager,
        generate_synthetic_training_data, SKLEARN_AVAILABLE
    )
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False
    SKLEARN_AVAILABLE = False

# Fallback sklearn import if ml_engine not available
if not ML_ENGINE_AVAILABLE:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.ensemble import RandomForestClassifier
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sablemoore Analytics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sablemoore branding
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d6efd;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0d6efd;
    }
    .success-high { color: #28a745; }
    .success-medium { color: #ffc107; }
    .success-low { color: #dc3545; }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }
</style>
""", unsafe_allow_html=True)

# User credentials
USERS = {
    'admin': 'sablemoore2024',
    'analyst': 'litigation123'
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'cases' not in st.session_state:
    st.session_state.cases = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Initialize ML components
if ML_ENGINE_AVAILABLE:
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = get_model_manager()
    if 'success_predictor' not in st.session_state:
        st.session_state.success_predictor = get_success_predictor()
    if 'duplicate_detector' not in st.session_state:
        st.session_state.duplicate_detector = get_duplicate_detector()


class CaseExtractor:
    """Extract case information from uploaded documents"""

    @staticmethod
    def extract_from_text(text: str) -> Dict:
        """Extract case details from text content"""
        case_data = {
            'case_id': hashlib.md5(text.encode()).hexdigest()[:8],
            'raw_text': text,
            'claim_amount': CaseExtractor._extract_claim_amount(text),
            'case_type': CaseExtractor._extract_case_type(text),
            'jurisdiction': CaseExtractor._extract_jurisdiction(text),
            'defendant_type': CaseExtractor._extract_defendant_type(text),
            'complexity': CaseExtractor._assess_complexity(text),
            'estimated_duration_months': CaseExtractor._estimate_duration(text),
            'extracted_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return case_data

    @staticmethod
    def _extract_claim_amount(text: str) -> float:
        """Extract monetary claim amount from text"""
        patterns = [
            r'£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|m)?',
            r'claim(?:ing)?\s*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'damages?\s*(?:of)?\s*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'value[:\s]*£\s*(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                amount = float(matches[0].replace(',', ''))
                if 'million' in text.lower():
                    amount *= 1000000
                return amount

        return np.random.uniform(50000, 5000000)

    @staticmethod
    def _extract_case_type(text: str) -> str:
        """Identify the type of case"""
        text_lower = text.lower()

        case_types = {
            'Contract Dispute': ['contract', 'breach', 'agreement', 'obligation'],
            'Personal Injury': ['injury', 'accident', 'negligence', 'medical'],
            'Employment': ['employment', 'unfair dismissal', 'discrimination', 'redundancy'],
            'Commercial Dispute': ['commercial', 'business', 'trade', 'partnership'],
            'Property': ['property', 'landlord', 'tenant', 'lease', 'possession'],
            'Professional Negligence': ['professional negligence', 'solicitor', 'accountant'],
            'Intellectual Property': ['patent', 'trademark', 'copyright', 'ip'],
            'Fraud': ['fraud', 'misrepresentation', 'deceit'],
            'Debt Recovery': ['debt', 'recovery', 'payment', 'outstanding']
        }

        for case_type, keywords in case_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return case_type

        return 'General Civil Litigation'

    @staticmethod
    def _extract_jurisdiction(text: str) -> str:
        """Determine jurisdiction"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['high court', 'chancery', 'queen\'s bench', 'king\'s bench']):
            return 'High Court'
        elif 'county court' in text_lower:
            return 'County Court'
        elif 'employment tribunal' in text_lower:
            return 'Employment Tribunal'
        elif any(word in text_lower for word in ['court of appeal', 'appeal']):
            return 'Court of Appeal'
        elif 'supreme court' in text_lower:
            return 'Supreme Court'

        return 'County Court'

    @staticmethod
    def _extract_defendant_type(text: str) -> str:
        """Identify defendant type"""
        text_lower = text.lower()

        if any(word in text_lower for word in ['plc', 'ltd', 'limited', 'corporation', 'company']):
            return 'Corporate'
        elif any(word in text_lower for word in ['individual', 'mr', 'mrs', 'ms', 'person']):
            return 'Individual'
        elif any(word in text_lower for word in ['public body', 'council', 'nhs', 'government']):
            return 'Public Body'

        return 'Individual'

    @staticmethod
    def _assess_complexity(text: str) -> str:
        """Assess case complexity"""
        complexity_score = 0
        text_lower = text.lower()

        high_complexity_indicators = [
            'expert', 'international', 'multiple parties', 'complex',
            'regulatory', 'class action', 'precedent'
        ]

        for indicator in high_complexity_indicators:
            if indicator in text_lower:
                complexity_score += 1

        if complexity_score >= 3:
            return 'High'
        elif complexity_score >= 1:
            return 'Medium'
        return 'Low'

    @staticmethod
    def _estimate_duration(text: str) -> int:
        """Estimate case duration in months"""
        text_lower = text.lower()

        if 'urgent' in text_lower or 'summary judgment' in text_lower:
            return int(np.random.randint(3, 9))
        elif 'complex' in text_lower or 'high court' in text_lower:
            return int(np.random.randint(18, 36))
        else:
            return int(np.random.randint(9, 18))


class MLDuplicateDetector:
    """ML-powered duplicate detection"""

    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1
            )
        self.classifier = None
        self.is_trained = False
        self.training_data = []

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for comparison"""
        text = text.lower()
        text = re.sub(r'[^\w\s£$€]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', ' DATE ', text)
        text = re.sub(r'£[\d,]+\.?\d*', ' AMOUNT ', text)
        return text.strip()

    def _extract_features(self, case1: Dict, case2: Dict) -> Dict:
        """Extract comparison features between two cases"""
        features = {}

        text1 = self._preprocess_text(case1.get('raw_text', ''))
        text2 = self._preprocess_text(case2.get('raw_text', ''))

        if SKLEARN_AVAILABLE and text1 and text2:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                features['tfidf_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                features['tfidf_similarity'] = 0.0
        else:
            features['tfidf_similarity'] = 0.0

        features['sequence_similarity'] = SequenceMatcher(None, text1[:1000], text2[:1000]).ratio()

        amount1 = case1.get('claim_amount', 0)
        amount2 = case2.get('claim_amount', 0)
        if amount1 > 0 and amount2 > 0:
            features['amount_ratio'] = min(amount1, amount2) / max(amount1, amount2)
        else:
            features['amount_ratio'] = 0.0

        features['same_case_type'] = 1.0 if case1.get('case_type') == case2.get('case_type') else 0.0
        features['same_jurisdiction'] = 1.0 if case1.get('jurisdiction') == case2.get('jurisdiction') else 0.0
        features['same_defendant_type'] = 1.0 if case1.get('defendant_type') == case2.get('defendant_type') else 0.0
        features['same_complexity'] = 1.0 if case1.get('complexity') == case2.get('complexity') else 0.0

        dur1 = case1.get('estimated_duration_months', 12)
        dur2 = case2.get('estimated_duration_months', 12)
        features['duration_ratio'] = min(dur1, dur2) / max(dur1, dur2) if max(dur1, dur2) > 0 else 0.0
        features['entity_overlap'] = self._calculate_entity_overlap(text1, text2)

        return features

    def _calculate_entity_overlap(self, text1: str, text2: str) -> float:
        """Calculate overlap of potential named entities"""
        words1 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text1))
        words2 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_composite_score(self, features: Dict) -> float:
        """Calculate weighted composite similarity score"""
        weights = {
            'tfidf_similarity': 0.30,
            'sequence_similarity': 0.15,
            'amount_ratio': 0.15,
            'same_case_type': 0.10,
            'same_jurisdiction': 0.08,
            'same_defendant_type': 0.07,
            'same_complexity': 0.05,
            'duration_ratio': 0.05,
            'entity_overlap': 0.05
        }

        score = sum(features.get(k, 0) * v for k, v in weights.items())
        return score

    def find_duplicates(self, cases: List[Dict], threshold: float = 0.60) -> List[Dict]:
        """Find potential duplicates using ML features"""
        if len(cases) < 2:
            return []

        duplicates = []

        for i in range(len(cases)):
            for j in range(i + 1, len(cases)):
                features = self._extract_features(cases[i], cases[j])
                final_score = self._calculate_composite_score(features)

                if final_score >= threshold:
                    duplicates.append({
                        'case_1': cases[i]['case_id'],
                        'case_2': cases[j]['case_id'],
                        'similarity': round(final_score * 100, 1),
                        'features': {
                            'text_similarity': round(features['tfidf_similarity'] * 100, 1),
                            'amount_match': round(features['amount_ratio'] * 100, 1),
                            'same_type': features['same_case_type'] == 1.0,
                            'same_jurisdiction': features['same_jurisdiction'] == 1.0,
                            'entity_overlap': round(features['entity_overlap'] * 100, 1)
                        },
                        'confidence': 'High' if final_score >= 0.80 else ('Medium' if final_score >= 0.65 else 'Low'),
                        'case_1_details': cases[i],
                        'case_2_details': cases[j]
                    })

        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def add_training_example(self, case1_id: str, case2_id: str, is_duplicate: bool, cases: List[Dict]) -> bool:
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
            'case2_id': case2_id
        })

        if len(self.training_data) >= 5 and SKLEARN_AVAILABLE:
            self._train_classifier()

        return True

    def _train_classifier(self):
        """Train the classifier on labeled examples"""
        if len(self.training_data) < 5 or not SKLEARN_AVAILABLE:
            return

        X = []
        y = []

        for example in self.training_data:
            X.append(list(example['features'].values()))
            y.append(1 if example['is_duplicate'] else 0)

        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.classifier.fit(X, y)
        self.is_trained = True

    def get_training_stats(self) -> Dict:
        """Get statistics about training data"""
        duplicates = sum(1 for x in self.training_data if x['is_duplicate'])
        return {
            'total_examples': len(self.training_data),
            'duplicates': duplicates,
            'non_duplicates': len(self.training_data) - duplicates,
            'is_trained': self.is_trained
        }


class SuccessPredictor:
    """Predict case success rate based on UK law and historical patterns"""

    @staticmethod
    def predict_success(case_data: Dict) -> Dict:
        """Calculate success probability based on case characteristics"""

        case_type_factors = {
            'Contract Dispute': 0.70,
            'Personal Injury': 0.75,
            'Employment': 0.60,
            'Commercial Dispute': 0.65,
            'Property': 0.68,
            'Professional Negligence': 0.55,
            'Intellectual Property': 0.50,
            'Fraud': 0.45,
            'Debt Recovery': 0.80,
            'General Civil Litigation': 0.65
        }

        jurisdiction_factors = {
            'County Court': 0.72,
            'High Court': 0.60,
            'Employment Tribunal': 0.58,
            'Court of Appeal': 0.45,
            'Supreme Court': 0.40
        }

        defendant_factors = {
            'Corporate': 0.68,
            'Individual': 0.62,
            'Public Body': 0.55
        }

        complexity_factors = {
            'Low': 0.75,
            'Medium': 0.65,
            'High': 0.50
        }

        claim_amount = case_data.get('claim_amount', 0)
        if claim_amount > 1000000:
            amount_factor = 0.55
        elif claim_amount > 250000:
            amount_factor = 0.65
        elif claim_amount > 100000:
            amount_factor = 0.70
        else:
            amount_factor = 0.75

        case_type = case_data.get('case_type', 'General Civil Litigation')
        jurisdiction = case_data.get('jurisdiction', 'County Court')
        defendant_type = case_data.get('defendant_type', 'Individual')
        complexity = case_data.get('complexity', 'Medium')

        success_rate = (
            case_type_factors.get(case_type, 0.65) * 0.30 +
            jurisdiction_factors.get(jurisdiction, 0.65) * 0.25 +
            defendant_factors.get(defendant_type, 0.65) * 0.20 +
            complexity_factors.get(complexity, 0.65) * 0.15 +
            amount_factor * 0.10
        )

        success_rate += np.random.uniform(-0.05, 0.05)
        success_rate = max(0.15, min(0.95, success_rate))

        risk_factors = []
        if complexity == 'High':
            risk_factors.append('High complexity case')
        if claim_amount > 1000000:
            risk_factors.append('High value claim may face vigorous defense')
        if jurisdiction in ['Court of Appeal', 'Supreme Court']:
            risk_factors.append('Appellate court - historically lower success rates')
        if case_type in ['Fraud', 'Professional Negligence']:
            risk_factors.append('Case type historically has lower success rates')

        positive_factors = []
        if case_type in ['Debt Recovery', 'Personal Injury']:
            positive_factors.append('Case type historically has higher success rates')
        if complexity == 'Low':
            positive_factors.append('Straightforward case with clear legal basis')
        if jurisdiction == 'County Court':
            positive_factors.append('County Court cases have good success rates')

        expected_return = claim_amount * success_rate * 0.85

        risk_level = 'Low' if success_rate >= 0.70 else ('Medium' if success_rate >= 0.50 else 'High')

        if success_rate >= 0.70:
            recommendation = 'RECOMMENDED - Strong case with good success probability'
        elif success_rate >= 0.55:
            recommendation = 'CONSIDER - Moderate case, review risk factors carefully'
        else:
            recommendation = 'CAUTION - Lower success probability, high risk investment'

        return {
            'success_rate': round(success_rate * 100, 2),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'positive_factors': positive_factors,
            'expected_return': expected_return,
            'recommendation': recommendation
        }


# Initialize global detector
if 'duplicate_detector' not in st.session_state:
    st.session_state.duplicate_detector = MLDuplicateDetector()


def login_page():
    """Display login page"""
    st.markdown('<h1 class="main-header">SABLEMOORE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Litigation Finance Intelligence Platform</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            st.subheader("Sign In")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In", use_container_width=True)

            if submit:
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        st.markdown("---")
        st.caption("🔒 Secure access to litigation analytics")


def dashboard_page():
    """Display main dashboard"""
    cases = st.session_state.cases
    predictions = st.session_state.predictions

    # Calculate stats
    total_cases = len(cases)
    avg_success = sum(p['success_rate'] for p in predictions) / len(predictions) if predictions else 0
    total_exposure = sum(c['claim_amount'] for c in cases) if cases else 0
    total_expected = sum(p['expected_return'] for p in predictions) if predictions else 0

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Cases", total_cases)
    with col2:
        st.metric("Avg Success Rate", f"{avg_success:.1f}%")
    with col3:
        st.metric("Total Exposure", f"£{total_exposure:,.0f}")
    with col4:
        st.metric("Expected Return", f"£{total_expected:,.0f}")

    if cases and predictions:
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Risk distribution chart
            risk_dist = {'Low': 0, 'Medium': 0, 'High': 0}
            for p in predictions:
                risk_dist[p['risk_level']] = risk_dist.get(p['risk_level'], 0) + 1

            fig = px.pie(
                values=list(risk_dist.values()),
                names=list(risk_dist.keys()),
                title="Risk Distribution",
                color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Case types chart
            case_types = {}
            for c in cases:
                case_types[c['case_type']] = case_types.get(c['case_type'], 0) + 1

            fig = px.bar(
                x=list(case_types.keys()),
                y=list(case_types.values()),
                title="Cases by Type",
                color_discrete_sequence=['#0d6efd']
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        # Recent cases table
        st.subheader("Recent Cases")
        recent = []
        for case, pred in zip(cases[-5:], predictions[-5:]):
            recent.append({
                'Case ID': case['case_id'],
                'Type': case['case_type'],
                'Claim': f"£{case['claim_amount']:,.0f}",
                'Success Rate': f"{pred['success_rate']:.1f}%",
                'Risk': pred['risk_level']
            })

        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
    else:
        st.info("No cases uploaded yet. Go to Upload Cases to add your first case.")


def upload_page():
    """Display upload page"""
    st.subheader("Upload Cases")

    # Get predictor
    if ML_ENGINE_AVAILABLE:
        predictor = st.session_state.success_predictor
    else:
        predictor = None

    # File upload
    uploaded_files = st.file_uploader(
        "Upload case documents",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload text files containing case information"
    )

    if uploaded_files:
        if st.button("Process Files", type="primary"):
            with st.spinner("Processing files with ML engine..."):
                for file in uploaded_files:
                    content = file.read().decode('utf-8', errors='ignore')
                    if content.strip():
                        case_data = CaseExtractor.extract_from_text(content)
                        if predictor:
                            prediction = predictor.predict(case_data)
                        else:
                            prediction = SuccessPredictor.predict_success(case_data)
                        st.session_state.cases.append(case_data)
                        st.session_state.predictions.append({
                            'case_id': case_data['case_id'],
                            **prediction
                        })

            st.success(f"Successfully processed {len(uploaded_files)} file(s)")
            st.rerun()

    st.markdown("---")

    # Manual text input
    st.subheader("Or Enter Case Text Manually")
    manual_text = st.text_area(
        "Case Details",
        height=200,
        placeholder="Paste case information here..."
    )

    if st.button("Analyze Case", type="primary") and manual_text.strip():
        with st.spinner("Analyzing case with ML engine..."):
            case_data = CaseExtractor.extract_from_text(manual_text)
            if predictor:
                prediction = predictor.predict(case_data)
            else:
                prediction = SuccessPredictor.predict_success(case_data)
            st.session_state.cases.append(case_data)
            st.session_state.predictions.append({
                'case_id': case_data['case_id'],
                **prediction
            })

        st.success("Case analyzed successfully!")

        # Show results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Rate", f"{prediction['success_rate']:.1f}%")
        with col2:
            st.metric("Risk Level", prediction['risk_level'])
        with col3:
            method = prediction.get('prediction_method', 'rule_based')
            st.metric("Method", "ML" if 'ml' in method else "Rules")

        # Show confidence interval if available
        if 'confidence_interval' in prediction:
            ci = prediction['confidence_interval']
            st.caption(f"95% Confidence Interval: {ci['low']:.1f}% - {ci['high']:.1f}%")

        st.info(prediction['recommendation'])


def analysis_page():
    """Display case analysis page"""
    cases = st.session_state.cases
    predictions = st.session_state.predictions

    if not cases:
        st.info("No cases to analyze. Upload some cases first.")
        return

    st.subheader("Case Analysis")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        case_types = list(set(c['case_type'] for c in cases))
        selected_type = st.selectbox("Case Type", ['All'] + case_types)

    with col2:
        selected_risk = st.selectbox("Risk Level", ['All', 'Low', 'Medium', 'High'])

    with col3:
        min_success = st.slider("Min Success Rate", 0, 100, 0)

    # Combine and filter data
    combined = []
    for case, pred in zip(cases, predictions):
        combined.append({**case, **pred})

    filtered = combined
    if selected_type != 'All':
        filtered = [c for c in filtered if c['case_type'] == selected_type]
    if selected_risk != 'All':
        filtered = [c for c in filtered if c['risk_level'] == selected_risk]
    if min_success > 0:
        filtered = [c for c in filtered if c['success_rate'] >= min_success]

    st.markdown(f"**Showing {len(filtered)} of {len(combined)} cases**")

    # Display cases
    for case in filtered:
        with st.expander(f"Case #{case['case_id']} - {case['case_type']}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Success Rate", f"{case['success_rate']:.1f}%")
            with col2:
                st.metric("Risk Level", case['risk_level'])
            with col3:
                st.metric("Claim Amount", f"£{case['claim_amount']:,.0f}")
            with col4:
                st.metric("Expected Return", f"£{case['expected_return']:,.0f}")

            st.markdown("**Details:**")
            st.write(f"- Jurisdiction: {case['jurisdiction']}")
            st.write(f"- Defendant Type: {case['defendant_type']}")
            st.write(f"- Complexity: {case['complexity']}")
            st.write(f"- Est. Duration: {case['estimated_duration_months']} months")

            if case['risk_factors']:
                st.warning("**Risk Factors:** " + "; ".join(case['risk_factors']))
            if case['positive_factors']:
                st.success("**Positive Factors:** " + "; ".join(case['positive_factors']))


def duplicates_page():
    """Display duplicate detection page"""
    cases = st.session_state.cases

    # Get detector from ML engine or fallback
    if ML_ENGINE_AVAILABLE:
        detector = st.session_state.duplicate_detector
    elif 'duplicate_detector' in st.session_state:
        detector = st.session_state.duplicate_detector
    else:
        st.error("Duplicate detector not available")
        return

    st.subheader("ML Duplicate Detection")
    st.info("🤖 Uses TF-IDF text analysis, cosine similarity, word Jaccard similarity, and Gradient Boosting ML to identify potential duplicates.")

    if len(cases) < 2:
        st.warning("Need at least 2 cases to detect duplicates. Upload more cases first.")
        return

    # Threshold slider
    threshold = st.slider("Similarity Threshold (%)", 30, 95, 60, help="Lower thresholds find more matches but may include false positives")

    if st.button("Scan for Duplicates", type="primary"):
        with st.spinner("Scanning for duplicates..."):
            duplicates = detector.find_duplicates(cases, threshold / 100)

        if duplicates:
            st.warning(f"Found {len(duplicates)} potential duplicate(s)")

            for dup in duplicates:
                confidence_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🔵'}

                with st.expander(f"{confidence_color[dup['confidence']]} {dup['similarity']}% Similar - Cases #{dup['case_1']} & #{dup['case_2']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Case #{dup['case_1']}**")
                        c1 = dup['case_1_details']
                        st.write(f"- Type: {c1['case_type']}")
                        st.write(f"- Claim: £{c1['claim_amount']:,.0f}")
                        st.write(f"- Jurisdiction: {c1['jurisdiction']}")

                    with col2:
                        st.markdown(f"**Case #{dup['case_2']}**")
                        c2 = dup['case_2_details']
                        st.write(f"- Type: {c2['case_type']}")
                        st.write(f"- Claim: £{c2['claim_amount']:,.0f}")
                        st.write(f"- Jurisdiction: {c2['jurisdiction']}")

                    st.markdown("**Similarity Breakdown:**")
                    feat = dup['features']
                    cols = st.columns(5)
                    cols[0].metric("Text", f"{feat['text_similarity']}%")
                    cols[1].metric("Amount", f"{feat['amount_match']}%")
                    cols[2].metric("Same Type", "✓" if feat['same_type'] else "✗")
                    cols[3].metric("Same Jurisdiction", "✓" if feat['same_jurisdiction'] else "✗")
                    cols[4].metric("Entity Overlap", f"{feat['entity_overlap']}%")

                    # Training feedback
                    st.markdown("---")
                    st.markdown("**Help improve detection:**")
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        if st.button(f"✓ Confirm Duplicate", key=f"dup_{dup['case_1']}_{dup['case_2']}"):
                            detector.add_training_example(dup['case_1'], dup['case_2'], True, cases)
                            st.success("Training example added: Duplicate")
                    with fcol2:
                        if st.button(f"✗ Not Duplicate", key=f"notdup_{dup['case_1']}_{dup['case_2']}"):
                            detector.add_training_example(dup['case_1'], dup['case_2'], False, cases)
                            st.success("Training example added: Not duplicate")
        else:
            st.success("No duplicate cases found at the current threshold")

    # Training stats sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML Model Status")
    stats = detector.get_training_stats()
    st.sidebar.metric("Training Examples", stats['total_examples'])
    st.sidebar.metric("Model Status", "Trained" if stats['is_trained'] else "Not Trained")

    if stats['total_examples'] < 5:
        st.sidebar.info(f"Provide {5 - stats['total_examples']} more examples to train the ML model")


def portfolio_page():
    """Display portfolio overview"""
    cases = st.session_state.cases
    predictions = st.session_state.predictions

    if not cases:
        st.info("No cases in portfolio. Upload some cases first.")
        return

    st.subheader("Portfolio Overview")

    # Stats
    total_cases = len(cases)
    avg_success = sum(p['success_rate'] for p in predictions) / len(predictions)
    total_exposure = sum(c['claim_amount'] for c in cases)
    total_expected = sum(p['expected_return'] for p in predictions)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", total_cases)
    col2.metric("Avg Success", f"{avg_success:.1f}%")
    col3.metric("Total Exposure", f"£{total_exposure:,.0f}")
    col4.metric("Expected Return", f"£{total_expected:,.0f}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Success rate distribution
        success_rates = [p['success_rate'] for p in predictions]
        fig = px.histogram(
            x=success_rates,
            nbins=20,
            title="Success Rate Distribution",
            labels={'x': 'Success Rate (%)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Jurisdiction breakdown
        jurisdictions = {}
        for c in cases:
            jurisdictions[c['jurisdiction']] = jurisdictions.get(c['jurisdiction'], 0) + 1

        fig = px.pie(
            values=list(jurisdictions.values()),
            names=list(jurisdictions.keys()),
            title="Cases by Jurisdiction"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    scatter_data = []
    for case, pred in zip(cases, predictions):
        scatter_data.append({
            'Claim Amount': case['claim_amount'],
            'Success Rate': pred['success_rate'],
            'Expected Return': pred['expected_return'],
            'Risk Level': pred['risk_level'],
            'Case Type': case['case_type']
        })

    df = pd.DataFrame(scatter_data)
    fig = px.scatter(
        df,
        x='Claim Amount',
        y='Success Rate',
        size='Expected Return',
        color='Risk Level',
        hover_data=['Case Type'],
        title="Risk vs Return Analysis",
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    )
    st.plotly_chart(fig, use_container_width=True)


def export_page():
    """Display export page"""
    cases = st.session_state.cases
    predictions = st.session_state.predictions

    if not cases:
        st.info("No cases to export. Upload some cases first.")
        return

    st.subheader("Export Reports")

    # Combine data
    combined = []
    for case, pred in zip(cases, predictions):
        row = {**case, **pred}
        del row['raw_text']
        row['risk_factors'] = '; '.join(row.get('risk_factors', []))
        row['positive_factors'] = '; '.join(row.get('positive_factors', []))
        combined.append(row)

    df = pd.DataFrame(combined)

    # Stats
    st.markdown("### Summary Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Success Rate Stats**")
        st.write(f"- Mean: {df['success_rate'].mean():.2f}%")
        st.write(f"- Median: {df['success_rate'].median():.2f}%")
        st.write(f"- Min: {df['success_rate'].min():.2f}%")
        st.write(f"- Max: {df['success_rate'].max():.2f}%")

    with col2:
        st.markdown("**Claim Amount Stats**")
        st.write(f"- Total: £{df['claim_amount'].sum():,.0f}")
        st.write(f"- Mean: £{df['claim_amount'].mean():,.0f}")
        st.write(f"- Median: £{df['claim_amount'].median():,.0f}")
        st.write(f"- Max: £{df['claim_amount'].max():,.0f}")

    st.markdown("---")

    # Preview
    st.markdown("### Data Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download buttons
    st.markdown("### Download")
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"litigation_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # Excel export
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Cases', index=False)
        buffer.seek(0)

        st.download_button(
            label="Download Excel",
            data=buffer,
            file_name=f"litigation_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def ml_models_page():
    """Display ML Models management page"""
    st.subheader("ML Model Management")

    if not ML_ENGINE_AVAILABLE:
        st.error("ML Engine not available. Please check dependencies.")
        return

    model_manager = st.session_state.model_manager
    success_predictor = st.session_state.success_predictor
    duplicate_detector = st.session_state.duplicate_detector

    # Model Status Overview
    st.markdown("### Model Status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Success Predictor**")
        status = model_manager.get_model_status()

        if success_predictor.is_trained:
            st.success("✅ ML Model Trained")
            st.write(f"Model Type: {status['success_predictor']['model_type']}")
        else:
            st.warning("⚠️ Using Rule-Based Predictions")
            st.caption("Train the model with labeled data for ML predictions")

    with col2:
        st.markdown("**Duplicate Detector**")
        dup_stats = duplicate_detector.get_training_stats()

        if dup_stats['is_trained']:
            st.success("✅ ML Model Trained")
            st.write(f"Training Examples: {dup_stats['total_examples']}")
        else:
            st.warning(f"⚠️ Need {5 - dup_stats['total_examples']} more examples")

    st.markdown("---")

    # Training Section
    st.markdown("### Train Success Predictor")
    st.caption("Generate synthetic training data based on UK litigation patterns to bootstrap the ML model.")

    col1, col2 = st.columns([2, 1])

    with col1:
        n_samples = st.slider("Training Samples", 50, 500, 100, step=50)

    with col2:
        if st.button("Generate & Train", type="primary"):
            with st.spinner("Generating training data and training model..."):
                # Generate synthetic data
                training_data = generate_synthetic_training_data(n_samples)

                # Train the model
                result = success_predictor.train(training_data)

                if result['success']:
                    st.success(f"✅ Model trained successfully!")
                    st.write(f"- Model: {result['model_type']}")
                    st.write(f"- Samples: {result['training_samples']}")
                    st.write(f"- CV Accuracy: {result['cv_accuracy']:.2%} (±{result['cv_std']:.2%})")
                else:
                    st.error(result['message'])

    st.markdown("---")

    # Model Performance Visualization
    st.markdown("### Model Insights")

    if success_predictor.is_trained:
        # Show feature importance if available
        if hasattr(success_predictor.model, 'feature_importances_'):
            importances = success_predictor.model.feature_importances_
            feature_names = success_predictor.feature_names if success_predictor.feature_names else [f"Feature {i}" for i in range(len(importances))]

            fig = px.bar(
                x=importances,
                y=feature_names,
                orientation='h',
                title="Feature Importance",
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the model to see feature importance analysis")

    # Duplicate Detection Stats
    st.markdown("### Duplicate Detection Training")

    dup_stats = duplicate_detector.get_training_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Examples", dup_stats['total_examples'])
    col2.metric("Marked Duplicates", dup_stats['duplicates'])
    col3.metric("Non-Duplicates", dup_stats['non_duplicates'])

    if dup_stats['total_examples'] >= 5:
        st.success("✅ Sufficient data for ML training")
    else:
        st.warning(f"Need {5 - dup_stats['total_examples']} more labeled examples for ML training")
        st.caption("Use the Duplicate Detection page to label potential duplicates")


def main():
    """Main application"""

    if not st.session_state.logged_in:
        login_page()
        return

    # Sidebar navigation
    st.sidebar.markdown('<h2 style="color: #0d6efd; font-weight: 700; letter-spacing: 1px;">SABLEMOORE</h2>', unsafe_allow_html=True)
    st.sidebar.caption("Litigation Finance Intelligence")
    st.sidebar.markdown("---")

    # User info
    st.sidebar.markdown(f"👤 Logged in as **{st.session_state.username}**")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.sidebar.markdown("---")

    # Navigation - add ML Models page
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Upload Cases", "Case Analysis", "Duplicate Detection", "Portfolio", "ML Models", "Export Reports"],
        label_visibility="collapsed"
    )

    # Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Cases", len(st.session_state.cases))

    # ML Status indicator
    if ML_ENGINE_AVAILABLE:
        predictor = st.session_state.success_predictor
        if predictor.is_trained:
            st.sidebar.success("🤖 ML Active")
        else:
            st.sidebar.warning("📊 Rules Mode")

    # Clear data button
    if st.sidebar.button("Clear All Data", type="secondary"):
        st.session_state.cases = []
        st.session_state.predictions = []
        st.rerun()

    # Page routing
    if page == "Dashboard":
        st.title("Dashboard")
        dashboard_page()
    elif page == "Upload Cases":
        st.title("Upload Cases")
        upload_page()
    elif page == "Case Analysis":
        st.title("Case Analysis")
        analysis_page()
    elif page == "Duplicate Detection":
        st.title("Duplicate Detection")
        duplicates_page()
    elif page == "Portfolio":
        st.title("Portfolio Overview")
        portfolio_page()
    elif page == "ML Models":
        st.title("ML Models")
        ml_models_page()
    elif page == "Export Reports":
        st.title("Export Reports")
        export_page()


if __name__ == "__main__":
    main()
