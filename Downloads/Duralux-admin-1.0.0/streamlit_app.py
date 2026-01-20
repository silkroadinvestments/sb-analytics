"""
Sablemoore Analytics - Litigation Finance Intelligence Platform
Streamlit Application with AI Agent and Enhanced ML Engine

This app can be deployed to Streamlit Cloud for hosting.
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

# Import AI Assistant
try:
    from ai_assistant import get_ai_assistant, AIAssistant
    AI_ASSISTANT_AVAILABLE = True
except ImportError:
    AI_ASSISTANT_AVAILABLE = False

# Import AI Agent
try:
    from ai_agent import get_ai_agent, get_intelligent_duplicate_detector, LitigationAIAgent
    AI_AGENT_AVAILABLE = True
except ImportError:
    AI_AGENT_AVAILABLE = False

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
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .agent-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .confidence-definite { color: #d32f2f; font-weight: bold; }
    .confidence-likely { color: #f57c00; font-weight: bold; }
    .confidence-possible { color: #1976d2; }
    .confidence-unlikely { color: #757575; }
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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize ML components
if ML_ENGINE_AVAILABLE:
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = get_model_manager()
    if 'success_predictor' not in st.session_state:
        st.session_state.success_predictor = get_success_predictor()
    if 'duplicate_detector' not in st.session_state:
        st.session_state.duplicate_detector = get_duplicate_detector()

# Initialize AI Assistant
if AI_ASSISTANT_AVAILABLE:
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = get_ai_assistant()

# Initialize AI Agent
if AI_AGENT_AVAILABLE:
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = get_ai_agent()
    if 'intelligent_detector' not in st.session_state:
        st.session_state.intelligent_detector = get_intelligent_duplicate_detector()


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
        st.caption("Secure access to litigation analytics")


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


def ai_agent_page():
    """AI Agent chat interface"""
    st.subheader("AI Agent")

    if not AI_AGENT_AVAILABLE:
        st.error("AI Agent module not available. Please check installation.")
        return

    agent = st.session_state.ai_agent
    cases = st.session_state.cases
    predictions = st.session_state.predictions

    # Agent status
    status = agent.get_status()
    col1, col2, col3 = st.columns(3)
    with col1:
        if status['llm_available']:
            st.success(f"Provider: {status['provider'].title()}")
        else:
            st.info("Provider: Rule-based")
    with col2:
        st.info(f"Model: {status['model'] or 'N/A'}")
    with col3:
        st.metric("Cases Loaded", len(cases))

    # API Key configuration
    with st.expander("Configure AI Provider (Optional)"):
        st.caption("Enter an API key to enable LLM-powered analysis.")
        col1, col2 = st.columns(2)
        with col1:
            openai_key = st.text_input("OpenAI API Key", type="password", key="agent_openai")
        with col2:
            anthropic_key = st.text_input("Anthropic API Key", type="password", key="agent_anthropic")

        if st.button("Update API Keys", key="update_agent_keys"):
            if anthropic_key:
                os.environ['ANTHROPIC_API_KEY'] = anthropic_key
                st.session_state.ai_agent = get_ai_agent()
                st.success("Anthropic API key configured!")
                st.rerun()
            elif openai_key:
                os.environ['OPENAI_API_KEY'] = openai_key
                st.session_state.ai_agent = get_ai_agent()
                st.success("OpenAI API key configured!")
                st.rerun()

    st.markdown("---")

    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Scan for Duplicates", use_container_width=True):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'Scan my portfolio for duplicate cases'
            })
            st.rerun()

    with col2:
        if st.button("Portfolio Summary", use_container_width=True):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'Give me a portfolio overview'
            })
            st.rerun()

    with col3:
        if st.button("Risk Analysis", use_container_width=True):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': 'Analyze the risk in my portfolio'
            })
            st.rerun()

    with col4:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            agent.clear_history()
            st.rerun()

    st.markdown("---")

    # Chat interface
    st.markdown("### Chat with AI Agent")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>AI Agent:</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)

    # Process pending messages
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
        with st.spinner("AI Agent is thinking..."):
            last_message = st.session_state.chat_history[-1]['content']
            response = agent.chat(last_message, cases, predictions)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['response']
            })
            st.rerun()

    # Chat input
    user_input = st.text_input(
        "Ask the AI Agent...",
        placeholder="e.g., 'Find duplicates', 'Analyze case #abc123', 'Give me portfolio insights'",
        key="agent_input"
    )

    if st.button("Send", type="primary") and user_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        st.rerun()


def ai_duplicates_page():
    """AI-powered intelligent duplicate detection"""
    st.subheader("AI Duplicate Scanner")

    if not AI_AGENT_AVAILABLE:
        st.warning("AI Agent not available. Using ML-based detection instead.")
        duplicates_page()
        return

    cases = st.session_state.cases
    detector = st.session_state.intelligent_detector

    st.info("""
    **Intelligent Duplicate Detection** uses AI to understand case context, not just match keywords.
    It can detect:
    - Same parties with different name spellings (Smith/Smyth, Ltd/Limited)
    - Same incident described differently
    - Related cases (appeals, counterclaims)
    - Cases sharing key facts
    """)

    if len(cases) < 2:
        st.warning("Need at least 2 cases to detect duplicates. Upload more cases first.")
        return

    # Detection settings
    st.markdown("### Detection Settings")
    col1, col2 = st.columns(2)

    with col1:
        confidence_level = st.select_slider(
            "Minimum Confidence",
            options=['unlikely', 'possible', 'likely', 'definite'],
            value='possible',
            help="Higher confidence = fewer but more certain matches"
        )

    with col2:
        st.markdown("**Confidence Levels:**")
        st.markdown("""
        - **Definite**: Almost certainly the same case
        - **Likely**: Probably duplicates, review recommended
        - **Possible**: May be related, needs investigation
        - **Unlikely**: Probably different cases
        """)

    if st.button("Scan for Duplicates", type="primary"):
        with st.spinner("AI is analyzing cases for duplicates..."):
            duplicates = detector.find_duplicates(cases, min_confidence=confidence_level)

        if duplicates:
            st.warning(f"Found **{len(duplicates)}** potential duplicate(s)")

            for i, dup in enumerate(duplicates):
                confidence = dup.get('confidence', 'possible')
                confidence_class = f"confidence-{confidence}"

                with st.expander(
                    f"Match {i+1}: Cases #{dup['case_1']} & #{dup['case_2']} - "
                    f"{confidence.upper()} ({dup['similarity']:.1f}%)"
                ):
                    # Case comparison
                    col1, col2 = st.columns(2)

                    c1 = dup.get('case_1_details', {})
                    c2 = dup.get('case_2_details', {})

                    with col1:
                        st.markdown(f"**Case #{dup['case_1']}**")
                        st.write(f"- Type: {c1.get('case_type', 'Unknown')}")
                        st.write(f"- Claim: £{c1.get('claim_amount', 0):,.0f}")
                        st.write(f"- Jurisdiction: {c1.get('jurisdiction', 'Unknown')}")

                    with col2:
                        st.markdown(f"**Case #{dup['case_2']}**")
                        st.write(f"- Type: {c2.get('case_type', 'Unknown')}")
                        st.write(f"- Claim: £{c2.get('claim_amount', 0):,.0f}")
                        st.write(f"- Jurisdiction: {c2.get('jurisdiction', 'Unknown')}")

                    # AI Analysis
                    st.markdown("---")
                    st.markdown("**AI Analysis:**")
                    st.markdown(f"<span class='{confidence_class}'>Confidence: {confidence.upper()}</span>",
                               unsafe_allow_html=True)

                    if dup.get('explanation'):
                        st.info(dup['explanation'])

                    if dup.get('recommendation'):
                        st.success(f"**Recommendation:** {dup['recommendation']}")

                    # Matching factors
                    if dup.get('matching_factors'):
                        st.markdown("**Matching Factors:**")
                        for factor in dup['matching_factors']:
                            if isinstance(factor, dict):
                                icon = "✓" if factor.get('match') else "✗"
                                st.write(f"{icon} {factor.get('factor', factor.get('details', ''))}")
                            else:
                                st.write(f"• {factor}")
        else:
            st.success("No potential duplicates found at the selected confidence level.")

    # Stats
    st.markdown("---")
    stats = detector.get_training_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Provider", stats.get('provider', 'rule_based').title())
    col2.metric("Model", stats.get('model', 'N/A') or 'Rule-based')
    col3.metric("AI Ready", "Yes" if stats.get('is_trained') else "No")


def duplicates_page():
    """Display ML duplicate detection page (fallback)"""
    cases = st.session_state.cases

    # Get detector from ML engine or fallback
    if ML_ENGINE_AVAILABLE:
        detector = st.session_state.duplicate_detector
    else:
        st.error("Duplicate detector not available")
        return

    st.subheader("ML Duplicate Detection")
    st.info("Using ML-based detection with fuzzy name matching.")

    if len(cases) < 2:
        st.warning("Need at least 2 cases to detect duplicates. Upload more cases first.")
        return

    # Detection settings
    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider("Similarity Threshold (%)", 50, 95, 75)

    with col2:
        filter_by_party = st.checkbox("Filter by Party Names", value=False)
        if filter_by_party:
            party_threshold = st.slider("Party Match Threshold (%)", 30, 80, 50)
        else:
            party_threshold = 50

    if st.button("Scan for Duplicates", type="primary"):
        with st.spinner("Scanning for duplicates..."):
            duplicates = detector.find_duplicates(
                cases,
                threshold / 100,
                filter_by_party=filter_by_party,
                party_threshold=party_threshold / 100
            )

        if duplicates:
            st.warning(f"Found {len(duplicates)} potential duplicate(s)")

            for dup in duplicates:
                confidence_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🔵'}

                with st.expander(f"{confidence_color.get(dup['confidence'], '🔵')} {dup['similarity']}% Similar - Cases #{dup['case_1']} & #{dup['case_2']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**Case #{dup['case_1']}**")
                        c1 = dup.get('case_1_details', {})
                        st.write(f"- Type: {c1.get('case_type', 'Unknown')}")
                        st.write(f"- Claim: £{c1.get('claim_amount', 0):,.0f}")

                    with col2:
                        st.markdown(f"**Case #{dup['case_2']}**")
                        c2 = dup.get('case_2_details', {})
                        st.write(f"- Type: {c2.get('case_type', 'Unknown')}")
                        st.write(f"- Claim: £{c2.get('claim_amount', 0):,.0f}")

                    st.markdown("**Similarity Breakdown:**")
                    feat = dup.get('features', {})
                    cols = st.columns(4)
                    cols[0].metric("Text", f"{feat.get('text_similarity', 0)}%")
                    cols[1].metric("Amount", f"{feat.get('amount_match', 0)}%")
                    cols[2].metric("Same Type", "Yes" if feat.get('same_type') else "No")
                    cols[3].metric("Entity Overlap", f"{feat.get('entity_overlap', 0)}%")
        else:
            st.success("No duplicate cases found at the current threshold")


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
        # JSON export
        json_data = json.dumps(combined, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"litigation_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


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
    st.sidebar.markdown(f"Logged in as **{st.session_state.username}**")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.sidebar.markdown("---")

    # Navigation
    pages = [
        "Dashboard",
        "Upload Cases",
        "Case Analysis",
        "AI Agent",
        "AI Duplicate Scanner",
        "Portfolio",
        "Export Reports"
    ]

    # Add ML Duplicates as fallback option
    if ML_ENGINE_AVAILABLE:
        pages.insert(5, "ML Duplicates")

    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    # Stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Cases", len(st.session_state.cases))

    # AI Status indicator
    if AI_AGENT_AVAILABLE:
        agent_status = st.session_state.ai_agent.get_status()
        if agent_status['llm_available']:
            st.sidebar.success(f"AI: {agent_status['provider'].title()}")
        else:
            st.sidebar.info("AI: Rule-based")

    # Clear data button
    if st.sidebar.button("Clear All Data", type="secondary"):
        st.session_state.cases = []
        st.session_state.predictions = []
        st.session_state.chat_history = []
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
    elif page == "AI Agent":
        st.title("AI Agent")
        ai_agent_page()
    elif page == "AI Duplicate Scanner":
        st.title("AI Duplicate Scanner")
        ai_duplicates_page()
    elif page == "ML Duplicates":
        st.title("ML Duplicate Detection")
        duplicates_page()
    elif page == "Portfolio":
        st.title("Portfolio Overview")
        portfolio_page()
    elif page == "Export Reports":
        st.title("Export Reports")
        export_page()


if __name__ == "__main__":
    main()
