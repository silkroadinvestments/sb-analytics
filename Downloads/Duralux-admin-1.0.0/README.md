# Sablemoore Analytics

**Litigation Finance Intelligence Platform**

A comprehensive litigation finance case prediction dashboard powered by AI and Machine Learning.

## Features

- **AI Agent** - Chat with an intelligent agent that can analyze cases, find duplicates, and provide portfolio insights
- **AI Duplicate Scanner** - Intelligent duplicate detection using LLM understanding (not just keyword matching)
- **ML-Powered Predictions** - Success rate predictions based on UK litigation patterns
- **Portfolio Management** - Track and analyze your entire case portfolio
- **Export Reports** - Download data in CSV or JSON format

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set **Main file path** to: `streamlit_app.py`
6. Click Deploy

### Configure AI Features (Optional)

To enable LLM-powered features:

1. In Streamlit Cloud, go to App Settings > Secrets
2. Add your API keys:

```toml
OPENAI_API_KEY = "sk-your-key-here"
# OR
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

## Login Credentials

- **Username:** `admin` | **Password:** `sablemoore2024`
- **Username:** `analyst` | **Password:** `litigation123`

## Project Structure

```
├── streamlit_app.py      # Main Streamlit application
├── ai_agent.py           # AI Agent with intelligent duplicate detection
├── ai_assistant.py       # AI Assistant for case analysis
├── ml_engine.py          # ML models for prediction and detection
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md
```

## Tech Stack

- **Frontend:** Streamlit
- **ML:** scikit-learn, XGBoost
- **AI:** OpenAI GPT-4 / Anthropic Claude (optional)
- **Visualization:** Plotly

## License

Proprietary - Sablemoore Analytics
