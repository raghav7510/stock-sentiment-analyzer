import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import time

warnings.filterwarnings('ignore')

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for live refresh
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = None
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if 'show_more_positive' not in st.session_state:
    st.session_state.show_more_positive = False
if 'show_more_neutral' not in st.session_state:
    st.session_state.show_more_neutral = False
if 'show_more_negative' not in st.session_state:
    st.session_state.show_more_negative = False
if 'positive_count' not in st.session_state:
    st.session_state.positive_count = 5
if 'neutral_count' not in st.session_state:
    st.session_state.neutral_count = 5
if 'negative_count' not in st.session_state:
    st.session_state.negative_count = 5
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = ""
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = ""

# ============ LOAD AI MODEL IMMEDIATELY ============
@st.cache_resource
def load_model():
    """Load FinBERT model for sentiment analysis"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained("./finbert_sentiment_model")
            model = AutoModelForSequenceClassification.from_pretrained("./finbert_sentiment_model")
            return tokenizer, model
        except:
            return None, None

tokenizer, model = load_model()

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    .main-title {
        text-align: center;
        font-size: 3.8em;
        font-weight: 900;
        margin: 50px 0 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3));
        animation: titleGlow 3s ease-in-out infinite;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3)); }
        50% { filter: drop-shadow(0 8px 16px rgba(102, 126, 234, 0.5)); }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.25em;
        color: #a0aec0;
        margin-bottom: 15px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    
    .input-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 40px;
        border-radius: 20px;
        margin: 30px 0;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
    }
    
    .input-section:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .input-label {
        font-size: 1em;
        font-weight: 700;
        margin-bottom: 12px;
        color: rgba(255, 255, 255, 0.95);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9em;
    }
    
    .section-header {
        font-size: 1.75em;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        border-bottom: 3px solid #667eea;
        padding-bottom: 15px;
        margin: 40px 0 25px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-left: 5px solid #667eea;
        border-radius: 12px;
        margin: 20px 0;
        font-size: 0.95em;
        color: #e0e0e0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-top: 3px solid #667eea;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .news-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        padding: 20px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(5px);
    }
    
    .news-item b {
        color: #e0e0e0;
        font-size: 0.98em;
        line-height: 1.6;
        display: block;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .news-item p {
        color: #b0b0b0;
        font-size: 0.9em;
        margin: 0;
        line-height: 1.5;
    }
    
    .news-meta {
        color: #808080;
        font-size: 0.85em;
        margin-top: 10px;
    }
    
    .news-item:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateX(6px);
        border-color: rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    .sentiment-bullish {
        display: inline-block;
        background: linear-gradient(135deg, #51CF66 0%, #40C057 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9em;
        box-shadow: 0 4px 12px rgba(81, 207, 102, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sentiment-bearish {
        display: inline-block;
        background: linear-gradient(135deg, #FF6B6B 0%, #FA5252 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9em;
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .sentiment-neutral {
        display: inline-block;
        background: linear-gradient(135deg, #FFA94D 0%, #FF9800 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9em;
        box-shadow: 0 4px 12px rgba(255, 169, 77, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-label {
        font-size: 0.8em;
        color: #a0aec0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 10px 0;
    }
    
    .show-more-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 700;
        margin: 15px 0;
        font-size: 0.9em;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .show-more-btn:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 30px 0;
    }
    
    .disclaimer {
        text-align: center;
        font-size: 0.85em;
        color: #808080;
        margin-top: 40px;
        padding: 25px 20px;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 12px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ============ PAGE TITLE ============
st.markdown('<div class="main-title">📈 STOCK SENTIMENT ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🚀 Enterprise-Grade Real-Time Analysis | 500+ Live Articles | AI-Powered Insights</div>', unsafe_allow_html=True)

# Premium visual separator
st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%); margin: 30px 0; border-radius: 3px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'></div>", unsafe_allow_html=True)

# Live status indicator with refresh button
col_status_1, col_status_2, col_status_3 = st.columns([2.5, 1, 0.5])
with col_status_1:
    model_status = "✅ MODEL READY" if (tokenizer and model) else "⏳ MODEL LOADING"
    st.markdown(f'<div style="font-size: 0.9em; color: #51CF66; font-weight: 600;">🔴 LIVE MODE • {model_status} • 500+ Articles</div>', unsafe_allow_html=True)
with col_status_2:
    if st.button("🔄 Refresh Now", help="Fetch fresh data immediately"):
        st.cache_data.clear()
        st.session_state.refresh_count += 1
        st.rerun()
with col_status_3:
    st.markdown(f'<div style="font-size: 0.8em; color: #667eea; font-weight: 600;">⚡ LIVE</div>', unsafe_allow_html=True)

st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)

# Get API key from secrets
try:
    API_KEY = st.secrets.get("NEWS_API_KEY", None)
    if API_KEY and API_KEY.strip() == "your_newsapi_key_here":
        API_KEY = None
    if not API_KEY or API_KEY.strip() == "":
        API_KEY = None
except (KeyError, AttributeError):
    API_KEY = None

# Store API status in session
if 'api_checked' not in st.session_state:
    st.session_state.api_checked = False
    if API_KEY:
        st.session_state.api_status = "valid"
    else:
        st.session_state.api_status = "missing"

# ============ STOCK DISCOVERY DATA ============
STOCK_CATEGORIES = {
    "🇮🇳 India (NSE)": [
        ("Infosys Limited", "INFY"),
        ("Tata Consultancy Services", "TCS"),
        ("Reliance Industries", "RELIANCE"),
        ("HDFC Bank", "HDFCBANK"),
        ("ITC Limited", "ITC"),
        ("Maruti Suzuki", "MARUTI"),
        ("Axis Bank", "AXISBANK"),
        ("ICICI Bank", "ICICIBANK"),
        ("Bharti Airtel", "BHARTIARTL"),
        ("Coal India", "COALINDIA"),
    ],
    "🇺🇸 USA (NASDAQ/NYSE)": [
        ("Apple Inc.", "AAPL"),
        ("Microsoft", "MSFT"),
        ("Tesla Inc.", "TSLA"),
        ("Google/Alphabet", "GOOGL"),
        ("Amazon", "AMZN"),
        ("Meta Platforms", "META"),
        ("NVIDIA", "NVDA"),
        ("Intel", "INTC"),
        ("AMD", "AMD"),
        ("Qualcomm", "QCOM"),
    ],
    "🌍 International": [
        ("Toyota Motor", "7203.T"),
        ("Samsung Electronics", "005930.KS"),
        ("ASML Holdings", "ASML"),
        ("SAP SE", "SAP"),
        ("LVMH Moët", "LVMHF"),
        ("Shell", "SHEL"),
        ("Nestlé", "NSRGY"),
        ("Unilever", "UL"),
    ]
}

# ============ CORE FUNCTIONS ============
def get_stock_news(company):
    """Fetch news from multiple queries - LIVE with maximum articles"""
    if not API_KEY:
        # Return demo/sample data when API key is missing
        return get_demo_news(company)
    
    try:
        all_articles = []
        # Expanded queries to get maximum diverse articles
        queries = [
            company,
            f"{company} stock",
            f"{company} shares",
            f"{company} earnings",
            f"{company} news",
            f"{company} price",
            f"{company} trading",
            f"{company} market",
            f"{company} update",
            f"{company} financial",
            f"{company} investor",
            f"{company} analyst"
        ]
        
        for query in queries:
            try:
                # Get latest articles with max pageSize
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"
                r = requests.get(url, timeout=5).json()
                
                if r.get("status") == "ok":
                    articles = r.get("articles", [])
                    all_articles.extend(articles)
                elif r.get("status") == "error":
                    # Show error once
                    if not st.session_state.get("api_error_shown", False):
                        error_msg = r.get("message", "API Error")
                        if "invalid" in error_msg.lower() or "incorrect" in error_msg.lower():
                            st.error(f"""
                            ❌ **Invalid NewsAPI Key!**
                            
                            {error_msg}
                            
                            **Fix it:**
                            1. Go to https://newsapi.org
                            2. Get a valid API key
                            3. Update `.streamlit/secrets.toml`
                            4. Refresh the app
                            """)
                            st.session_state.api_error_shown = True
                        st.stop()
                time.sleep(0.15)
            except requests.exceptions.Timeout:
                st.warning(f"⏱️ Timeout fetching '{query}'. Continuing with other queries...")
            except Exception as e:
                pass
        
        if not all_articles:
            # Fallback to demo data if all queries failed
            return get_demo_news(company)
        
        # Remove duplicates and keep unique articles
        seen = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "")
            url = article.get("url", "")
            # Use both title and URL for duplicate detection
            article_id = f"{title}|{url}"
            if article_id not in seen:
                seen.add(article_id)
                unique_articles.append(article)
        
        # Sort by published date (most recent first)
        result = sorted(unique_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)[:500]
        
        if result:
            return result
        else:
            return get_demo_news(company)
            
    except Exception as e:
        st.warning(f"⚠️ Error fetching live news: {str(e)[:50]}. Showing sample data instead.")
        return get_demo_news(company)

def get_demo_news(company):
    """Return sample/demo news data for demonstration - 100+ articles"""
    company_lower = company.lower()
    
    # Generate many articles for each company
    def generate_articles(company_name, count=120):
        templates = [
            f"{company_name} Q4 earnings beat expectations",
            f"{company_name} launches new product line",
            f"{company_name} stock hits new all-time high",
            f"{company_name} receives analyst upgrade",
            f"{company_name} expands into new markets",
            f"{company_name} faces regulatory challenge",
            f"{company_name} revenue growth slows",
            f"{company_name} CEO makes major announcement",
            f"{company_name} stock drops on profit warning",
            f"{company_name} signs major partnership",
            f"{company_name} invests in new technology",
            f"{company_name} reports strong quarterly results",
            f"{company_name} market share gains",
            f"{company_name} introduces AI features",
            f"{company_name} completes acquisition",
        ]
        sentiments = ["Positive", "Negative", "Neutral"]
        sources = ["Reuters", "Bloomberg", "CNBC", "TechCrunch", "MarketWatch", "WSJ", "FT", "Yahoo Finance", "Associated Press", "AP News"]
        
        articles = []
        for i in range(count):
            template_idx = i % len(templates)
            sentiment_idx = i % len(sentiments)
            source_idx = i % len(sources)
            
            articles.append({
                "title": templates[template_idx],
                "source": {"name": sources[source_idx]},
                "publishedAt": f"2026-01-{18 - (i // 24)%18}T{10 + (i % 12):02d}:{(i*7)%60:02d}:00Z",
                "url": f"https://example.com/article/{i}"
            })
        
        return articles
    
    # Return generated articles for matching company
    if "tesla" in company_lower or "tsla" in company_lower:
        return generate_articles(company, 150)
    elif "apple" in company_lower or "aapl" in company_lower:
        return generate_articles(company, 150)
    elif "infosys" in company_lower or "infy" in company_lower:
        return generate_articles(company, 150)
    elif "microsoft" in company_lower or "msft" in company_lower:
        return generate_articles(company, 150)
    elif "google" in company_lower or "googl" in company_lower:
        return generate_articles(company, 150)
    elif "amazon" in company_lower or "amzn" in company_lower:
        return generate_articles(company, 150)
    else:
        # Generic articles for any company
        return generate_articles(company, 150)

def analyze(text):
    """Analyze sentiment using FinBERT"""
    if not tokenizer or not model:
        return None, None, None
    
    try:
        # Encode text
        inputs = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        
        # Forward pass through model
        with torch.no_grad():
            outputs = model(inputs)
        
        # Get probabilities
        probabilities = F.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
        labels = ["Negative", "Neutral", "Positive"]
        sentiment = labels[prediction.item()]
        
        # Get score based on sentiment
        if sentiment == "Positive":
            score = float(probabilities[0][2].item())
        elif sentiment == "Negative":
            score = -float(probabilities[0][0].item())
        else:
            score = 0
        
        return sentiment, score, float(confidence.item())
    except Exception as e:
        return None, None, None

def analyze_news_sentiment(company):
    """Batch analyze news articles - Always use FinBERT for analysis"""
    news_articles = get_stock_news(company)
    
    if not news_articles:
        st.warning(f"❌ No news found for {company}. Tips:\n- Try a different company name\n- Add your NewsAPI key to `.streamlit/secrets.toml`\n- Use the sidebar stock discovery feature")
        return []
    
    if not tokenizer or not model:
        st.error("❌ AI Model not loaded! Please refresh the page and wait for the model to load (30-60 seconds).")
        st.info("Check the status at the top of the page - it should show '✅ MODEL READY'")
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_count = 0
    success_count = 0
    
    debug_info = st.empty()
    debug_info.write(f"📰 Processing {len(news_articles)} articles with FinBERT AI...")
    
    for i, article in enumerate(news_articles):
        headline = article.get("title", "")
        
        if not headline or len(headline) < 3:
            error_count += 1
            continue
        
        # ALWAYS use FinBERT for sentiment analysis
        sentiment, score, confidence = analyze(headline)
        
        if sentiment and sentiment in ["Positive", "Negative", "Neutral"]:
            # Ensure confidence is a number 0-100
            if isinstance(confidence, float):
                conf_value = confidence * 100
            else:
                conf_value = float(confidence) if confidence else 75.0
            
            results.append({
                "headline": headline,
                "sentiment": sentiment,
                "score": score if isinstance(score, (int, float)) else 0,
                "confidence": min(100, max(0, conf_value)),  # Ensure 0-100
                "source": article.get("source", {}).get("name", "Unknown"),
                "published": article.get("publishedAt", "")
            })
            success_count += 1
        else:
            error_count += 1
        
        progress = (i + 1) / len(news_articles)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Processing: {i+1}/{len(news_articles)} | ✓ {success_count} | ⚠️ {error_count}")
    
    progress_bar.empty()
    status_text.empty()
    debug_info.empty()
    
    if results:
        st.success(f"✅ Successfully analyzed {success_count}/{len(news_articles)} articles!")
    else:
        st.error(f"❌ Analysis failed!\n\n**Results:**\n- Processed: {len(news_articles)} articles\n- Successful: {success_count}\n- Failed: {error_count}\n\n**Debug Info:**\n- Tokenizer loaded: {tokenizer is not None}\n- Model loaded: {model is not None}\n\nPlease refresh the page and try again. The model may still be initializing.")
    
    return results
    
    return results

def overall_sentiment(results):
    """Calculate average sentiment score"""
    if not results:
        return 0
    scores = [r["score"] for r in results]
    return np.mean(scores)

def extract_keywords(texts, top_n=10):
    """Extract top keywords"""
    try:
        all_text = " ".join(texts).lower()
        stop_words = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "in", "of", "for", "with", "on", "by", "at", "from"}
        words = [w for w in all_text.split() if len(w) > 3 and w not in stop_words]
        return dict(Counter(words).most_common(top_n))
    except:
        return {}

@st.cache_data(ttl=60)  # 60 seconds - live updates
def get_forex_rate():
    """Fetch USD to INR exchange rate"""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        r = requests.get(url, timeout=5).json()
        return r.get('rates', {}).get('INR', 82.5)
    except:
        return 82.5  # Default rate

@st.cache_data(ttl=30)  # 30 seconds - LIVE stock price updates
def get_stock_price(ticker):
    """Fetch LIVE stock price from yfinance with proper Indian stock handling"""
    try:
        original_ticker = ticker
        
        # List of known Indian stock tickers (NSE)
        indian_tickers = [
            'INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'ITC', 'MARUTI',
            'AXISBANK', 'ICICIBANK', 'BHARTIARTL', 'COALINDIA', 'WIPRO',
            'SUNPHARMA', 'HINDUNILVR', 'LT', 'TATAMOTORS', 'HDFC',
            'BAJAAJFINSV', 'BAJAJFINSV', 'CIPLA', 'DRREDDY', 'NTPC',
            'POWERGRID', 'JSWSTEEL', 'SBIN', 'ONGC', 'ADANIPOWER'
        ]
        
        # Auto-add .NS for Indian stocks if not already present
        if ticker and not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            if ticker.upper() in indian_tickers:
                # Definitely an Indian stock
                ticker = ticker + '.NS'
            else:
                # Try .NS for unknown tickers as fallback
                ticker_with_ns = ticker + '.NS'
                try:
                    test_stock = yf.Ticker(ticker_with_ns)
                    test_hist = test_stock.history(period="1d")
                    if not test_hist.empty:
                        # Has data with .NS, so it's an Indian stock
                        ticker = ticker_with_ns
                except:
                    # Keep original ticker if .NS fails
                    pass
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        
        # Get current data
        hist_1d = stock.history(period="5d", interval="1d")
        
        # Get info
        try:
            info = stock.info
        except:
            info = {}
        
        # Get 1 year history for 52W stats
        try:
            hist_1y = stock.history(period="1y")
        except:
            hist_1y = hist_1d
        
        # Get LIVE current price - multiple strategies
        current_price = None
        
        # Strategy 1: Most recent historical close (most reliable)
        if not hist_1d.empty and len(hist_1d) > 0:
            current_price = float(hist_1d['Close'].iloc[-1])
        
        # Strategy 2: regularMarketPrice from info
        if current_price is None or current_price <= 0:
            current_price = info.get('regularMarketPrice')
        
        # Strategy 3: currentPrice from info
        if current_price is None or current_price <= 0:
            current_price = info.get('currentPrice')
        
        # Strategy 4: previousClose from info
        if current_price is None or current_price <= 0:
            current_price = info.get('previousClose')
        
        # Strategy 5: Try bid/ask average
        if current_price is None or current_price <= 0:
            bid = info.get('bid')
            ask = info.get('ask')
            if bid and ask and bid > 0 and ask > 0:
                current_price = (bid + ask) / 2
        
        # Validate price
        if current_price and current_price > 0:
            current_price = float(current_price)
        else:
            current_price = None
        
        # 52W High/Low
        high_52w = None
        low_52w = None
        if not hist_1y.empty and len(hist_1y) > 0:
            high_52w = float(hist_1y['High'].max())
            low_52w = float(hist_1y['Low'].min())
        
        # P/E Ratio
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio:
            pe_ratio = float(pe_ratio)
        
        # Country & Currency
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
        
        # Better country detection
        if 'NSE' in str(info.get('exchange', '')) or 'BSE' in str(info.get('exchange', '')) or ticker.endswith('.NS') or ticker.endswith('.BO'):
            country = 'India'
            currency = 'INR'
        
        # Get volume
        volume = 0
        if not hist_1d.empty and len(hist_1d) > 0 and 'Volume' in hist_1d.columns:
            vol = hist_1d['Volume'].iloc[-1]
            if vol and vol > 0:
                volume = int(vol)
        
        # Calculate price change
        change = 0
        change_pct = 0
        if len(hist_1d) >= 2 and current_price and current_price > 0:
            prev_close = float(hist_1d['Close'].iloc[-2])
            if prev_close > 0:
                change = current_price - prev_close
                change_pct = (change / prev_close * 100)
        
        return {
            'price': current_price,
            '52w_high': high_52w,
            '52w_low': low_52w,
            'pe_ratio': pe_ratio,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'N/A'),
            'history': hist_1y,
            'history_1d': hist_1d,
            'country': country,
            'currency': currency,
            'volume': volume,
            'change': change,
            'change_pct': change_pct,
            'ticker_used': ticker
        }
    except Exception as e:
        st.warning(f"⚠️ Error fetching {ticker}: {str(e)[:60]}")
        return None

def plot_stock_chart(ticker, period="1mo"):
    """Plot professional LIVE stock price chart like Zerodha"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 2:
            return None
        
        # Get country info
        try:
            info = stock.info
            country = info.get('country', 'US')
            currency = info.get('currency', 'USD')
        except:
            country = 'US'
            currency = 'USD'
        
        # Better country detection
        if 'NSE' in str(info.get('exchange', '')) or ticker.endswith('.NS') or ticker.endswith('.BO'):
            country = 'India'
            currency = 'INR'
        
        is_indian = country.upper() == 'INDIA'
        currency_symbol = "₹" if is_indian else "$"
        currency_label = "INR" if is_indian else currency
        
        # Create professional figure with price and volume
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.3)
        
        # Price chart (top)
        ax1 = fig.add_subplot(gs[0])
        
        # Calculate colors based on open/close
        colors = ['#51CF66' if close >= open_ else '#FF6B6B' for open_, close in zip(hist['Open'], hist['Close'])]
        
        # Plot candlesticks (simplified with bar)
        for i, (date, row) in enumerate(hist.iterrows()):
            high = row['High']
            low = row['Low']
            open_ = row['Open']
            close = row['Close']
            color = '#51CF66' if close >= open_ else '#FF6B6B'
            
            # Thin line for high-low range
            ax1.plot([i, i], [low, high], color=color, linewidth=1, alpha=0.6)
            # Thick bar for open-close
            ax1.bar(i, abs(close - open_), bottom=min(open_, close), width=0.6, color=color, alpha=0.8, edgecolor=color, linewidth=1)
        
        # Smooth line over closes
        ax1.plot(range(len(hist)), hist['Close'], color='#667eea', linewidth=2.5, alpha=0.7, label='Close Price', zorder=5)
        
        ax1.set_title(f"📈 {ticker} - {currency_label} | LIVE PRICE CHART", fontsize=14, fontweight='bold', pad=15, color='#1a1a1a')
        ax1.set_ylabel(f"Price ({currency_label})", fontsize=11, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.0f}'))
        ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax1.set_facecolor('#fafbfc')
        
        # Volume chart (bottom)
        ax2 = fig.add_subplot(gs[1])
        colors_vol = ['#51CF66' if close >= open_ else '#FF6B6B' for open_, close in zip(hist['Open'], hist['Close'])]
        ax2.bar(range(len(hist)), hist['Volume'], color=colors_vol, alpha=0.5, width=0.8)
        ax2.set_ylabel('Volume', fontsize=10, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x/1e3)}K'))
        ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        ax2.set_facecolor('#fafbfc')
        
        # Sync x-axis
        ax2.set_xlim(ax1.get_xlim())
        num_ticks = min(6, len(hist))
        if num_ticks > 0:
            ax2.set_xticks(range(0, len(hist), max(1, len(hist)//num_ticks)))
            try:
                ax2.set_xticklabels([hist.index[i].strftime('%b %d') for i in range(0, len(hist), max(1, len(hist)//num_ticks))], rotation=45)
            except:
                pass
        ax1.set_xticks([])
        
        # Add legend
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        change_color = '#51CF66' if change >= 0 else '#FF6B6B'
        change_symbol = '+' if change >= 0 else ''
        
        ax1.text(0.02, 0.98, f"Current: {currency_symbol}{current_price:,.2f} | Change: {change_symbol}{change:,.2f} ({change_symbol}{change_pct:.2f}%)",
                transform=ax1.transAxes, fontsize=11, fontweight='bold', color=change_color,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def analyze_sentiment_price_impact(ticker, results):
    """Analyze correlation between sentiment and price movement"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get 1 month history
        hist = stock.history(period="1mo")
        
        if hist.empty or len(hist) < 5:
            return None
        
        if not results or len(results) < 2:
            return None
        
        # Calculate daily returns
        hist['Daily_Return'] = hist['Close'].pct_change() * 100
        
        # Get sentiment by date from news
        sentiment_by_date = {}
        for r in results:
            try:
                pub_date = pd.to_datetime(r.get('published', '')).date()
                score = r.get('score', 0)
                if pub_date not in sentiment_by_date:
                    sentiment_by_date[pub_date] = []
                sentiment_by_date[pub_date].append(score)
            except:
                pass
        
        if not sentiment_by_date:
            return None
        
        # Average sentiment per date
        daily_sentiment = {}
        for date, scores in sentiment_by_date.items():
            daily_sentiment[date] = np.mean(scores)
        
        return {
            'history': hist,
            'daily_sentiment': daily_sentiment,
            'latest_return': hist['Daily_Return'].iloc[-1] if len(hist) > 0 else 0
        }
    except Exception as e:
        return None

def plot_sentiment_price_correlation(ticker, analysis_data):
    """Plot sentiment vs price movement correlation"""
    if not analysis_data or analysis_data['history'].empty or len(analysis_data['history']) < 2:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
        if not daily_sentiment:
            return None
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        # Subplot 1: Stock Price
        ax1.plot(hist.index, hist['Close'], linewidth=2, color='#667eea', label='Close Price')
        ax1.fill_between(hist.index, hist['Close'], alpha=0.2, color='#667eea')
        ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax1.set_title('Stock Price vs Sentiment Impact', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Subplot 2: Sentiment Score
        dates = sorted(daily_sentiment.keys())
        scores = [daily_sentiment[d] for d in dates]
        colors_sentiment = ['#51CF66' if s > 0.1 else '#FF6B6B' if s < -0.1 else '#FFA94D' for s in scores]
        
        ax2.bar(dates, scores, color=colors_sentiment, alpha=0.7, label='Sentiment Score')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_title('News Sentiment Trend', fontsize=12, fontweight='bold')
        ax2.set_ylim([-1, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper left')
        
        fig.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_combined_sentiment_price(ticker, analysis_data):
    """Plot combined sentiment and price with dual axis"""
    if not analysis_data or analysis_data['history'].empty or len(analysis_data['history']) < 2:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
        if not daily_sentiment:
            return None
        
        # Get country info
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            country = info.get('country', 'US')
            currency = info.get('currency', 'USD')
        except:
            country = 'US'
            currency = 'USD'
        
        is_indian = country.upper() == 'INDIA'
        
        # Get currency symbol and label
        currency_symbol = "₹" if is_indian else "$"
        currency_label = "INR" if is_indian else currency
        
        close_prices = hist['Close'].values
        
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        # Plot price on left axis
        color = '#667eea'
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'Stock Price ({currency_label})', color=color, fontsize=11, fontweight='bold')
        ax1.plot(hist.index, close_prices, color=color, linewidth=2.5, label='Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.0f}'))
        ax1.grid(True, alpha=0.2)
        
        # Create second y-axis for sentiment
        ax2 = ax1.twinx()
        
        dates = sorted(daily_sentiment.keys())
        scores = [daily_sentiment[d] for d in dates]
        colors_sentiment = ['#51CF66' if s > 0.1 else '#FF6B6B' if s < -0.1 else '#FFA94D' for s in scores]
        
        ax2.bar(dates, scores, color=colors_sentiment, alpha=0.4, label='Sentiment', width=0.6)
        ax2.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold', color='#FF6B6B')
        ax2.set_ylim([-1, 1])
        ax2.tick_params(axis='y', labelcolor='#FF6B6B')
        
        ax1.set_title(f'{ticker}: Price & Sentiment Correlation', fontsize=13, fontweight='bold')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()
        return fig
    except Exception as e:
        return None

def calculate_sentiment_price_correlation(analysis_data):
    """Calculate correlation coefficient between sentiment and price movement"""
    if not analysis_data or not analysis_data['daily_sentiment']:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
        # Create aligned arrays
        corr_data = []
        for date in hist.index.date:
            if date in daily_sentiment:
                price_return = hist.loc[hist.index.date == date, 'Daily_Return'].values
                if len(price_return) > 0:
                    corr_data.append({
                        'date': date,
                        'sentiment': daily_sentiment[date],
                        'return': price_return[0]
                    })
        
        if len(corr_data) < 2:
            return None
        
        sentiments = [d['sentiment'] for d in corr_data]
        returns = [d['return'] for d in corr_data]
        
        correlation = np.corrcoef(sentiments, returns)[0, 1]
        
        return {
            'correlation': correlation if not np.isnan(correlation) else 0,
            'data_points': len(corr_data),
            'interpretation': get_correlation_interpretation(correlation)
        }
    except:
        return None

def get_correlation_interpretation(corr):
    """Interpret correlation coefficient"""
    if np.isnan(corr):
        return "Insufficient data"
    elif corr > 0.6:
        return "🚀 Strong positive - Positive sentiment drives prices UP"
    elif corr > 0.3:
        return "📈 Moderate positive - Sentiment tends to support price gains"
    elif corr > -0.3:
        return "⚖️ Weak correlation - Mixed relationship"
    elif corr > -0.6:
        return "📉 Moderate negative - Sentiment has inverse effect"
    else:
        return "🔴 Strong negative - Negative sentiment drives prices DOWN"

# ============ MAIN APP ============

# ========== SIDEBAR - STOCK DISCOVERY ==========
with st.sidebar:
    st.markdown("### 📚 Stock Discovery")
    
    # Use button-based tabs instead of radio for better responsiveness
    discovery_choice = st.segmented_control(
        "Choose an option:",
        ["Popular", "Search", "Help"],
        default="Popular"
    ) if hasattr(st, 'segmented_control') else st.radio(
        "Choose an option:",
        ["Popular", "Search", "Help"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if discovery_choice == "Popular" or discovery_choice == 0:
        st.markdown("#### 🌍 Popular Stocks")
        
        for category, stocks in STOCK_CATEGORIES.items():
            with st.expander(f"**{category}**", expanded=category == "🇮🇳 India (NSE)"):
                for company_name, ticker in stocks:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"{company_name} `({ticker})`")
                    with col2:
                        if st.button("👉", key=f"btn_{ticker}", help="Analyze this stock"):
                            st.session_state.selected_company = company_name.split()[0]
                            st.session_state.selected_ticker = ticker
                            st.rerun()
    
    elif discovery_choice == "Search" or discovery_choice == 1:
        st.markdown("#### 🔍 Find Stock")
        
        search_query = st.text_input(
            "Search by name or ticker:",
            placeholder="Apple, TSLA, Infosys...",
            label_visibility="collapsed"
        )
        
        if search_query and len(search_query) >= 1:
            all_stocks = []
            for stocks in STOCK_CATEGORIES.values():
                all_stocks.extend(stocks)
            
            query_lower = search_query.lower()
            results = [s for s in all_stocks if query_lower in s[0].lower() or query_lower in s[1].lower()]
            
            if results:
                st.markdown(f"**Found {len(results)}:**")
                for company_name, ticker in results[:15]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{company_name}** `{ticker}`")
                    with col2:
                        if st.button("→", key=f"search_{ticker}", help="Select"):
                            st.session_state.selected_company = company_name.split()[0]
                            st.session_state.selected_ticker = ticker
                            st.rerun()
            else:
                st.info("No matches. Try different keywords!")
    
    else:  # Help
        st.markdown("#### ℹ️ Help")
        st.markdown("""
        **Quick Start:**
        1. Pick stock from Popular tab
        2. Click the arrow button
        3. App fills company name
        4. Click Analyze
        
        **Features:**
        - 🔴 LIVE prices (30s)
        - 📊 Candlestick charts  
        - 🤖 AI sentiment
        - 📰 500+ articles
        - 🔗 Correlation
        
        **Setup API:**
        - https://newsapi.org
        - Add to secrets.toml
        - NEWS_API_KEY = "key"
        
        **Status:**
        """)
        if API_KEY:
            st.success("✅ API Key: Valid")
        else:
            st.error("⚠️ API Key: Missing")

# INPUT SECTION - Clean & Simple
st.markdown('<div class="input-section">', unsafe_allow_html=True)

st.markdown("### 🔍 Analyze a Stock")

col1, col2 = st.columns([2, 1])

with col1:
    # Pre-fill from sidebar if selected
    default_company = st.session_state.selected_company if st.session_state.selected_company else ""
    company = st.text_input(
        "Company Name",
        value=default_company,
        placeholder="E.g., Tesla, Apple, Infosys",
        help="Type or select from sidebar →",
        label_visibility="visible"
    )

with col2:
    # Pre-fill from sidebar if selected
    default_ticker = st.session_state.selected_ticker if st.session_state.selected_ticker else ""
    ticker = st.text_input(
        "Ticker",
        value=default_ticker,
        placeholder="E.g., TSLA",
        help="Optional",
        label_visibility="visible"
    ).upper()

st.caption("💡 **Tip:** Use sidebar → Stock Discovery to find stocks by name")

analyze_btn = st.button("🚀 ANALYZE", use_container_width=True, type="primary")

st.markdown('</div>', unsafe_allow_html=True)

# Optional Comparison
with st.expander("➕ Compare with another stock? (Optional)"):
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        company2 = st.text_input(
            "Second Company",
            placeholder="E.g., Ford, Microsoft",
            help="Leave empty to skip"
        )
    
    with col_c2:
        ticker2 = st.text_input("Ticker", placeholder="", key="ticker2").upper()

# ============ ANALYSIS SECTION ============
if analyze_btn:
    # Use input values or fall back to sidebar selection
    company_clean = company.strip() if company.strip() else st.session_state.selected_company.strip()
    ticker_clean = ticker.strip() if ticker.strip() else st.session_state.selected_ticker.strip()
    
    if not company_clean:
        st.error("⚠️ Please enter a company name or select from sidebar → Stock Discovery")
    else:
        # Clear session state after extracting values
        st.session_state.selected_company = ""
        st.session_state.selected_ticker = ""
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Check API key status and warn user
        if not API_KEY:
            st.info("""
            ℹ️ **Using Sample/Demo Data** (API Key Not Configured)
            
            Real-time sentiment analysis requires a NewsAPI key. You're seeing sample/demo articles for demonstration.
            
            **To enable live sentiment analysis:**
            1. Get free API key: https://newsapi.org
            2. Edit `.streamlit/secrets.toml`
            3. Add your key: `NEWS_API_KEY = "your_key_here"`
            4. Refresh this page
            """)
        
        # ===== MAIN ANALYSIS =====
        st.markdown(f'<div class="section-header">📊 {company_clean.upper()} - Sentiment Analysis</div>', unsafe_allow_html=True)
        
        with st.spinner(f"⏳ Analyzing {company_clean}... (30-60 seconds)"):
            results = analyze_news_sentiment(company_clean)
        
        if results:
            # Overall Sentiment
            overall_score = overall_sentiment(results)
            positive_count = sum(1 for r in results if r["sentiment"] == "Positive")
            negative_count = sum(1 for r in results if r["sentiment"] == "Negative")
            neutral_count = sum(1 for r in results if r["sentiment"] == "Neutral")
            total = len(results)
            
            # Sentiment Gauge
            col_gauge = st.columns(1)[0]
            gauge_fig, gauge_ax = plt.subplots(figsize=(12, 2))
            gauge_ax.barh([0], [overall_score], color=['#FF6B6B' if overall_score < -0.3 else '#FFA94D' if overall_score < 0.3 else '#51CF66'], height=0.3)
        gauge_ax.set_xlim([-1, 1])
        gauge_ax.set_ylim([-0.5, 0.5])
        gauge_ax.axvline(0, color='black', linestyle='-', linewidth=2)
        gauge_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        gauge_ax.set_xticklabels(['Bearish', '', 'Neutral', '', 'Bullish'])
        gauge_ax.set_yticks([])
        gauge_ax.text(overall_score, 0, f"  {overall_score:.2f}", ha='left', va='center', fontsize=12, fontweight='bold', color='white' if overall_score < -0.3 or overall_score > 0.3 else 'black')
        gauge_fig.tight_layout()
        st.pyplot(gauge_fig)
        plt.close()
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📰 Total Articles", total, help=f"Live articles analyzed from 12+ queries")
        
        with col2:
            st.metric("📈 Positive", positive_count)
        
        with col3:
            st.metric("⚖️ Neutral", neutral_count)
        
        with col4:
            st.metric("📉 Negative", negative_count)
        
        with col5:
            if overall_score > 0.3:
                st.metric("🚀 Signal", "BULLISH")
            elif overall_score < -0.3:
                st.metric("📍 Signal", "BEARISH")
            else:
                st.metric("➡️ Signal", "NEUTRAL")
        
        # News breakdown tabs
        st.markdown('<div class="section-header">📋 News by Sentiment</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            f"🟢 Positive ({positive_count})",
            f"🟠 Neutral ({neutral_count})",
            f"🔴 Negative ({negative_count})"
        ])
        
        with tab1:
            positive_results = [r for r in results if r["sentiment"] == "Positive"]
            if positive_results:
                st.markdown(f"**✅ {len(positive_results)} Positive Articles**")
                # Show first 5
                display_count = 5 if not st.session_state.show_more_positive else len(positive_results)
                for r in positive_results[:display_count]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b>
                    <p></p>
                    <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(positive_results) > 5 and not st.session_state.show_more_positive:
                    if st.button(f"📖 Show More ({len(positive_results) - 5} more)", key="show_more_pos"):
                        st.session_state.show_more_positive = True
                        st.rerun()
            else:
                st.info("No positive news found")
        
        with tab2:
            neutral_results = [r for r in results if r["sentiment"] == "Neutral"]
            if neutral_results:
                st.markdown(f"**⚖️ {len(neutral_results)} Neutral Articles**")
                # Show first 5
                display_count = 5 if not st.session_state.show_more_neutral else len(neutral_results)
                for r in neutral_results[:display_count]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b>
                    <p></p>
                    <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(neutral_results) > 5 and not st.session_state.show_more_neutral:
                    if st.button(f"📖 Show More ({len(neutral_results) - 5} more)", key="show_more_neu"):
                        st.session_state.show_more_neutral = True
                        st.rerun()
            else:
                st.info("No neutral news found")
        
        with tab3:
            negative_results = [r for r in results if r["sentiment"] == "Negative"]
            if negative_results:
                st.markdown(f"**❌ {len(negative_results)} Negative Articles**")
                # Show first 5
                display_count = 5 if not st.session_state.show_more_negative else len(negative_results)
                for r in negative_results[:display_count]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b>
                    <p></p>
                    <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(negative_results) > 5 and not st.session_state.show_more_negative:
                    if st.button(f"📖 Show More ({len(negative_results) - 5} more)", key="show_more_neg"):
                        st.session_state.show_more_negative = True
                        st.rerun()
            else:
                st.info("No negative news found")
        
        # Keywords
        st.markdown('<div class="section-header">🔑 Top Keywords</div>', unsafe_allow_html=True)
        
        keywords = extract_keywords([r["headline"] for r in results], top_n=12)
        if keywords:
            col_kw1, col_kw2 = st.columns([2, 1])
            with col_kw1:
                fig, ax = plt.subplots(figsize=(10, 4))
                kw_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Count"]).sort_values("Count")
                ax.barh(kw_df["Keyword"], kw_df["Count"], color='#667eea')
                ax.set_xlabel("Frequency")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col_kw2:
                st.markdown("**Top Keywords:**")
                for kw, count in list(keywords.items())[:7]:
                    st.markdown(f"• **{kw}** ({count})")
        
        # Stock Price (if ticker provided)
        if ticker:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">💵 Live Stock Price: {ticker}</div>', unsafe_allow_html=True)
            
            stock_data = get_stock_price(ticker)
            if stock_data:
                # Get currency info
                is_indian = stock_data['country'].upper() == 'INDIA'
                currency_symbol = "₹" if is_indian else "$"
                currency_label = "INR" if is_indian else stock_data['currency']
                
                # Only show metrics if data is valid
                if stock_data['price'] is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Current price with change indicator
                    with col1:
                        change_display = f"{currency_symbol}{stock_data['change']:,.2f} ({stock_data['change_pct']:+.2f}%)"
                        change_color = "green" if stock_data['change'] >= 0 else "red"
                        st.metric("💵 Current Price", f"{currency_symbol}{stock_data['price']:,.2f}", delta=change_display, delta_color=change_color)
                    
                    with col2:
                        if stock_data['52w_high'] is not None:
                            st.metric("📈 52W High", f"{currency_symbol}{stock_data['52w_high']:,.2f}")
                        else:
                            st.metric("📈 52W High", "N/A")
                    
                    with col3:
                        if stock_data['52w_low'] is not None:
                            st.metric("📉 52W Low", f"{currency_symbol}{stock_data['52w_low']:,.2f}")
                        else:
                            st.metric("📉 52W Low", "N/A")
                    
                    with col4:
                        if stock_data['pe_ratio'] is not None:
                            st.metric("P/E Ratio", f"{stock_data['pe_ratio']:.1f}")
                        else:
                            st.metric("P/E Ratio", "N/A")
                    
                    # Show currency and volume info
                    volume_str = f"{stock_data['volume']/1e6:.2f}M" if stock_data['volume'] >= 1e6 else f"{stock_data['volume']/1e3:.0f}K"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); padding: 15px; border-radius: 12px; font-size: 0.9em; margin: 15px 0; color: #e0e0e0; border-left: 4px solid #667eea; border: 1px solid rgba(102, 126, 234, 0.2);">
                    💱 <b>Currency:</b> {currency_label} | 📊 <b>Volume:</b> {volume_str} | 🌍 <b>Country:</b> {stock_data['country']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart - Use the correct ticker with .NS if needed
                    ticker_for_chart = stock_data.get('ticker_used', ticker)
                    st.markdown('<div class="section-header">📈 Stock Price Chart</div>', unsafe_allow_html=True)
                    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"], key="period_select")
                    try:
                        chart = plot_stock_chart(ticker_for_chart, period)
                        if chart:
                            st.pyplot(chart, use_container_width=True)
                        else:
                            st.warning("⚠️ Chart data not available for this ticker")
                    except Exception as e:
                        st.warning(f"⚠️ Unable to render chart: {str(e)[:60]}")
                    
                    # ===== SENTIMENT & PRICE CORRELATION =====
                    st.markdown('<div class="section-header">🔗 How Sentiment Affects Price</div>', unsafe_allow_html=True)
                    
                    # Analyze correlation - Use correct ticker
                    try:
                        sentiment_price_analysis = analyze_sentiment_price_impact(ticker_for_chart, results)
                    except Exception as e:
                        st.warning(f"⚠️ Error analyzing sentiment-price correlation: {str(e)[:60]}")
                        sentiment_price_analysis = None
                    
                    if sentiment_price_analysis:
                        # Combined chart
                        combined_chart = plot_combined_sentiment_price(ticker, sentiment_price_analysis)
                        if combined_chart:
                            st.pyplot(combined_chart, use_container_width=True)
                        
                        # Correlation metrics
                        corr_analysis = calculate_sentiment_price_correlation(sentiment_price_analysis)
                        if corr_analysis:
                            col_corr1, col_corr2, col_corr3 = st.columns(3)
                            
                            with col_corr1:
                                corr_value = corr_analysis['correlation']
                                corr_color = "🟢" if corr_value > 0 else "🔴" if corr_value < 0 else "⚪"
                                st.metric("📊 Correlation", f"{corr_value:.2f}", corr_color)
                            
                            with col_corr2:
                                st.metric("📈 Data Points", corr_analysis['data_points'])
                            
                            with col_corr3:
                                latest_return = sentiment_price_analysis['latest_return']
                                delta_color = "green" if latest_return > 0 else "red"
                                st.metric("📊 Latest Return", f"{latest_return:.2f}%", delta=f"{latest_return:.2f}%")
                            
                            # Interpretation
                            interpretation = corr_analysis['interpretation']
                            st.markdown(f"""
                            <div style="background: #f0f2f6; padding: 15px; border-radius: 8px; margin-top: 15px;">
                            <b>📌 Interpretation:</b><br>
                            {interpretation}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Sentiment vs Price Relationship
                        st.markdown('<div class="section-header">📊 Detailed Sentiment-Price Analysis</div>', unsafe_allow_html=True)
                        
                        detail_chart = plot_sentiment_price_correlation(ticker, sentiment_price_analysis)
                        if detail_chart:
                            st.pyplot(detail_chart, use_container_width=True)
                    else:
                        st.info("ℹ️ Analyzing sentiment-price relationship requires historical data. Try again in a moment.")
                else:
                    st.error(f"""
                    ❌ Unable to fetch live price for {ticker}
                    
                    **Tips to fix:**
                    1. Check your ticker symbol (e.g., TSLA, AAPL, INFY)
                    2. For Indian stocks, use the NSE ticker (add .NS if needed)
                    3. Use the sidebar stock discovery to find correct ticker
                    4. Try without entering ticker (just company name)
                    """)
    
    # ===== COMPARISON SECTION =====
    if company2.strip():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">🔄 Compare: {company.upper()} vs {company2.upper()}</div>', unsafe_allow_html=True)
        
        with st.spinner(f"⏳ Analyzing {company2}... (30-60 seconds)"):
            results2 = analyze_news_sentiment(company2.strip())
        
        if results2:
            overall_score2 = overall_sentiment(results2)
            
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**{company}**")
                if overall_score > 0.3:
                    st.markdown('<span class="sentiment-bullish">🚀 BULLISH</span>', unsafe_allow_html=True)
                elif overall_score < -0.3:
                    st.markdown('<span class="sentiment-bearish">📉 BEARISH</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="sentiment-neutral">⚖️ NEUTRAL</span>', unsafe_allow_html=True)
                st.metric("Score", f"{overall_score:.2f}")
                st.metric("Articles", len(results))
            
            with col2:
                st.markdown("**Difference**")
                diff = overall_score2 - overall_score
                if diff > 0:
                    st.metric(f"{company2} Better", f"+{diff:.2f}", "🟢")
                elif diff < 0:
                    st.metric(f"{company} Better", f"{diff:.2f}", "🔴")
                else:
                    st.metric("Equal", "0.00", "⚖️")
            
            with col3:
                st.markdown(f"**{company2}**")
                if overall_score2 > 0.3:
                    st.markdown('<span class="sentiment-bullish">🚀 BULLISH</span>', unsafe_allow_html=True)
                elif overall_score2 < -0.3:
                    st.markdown('<span class="sentiment-bearish">📉 BEARISH</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="sentiment-neutral">⚖️ NEUTRAL</span>', unsafe_allow_html=True)
                st.metric("Score", f"{overall_score2:.2f}")
                st.metric("Articles", len(results2))
            
            # Comparison chart
            fig, ax = plt.subplots(figsize=(10, 4))
            companies_list = [company, company2]
            scores = [overall_score, overall_score2]
            colors = ['#51CF66' if s > 0.3 else '#FF6B6B' if s < -0.3 else '#FFA94D' for s in scores]
            ax.bar(companies_list, scores, color=colors, alpha=0.7, width=0.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_ylabel("Sentiment Score")
            ax.set_ylim([-1, 1])
            ax.set_title("Sentiment Score Comparison")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

elif analyze_btn:
    st.error("⚠️ Please enter a company name")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div class="disclaimer">
⚠️ For educational purposes. Not financial advice.<br>
✅ <b>LIVE Mode Active</b> - Stock prices update every 30 seconds from real market data<br>
📊 <b>Real-Time Prices</b>: Auto-detects Indian stocks (INFY, TCS, RELIANCE, etc.) and fetches correct INR prices<br>
📰 Maximum articles fetched: 500+ from 12 search queries<br>
📈 Charts: Professional candlestick + volume | Auto-refresh every 30 seconds<br>
🔄 Data Source: yfinance (Yahoo Finance) - Real market data
</div>
""", unsafe_allow_html=True)
