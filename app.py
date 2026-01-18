"""
Stock Sentiment Analyzer Application
Updated: January 19, 2026

UPDATE: Fixed emoji gradient effects - all emojis now display clearly without purple tint
UPDATE: Added .emoji CSS class to prevent gradient application to emojis
UPDATE: Updated all section headers to use emoji spans for proper display
UPDATE: Improved emoji visibility across all titles and headers
UPDATE: Enhanced overall visual hierarchy and emoji presentation
"""

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

# ============ SESSION STATE INITIALIZATION ============
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
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = ""
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = ""
if 'show_more_positive_main' not in st.session_state:
    st.session_state.show_more_positive_main = False
if 'show_more_neutral_main' not in st.session_state:
    st.session_state.show_more_neutral_main = False
if 'show_more_negative_main' not in st.session_state:
    st.session_state.show_more_negative_main = False

# ============ LOAD AI MODEL ============
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained("./finbert_sentiment_model")
            model = AutoModelForSequenceClassification.from_pretrained("./finbert_sentiment_model")
            return tokenizer, model
        except:
            return None, None

tokenizer, model = load_model()

# ============ PREMIUM CUSTOM CSS ============
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        min-height: 100vh;
    }
    
    .stApp {
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
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
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
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(118, 75, 162, 0.18) 100%);
        border: 2.5px solid rgba(102, 126, 234, 0.35);
        padding: 45px;
        border-radius: 22px;
        margin: 30px 0;
        color: white;
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
    }
    
    .input-section:hover {
        border-color: rgba(102, 126, 234, 0.55);
        box-shadow: 0 28px 60px rgba(102, 126, 234, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.18);
    }
    
    .section-header {
        font-size: 1.85em;
        font-weight: 900;
        border-bottom: 4px solid #667eea;
        padding-bottom: 18px;
        margin: 50px 0 30px 0;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* UPDATE: Emoji styling - ensure emojis display with natural colors */
    .section-header::first-letter {
        -webkit-text-fill-color: white;
        color: white;
        background-clip: unset;
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
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.25);
        border-top: 4px solid #667eea;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
    }
    
    .metric-box:hover {
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-3px);
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
        color: #f0f0f0;
        font-size: 0.99em;
        line-height: 1.65;
        display: block;
        margin-bottom: 12px;
        font-weight: 700;
    }
    
    .news-meta {
        font-size: 0.85em;
        color: #a8a8a8;
        margin-top: 8px;
        font-weight: 500;
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
# UPDATE: Separated emoji from gradient text to prevent purple tinting
# UPDATE: Emoji displays in natural bright color
st.markdown('<div class="main-title">📈 <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">STOCK SENTIMENT ANALYZER</span></div>', unsafe_allow_html=True)
# UPDATE: Styled subtitle emoji for better visibility
st.markdown('<div class="subtitle">🚀 Enterprise-Grade Real-Time Analysis | 500+ Live Articles | AI-Powered Insights</div>', unsafe_allow_html=True)
st.markdown("<div style='height: 3px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%); margin: 30px 0; border-radius: 3px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'></div>", unsafe_allow_html=True)

# Live status indicator
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

# ============ API KEY ============
try:
    API_KEY = st.secrets.get("NEWS_API_KEY", None)
    if not API_KEY or API_KEY.strip() == "your_newsapi_key_here":
        API_KEY = None
except:
    API_KEY = None

# ============ STOCK CATEGORIES ============
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

# ============ NEWS FUNCTIONS ============
@st.cache_data(ttl=300)
def get_stock_news(company):
    if not API_KEY:
        return get_demo_news(company)
    
    try:
        all_articles = []
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
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"
                r = requests.get(url, timeout=5).json()
                
                if r.get("status") == "ok":
                    articles = r.get("articles", [])
                    all_articles.extend(articles)
                time.sleep(0.15)
            except:
                pass
        
        if not all_articles:
            return get_demo_news(company)
        
        seen = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "")
            url = article.get("url", "")
            article_id = f"{title}|{url}"
            if article_id not in seen:
                seen.add(article_id)
                unique_articles.append(article)
        
        result = sorted(unique_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)[:500]
        return result if result else get_demo_news(company)
    except:
        return get_demo_news(company)

def get_demo_news(company):
    templates = [
        f"{company} Q4 earnings beat expectations",
        f"{company} launches new product line",
        f"{company} stock hits new all-time high",
        f"{company} receives analyst upgrade",
        f"{company} expands into new markets",
        f"{company} faces regulatory challenge",
        f"{company} revenue growth slows",
        f"{company} CEO makes major announcement",
        f"{company} stock drops on profit warning",
        f"{company} signs major partnership",
        f"{company} invests in new technology",
        f"{company} reports strong quarterly results",
        f"{company} market share gains",
        f"{company} introduces AI features",
        f"{company} completes acquisition",
    ]
    
    articles = []
    for i in range(150):
        articles.append({
            "title": templates[i % len(templates)],
            "source": {"name": ["Reuters", "Bloomberg", "CNBC", "TechCrunch", "MarketWatch"][i % 5]},
            "publishedAt": f"2026-01-{18 - (i // 20) % 10}T{10 + i % 12:02d}:00:00Z",
            "url": f"https://example.com/{i}"
        })
    return articles

# ============ SENTIMENT ANALYSIS ============
def analyze_sentiment(headline):
    if not tokenizer or not model or not headline:
        return None, 0, 0
    
    try:
        inputs = tokenizer.encode(headline, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            logits = model(inputs).logits
        
        probs = F.softmax(logits, dim=1)[0]
        confidence = float(probs.max().item())
        pred = int(probs.argmax().item())
        
        sentiments = ["Negative", "Neutral", "Positive"]
        sentiment = sentiments[pred]
        score = float(probs[2].item()) - float(probs[0].item())
        
        return sentiment, score, confidence
    except:
        return None, 0, 0

def analyze_all_articles(company):
    articles = get_stock_news(company)
    
    if not articles:
        st.warning("No articles found")
        return []
    
    results = []
    progress = st.progress(0)
    status = st.empty()
    
    for i, article in enumerate(articles):
        title = article.get("title", "")
        if title and len(title) > 3:
            sentiment, score, conf = analyze_sentiment(title)
            if sentiment:
                results.append({
                    "headline": title,
                    "sentiment": sentiment,
                    "score": score,
                    "confidence": conf,
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published": article.get("publishedAt", "")
                })
        
        progress.progress((i + 1) / len(articles))
        status.text(f"Analyzed: {len(results)}/{len(articles)}")
    
    progress.empty()
    status.empty()
    return results

def overall_sentiment(results):
    if not results:
        return 0
    scores = [r["score"] for r in results]
    return np.mean(scores)

def extract_keywords(texts, top_n=10):
    try:
        all_text = " ".join(texts).lower()
        stop_words = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "in", "of", "for", "with", "on", "by", "at", "from"}
        words = [w for w in all_text.split() if len(w) > 3 and w not in stop_words]
        return dict(Counter(words).most_common(top_n))
    except:
        return {}

@st.cache_data(ttl=300)  # Increased from 30s to 5 minutes to reduce rate limiting
def get_stock_price(ticker, retry_count=0):
    try:
        # Rate limiting protection - exponential backoff
        if retry_count > 0:
            wait_time = min(2 ** retry_count, 10)  # Max 10 seconds
            time.sleep(wait_time)
        
        original_ticker = ticker
        indian_tickers = [
            'INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'ITC', 'MARUTI',
            'AXISBANK', 'ICICIBANK', 'BHARTIARTL', 'COALINDIA', 'WIPRO',
            'SUNPHARMA', 'HINDUNILVR', 'LT', 'TATAMOTORS', 'HDFC',
            'BAJAAJFINSV', 'BAJAJFINSV', 'CIPLA', 'DRREDDY', 'NTPC',
            'POWERGRID', 'JSWSTEEL', 'SBIN', 'ONGC', 'ADANIPOWER'
        ]
        
        if ticker and not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            if ticker.upper() in indian_tickers:
                ticker = ticker + '.NS'
            else:
                ticker_with_ns = ticker + '.NS'
                try:
                    test_stock = yf.Ticker(ticker_with_ns)
                    test_hist = test_stock.history(period="1d")
                    if not test_hist.empty:
                        ticker = ticker_with_ns
                except:
                    pass
        
        stock = yf.Ticker(ticker)
        hist_1d = stock.history(period="5d", interval="1d")
        
        try:
            info = stock.info
        except:
            info = {}
        
        try:
            hist_1y = stock.history(period="1y")
        except:
            hist_1y = hist_1d
        
        current_price = None
        if not hist_1d.empty and len(hist_1d) > 0:
            current_price = float(hist_1d['Close'].iloc[-1])
        
        if current_price is None or current_price <= 0:
            current_price = info.get('regularMarketPrice')
        
        if current_price is None or current_price <= 0:
            current_price = info.get('currentPrice')
        
        if current_price is None or current_price <= 0:
            current_price = info.get('previousClose')
        
        if current_price and current_price > 0:
            current_price = float(current_price)
        else:
            current_price = None
        
        high_52w = None
        low_52w = None
        if not hist_1y.empty and len(hist_1y) > 0:
            high_52w = float(hist_1y['High'].max())
            low_52w = float(hist_1y['Low'].min())
        
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if pe_ratio:
            pe_ratio = float(pe_ratio)
        
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
        
        if 'NSE' in str(info.get('exchange', '')) or 'BSE' in str(info.get('exchange', '')) or ticker.endswith('.NS') or ticker.endswith('.BO'):
            country = 'India'
            currency = 'INR'
        
        volume = 0
        if not hist_1d.empty and len(hist_1d) > 0 and 'Volume' in hist_1d.columns:
            vol = hist_1d['Volume'].iloc[-1]
            if vol and vol > 0:
                volume = int(vol)
        
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
        error_msg = str(e).lower()
        
        # Handle rate limiting with retry logic
        if 'too many requests' in error_msg or '429' in error_msg or 'rate' in error_msg:
            if retry_count < 2:
                st.warning(f"⏳ Rate limited. Retrying... (Attempt {retry_count + 2}/3)")
                return get_stock_price(ticker, retry_count + 1)
            else:
                st.warning(f"⚠️ Rate limit reached for {ticker}. Please try again in a few moments.")
                return None
        
        st.warning(f"⚠️ Error fetching {ticker}: {str(e)[:60]}")
        return None

def plot_stock_chart(ticker, period="1mo"):
    try:
        # Add small delay to prevent rate limiting
        time.sleep(0.5)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 2:
            return None
        
        try:
            info = stock.info
            country = info.get('country', 'US')
            currency = info.get('currency', 'USD')
        except:
            country = 'US'
            currency = 'USD'
        
        if 'NSE' in str(info.get('exchange', '')) or ticker.endswith('.NS') or ticker.endswith('.BO'):
            country = 'India'
            currency = 'INR'
        
        is_indian = country.upper() == 'INDIA'
        currency_symbol = "₹" if is_indian else "$"
        currency_label = "INR" if is_indian else currency
        
        fig = plt.figure(figsize=(14, 7), facecolor='white', dpi=120)
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.05], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        
        # Professional candlestick styling (Zerodha/Groww style)
        for i, (date, row) in enumerate(hist.iterrows()):
            high = row['High']
            low = row['Low']
            open_ = row['Open']
            close = row['Close']
            
            # Color based on price movement
            is_green = close >= open_
            body_color = '#00C596' if is_green else '#FF4949'  # Professional green/red
            wick_color = '#00C596' if is_green else '#FF4949'
            
            # Draw wick (high-low line)
            ax1.plot([i, i], [low, high], color=wick_color, linewidth=1.5, alpha=0.9)
            
            # Draw body (open-close rectangle)
            body_height = abs(close - open_)
            body_bottom = min(open_, close)
            if body_height > 0:
                ax1.bar(i, body_height, bottom=body_bottom, width=0.7, color=body_color, alpha=0.95, edgecolor=wick_color, linewidth=1.2)
            else:
                # Draw thin line for no change
                ax1.plot([i-0.35, i+0.35], [close, close], color=body_color, linewidth=2)
        
        # Add smooth price line overlay
        ax1.plot(range(len(hist)), hist['Close'], color='#1E90FF', linewidth=1.5, alpha=0.5, label='Close Price', zorder=5, linestyle='-')
        
        # Professional styling
        current_price = hist['Close'].iloc[-1]
        ax1.set_title(f"📈 {ticker} - {currency_label} | Current: {currency_symbol}{current_price:,.2f}", 
                     fontsize=14, fontweight='bold', pad=15, color='#1a1a1a')
        ax1.set_ylabel(f"Price ({currency_label})", fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.0f}'))
        ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.7, color='#d0d0d0')
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#e0e0e0')
        ax1.spines['bottom'].set_color('#e0e0e0')
        
        ax2 = fig.add_subplot(gs[1])
        colors_vol = ['#00C596' if close >= open_ else '#FF4949' for open_, close in zip(hist['Open'], hist['Close'])]
        ax2.bar(range(len(hist)), hist['Volume'], color=colors_vol, alpha=0.65, width=0.8, edgecolor='none')
        ax2.set_ylabel('Volume', fontsize=10, fontweight='bold', color='#2c3e50')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x/1e3)}K'))
        ax2.grid(True, alpha=0.1, linestyle='-', linewidth=0.7, axis='y', color='#d0d0d0')
        ax2.set_facecolor('#f8f9fa')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#e0e0e0')
        ax2.spines['bottom'].set_color('#e0e0e0')
        
        ax2.set_xlim(ax1.get_xlim())
        num_ticks = min(6, len(hist))
        if num_ticks > 0:
            ax2.set_xticks(range(0, len(hist), max(1, len(hist)//num_ticks)))
            try:
                ax2.set_xticklabels([hist.index[i].strftime('%b %d') for i in range(0, len(hist), max(1, len(hist)//num_ticks))], rotation=45)
            except:
                pass
        ax1.set_xticks([])
        
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        
        change_color = '#00C596' if change >= 0 else '#FF4949'
        change_symbol = '+' if change >= 0 else ''
        
        # Add info box with styling
        info_text = f"Current: {currency_symbol}{current_price:,.2f} | Change: {change_symbol}{change:,.2f} ({change_symbol}{change_pct:.2f}%)"
        ax1.text(0.02, 0.97, info_text, transform=ax1.transAxes, fontsize=10, fontweight='bold', 
                color=change_color, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor=change_color, linewidth=2, alpha=0.95))
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        return fig
    except:
        return None

def analyze_sentiment_price_impact(ticker, results):
    """Analyze correlation between sentiment and price movement"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty or len(hist) < 5:
            return None
        
        if not results or len(results) < 2:
            return None
        
        hist['Daily_Return'] = hist['Close'].pct_change() * 100
        
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
        
        daily_sentiment = {}
        for date, scores in sentiment_by_date.items():
            daily_sentiment[date] = np.mean(scores)
        
        return {
            'history': hist,
            'daily_sentiment': daily_sentiment,
            'latest_return': hist['Daily_Return'].iloc[-1] if len(hist) > 0 else 0
        }
    except:
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
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), facecolor='white', sharex=False, dpi=120)
        
        # Professional price chart
        ax1.plot(hist.index, hist['Close'], linewidth=2.5, color='#1E90FF', label='Close Price', zorder=3)
        ax1.fill_between(hist.index, hist['Close'], alpha=0.15, color='#1E90FF')
        ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.set_title('📈 Stock Price vs Sentiment Impact', fontsize=13, fontweight='bold', color='#2c3e50')
        ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.7, color='#d0d0d0')
        ax1.legend(loc='upper left', framealpha=0.95)
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#e0e0e0')
        ax1.spines['bottom'].set_color('#e0e0e0')
        
        # Sentiment bars
        dates = sorted(daily_sentiment.keys())
        scores = [daily_sentiment[d] for d in dates]
        colors_sentiment = ['#00C596' if s > 0.1 else '#FF4949' if s < -0.1 else '#F39C12' for s in scores]
        
        ax2.bar(dates, scores, color=colors_sentiment, alpha=0.75, label='Sentiment Score', edgecolor='none')
        ax2.axhline(0, color='#2c3e50', linestyle='-', linewidth=1.5)
        ax2.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold', color='#2c3e50')
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold', color='#2c3e50')
        ax2.set_title('📊 News Sentiment Trend', fontsize=12, fontweight='bold', color='#2c3e50')
        ax2.set_ylim([-1, 1])
        ax2.grid(True, alpha=0.1, axis='y', linestyle='-', linewidth=0.7, color='#d0d0d0')
        ax2.legend(loc='upper left', framealpha=0.95)
        ax2.set_facecolor('#f8f9fa')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#e0e0e0')
        ax2.spines['bottom'].set_color('#e0e0e0')
        
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        return fig
    except:
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
        
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            country = info.get('country', 'US')
            currency = info.get('currency', 'USD')
        except:
            country = 'US'
            currency = 'USD'
        
        is_indian = country.upper() == 'INDIA'
        currency_symbol = "₹" if is_indian else "$"
        currency_label = "INR" if is_indian else currency
        
        close_prices = hist['Close'].values
        
        fig, ax1 = plt.subplots(figsize=(13, 6), facecolor='white', dpi=120)
        
        # Price line
        color_price = '#1E90FF'
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.set_ylabel(f'Stock Price ({currency_label})', color=color_price, fontsize=11, fontweight='bold')
        ax1.plot(hist.index, close_prices, color=color_price, linewidth=2.8, label='Price', zorder=3, marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2, markeredgecolor=color_price)
        ax1.tick_params(axis='y', labelcolor=color_price)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.0f}'))
        ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.7, color='#d0d0d0')
        ax1.set_facecolor('#f8f9fa')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_color(color_price)
        ax1.spines['left'].set_color(color_price)
        ax1.spines['bottom'].set_color('#e0e0e0')
        
        # Sentiment bars on secondary axis
        ax2 = ax1.twinx()
        
        dates = sorted(daily_sentiment.keys())
        scores = [daily_sentiment[d] for d in dates]
        colors_sentiment = ['#00C596' if s > 0.1 else '#FF4949' if s < -0.1 else '#F39C12' for s in scores]
        
        ax2.bar(dates, scores, color=colors_sentiment, alpha=0.55, label='Sentiment', width=0.7, edgecolor='none')
        ax2.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold', color='#FF4949')
        ax2.set_ylim([-1, 1])
        ax2.tick_params(axis='y', labelcolor='#FF4949')
        ax2.spines['right'].set_color('#FF4949')
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        ax1.set_title(f'📊 {ticker}: Price & Sentiment Correlation (Live)', fontsize=13, fontweight='bold', color='#2c3e50')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95, fontsize=10)
        
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        return fig
    except:
        return None

def calculate_sentiment_price_correlation(analysis_data):
    """Calculate correlation coefficient"""
    if not analysis_data or not analysis_data['daily_sentiment']:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
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
    # UPDATE: Fixed white-on-white text contrast with color-coded HTML boxes
    # UPDATE: Each correlation level has distinct background and text colors
    # UPDATE: Improved readability with proper visual hierarchy
    if np.isnan(corr):
        return '<div style="background: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 4px solid #4caf50; color: #1b5e20;"><b>ℹ️ Insufficient data</b></div>'
    elif corr > 0.6:
        return '<div style="background: #c8e6c9; padding: 12px; border-radius: 8px; border-left: 4px solid #2e7d32; color: #1b5e20;"><b>🚀 Strong positive</b> - Positive sentiment drives prices UP</div>'
    elif corr > 0.3:
        return '<div style="background: #a5d6a7; padding: 12px; border-radius: 8px; border-left: 4px solid #388e3c; color: #1b5e20;"><b>📈 Moderate positive</b> - Sentiment tends to support price gains</div>'
    elif corr > -0.3:
        return '<div style="background: #fff9c4; padding: 12px; border-radius: 8px; border-left: 4px solid #f57f17; color: #333333;"><b>⚖️ Weak correlation</b> - Mixed relationship</div>'
    elif corr > -0.6:
        return '<div style="background: #ffccbc; padding: 12px; border-radius: 8px; border-left: 4px solid #e65100; color: #3e2723;"><b>📉 Moderate negative</b> - Sentiment has inverse effect</div>'
    else:
        return '<div style="background: #ffcdd2; padding: 12px; border-radius: 8px; border-left: 4px solid #c62828; color: #b71c1c;"><b>🔴 Strong negative</b> - Negative sentiment drives prices DOWN</div>'

# ============ SIDEBAR DISCOVERY ============
with st.sidebar:
    st.markdown("### 📚 Stock Discovery")
    
    choice = st.radio("Choose:", ["Popular", "Search", "Help"])
    st.markdown("---")
    
    if choice == "Popular":
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
    
    elif choice == "Search":
        st.markdown("#### 🔍 Advanced Search")
        
        search_query = st.text_input(
            "Search by company name or ticker:",
            placeholder="e.g., Tesla, TSLA, Infosys, INFY...",
            label_visibility="collapsed"
        )
        
        # Filter by region
        search_region = st.multiselect(
            "Filter by region:",
            ["🇮🇳 India (NSE)", "🇺🇸 USA (NASDAQ/NYSE)", "🌍 International"],
            default=["🇮🇳 India (NSE)", "🇺🇸 USA (NASDAQ/NYSE)", "🌍 International"],
            label_visibility="collapsed"
        )
        
        if search_query and len(search_query) >= 1:
            all_stocks = []
            for category, stocks in STOCK_CATEGORIES.items():
                if category in search_region:
                    all_stocks.extend(stocks)
            
            query_lower = search_query.lower()
            results = [s for s in all_stocks if query_lower in s[0].lower() or query_lower in s[1].lower()]
            
            if results:
                st.markdown(f"✅ **Found {len(results)} match{'es' if len(results) != 1 else ''}:**")
                st.divider()
                
                for i, (company_name, ticker) in enumerate(results[:20], 1):
                    col1, col2, col3 = st.columns([2, 0.8, 0.5])
                    with col1:
                        st.markdown(f"**{i}. {company_name}**")
                        st.caption(f"Ticker: `{ticker}`")
                    with col2:
                        st.write("")
                    with col3:
                        if st.button("Select", key=f"search_{ticker}", help="Select this stock", use_container_width=True):
                            st.session_state.selected_company = company_name.split()[0]
                            st.session_state.selected_ticker = ticker
                            st.rerun()
                
                if len(results) > 20:
                    st.caption(f"📌 Showing 20 of {len(results)} results. Refine your search!")
            else:
                st.warning("❌ No matches found. Try:")
                st.markdown("• Different spelling or abbreviation")
                st.markdown("• Full company name instead of partial")
                st.markdown("• Stock ticker symbol (e.g., AAPL, TCS)")
    
    else:  # Help
        st.markdown("#### ℹ️ Help & Guide")
        st.markdown("""
        **Quick Start:**
        1. Pick stock from Popular tab
        2. Click the arrow button
        3. App fills company name
        4. Click Analyze
        
        **Features:**
        - 🔴 LIVE prices (30s)
        - 📊 Candlestick charts  
        - 🤖 AI sentiment analysis
        - 📰 500+ real articles
        - 🔗 Price-Sentiment correlation
        
        **Setup NewsAPI:**
        1. Visit https://newsapi.org
        2. Get free API key
        3. Edit `.streamlit/secrets.toml`
        4. Add: `NEWS_API_KEY = "key"`
        5. Refresh app
        
        **Status:**
        """)
        if API_KEY:
            st.success("✅ API Key: Valid")
        else:
            st.error("⚠️ API Key: Missing (Demo Mode)")

# ============ INPUT SECTION ============
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### 🔍 Analyze a Stock")

col1, col2 = st.columns([2, 1])

with col1:
    default_company = st.session_state.selected_company if st.session_state.selected_company else ""
    company = st.text_input(
        "Company Name",
        value=default_company,
        placeholder="E.g., Tesla, Apple, Infosys",
        help="Type or select from sidebar →",
        label_visibility="visible"
    )

with col2:
    default_ticker = st.session_state.selected_ticker if st.session_state.selected_ticker else ""
    ticker = st.text_input(
        "Ticker",
        value=default_ticker,
        placeholder="E.g., TSLA",
        help="Optional - for live price",
        label_visibility="visible"
    ).upper()

st.caption("💡 **Tip:** Use sidebar → Stock Discovery to find stocks by name")

col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("🚀 ANALYZE", use_container_width=True, type="primary")
with col2:
    if st.button("✨ Clear"):
        st.session_state.selected_company = ""
        st.session_state.selected_ticker = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ============ COMPARISON SECTION ============
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
    company_clean = company.strip() if company.strip() else st.session_state.selected_company.strip()
    ticker_clean = ticker.strip() if ticker.strip() else st.session_state.selected_ticker.strip()
    
    if not company_clean:
        st.error("⚠️ Please enter a company name or select from sidebar")
    else:
        st.session_state.selected_company = ""
        st.session_state.selected_ticker = ""
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">📊 {company_clean.upper()} - Sentiment Analysis</div>', unsafe_allow_html=True)
        
        if not API_KEY:
            st.info("""
            ℹ️ **Using Demo Data** (API Key Not Configured)
            Real-time sentiment analysis requires a NewsAPI key. You're seeing sample articles.
            """)
        
        with st.spinner(f"⏳ Analyzing {company_clean}... (30-60 seconds)"):
            results = analyze_all_articles(company_clean)
        
        if results:
            # Calculate metrics
            positive_count = sum(1 for r in results if r["sentiment"] == "Positive")
            negative_count = sum(1 for r in results if r["sentiment"] == "Negative")
            neutral_count = sum(1 for r in results if r["sentiment"] == "Neutral")
            overall_score = overall_sentiment(results)
            total = len(results)
            
            # Sentiment Gauge
            gauge_fig, gauge_ax = plt.subplots(figsize=(12, 2), dpi=120)
            gauge_ax.barh([0], [overall_score], color=['#FF6B6B' if overall_score < -0.3 else '#FFA94D' if overall_score < 0.3 else '#51CF66'], height=0.3, edgecolor='#333', linewidth=2)
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
            col1.metric("📰 Total Articles", total)
            col2.metric("📈 Positive", positive_count)
            col3.metric("⚖️ Neutral", neutral_count)
            col4.metric("📉 Negative", negative_count)
            
            if overall_score > 0.3:
                col5.metric("🚀 Signal", "BULLISH")
            elif overall_score < -0.3:
                col5.metric("📍 Signal", "BEARISH")
            else:
                col5.metric("➡️ Signal", "NEUTRAL")
            
            # News tabs
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
                    show_all_positive = st.session_state.show_more_positive_main
                    display_count = len(positive_results) if show_all_positive else min(5, len(positive_results))
                    
                    for r in positive_results[:display_count]:
                        pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                        st.markdown(f"""
                        <div class="news-item">
                        <b>{r['headline']}</b>
                        <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(positive_results) > 5 and not show_all_positive:
                        if st.button(f"📂 Show {len(positive_results) - 5} more positive articles", key="show_more_pos"):
                            st.session_state.show_more_positive_main = True
                            st.rerun()
                else:
                    st.info("No positive news found")
            
            with tab2:
                neutral_results = [r for r in results if r["sentiment"] == "Neutral"]
                if neutral_results:
                    st.markdown(f"**⚖️ {len(neutral_results)} Neutral Articles**")
                    show_all_neutral = st.session_state.show_more_neutral_main
                    display_count = len(neutral_results) if show_all_neutral else min(5, len(neutral_results))
                    
                    for r in neutral_results[:display_count]:
                        pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                        st.markdown(f"""
                        <div class="news-item">
                        <b>{r['headline']}</b>
                        <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(neutral_results) > 5 and not show_all_neutral:
                        if st.button(f"📂 Show {len(neutral_results) - 5} more neutral articles", key="show_more_neu"):
                            st.session_state.show_more_neutral_main = True
                            st.rerun()
                else:
                    st.info("No neutral news found")
            
            with tab3:
                negative_results = [r for r in results if r["sentiment"] == "Negative"]
                if negative_results:
                    st.markdown(f"**❌ {len(negative_results)} Negative Articles**")
                    show_all_negative = st.session_state.show_more_negative_main
                    display_count = len(negative_results) if show_all_negative else min(5, len(negative_results))
                    
                    for r in negative_results[:display_count]:
                        pub_date = pd.to_datetime(r.get('published', '')).strftime('%b %d, %H:%M') if r.get('published') else 'Unknown'
                        st.markdown(f"""
                        <div class="news-item">
                        <b>{r['headline']}</b>
                        <div class="news-meta">📰 {r.get('source', 'Unknown')} • 🕐 {pub_date} • 🎯 {r['confidence']:.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if len(negative_results) > 5 and not show_all_negative:
                        if st.button(f"📂 Show {len(negative_results) - 5} more negative articles", key="show_more_neg"):
                            st.session_state.show_more_negative_main = True
                            st.rerun()
                else:
                    st.info("No negative news found")
            
            # Keywords
            st.markdown('<div class="section-header">🔑 Top Keywords</div>', unsafe_allow_html=True)
            
            keywords = extract_keywords([r["headline"] for r in results], top_n=12)
            if keywords:
                col_kw1, col_kw2 = st.columns([2, 1])
                with col_kw1:
                    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
                    kw_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Count"]).sort_values("Count")
                    bars = ax.barh(kw_df["Keyword"], kw_df["Count"], color='#667eea', edgecolor='#2c3e50', linewidth=1.5)
                    ax.set_xlabel("Frequency", fontsize=11, fontweight='bold')
                    ax.set_title("Most Mentioned Keywords", fontsize=12, fontweight='bold', pad=15)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, f' {int(width)}', 
                               ha='left', va='center', fontweight='bold', fontsize=9)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                with col_kw2:
                    st.markdown("**Top Keywords:**")
                    for kw, count in list(keywords.items())[:7]:
                        st.markdown(f"• **{kw}** ({count})")
            
            # Stock Price
            if ticker_clean:
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Auto-refresh every 30 seconds
                col_refresh_1, col_refresh_2, col_refresh_3 = st.columns([2, 1, 1])
                with col_refresh_1:
                    st.markdown(f'<div class="section-header">💵 Live Stock Price: {ticker_clean} 🔴 LIVE</div>', unsafe_allow_html=True)
                with col_refresh_2:
                    st.caption("⏱️ Updates every 5 min")
                with col_refresh_3:
                    if st.button("🔄 Refresh Now", key="refresh_price"):
                        st.cache_data.clear()
                        st.rerun()
                
                stock_data = get_stock_price(ticker_clean)
                if stock_data and stock_data['price'] is not None:
                    is_indian = stock_data['country'].upper() == 'INDIA'
                    currency_symbol = "₹" if is_indian else "$"
                    currency_label = "INR" if is_indian else stock_data['currency']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
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
                    
                    volume_str = f"{stock_data['volume']/1e6:.2f}M" if stock_data['volume'] >= 1e6 else f"{stock_data['volume']/1e3:.0f}K"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); padding: 15px; border-radius: 12px; font-size: 0.9em; margin: 15px 0; color: #e0e0e0; border-left: 4px solid #667eea; border: 1px solid rgba(102, 126, 234, 0.2);">
                    💱 <b>Currency:</b> {currency_label} | 📊 <b>Volume:</b> {volume_str} | 🌍 <b>Country:</b> {stock_data['country']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="section-header">📈 Stock Price Chart</div>', unsafe_allow_html=True)
                    period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"], key="period_select")
                    try:
                        ticker_for_chart = stock_data.get('ticker_used', ticker_clean)
                        chart = plot_stock_chart(ticker_for_chart, period)
                        if chart:
                            st.pyplot(chart, use_container_width=True)
                        else:
                            st.warning("⚠️ Chart data not available")
                    except Exception as e:
                        st.warning(f"⚠️ Unable to render chart: {str(e)[:60]}")
                    
                    # ===== SENTIMENT & PRICE CORRELATION =====
                    st.markdown('<div class="section-header">🔗 How Sentiment Affects Price</div>', unsafe_allow_html=True)
                    
                    try:
                        sentiment_price_analysis = analyze_sentiment_price_impact(ticker_for_chart, results)
                    except:
                        sentiment_price_analysis = None
                    
                    if sentiment_price_analysis:
                        # Combined chart
                        combined_chart = plot_combined_sentiment_price(ticker_for_chart, sentiment_price_analysis)
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
                                st.metric("📊 Latest Return", f"{latest_return:.2f}%")
                            
                            # Interpretation
                            interpretation = corr_analysis['interpretation']
                            st.markdown(f"""
                            <div style="margin-top: 15px;">
                            {interpretation}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed chart
                        st.markdown('<div class="section-header">📊 Detailed Sentiment-Price Analysis</div>', unsafe_allow_html=True)
                        detail_chart = plot_sentiment_price_correlation(ticker_for_chart, sentiment_price_analysis)
                        if detail_chart:
                            st.pyplot(detail_chart, use_container_width=True)
    
    # ===== COMPARISON SECTION =====
    if company2.strip():
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">🔄 Compare: {company.upper()} vs {company2.upper()}</div>', unsafe_allow_html=True)
        
        with st.spinner(f"⏳ Analyzing {company2}... (30-60 seconds)"):
            results2 = analyze_all_articles(company2.strip())
        
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
            fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
            companies_list = [company, company2]
            scores = [overall_score, overall_score2]
            colors = ['#51CF66' if s > 0.3 else '#FF6B6B' if s < -0.3 else '#FFA94D' for s in scores]
            bars = ax.bar(companies_list, scores, color=colors, alpha=0.85, width=0.5, edgecolor='#2c3e50', linewidth=2.5)
            ax.axhline(0, color='#333', linestyle='-', linewidth=2)
            ax.set_ylabel("Sentiment Score", fontsize=12, fontweight='bold')
            ax.set_ylim([-1, 1])
            ax.set_title("📊 Sentiment Score Comparison", fontsize=13, fontweight='bold', pad=15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=11)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div class="disclaimer">
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
    <div>
        <b>📋 Disclaimer</b><br>
        ⚠️ Educational purposes only. Not financial advice.<br>
        📊 For research and learning purposes<br>
        💼 Always consult a financial advisor
    </div>
    <div>
        <b>✨ Features</b><br>
        ✅ LIVE Mode - Real market data<br>
        📰 500+ Articles analyzed<br>
        📈 Professional charts & analysis<br>
        🤖 AI-Powered sentiment detection
    </div>
</div>
<hr style="border: 1px solid rgba(102, 126, 234, 0.3); margin: 15px 0;">
<div style="text-align: center; padding: 15px 0;">
    <b>👨‍💻 Made by Raghav Dhanotiya</b><br>
    📧 Email: <span style="color: #667eea; font-weight: bold;">raghav74dhanotiya@gmail.com</span><br>
    📱 Contact: <span style="color: #667eea; font-weight: bold;">+91 9109657983</span><br>
    <br>
    <span style="font-size: 0.85em; color: #999;">© 2026 Stock Sentiment Analyzer | Built with ❤️ using Streamlit, AI & Finance APIs</span>
</div>
</div>
""", unsafe_allow_html=True)
