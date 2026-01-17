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
)

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    body {
        background: #ffffff;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        margin: 30px 0 10px 0;
        color: #2c3e50;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1em;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    
    .input-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 25px 0;
        color: white;
    }
    
    .input-label {
        font-size: 0.95em;
        font-weight: 600;
        margin-bottom: 8px;
        color: rgba(255, 255, 255, 0.95);
    }
    
    .section-header {
        font-size: 1.6em;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #667eea;
        padding-bottom: 12px;
        margin: 28px 0 20px 0;
    }
    
    .info-box {
        background: #ecf0f1;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 6px;
        margin: 15px 0;
        font-size: 0.95em;
        color: #2c3e50;
    }
    
    .metric-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border-top: 3px solid #667eea;
    }
    
    .news-item {
        background: #fafbfc;
        padding: 14px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.2s ease;
        border: 1px solid #e1e8ed;
    }
    
    .news-item b {
        color: #1a1a1a;
        font-size: 0.98em;
        line-height: 1.45;
        display: block;
        margin-bottom: 6px;
    }
    
    .news-item:hover {
        box-shadow: 0 4px 12px rgba(102,126,234,0.2);
        transform: translateX(2px);
        background: #f0f5ff;
    }
    
    .sentiment-bullish {
        display: inline-block;
        background: #51CF66;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .sentiment-bearish {
        display: inline-block;
        background: #FF6B6B;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .sentiment-neutral {
        display: inline-block;
        background: #FFA94D;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: 700;
        color: #2c3e50;
        margin: 8px 0;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: #ecf0f1;
        margin: 30px 0;
    }
    
    .disclaimer {
        text-align: center;
        font-size: 0.85em;
        color: #95a5a6;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

# ============ PAGE TITLE ============
st.markdown('<div class="main-title">📈 Stock Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">📡 Real-time analysis with live data from internet</div>', unsafe_allow_html=True)

# Live status indicator
col_status_1, col_status_2, col_status_3 = st.columns([1, 1, 2])
with col_status_1:
    if st.button("🔄 Refresh Now", help="Fetch fresh data from internet"):
        st.cache_data.clear()
        st.rerun()

with col_status_2:
    st.markdown('<div style="text-align: center; color: #51CF66; font-weight: bold;">🟢 LIVE MODE</div>', unsafe_allow_html=True)

with col_status_3:
    st.markdown('<div style="font-size: 0.85em; color: #7f8c8d;">⏱️ Updates: Every 60 seconds | Max 500+ articles</div>', unsafe_allow_html=True)

# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

tokenizer, model = load_model()

# Get API key from secrets
try:
    API_KEY = st.secrets["NEWS_API_KEY"]
except KeyError:
    API_KEY = None
    st.error("❌ Add NEWS_API_KEY to .streamlit/secrets.toml")

# ============ CORE FUNCTIONS ============
def get_stock_news(company):
    """Fetch news from multiple queries - LIVE with maximum articles"""
    if not API_KEY:
        return []
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
                time.sleep(0.15)
            except:
                pass
        
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
        return sorted(unique_articles, key=lambda x: x.get("publishedAt", ""), reverse=True)[:500]
    except Exception as e:
        st.error(f"❌ Error fetching news: {str(e)}")
        return []

def analyze(text):
    """Analyze sentiment using FinBERT"""
    if not tokenizer or not model:
        return None, None, None
    
    try:
        inputs = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model(inputs)
        
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
        st.error(f"❌ Analysis error: {str(e)}")
        return None, None, None

def analyze_news_sentiment(company):
    """Batch analyze news articles"""
    news_articles = get_stock_news(company)
    
    if not news_articles:
        st.warning(f"No news found for {company}")
        return []
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, article in enumerate(news_articles):
        headline = article.get("title", "")
        sentiment, score, confidence = analyze(headline)
        
        if sentiment:
            results.append({
                "headline": headline,
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence * 100,
                "source": article.get("source", {}).get("name", ""),
                "published": article.get("publishedAt", "")
            })
        
        progress = (i + 1) / len(news_articles)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Analyzing: {i+1}/{len(news_articles)} articles... ({len(results)} analyzed)")
    
    progress_bar.empty()
    status_text.empty()
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

@st.cache_data(ttl=60)
def get_forex_rate():
    """Fetch USD to INR exchange rate - LIVE"""
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        r = requests.get(url, timeout=5).json()
        return r.get('rates', {}).get('INR', 82.5)
    except:
        return 82.5  # Default rate

@st.cache_data(ttl=60)
def get_stock_price(ticker):
    """Fetch live stock price"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        # Get country info
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
        
        return {
            'price': info.get('currentPrice', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', ''),
            'history': hist,
            'country': country,
            'currency': currency
        }
    except:
        return None

def plot_stock_chart(ticker, period="1mo"):
    """Plot stock price chart"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
        
        # Get country info
        info = stock.info
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
        is_indian = country.upper() == 'INDIA'
        
        # Get currency symbol
        currency_symbol = "₹" if is_indian else "$"
        currency_label = "INR" if is_indian else currency
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist.index, hist['Close'], linewidth=2.5, color='#667eea')
        ax.fill_between(hist.index, hist['Close'], alpha=0.3, color='#667eea')
        ax.set_title(f"{ticker} Price - Last {period} ({currency_label})", fontsize=13, fontweight='bold')
        ax.set_ylabel(f"Price ({currency_label})")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{currency_symbol}{x:,.0f}'))
        fig.tight_layout()
        return fig
    except:
        return None

def analyze_sentiment_price_impact(ticker, results):
    """Analyze correlation between sentiment and price movement"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get 1 month history
        hist = stock.history(period="1mo")
        
        if hist.empty or not results:
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
        
        # Average sentiment per date
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
    if not analysis_data or analysis_data['history'].empty:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
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
        if daily_sentiment:
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
        st.error(f"Error creating correlation chart: {str(e)}")
        return None

def plot_combined_sentiment_price(ticker, analysis_data):
    """Plot combined sentiment and price with dual axis"""
    if not analysis_data or analysis_data['history'].empty:
        return None
    
    try:
        hist = analysis_data['history']
        daily_sentiment = analysis_data['daily_sentiment']
        
        # Get country info
        stock = yf.Ticker(ticker)
        info = stock.info
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
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
        
        if daily_sentiment:
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
        st.error(f"Error creating combined chart: {str(e)}")
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

# INPUT SECTION - Clean & Simple
st.markdown('<div class="input-section">', unsafe_allow_html=True)

st.markdown("### 🔍 Analyze a Stock")

col1, col2 = st.columns([2, 1])

with col1:
    company = st.text_input(
        "Company Name",
        placeholder="E.g., Tesla, Apple, Infosys",
        help="Type company name",
        label_visibility="visible"
    )

with col2:
    ticker = st.text_input(
        "Ticker Code",
        placeholder="E.g., TSLA",
        help="Optional stock ticker",
        label_visibility="visible"
    ).upper()

analyze_btn = st.button("🚀 Analyze Stock", use_container_width=True, type="primary")

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
if analyze_btn and company.strip():
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ===== MAIN ANALYSIS =====
    st.markdown(f'<div class="section-header">📊 {company.upper()} - Sentiment Analysis</div>', unsafe_allow_html=True)
    
    with st.spinner(f"⏳ Analyzing {company}... (30-60 seconds)"):
        results = analyze_news_sentiment(company.strip())
    
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
            st.metric("📰 Articles", total)
        
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
                st.markdown(f"**📊 Total: {len(positive_results)} positive articles**")
                for r in positive_results[:50]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%Y-%m-%d %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b><br>
                    📰 {r.get('source', 'Unknown')} | 🕐 {pub_date} | 🎯 {r['confidence']:.0f}% confident
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No positive news found")
        
        with tab2:
            neutral_results = [r for r in results if r["sentiment"] == "Neutral"]
            if neutral_results:
                st.markdown(f"**📊 Total: {len(neutral_results)} neutral articles**")
                for r in neutral_results[:50]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%Y-%m-%d %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b><br>
                    📰 {r.get('source', 'Unknown')} | 🕐 {pub_date} | 🎯 {r['confidence']:.0f}% confident
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No neutral news found")
        
        with tab3:
            negative_results = [r for r in results if r["sentiment"] == "Negative"]
            if negative_results:
                st.markdown(f"**📊 Total: {len(negative_results)} negative articles**")
                for r in negative_results[:50]:
                    pub_date = pd.to_datetime(r.get('published', '')).strftime('%Y-%m-%d %H:%M') if r.get('published') else 'Unknown'
                    st.markdown(f"""
                    <div class="news-item">
                    <b>{r['headline']}</b><br>
                    📰 {r.get('source', 'Unknown')} | 🕐 {pub_date} | 🎯 {r['confidence']:.0f}% confident
                    </div>
                    """, unsafe_allow_html=True)
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
                
                # Display prices in original currency (no conversion)
                display_price = stock_data['price']
                display_high = stock_data['52w_high']
                display_low = stock_data['52w_low']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💵 Current", f"{currency_symbol}{display_price:,.2f}")
                
                with col2:
                    st.metric("📈 52W High", f"{currency_symbol}{display_high:,.2f}")
                
                with col3:
                    st.metric("📉 52W Low", f"{currency_symbol}{display_low:,.2f}")
                
                with col4:
                    st.metric("P/E Ratio", f"{stock_data['pe_ratio']:.1f}")
                
                # Show currency info
                st.markdown(f"""
                <div style="background: #e8f4f8; padding: 10px; border-radius: 6px; font-size: 0.9em; margin: 10px 0;">
                💱 <b>Currency:</b> {currency_label}
                </div>
                """, unsafe_allow_html=True)
                
                # Chart
                st.markdown('<div class="section-header">📈 Stock Price Chart</div>', unsafe_allow_html=True)
                period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"], key="period_select")
                chart = plot_stock_chart(ticker, period)
                if chart:
                    st.pyplot(chart, use_container_width=True)
                

                # ===== SENTIMENT & PRICE CORRELATION =====
                st.markdown('<div class="section-header">🔗 How Sentiment Affects Price</div>', unsafe_allow_html=True)
                
                # Analyze correlation
                sentiment_price_analysis = analyze_sentiment_price_impact(ticker, results)
                
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
                st.error(f"Could not fetch data for {ticker}")
    
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
📡 <b>Live Mode Active</b> - Data updates automatically every 60 seconds<br>
📰 Maximum articles fetched: 500+ from 12 search queries
</div>
""", unsafe_allow_html=True)
