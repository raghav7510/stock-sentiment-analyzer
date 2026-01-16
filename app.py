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
import csv
from io import StringIO
import yfinance as yf
import time

warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

st.set_page_config(
    page_title="Live Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

st.markdown("# 📈 Live Stock Sentiment Analyzer")
st.markdown("Analyze **real-time stock market news sentiment** using **FinBERT (financial NLP model)** powered by AI.")

# Add quick guide in sidebar
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Enter Company Name** - Type Tesla, Apple, Infosys, etc.
    2. **Click "Start Analysis"** - Wait 30-60 seconds
    3. **View Results** - See sentiment breakdown
    4. **Optional: Add Stock Ticker** - Get live prices
    5. **Check Tabs** - View by sentiment type
    
    ### 💡 Tips
    - **Positive articles** = Stock likely to go UP 📈
    - **Negative articles** = Stock likely to go DOWN 📉
    - **High confidence** = More reliable signal
    - **100+ articles** = Better accuracy
    """)
    st.markdown("---")

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
    st.warning("⚠️ NEWS_API_KEY not configured. Please add it to .streamlit/secrets.toml")

def get_stock_news(company):
    """Fetch more comprehensive news from multiple queries"""
    if not API_KEY:
        st.error("❌ API Key not configured")
        return []
    try:
        all_articles = []
        
        # Multiple search queries for better coverage
        queries = [
            company,                          # Direct company name
            f"{company} stock",               # Stock specific
            f"{company} shares",              # Shares
            f"{company} trading",             # Trading
            f"{company} earnings",            # Earnings
            f"{company} price"                # Price
        ]
        
        for query in queries:
            try:
                # Get up to 100 articles per query (higher than before)
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=50&apiKey={API_KEY}"
                r = requests.get(url, timeout=5).json()
                
                if r.get("status") == "ok":
                    articles = r.get("articles", [])
                    all_articles.extend(articles)
                time.sleep(0.2)  # Avoid rate limiting
            except:
                pass
        
        # Remove duplicates and limit to unique articles
        seen_titles = set()
        unique_articles = []
        for a in all_articles:
            title = a.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append({
                    "title": title,
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "published": a.get("publishedAt", ""),
                    "url": a.get("url", "")
                })
        
        # Return sorted by date (newest first) and limit to top 100
        return sorted(unique_articles, key=lambda x: x["published"], reverse=True)[:100]
        
    except Exception as e:
        st.error(f"❌ Network error: {str(e)}")
        return []

def analyze(text):
    if not tokenizer or not model:
        st.error("❌ Model not loaded")
        return "Error", 0, 0
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            out = model(**enc)
        probs = F.softmax(out.logits, dim=1)[0]
        labels = ["Negative","Neutral","Positive"]
        sentiment = labels[torch.argmax(probs)]
        score = {"Positive":1,"Neutral":0,"Negative":-1}[sentiment]
        confidence = round(torch.max(probs).item()*100, 2)
        return sentiment, score, confidence
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        return "Error", 0, 0

def analyze_news_sentiment(company):
    """Analyze news for a company and return results as list of dicts"""
    news_articles = get_stock_news(company)
    if not news_articles:
        return []
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, article in enumerate(news_articles):
        headline = article["title"]
        sentiment, score, confidence = analyze(headline)
        results.append({
            "headline": headline,
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "source": article.get("source", "Unknown"),
            "published": article.get("published", "")
        })
        progress_bar.progress((idx + 1) / len(news_articles))
    
    progress_bar.empty()
    return results

def overall_sentiment(results):
    """Calculate average sentiment score"""
    if not results:
        return 0
    scores = [r["score"] for r in results]
    return sum(scores) / len(scores)

# ========== NEW FUNCTIONS ==========

def sentiment_trend(company, num_periods=5):
    """Track sentiment changes over multiple time periods"""
    try:
        trends = []
        for i in range(num_periods):
            results = analyze_news_sentiment(company)
            if results:
                score = overall_sentiment(results)
                trends.append({
                    "period": i + 1,
                    "score": score,
                    "timestamp": datetime.now() - timedelta(days=i)
                })
        return trends
    except Exception as e:
        st.error(f"❌ Trend analysis error: {str(e)}")
        return []

def extract_keywords(texts, top_n=10):
    """Extract most common words from headlines"""
    try:
        # Combine all text
        all_text = " ".join(texts).lower()
        # Split into words and remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "in", "of", "for", "with", "on", "by", "at", "from"}
        words = [w for w in all_text.split() if len(w) > 3 and w not in stop_words]
        
        # Count and return top words
        word_freq = Counter(words).most_common(top_n)
        return dict(word_freq)
    except Exception as e:
        st.error(f"❌ Keyword extraction error: {str(e)}")
        return {}

def calculate_risk_score(results):
    """Calculate volatility based on confidence scores"""
    try:
        if not results:
            return 0
        
        confidences = [r["confidence"] for r in results]
        variance = np.var(confidences)
        # Normalize risk score (0-100)
        risk_score = min(100, (variance / 100) * 100)
        return round(risk_score, 2)
    except Exception as e:
        st.error(f"❌ Risk calculation error: {str(e)}")
        return 0

def analyze_portfolio(companies):
    """Analyze sentiment for multiple companies"""
    try:
        portfolio_results = []
        for company in companies:
            if company.strip():
                results = analyze_news_sentiment(company.strip())
                if results:
                    score = overall_sentiment(results)
                    portfolio_results.append({
                        "company": company.strip(),
                        "score": score,
                        "articles": len(results),
                        "sentiment": "Bullish" if score > 0.2 else "Bearish" if score < -0.2 else "Neutral"
                    })
        return portfolio_results
    except Exception as e:
        st.error(f"❌ Portfolio analysis error: {str(e)}")
        return []

def sentiment_pie_chart(df):
    """Create pie chart of sentiment distribution"""
    try:
        sentiment_counts = df["Sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#FF6B6B", "#FFA94D", "#51CF66"]
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%", 
               colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title("Sentiment Distribution", fontsize=14, weight='bold')
        return fig
    except Exception as e:
        st.error(f"❌ Pie chart error: {str(e)}")
        return None

def generate_wordcloud(texts, sentiment_type="all"):
    """Generate word cloud from texts"""
    if not HAS_WORDCLOUD:
        st.warning("⚠️ WordCloud library not installed. Install with: pip install wordcloud")
        return None
    
    try:
        text = " ".join(texts)
        if not text.strip():
            return None
        
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='RdYlGn' if sentiment_type == "mixed" else 'Greens').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Word Cloud - {sentiment_type.capitalize()}", fontsize=14, weight='bold')
        return fig
    except Exception as e:
        st.error(f"❌ WordCloud error: {str(e)}")
        return None

def export_to_csv(results, company_name):
    """Convert results to CSV string for download"""
    try:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Company", "Headline", "Sentiment", "Score", "Confidence (%)", "Timestamp"])
        
        for r in results:
            writer.writerow([
                company_name,
                r.get("headline", ""),
                r.get("sentiment", ""),
                r.get("score", ""),
                r.get("confidence", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
        
        return output.getvalue()
    except Exception as e:
        st.error(f"❌ Export error: {str(e)}")
        return None

def find_correlated_companies(reference_company, all_companies):
    """Find companies with similar sentiment patterns"""
    try:
        ref_results = analyze_news_sentiment(reference_company)
        if not ref_results:
            return []
        
        ref_score = overall_sentiment(ref_results)
        correlations = []
        
        for company in all_companies:
            if company.strip() and company.strip() != reference_company.strip():
                comp_results = analyze_news_sentiment(company.strip())
                if comp_results:
                    comp_score = overall_sentiment(comp_results)
                    # Calculate correlation (similarity in sentiment)
                    correlation = 1 - abs(ref_score - comp_score)
                    correlations.append({
                        "company": company.strip(),
                        "score": comp_score,
                        "correlation": round(correlation, 2)
                    })
        
        return sorted(correlations, key=lambda x: x["correlation"], reverse=True)
    except Exception as e:
        st.error(f"❌ Correlation analysis error: {str(e)}")
        return []

def check_sentiment_alert(score, threshold=0.7):
    """Check if sentiment crosses alert threshold"""
    try:
        if score > threshold:
            return {
                "alert": True,
                "type": "EXTREME_BULLISH",
                "message": f"🚀 Extreme bullish sentiment detected! (Score: {score:.2f})"
            }
        elif score < -threshold:
            return {
                "alert": True,
                "type": "EXTREME_BEARISH",
                "message": f"⚠️ Extreme bearish sentiment detected! (Score: {score:.2f})"
            }
        return {"alert": False, "type": None, "message": ""}
    except Exception as e:
        st.error(f"❌ Alert check error: {str(e)}")
        return {"alert": False, "type": None, "message": ""}

def calculate_impact_score(confidence, sentiment_score):
    """Calculate potential market impact score"""
    try:
        # Combine confidence and sentiment magnitude
        sentiment_magnitude = abs(sentiment_score)
        impact = (confidence / 100) * sentiment_magnitude * 100
        
        if impact > 50:
            impact_level = "🔴 HIGH"
        elif impact > 25:
            impact_level = "🟡 MEDIUM"
        else:
            impact_level = "🟢 LOW"
        
        return {
            "score": round(impact, 2),
            "level": impact_level,
            "description": f"Impact Score: {impact:.2f}/100"
        }
    except Exception as e:
        st.error(f"❌ Impact calculation error: {str(e)}")
        return {"score": 0, "level": "🟢 LOW", "description": "Error calculating impact"}

# ========== STOCK PRICE FUNCTIONS ==========

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_price(ticker):
    """Fetch live stock price and info"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        
        if history.empty:
            return None
        
        current_price = info.get("currentPrice") or info.get("previousClose")
        
        return {
            "ticker": ticker,
            "price": current_price,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "company_name": info.get("longName")
        }
    except Exception as e:
        st.error(f"❌ Error fetching stock data for {ticker}: {str(e)}")
        return None

def get_stock_history(ticker, period="1mo"):
    """Get historical stock price data"""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        return history
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {str(e)}")
        return None

def calculate_price_change(ticker, period="1d"):
    """Calculate price change percentage"""
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        if len(history) < 2:
            return None
        
        old_price = history.iloc[0]['Close']
        new_price = history.iloc[-1]['Close']
        change = ((new_price - old_price) / old_price) * 100
        
        return {
            "old_price": old_price,
            "new_price": new_price,
            "change_percent": round(change, 2),
            "change_amount": round(new_price - old_price, 2)
        }
    except Exception as e:
        st.error(f"❌ Error calculating price change: {str(e)}")
        return None

def plot_stock_chart(ticker, period="1mo"):
    """Create interactive stock price chart"""
    try:
        history = get_stock_history(ticker, period)
        
        if history is None or history.empty:
            st.warning(f"No data available for {ticker}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history.index, history['Close'], linewidth=2, color='steelblue', label='Close Price')
        ax.fill_between(history.index, history['Close'], alpha=0.3)
        ax.set_title(f"{ticker} - Stock Price ({period})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        st.error(f"❌ Error plotting chart: {str(e)}")
        return None

def correlate_sentiment_price(sentiment_score, ticker):
    """Compare sentiment sentiment vs price movement"""
    try:
        price_change = calculate_price_change(ticker, "1mo")
        
        if not price_change:
            return None
        
        price_direction = "📈 UP" if price_change["change_percent"] > 0 else "📉 DOWN"
        sentiment_direction = "📈 BULLISH" if sentiment_score > 0.2 else "📉 BEARISH" if sentiment_score < -0.2 else "⚖️ NEUTRAL"
        
        # Check if sentiment matches price direction
        match = ((sentiment_score > 0.2 and price_change["change_percent"] > 0) or 
                (sentiment_score < -0.2 and price_change["change_percent"] < 0) or
                (abs(sentiment_score) <= 0.2 and abs(price_change["change_percent"]) <= 2))
        
        return {
            "sentiment_direction": sentiment_direction,
            "price_direction": price_direction,
            "price_change": price_change["change_percent"],
            "match": match,
            "conflict": not match
        }
    except Exception as e:
        st.error(f"❌ Error correlating sentiment/price: {str(e)}")
        return None

def get_sentiment_price_alert(sentiment_score, ticker):
    """Generate alert if sentiment conflicts with price"""
    try:
        correlation = correlate_sentiment_price(sentiment_score, ticker)
        
        if not correlation:
            return None
        
        if correlation["conflict"]:
            if correlation["sentiment_direction"] == "📈 BULLISH" and correlation["price_direction"] == "📉 DOWN":
                return {
                    "alert": True,
                    "type": "BULLISH_DIVERGENCE",
                    "message": "⚠️ **Bullish Sentiment but Price Falling!** Possible reversal signal or market correction ahead.",
                    "severity": "HIGH"
                }
            elif correlation["sentiment_direction"] == "📉 BEARISH" and correlation["price_direction"] == "📈 UP":
                return {
                    "alert": True,
                    "type": "BEARISH_DIVERGENCE",
                    "message": "⚠️ **Bearish Sentiment but Price Rising!** Strong bullish momentum overcoming negative sentiment.",
                    "severity": "HIGH"
                }
        else:
            return {
                "alert": False,
                "type": "ALIGNED",
                "message": "✅ Sentiment and price movement are aligned.",
                "severity": "LOW"
            }
    except Exception as e:
        st.error(f"❌ Error generating alert: {str(e)}")
        return None

def plot_price_vs_sentiment(ticker, period="1mo"):
    """Plot price movement and overlay sentiment score"""
    try:
        history = get_stock_history(ticker, period)
        
        if history is None or history.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(history.index, history['Close'], linewidth=2, color='steelblue', label='Stock Price')
        ax.fill_between(history.index, history['Close'], alpha=0.2, color='steelblue')
        
        ax.set_title(f"{ticker} - Price vs Sentiment Correlation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        return fig
    except Exception as e:
        st.error(f"❌ Error plotting correlation: {str(e)}")
        return None

def get_multiple_stocks_data(tickers):
    """Fetch data for multiple stocks"""
    try:
        stocks_data = []
        for ticker in tickers:
            if ticker.strip():
                data = get_stock_price(ticker.strip().upper())
                if data:
                    stocks_data.append(data)
        return stocks_data
    except Exception as e:
        st.error(f"❌ Error fetching multiple stocks: {str(e)}")
        return []

# ========== TRADING STRATEGY FUNCTIONS ==========

def get_trading_recommendation(sentiment_score, price_change_percent, confidence):
    """Get trading strategy recommendation based on sentiment and price"""
    try:
        recommendations = []
        
        # INTRADAY TRADING
        if confidence > 70:  # High confidence
            if sentiment_score > 0.5 and price_change_percent > 0:
                recommendations.append({
                    "strategy": "Intraday Trading",
                    "action": "BUY (Momentum Play)",
                    "confidence": "🟢 HIGH",
                    "description": "Strong bullish sentiment with positive price action. Good for intraday momentum traders.",
                    "risk": "Medium - High volatility possible",
                    "target_hold": "Few hours to 1 day",
                    "sl_percent": 2.0,
                    "tp_percent": 5.0
                })
            elif sentiment_score < -0.5 and price_change_percent < 0:
                recommendations.append({
                    "strategy": "Intraday Trading",
                    "action": "SELL/SHORT",
                    "confidence": "🟢 HIGH",
                    "description": "Strong bearish sentiment with negative price action. Good for short selling.",
                    "risk": "High - Unlimited in case of short squeeze",
                    "target_hold": "Few hours to 1 day",
                    "sl_percent": 2.0,
                    "tp_percent": 5.0
                })
        
        # SWING TRADING (2-5 days)
        if confidence > 60:
            if sentiment_score > 0.3:
                recommendations.append({
                    "strategy": "Swing Trading",
                    "action": "BUY",
                    "confidence": "🟡 MEDIUM-HIGH",
                    "description": "Positive sentiment for medium-term gains. Good for 2-5 day holding.",
                    "risk": "Medium",
                    "target_hold": "2-5 days",
                    "sl_percent": 3.0,
                    "tp_percent": 8.0
                })
            elif sentiment_score < -0.3:
                recommendations.append({
                    "strategy": "Swing Trading",
                    "action": "SELL/SHORT",
                    "confidence": "🟡 MEDIUM-HIGH",
                    "description": "Negative sentiment for downside. Good for bearish swing traders.",
                    "risk": "Medium-High",
                    "target_hold": "2-5 days",
                    "sl_percent": 3.0,
                    "tp_percent": 8.0
                })
        
        # LONG-TERM INVESTING (>1 month)
        if confidence > 50:
            if sentiment_score > 0.2:
                recommendations.append({
                    "strategy": "Long-term Investing",
                    "action": "BUY & HOLD",
                    "confidence": "🟠 MEDIUM",
                    "description": "Bullish sentiment over time. Suitable for buy-and-hold investors.",
                    "risk": "Low-Medium",
                    "target_hold": "3+ months",
                    "sl_percent": 7.0,
                    "tp_percent": 15.0
                })
            elif sentiment_score < -0.2:
                recommendations.append({
                    "strategy": "Long-term Investing",
                    "action": "AVOID / WAIT",
                    "confidence": "🟠 MEDIUM",
                    "description": "Bearish sentiment. Wait for reversal or invest elsewhere.",
                    "risk": "Medium",
                    "target_hold": "3+ months",
                    "sl_percent": 10.0,
                    "tp_percent": 20.0
                })
        
        # F&O (FUTURES & OPTIONS)
        if confidence > 75:
            if sentiment_score > 0.6:
                recommendations.append({
                    "strategy": "F&O - Call Options",
                    "action": "BUY CALL / BULL CALL SPREAD",
                    "confidence": "🟢 HIGH",
                    "description": "Strong bullish sentiment. Good for leveraged bullish bets using options.",
                    "risk": "🔴 HIGH - Leverage risk",
                    "target_hold": "Few days to 2 weeks",
                    "leverage": "10-50x possible",
                    "warning": "⚠️ Options expire - time decay applies"
                })
            elif sentiment_score < -0.6:
                recommendations.append({
                    "strategy": "F&O - Put Options",
                    "action": "BUY PUT / BEAR PUT SPREAD",
                    "confidence": "🟢 HIGH",
                    "description": "Strong bearish sentiment. Good for leveraged bearish bets using options.",
                    "risk": "🔴 HIGH - Leverage risk",
                    "target_hold": "Few days to 2 weeks",
                    "leverage": "10-50x possible",
                    "warning": "⚠️ Options expire - time decay applies"
                })
        
        return recommendations
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return []

def calculate_position_sizing(capital, risk_percent, stop_loss_percent):
    """Calculate proper position size for risk management"""
    try:
        risk_amount = capital * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_percent / 100)
        return {
            "capital": capital,
            "risk_amount": round(risk_amount, 2),
            "position_size": round(position_size, 2),
            "risk_percent": risk_percent,
            "sl_percent": stop_loss_percent
        }
    except Exception as e:
        st.error(f"❌ Error calculating position size: {str(e)}")
        return None

def calculate_profit_loss(entry_price, exit_price, quantity):
    """Calculate P&L for a trade"""
    try:
        profit = (exit_price - entry_price) * quantity
        profit_percent = ((exit_price - entry_price) / entry_price) * 100
        
        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "profit": round(profit, 2),
            "profit_percent": round(profit_percent, 2),
            "status": "✅ PROFIT" if profit > 0 else "❌ LOSS" if profit < 0 else "➡️ BREAKEVEN"
        }
    except Exception as e:
        st.error(f"❌ Error calculating P&L: {str(e)}")
        return None

def get_investment_growth_strategy(sentiment_score, investment_horizon, capital):
    """Get investment growth strategy based on sentiment and time horizon"""
    try:
        strategies = []
        
        if investment_horizon == "1-3 months":
            strategies.append({
                "name": "Short-term Growth",
                "allocation": "High Risk (70-80% stocks, 20-30% debt/cash)",
                "approach": "Active trading, intraday/swing trading",
                "expected_return": "10-20% (if successful)",
                "risk": "🔴 HIGH",
                "tools": ["Zerodha - Intraday trading", "TradingView - Chart analysis", "Grow - Tracking"]
            })
        
        elif investment_horizon == "3-12 months":
            if sentiment_score > 0.2:
                strategies.append({
                    "name": "Bullish Accumulation",
                    "allocation": "Balanced (50-60% stocks, 40-50% debt/cash)",
                    "approach": "Buy on dips, SIP in good stocks, F&O covered calls",
                    "expected_return": "15-30% (annual)",
                    "risk": "🟡 MEDIUM",
                    "tools": ["Zerodha - SIP setup", "Grow - Portfolio tracking", "NSE - Stock screening"]
                })
            else:
                strategies.append({
                    "name": "Defensive Strategy",
                    "allocation": "Conservative (30-40% stocks, 60-70% debt/cash)",
                    "approach": "Hold cash, wait for reversal, defensive stocks",
                    "expected_return": "5-10% (annual)",
                    "risk": "🟢 LOW",
                    "tools": ["Zerodha - Watchlist", "Grow - Alerts", "SEBI - Mutual funds"]
                })
        
        else:  # 1+ year
            strategies.append({
                "name": "Long-term Wealth Building",
                "allocation": "Growth (70-80% stocks, 20-30% debt/bonds)",
                "approach": "SIP, buy quality stocks, reinvest dividends",
                "expected_return": "12-18% (CAGR)",
                "risk": "🟡 MEDIUM",
                "tools": ["Zerodha - SIP", "Grow - Goal tracking", "SEBI mutual funds", "Dividend reinvestment"]
            })
        
        return strategies
    except Exception as e:
        st.error(f"❌ Error generating strategy: {str(e)}")
        return []

st.divider()

# ============ SECTION 1: SINGLE COMPANY ANALYSIS ============
st.markdown("## 🎯 Quick Stock Analysis (Easiest Way to Start)")

# Simple input section
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    company = st.text_input(
        "📝 Enter Company Name",
        placeholder="E.g., Tesla, Apple, Infosys, Reliance",
        help="Type a company name and we'll analyze all available news"
    )
with col2:
    st.write("")
    analyze_button = st.button("🔍 Start Analysis", use_container_width=True, key="analyze_btn")
with col3:
    st.write("")
    stock_ticker = st.text_input("Stock Code", placeholder="TSLA", help="Optional: 3-4 letter ticker", key="ticker_input").upper()

if analyze_button and company.strip():
    st.info(f"📡 Analyzing {company}... This may take 30-60 seconds (gathering 100+ articles). Please wait...", icon="⏳")
    
    with st.spinner(f"🔍 Fetching latest news articles for {company}..."):
        news_articles = get_stock_news(company.strip())
    
    if news_articles:
        st.success(f"✅ Found {len(news_articles)} articles!", icon="📰")
        
        # Run sentiment analysis
        with st.spinner("🧠 Analyzing sentiment..."):
            results = analyze_news_sentiment(company.strip())
        
        if results:
            # ===== OVERALL SENTIMENT DISPLAY =====
            st.divider()
            st.markdown("### 📊 Overall Sentiment Score")
            
            overall_score = overall_sentiment(results)
            positive_count = sum(1 for r in results if r["sentiment"] == "Positive")
            negative_count = sum(1 for r in results if r["sentiment"] == "Negative")
            neutral_count = sum(1 for r in results if r["sentiment"] == "Neutral")
            total = len(results)
            
            # Display metrics with emojis
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📈 Positive", f"{positive_count}/{total}", f"+{(positive_count/total*100):.0f}%")
            
            with col2:
                st.metric("📉 Negative", f"{negative_count}/{total}", f"-{(negative_count/total*100):.0f}%")
            
            with col3:
                st.metric("⚖️ Neutral", f"{neutral_count}/{total}", f"{(neutral_count/total*100):.0f}%")
            
            with col4:
                avg_confidence = np.mean([r["confidence"] for r in results])
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%", "")
            
            with col5:
                if overall_score > 0.5:
                    st.metric("🚀 SIGNAL", "BULLISH", "")
                elif overall_score < -0.5:
                    st.metric("🔴 SIGNAL", "BEARISH", "")
                else:
                    st.metric("⚖️ SIGNAL", "NEUTRAL", "")
            
            # Sentiment gauge visualization
            st.divider()
            col_gauge = st.columns(1)[0]
            with col_gauge:
                st.markdown("### 📍 Sentiment Gauge")
                # Create gauge visualization
                gauge_fig, gauge_ax = plt.subplots(figsize=(10, 2))
                gauge_ax.barh([0], [overall_score], color=['red' if overall_score < -0.3 else 'orange' if overall_score < 0.3 else 'green'], height=0.3)
                gauge_ax.set_xlim([-1, 1])
                gauge_ax.set_ylim([-0.5, 0.5])
                gauge_ax.axvline(0, color='black', linestyle='-', linewidth=2)
                gauge_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                gauge_ax.set_xticklabels(['Extreme Bearish', 'Bearish', 'Neutral', 'Bullish', 'Extreme Bullish'])
                gauge_ax.set_yticks([])
                gauge_ax.text(overall_score, 0, f"  {overall_score:.2f}", ha='left', va='center', fontsize=12, fontweight='bold')
                gauge_fig.tight_layout()
                st.pyplot(gauge_fig)
                plt.close()
            
            # ===== SENTIMENT LEGEND =====
            st.divider()
            st.markdown("### 📍 What the Score Means")
            col_legend1, col_legend2, col_legend3, col_legend4, col_legend5 = st.columns(5)
            
            with col_legend1:
                st.markdown("**🚀 +0.7 to +1.0**\nExtreme Bullish\n(Strong BUY)")
            with col_legend2:
                st.markdown("**📈 +0.3 to +0.7**\nBullish\n(BUY)")
            with col_legend3:
                st.markdown("**⚖️ -0.3 to +0.3**\nNeutral\n(HOLD)")
            with col_legend4:
                st.markdown("**📉 -0.7 to -0.3**\nBearish\n(SELL)")
            with col_legend5:
                st.markdown("**🔴 -1.0 to -0.7**\nExtreme Bearish\n(SELL/SHORT)")
            
            # ===== ARTICLE-BY-ARTICLE BREAKDOWN =====
            st.divider()
            st.markdown("### 📰 News Sentiment Breakdown")
            # Create tabs for different sentiments
            tab1, tab2, tab3 = st.tabs([
                f"🟢 Positive ({positive_count})",
                f"🔴 Negative ({negative_count})", 
                f"⚪ Neutral ({neutral_count})"
            ])
            
            with tab1:
                positive_results = [r for r in results if r["sentiment"] == "Positive"]
                if positive_results:
                    for i, r in enumerate(positive_results[:20], 1):  # Show top 20
                        confidence_color = "🟢" if r["confidence"] > 80 else "🟡" if r["confidence"] > 60 else "🟠"
                        st.markdown(f"""
                        **{i}. {r['headline'][:100]}...**
                        - 📰 Source: {r.get('source', 'Unknown')}
                        - 🎯 Confidence: {confidence_color} {r['confidence']:.1f}%
                        """)
                        st.divider()
                else:
                    st.info("No positive articles found")
            
            with tab2:
                negative_results = [r for r in results if r["sentiment"] == "Negative"]
                if negative_results:
                    for i, r in enumerate(negative_results[:20], 1):
                        confidence_color = "🟢" if r["confidence"] > 80 else "🟡" if r["confidence"] > 60 else "🟠"
                        st.markdown(f"""
                        **{i}. {r['headline'][:100]}...**
                        - 📰 Source: {r.get('source', 'Unknown')}
                        - 🎯 Confidence: {confidence_color} {r['confidence']:.1f}%
                        """)
                        st.divider()
                else:
                    st.info("No negative articles found")
            
            with tab3:
                neutral_results = [r for r in results if r["sentiment"] == "Neutral"]
                if neutral_results:
                    for i, r in enumerate(neutral_results[:20], 1):
                        confidence_color = "🟢" if r["confidence"] > 80 else "🟡" if r["confidence"] > 60 else "🟠"
                        st.markdown(f"""
                        **{i}. {r['headline'][:100]}...**
                        - 📰 Source: {r.get('source', 'Unknown')}
                        - 🎯 Confidence: {confidence_color} {r['confidence']:.1f}%
                        """)
                        st.divider()
                else:
                    st.info("No neutral articles found")
            
            # ===== TOP KEYWORDS =====
            st.divider()
            st.markdown("### 🔑 Top Keywords from News")
            
            all_headlines = [r["headline"] for r in results]
            keywords = extract_keywords(all_headlines, top_n=15)
            
            if keywords:
                # Create keyword visualization
                keyword_df = pd.DataFrame(list(keywords.items()), columns=["Keyword", "Count"])
                keyword_df = keyword_df.sort_values("Count", ascending=True)
                
                col_kw1, col_kw2 = st.columns([2, 1])
                with col_kw1:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(keyword_df["Keyword"], keyword_df["Count"], color='steelblue')
                    ax.set_xlabel("Frequency")
                    ax.set_title("Top Keywords Frequency")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with col_kw2:
                    st.markdown("**Top 5 Keywords:**")
                    for i, (kw, count) in enumerate(list(keywords.items())[:5], 1):
                        st.markdown(f"**{i}. {kw}** - {count} mentions")
            
            # ===== LIVE STOCK PRICE (if ticker provided) =====
            if stock_ticker:
                st.divider()
                st.markdown(f"### 💰 Live Stock Price for {stock_ticker}")
                
                stock_data = get_stock_price(stock_ticker)
                
                if stock_data:
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💵 Current Price", f"${stock_data['price']:.2f}" if stock_data['price'] else "N/A")
                    
                    with col2:
                        price_change = calculate_price_change(stock_ticker, "1d")
                        if price_change:
                            delta_val = f"{price_change['change_percent']:.2f}%"
                            st.metric("📊 1D Change", delta_val, f"${price_change['change']:.2f}")
                        else:
                            st.metric("📊 1D Change", "N/A")
                    
                    with col3:
                        st.metric("📈 52W High", f"${stock_data['52w_high']:.2f}" if stock_data['52w_high'] else "N/A")
                    
                    with col4:
                        st.metric("📉 52W Low", f"${stock_data['52w_low']:.2f}" if stock_data['52w_low'] else "N/A")
                    
                    # More details
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("P/E Ratio", f"{stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "N/A")
                    
                    with col_info2:
                        market_cap = stock_data['market_cap']
                        if market_cap:
                            market_cap_b = market_cap / 1e9
                            st.metric("Market Cap", f"${market_cap_b:.1f}B")
                        else:
                            st.metric("Market Cap", "N/A")
                    
                    with col_info3:
                        st.metric("Industry", stock_data['sector'] or "N/A")
                    
                    # Stock price chart
                    st.subheader("📊 Stock Price Chart")
                    period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y"], key="stock_period")
                
                chart = plot_stock_chart(stock_ticker, period)
                if chart:
                    st.pyplot(chart, use_container_width=True)
                
                st.divider()
                
                # ===== SENTIMENT vs PRICE CORRELATION =====
                st.subheader("🔄 Sentiment vs Price Correlation")
                
                avg_sentiment = overall_sentiment(results)
                correlation = correlate_sentiment_price(avg_sentiment, stock_ticker)
                alert = get_sentiment_price_alert(avg_sentiment, stock_ticker)
                
                if correlation:
                    col_corr1, col_corr2, col_corr3 = st.columns(3)
                    
                    with col_corr1:
                        st.write(f"**Sentiment**: {correlation['sentiment_direction']}")
                    
                    with col_corr2:
                        st.write(f"**Price**: {correlation['price_direction']}")
                    
                    with col_corr3:
                        status = "✅ Aligned" if correlation['match'] else "⚠️ Diverged"
                        st.write(f"**Status**: {status}")
                    
                    if alert:
                        if alert["severity"] == "HIGH":
                            st.warning(alert["message"])
                        else:
                            st.success(alert["message"])
            else:
                st.error(f"❌ Could not fetch data for {stock_ticker}")
        
        st.divider()
        
        # Create dataframe from results for visualization
        df = pd.DataFrame(results)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            st.metric("Positive", (df["sentiment"] == "Positive").sum())
        with col3:
            st.metric("Neutral", (df["sentiment"] == "Neutral").sum())
        with col4:
            st.metric("Negative", (df["sentiment"] == "Negative").sum())
        
        # Visualization
        st.subheader("Sentiment Score Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["red" if s == "Negative" else "orange" if s == "Neutral" else "green" for s in df["sentiment"]]
        ax.bar(range(len(df)), df["score"], color=colors, alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Article Index")
        ax.set_ylabel("Sentiment Score")
        ax.set_title(f"Sentiment Scores for {company}")
        ax.set_ylim(-1.5, 1.5)
        st.pyplot(fig, use_container_width=True)
        
        # Average sentiment
        avg_score = df["score"].mean()
        st.subheader("Overall Sentiment")
        if avg_score > 0.3:
            st.success(f"📈 **Bullish** (Average Score: {avg_score:.2f})")
        elif avg_score < -0.3:
            st.error(f"📉 **Bearish** (Average Score: {avg_score:.2f})")
        else:
            st.info(f"⚖️ **Neutral** (Average Score: {avg_score:.2f})")

elif analyze_button:
    st.warning("⚠️ Please enter a company name")

elif analyze_button:
    st.warning("⚠️ Please enter a company name")

st.divider()
st.caption("⚠️ **Educational project only.** Not financial advice.")
st.divider()

# ============ SECTION 1.5: STOCK WATCHLIST SCREENER ============
st.markdown("### 📋 Stock Watchlist Screener")

watchlist_input = st.text_area(
    "Enter stock tickers (comma-separated)",
    placeholder="TSLA, AAPL, MSFT, NVDA, GOOGL",
    height=40
)

watchlist_btn = st.button("🔍 Scan Watchlist", use_container_width=True)

if watchlist_btn and watchlist_input:
    tickers = [t.strip().upper() for t in watchlist_input.split(",")]
    
    with st.spinner("Fetching stock data..."):
        stocks_data = get_multiple_stocks_data(tickers)
    
    if stocks_data:
        # Create DataFrame
        watchlist_df = pd.DataFrame(stocks_data)
        
        # Display watchlist table
        st.subheader("Watchlist Summary")
        display_cols = ["ticker", "price", "currency", "pe_ratio", "52w_high", "52w_low"]
        display_df = watchlist_df[display_cols].copy()
        display_df.columns = ["Ticker", "Price", "Currency", "P/E Ratio", "52W High", "52W Low"]
        st.dataframe(display_df, use_container_width=True)
        
        # Price chart comparison
        st.subheader("Price Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(watchlist_df["ticker"], watchlist_df["price"], color="steelblue", alpha=0.7)
        ax.set_ylabel("Price (USD)")
        ax.set_title("Stock Price Comparison")
        st.pyplot(fig, use_container_width=True)
        
        # Download watchlist
        watchlist_csv = watchlist_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Watchlist (CSV)",
            data=watchlist_csv,
            file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.divider()

# ============ SECTION 2: COMPANY SENTIMENT COMPARISON ============
st.markdown("### 📊 Company Sentiment Comparison")

c1, c2 = st.columns(2)

with c1:
    company_a = st.text_input("Company A", placeholder="Infosys")

with c2:
    company_b = st.text_input("Company B", placeholder="TCS")

compare_btn = st.button("📊 Compare Companies", use_container_width=True)

if compare_btn and company_a and company_b:
    with st.spinner("Comparing sentiment..."):
        res_a = analyze_news_sentiment(company_a)
        res_b = analyze_news_sentiment(company_b)

    if res_a and res_b:
        score_a = overall_sentiment(res_a)
        score_b = overall_sentiment(res_b)

        colA, colB = st.columns(2)

        with colA:
            st.subheader(company_a)
            if score_a > 0.2:
                st.success("📈 **Bullish sentiment**")
            elif score_a < -0.2:
                st.error("📉 **Bearish sentiment**")
            else:
                st.info("⚖️ **Neutral sentiment**")
            st.metric("Sentiment Score", f"{score_a:.2f}")

        with colB:
            st.subheader(company_b)
            if score_b > 0.2:
                st.success("📈 **Bullish sentiment**")
            elif score_b < -0.2:
                st.error("📉 **Bearish sentiment**")
            else:
                st.info("⚖️ **Neutral sentiment**")
            st.metric("Sentiment Score", f"{score_b:.2f}")

        # Visualization
        st.subheader("Sentiment Comparison Chart")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["green" if s > 0.2 else "red" if s < -0.2 else "orange" for s in [score_a, score_b]]
        bars = ax.bar([company_a, company_b], [score_a, score_b], color=colors, alpha=0.7, width=0.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("Average Sentiment Score")
        ax.set_title("Sentiment Score Comparison")
        ax.set_ylim(-1.5, 1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("⚠️ Could not fetch data for one or both companies")

elif compare_btn:
    st.warning("⚠️ Please enter both company names")

st.divider()

# ============ SECTION 3: PORTFOLIO ANALYSIS ============
st.markdown("### 🏢 Portfolio Analysis")

portfolio_input = st.text_area(
    "Enter companies to analyze (comma-separated)",
    placeholder="Tesla, Apple, Microsoft, Meta, Google",
    height=50
)

portfolio_btn = st.button("📈 Analyze Portfolio", use_container_width=True)

if portfolio_btn and portfolio_input:
    companies = [c.strip() for c in portfolio_input.split(",")]
    
    with st.spinner("Analyzing portfolio..."):
        portfolio_results = analyze_portfolio(companies)
    
    if portfolio_results:
        portfolio_df = pd.DataFrame(portfolio_results)
        
        # Display portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Companies", len(portfolio_df))
        with col2:
            bullish = len(portfolio_df[portfolio_df["sentiment"] == "Bullish"])
            st.metric("Bullish", bullish)
        with col3:
            bearish = len(portfolio_df[portfolio_df["sentiment"] == "Bearish"])
            st.metric("Bearish", bearish)
        with col4:
            neutral = len(portfolio_df[portfolio_df["sentiment"] == "Neutral"])
            st.metric("Neutral", neutral)
        
        # Portfolio table
        st.subheader("Portfolio Summary")
        st.dataframe(portfolio_df.style.highlight_max(subset=["score"], color='lightgreen')
                                     .highlight_min(subset=["score"], color='lightcoral'),
                     use_container_width=True)
        
        # Portfolio chart
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["green" if s > 0.2 else "red" if s < -0.2 else "orange" for s in portfolio_df["score"]]
        ax.barh(portfolio_df["company"], portfolio_df["score"], color=colors, alpha=0.7)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Sentiment Score")
        ax.set_title("Portfolio Sentiment Scores")
        st.pyplot(fig, use_container_width=True)
        
        # Export portfolio
        portfolio_csv = portfolio_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Report (CSV)",
            data=portfolio_csv,
            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

elif portfolio_btn:
    st.warning("⚠️ Please enter at least one company")

st.divider()

# ============ SECTION 5: LIVE STOCK ANALYSIS ============
st.markdown("### 📊 Live Stock Price Analysis")

stock_analysis_tabs = st.tabs(["Stock Price Chart", "Price vs Sentiment", "Stock Screener", "Real-time Alerts"])

with stock_analysis_tabs[0]:
    st.subheader("Stock Price Chart")
    stock_sym = st.text_input("Stock ticker", placeholder="AAPL", key="chart_ticker").upper()
    
    if stock_sym:
        period = st.selectbox("Time period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], key="chart_period")
        
        if st.button("📈 Load Chart"):
            chart = plot_stock_chart(stock_sym, period)
            if chart:
                st.pyplot(chart, use_container_width=True)
                
                price_change = calculate_price_change(stock_sym, period)
                if price_change:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${price_change['new_price']:.2f}")
                    with col2:
                        st.metric("Previous Price", f"${price_change['old_price']:.2f}")
                    with col3:
                        color = "normal" if price_change["change_percent"] > 0 else "inverse"
                        st.metric("Change %", f"{price_change['change_percent']:.2f}%", delta_color=color)

with stock_analysis_tabs[1]:
    st.subheader("Price Movement vs Sentiment Analysis")
    sentiment_ticker = st.text_input("Stock ticker", placeholder="TSLA", key="sentiment_ticker").upper()
    sentiment_text = st.text_area("Paste news headlines or sentiment text", height=100)
    
    if st.button("📊 Analyze Price vs Sentiment"):
        if sentiment_ticker and sentiment_text:
            # Analyze sentiment
            sentiment, score, confidence = analyze(sentiment_text)
            correlation = correlate_sentiment_price(score, sentiment_ticker)
            alert = get_sentiment_price_alert(score, sentiment_ticker)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Score", f"{score:.2f}")
            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            if correlation:
                st.write(f"**Price Direction**: {correlation['price_direction']}")
                st.write(f"**Status**: {'✅ Aligned' if correlation['match'] else '⚠️ Diverged'}")
            
            if alert and alert["severity"] == "HIGH":
                st.warning(alert["message"])
            elif alert:
                st.success(alert["message"])
            
            # Plot correlation
            chart = plot_price_vs_sentiment(sentiment_ticker, "1mo")
            if chart:
                st.pyplot(chart, use_container_width=True)

with stock_analysis_tabs[2]:
    st.subheader("Stock Screener")
    screener_tickers = st.text_area("Stock tickers (comma-separated)", placeholder="AAPL, MSFT, GOOGL, TSLA", height=60)
    
    if st.button("🔍 Screen Stocks"):
        if screener_tickers:
            tickers = [t.strip().upper() for t in screener_tickers.split(",")]
            
            with st.spinner("Screening stocks..."):
                stocks = get_multiple_stocks_data(tickers)
            
            if stocks:
                stocks_df = pd.DataFrame(stocks)
                stocks_df = stocks_df.sort_values("price", ascending=False)
                
                st.dataframe(stocks_df[["ticker", "price", "market_cap", "pe_ratio"]], use_container_width=True)

with stock_analysis_tabs[3]:
    st.subheader("Real-time Sentiment-Price Alerts")
    alert_ticker = st.text_input("Stock to monitor", placeholder="NVDA", key="alert_ticker").upper()
    alert_sentiment_input = st.slider("Sentiment threshold for alert", -1.0, 1.0, 0.5)
    
    if st.button("🚨 Check Alert Status"):
        if alert_ticker:
            alert_data = get_sentiment_price_alert(alert_sentiment_input, alert_ticker)
            
            if alert_data:
                if alert_data["severity"] == "HIGH":
                    st.error(alert_data["message"])
                else:
                    st.info(alert_data["message"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Alert Type", alert_data["type"])
                with col2:
                    st.metric("Severity", alert_data["severity"])

st.divider()

tabs = st.tabs(["📊 Sentiment Distribution", "🔑 Keywords", "⚠️ Risk & Impact", "🌐 Correlations", "☁️ Word Cloud"])

with tabs[0]:
    st.subheader("Sentiment Distribution Analysis")
    analysis_company = st.text_input("Company for distribution analysis", placeholder="Tesla")
    
    if st.button("Analyze Distribution"):
        results = analyze_news_sentiment(analysis_company)
        if results:
            df_dist = pd.DataFrame(results)
            fig = sentiment_pie_chart(df_dist)
            if fig:
                st.pyplot(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Keyword Extraction")
    keyword_company = st.text_input("Company for keyword analysis", placeholder="Apple", key="keyword_input")
    
    if st.button("Extract Keywords"):
        results = analyze_news_sentiment(keyword_company)
        if results:
            headlines = [r["headline"] for r in results]
            keywords = extract_keywords(headlines, top_n=15)
            
            if keywords:
                # Display keywords
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Keywords**")
                    for word, freq in list(keywords.items())[:8]:
                        st.write(f"- {word}: **{freq}** mentions")
                
                with col2:
                    # Bar chart of keywords
                    fig, ax = plt.subplots(figsize=(8, 5))
                    words, counts = zip(*keywords.items())
                    ax.barh(words, counts, color="steelblue", alpha=0.7)
                    ax.set_xlabel("Frequency")
                    ax.set_title(f"Top Keywords for {keyword_company}")
                    st.pyplot(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Risk & Market Impact Assessment")
    risk_company = st.text_input("Company for risk analysis", placeholder="Microsoft", key="risk_input")
    
    if st.button("Calculate Risk & Impact"):
        results = analyze_news_sentiment(risk_company)
        if results:
            risk_score = calculate_risk_score(results)
            avg_sentiment = overall_sentiment(results)
            impact = calculate_impact_score(np.mean([r["confidence"] for r in results]), avg_sentiment)
            alert = check_sentiment_alert(avg_sentiment)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score (Volatility)", f"{risk_score}/100")
            with col2:
                st.metric("Impact Level", impact["level"])
            with col3:
                st.metric("Impact Score", f"{impact['score']}/100")
            
            if alert["alert"]:
                if alert["type"] == "EXTREME_BULLISH":
                    st.success(alert["message"])
                else:
                    st.error(alert["message"])

with tabs[3]:
    st.subheader("Company Sentiment Correlations")
    ref_company = st.text_input("Reference company", placeholder="Tesla", key="ref_input")
    corr_companies = st.text_area("Companies to compare (comma-separated)", 
                                   placeholder="Apple, Microsoft, Google, Meta", height=50)
    
    if st.button("Find Correlations"):
        if ref_company and corr_companies:
            companies = [c.strip() for c in corr_companies.split(",")]
            correlations = find_correlated_companies(ref_company, companies)
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                st.dataframe(corr_df, use_container_width=True)
                
                # Correlation chart
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(corr_df["company"], corr_df["correlation"], color="skyblue", alpha=0.7)
                ax.set_xlabel("Correlation Score")
                ax.set_title(f"Sentiment Correlation with {ref_company}")
                ax.set_xlim(0, 1)
                st.pyplot(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Word Cloud Analysis")
    wc_company = st.text_input("Company for word cloud", placeholder="Google", key="wc_input")
    
    if st.button("Generate Word Cloud"):
        if not HAS_WORDCLOUD:
            st.warning("⚠️ Install wordcloud: pip install wordcloud")
        else:
            results = analyze_news_sentiment(wc_company)
            if results:
                headlines = [r["headline"] for r in results]
                fig = generate_wordcloud(headlines, "all")
                if fig:
                    st.pyplot(fig, use_container_width=True)

st.divider()

# ============ SECTION 6: TRADING STRATEGIES & RECOMMENDATIONS ============
st.markdown("### 💹 Trading Strategies & Recommendations")

strategy_tabs = st.tabs(["Trade Recommendations", "Investment Strategies", "Risk Management", "Platform Guides"])

with strategy_tabs[0]:
    st.subheader("🎯 Trade Recommendations")
    
    rec_company = st.text_input("Company for trade recommendation", placeholder="Apple", key="rec_ticker")
    rec_ticker = st.text_input("Stock ticker", placeholder="AAPL", key="rec_symbol").upper()
    rec_confidence = st.slider("Minimum confidence threshold", 0, 100, 60)
    
    if st.button("Get Trade Recommendations"):
        if rec_company and rec_ticker:
            results = analyze_news_sentiment(rec_company)
            if results:
                avg_sentiment = overall_sentiment(results)
                price_change = calculate_price_change(rec_ticker, "1d")
                avg_confidence = np.mean([r["confidence"] for r in results])
                
                price_change_pct = price_change["change_percent"] if price_change else 0
                
                recommendations = get_trading_recommendation(avg_sentiment, price_change_pct, avg_confidence)
                
                if recommendations:
                    st.subheader(f"Recommendations for {rec_ticker}")
                    
                    # Display each recommendation
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"**{i}. {rec['strategy']}** - {rec['action']}", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Action**: {rec['action']}")
                                st.write(f"**Confidence**: {rec['confidence']}")
                                st.write(f"**Hold Period**: {rec.get('target_hold', 'N/A')}")
                                st.write(f"**S/L %**: {rec.get('sl_percent', 'N/A')}%")
                                st.write(f"**T/P %**: {rec.get('tp_percent', 'N/A')}%")
                            
                            with col2:
                                st.write(f"**Risk**: {rec.get('risk', 'N/A')}")
                                st.write(f"**Leverage**: {rec.get('leverage', 'N/A')}")
                                if "warning" in rec:
                                    st.warning(rec["warning"])
                            
                            st.write(f"📝 {rec['description']}")
                else:
                    st.info("No strong recommendations at this confidence level")

with strategy_tabs[1]:
    st.subheader("📈 Investment Growth Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        inv_company = st.text_input("Company name", placeholder="Tesla", key="inv_comp")
        inv_capital = st.number_input("Capital available (₹)", min_value=1000, value=50000, step=1000)
    
    with col2:
        inv_horizon = st.selectbox("Investment horizon", ["1-3 months", "3-12 months", "1+ year"])
        st.write("")  # Spacing
    
    if st.button("Get Investment Strategy"):
        if inv_company:
            results = analyze_news_sentiment(inv_company)
            if results:
                avg_sentiment = overall_sentiment(results)
                
                strategies = get_investment_growth_strategy(avg_sentiment, inv_horizon, inv_capital)
                
                if strategies:
                    for strat in strategies:
                        st.success(f"### 🎯 {strat['name']}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Allocation**:\n{strat['allocation']}")
                        with col2:
                            st.write(f"**Approach**:\n{strat['approach']}")
                        with col3:
                            st.write(f"**Expected Return**:\n{strat['expected_return']}")
                        
                        st.write(f"**Risk Level**: {strat['risk']}")
                        st.write(f"**Recommended Tools**: {', '.join(strat['tools'])}")

with strategy_tabs[2]:
    st.subheader("🛡️ Risk Management & Position Sizing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        capital_pm = st.number_input("Total Capital (₹)", min_value=1000, value=100000, step=1000)
    with col2:
        risk_per_trade = st.number_input("Risk per trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    with col3:
        sl_points = st.number_input("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    
    if st.button("Calculate Position Size"):
        ps = calculate_position_sizing(capital_pm, risk_per_trade, sl_points)
        
        if ps:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Capital", f"₹{ps['capital']:,.0f}")
            with col2:
                st.metric("Risk Amount", f"₹{ps['risk_amount']:,.2f}")
            with col3:
                st.metric("Position Size", f"₹{ps['position_size']:,.2f}")
            with col4:
                st.metric("Risk %", f"{ps['risk_percent']}%")
            
            st.info(f"📌 For every ₹{capital_pm:,.0f} capital with {risk_per_trade}% risk, trade size should be ₹{ps['position_size']:,.2f} with {sl_points}% stop loss")
    
    st.divider()
    st.subheader("💰 Profit/Loss Calculator")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        entry = st.number_input("Entry Price (₹)", min_value=0.1, value=100.0)
    with col2:
        exit_p = st.number_input("Exit Price (₹)", min_value=0.1, value=105.0)
    with col3:
        qty = st.number_input("Quantity", min_value=1, value=100)
    with col4:
        st.write("")  # Spacing
    
    if st.button("Calculate P&L"):
        pl = calculate_profit_loss(entry, exit_p, qty)
        
        if pl:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entry", f"₹{pl['entry_price']}")
            with col2:
                st.metric("Exit", f"₹{pl['exit_price']}")
            with col3:
                color = "normal" if pl["profit"] > 0 else "inverse"
                st.metric("Profit/Loss", f"₹{pl['profit']:,.2f}", delta_color=color)
            with col4:
                color = "normal" if pl["profit_percent"] > 0 else "inverse"
                st.metric("P/L %", f"{pl['profit_percent']:.2f}%", delta_color=color)
            
            st.success(pl["status"])

with strategy_tabs[3]:
    st.subheader("📱 Platform Guides & Setup")
    
    platform_guide = st.selectbox(
        "Select a platform",
        ["Zerodha", "Grow", "SEBI", "NSE", "TradingView"]
    )
    
    if platform_guide == "Zerodha":
        st.markdown("""
        ### 🔵 Zerodha - Stock Brokerage & F&O
        
        **What is Zerodha?**
        - India's largest retail stock brokerage
        - Low brokerage: ₹20 per trade
        - Excellent for intraday, swing, F&O, and long-term investing
        
        **Key Features:**
        - **Kite App**: Desktop/Mobile trading platform
        - **Brokerage**: ₹20 flat per trade (intraday/delivery)
        - **Margin**: Intraday margin available at low interest
        - **F&O**: Full options and futures trading
        - **Holdings**: Automatic dividend updates
        
        **How to Use for Profit Growth:**
        1. **Setup SIP** in quality stocks (₹100-1000/month)
        2. **Use Alerts** for buy/sell signals
        3. **Margin Trading** for short-term gains (risky)
        4. **Covered Calls** in F&O to earn extra income
        5. **Stop Loss Orders** to manage risk
        
        **Steps:**
        1. Open account on zerodha.com
        2. KYC verification (Aadhaar/PAN)
        3. Get DP number
        4. Download Kite app
        5. Fund your account
        6. Start trading!
        
        **Best For:**
        - Intraday traders ✅
        - Swing traders ✅
        - Long-term investors ✅
        - F&O traders ✅
        """)
    
    elif platform_guide == "Grow":
        st.markdown("""
        ### 🟢 Grow - Investment & Portfolio Tracking
        
        **What is Grow?**
        - Free investment tracking & portfolio analysis app
        - Track stocks, mutual funds, crypto, ETFs
        - Real-time portfolio monitoring
        
        **Key Features:**
        - **Portfolio Tracking**: Real-time updates
        - **Goal Setting**: Investment goals and progress
        - **Alerts**: Price, dividend, earnings alerts
        - **Research**: Company analysis, news feed
        - **Tax Planning**: STT, ITR tracking
        
        **How to Use for Profit Growth:**
        1. **Add Holdings** - Track all investments
        2. **Set Goals** - Create financial targets
        3. **Enable Alerts** - Get notified of price changes
        4. **Monitor Dividends** - Track dividend income
        5. **Analyze Performance** - Compare returns
        6. **Rebalance** - Adjust portfolio allocation
        
        **Steps:**
        1. Download Grow app
        2. Sign up with email/phone
        3. Add your holdings
        4. Set investment goals
        5. Enable notifications
        6. Review regularly
        
        **Best For:**
        - Portfolio monitoring ✅
        - Long-term tracking ✅
        - Goal planning ✅
        """)
    
    elif platform_guide == "SEBI":
        st.markdown("""
        ### 🏛️ SEBI - Securities & Exchange Board of India
        
        **What is SEBI?**
        - Regulator of Indian stock market
        - Ensures investor protection
        - Regulates mutual funds, brokers, advisors
        
        **Important SEBI Regulations:**
        1. **Mutual Fund** - Invest through SEBI-registered AMCs
        2. **Research Reports** - Only from SEBI-registered analysts
        3. **Advisor License** - Financial advisors must be SEBI-registered
        4. **Fraud Protection** - SEBI investigates market manipulation
        5. **KYC Rules** - Know Your Customer requirements
        
        **How to Use for Safety & Growth:**
        1. **Check Broker Registration** - Verify on SEBI website
        2. **Use Registered Advisors** - For professional advice
        3. **Invest in SEBI-Approved** - Mutual funds, structured products
        4. **File Complaints** - If you face fraud
        5. **Follow Rules** - Don't violate position limits
        
        **Resources:**
        - SEBI Website: sebi.gov.in
        - Investor Charter: Know your rights
        - Complaint Portal: Redressal system
        - Educational Content: Free courses
        
        **Best For:**
        - Investor Protection ✅
        - Understanding Regulations ✅
        - Filing Complaints ✅
        """)
    
    elif platform_guide == "NSE":
        st.markdown("""
        ### 📊 NSE - National Stock Exchange
        
        **What is NSE?**
        - India's largest stock exchange
        - Official price discovery platform
        - Real-time market data source
        
        **Key Indices:**
        - **NIFTY 50** - Top 50 large-cap companies
        - **SENSEX** - BSE's top 30 companies (for reference)
        - **Sector Indices** - IT, Finance, Pharma, etc.
        - **Midcap/Smallcap** - NIFTY MIDCAP 50, NIFTY SMALLCAP 50
        
        **How to Use for Investment:**
        1. **Market Data** - NSE website for live prices
        2. **Index Tracking** - Follow NIFTY 50 performance
        3. **Corporate Actions** - Dividends, splits, mergers
        4. **Index Funds** - Invest in NIFTY 50 ETF
        5. **Market Scans** - Find breakout stocks
        
        **Trading Hours:**
        - Pre-open: 9:00 AM - 9:15 AM
        - Main session: 9:15 AM - 3:30 PM
        - Settlement: T+1 (next day)
        
        **Best For:**
        - Price information ✅
        - Index tracking ✅
        - Index investing ✅
        """)
    
    elif platform_guide == "TradingView":
        st.markdown("""
        ### 📈 TradingView - Technical Analysis Platform
        
        **What is TradingView?**
        - World's largest charting platform
        - Technical analysis tools
        - Community of traders & analysts
        
        **Key Features:**
        - **Charts**: Multiple timeframes and indicators
        - **Screener**: Find stocks based on criteria
        - **Ideas**: Trading ideas from community
        - **Alerts**: Price/indicator-based alerts
        - **Widget**: Add to your website
        
        **How to Use for Trading:**
        1. **Chart Analysis** - Identify support/resistance
        2. **Indicators** - RSI, MACD, Bollinger Bands, etc.
        3. **Patterns** - Find chart patterns
        4. **Screener** - Find bullish/bearish stocks
        5. **Alerts** - Set breakout alerts
        
        **Best Indicators for Sentiment Trading:**
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence)
        - Moving Averages (20, 50, 200 day)
        - Volume Analysis
        - Bollinger Bands
        
        **Steps:**
        1. Go to tradingview.com
        2. Sign up for free account
        3. Add Indian stocks (MCX:NIFTY1!, NSE:INFY)
        4. Draw charts with indicators
        5. Share ideas with community
        
        **Best For:**
        - Technical analysis ✅
        - Pattern recognition ✅
        - Community ideas ✅
        """)
    
    st.divider()
    st.info("💡 **Pro Tip**: Combine sentiment analysis + technical analysis on TradingView + position sizing on Zerodha for best results!")

st.divider()

st.markdown("""
    # 📈 Stock Sentiment Analyzer - Complete Trading System
    
    Your AI-powered stock sentiment analyzer with real-time trading recommendations!
    
    ## 🎯 What You Get
    ✅ Real-time sentiment analysis (FinBERT NLP)
    ✅ Live stock prices (Yahoo Finance)
    ✅ Trading signals (BUY/SELL/SHORT)
    ✅ Intraday, Swing, Long-term & F&O strategies
    ✅ Risk management tools (1% rule, position sizing)
    ✅ Platform guides (Zerodha, Grow, SEBI, NSE, TradingView)
    
    ## 📊 Features
    1. **Single Company Analysis** - Sentiment + live prices + correlations
    2. **Stock Watchlist** - Track multiple stocks with CSV export
    3. **Company Comparison** - Compare 2 stocks side-by-side
    4. **Portfolio Analysis** - Multi-stock screening & breakdown
    5. **Live Stock Analysis** - Charts, correlations, screener, alerts
    6. **Advanced Analysis** - Keywords, risk, correlations, word clouds
    7. **Trading Strategies** - Recommendations, investment plans, risk mgmt, platform guides
    
    ## 💡 Key Trading Signals
    | Sentiment | Signal | Action |
    |-----------|--------|--------|
    | +0.7 to +1.0 | EXTREME BULLISH | Strong BUY |
    | +0.3 to +0.7 | BULLISH | Moderate BUY |
    | -0.3 to +0.3 | NEUTRAL | HOLD/WAIT |
    | -0.7 to -0.3 | BEARISH | Moderate SELL |
    | -1.0 to -0.7 | EXTREME BEARISH | Strong SELL/SHORT |
    
    ## 🛡️ The 1% Risk Rule (MOST IMPORTANT!)
    - Risk only 1% of capital per trade
    - Example: ₹100,000 capital = ₹1,000 risk per trade
    - Use stop loss ALWAYS
    - Position size = Risk amount / Stop loss %
    - You can lose 100 trades and break even!
    
    ## 📈 Expected Returns by Strategy
    | Strategy | Time | Return | Risk | For |
    |----------|------|--------|------|-----|
    | Intraday | 1 day | 2-5% | Very High | Experts |
    | Swing | 2-5 days | 5-10% | High | Intermediate |
    | F&O | Few days | 10-50% | Very High | Experts |
    | Long-term | 1+ year | 12-18% CAGR | Medium | Everyone |
    | SIP | 3-5 years | 12-15% CAGR | Low | Beginners |
    
    ## 🎓 Recommended Portfolio Split
    - 70% Long-term (Index funds SIP + Blue chips)
    - 20% Medium-term (Swing trading 2-5 days)
    - 10% Short-term (Intraday/F&O)
    
    ## 🧠 Success Formula
    **SUCCESS = 70% Psychology + 30% Analysis**
    1. KNOWLEDGE - Learn properly
    2. DISCIPLINE - Follow your rules
    3. PATIENCE - Wait for good setups
    4. RISK MANAGEMENT - Always 1% rule
    5. CONSISTENCY - Same approach daily
    6. LEARNING - Review trades daily
    7. HUMILITY - Accept losses
    8. EMOTION CONTROL - Trade logically
    
    ## 📚 Documentation
    - **README.md** - Complete guide with examples
    - **QUICKSTART.md** - 5-minute setup
    - **TRADING_RULES.md** - Trading guidelines & checklists
    - **FEATURES.md** - All features explained
    - **SUMMARY.md** - Quick overview
    
    ## ⚠️ IMPORTANT DISCLAIMERS
    - **Educational Purpose Only** - Learning tool, not financial advice
    - **Not Guaranteed** - Past performance ≠ Future results
    - **Risk Warning** - You CAN lose money trading
    - **Use Registered Brokers** - Only Zerodha (SEBI registered)
    - **Follow Regulations** - Comply with SEBI & exchange rules
    - **Seek Professional Help** - Consult certified advisor
    
    ## 🔧 Tech Stack
    - **Framework**: Streamlit
    - **NLP Model**: FinBERT (HuggingFace Transformers)
    - **Deep Learning**: PyTorch
    - **Stock Data**: yfinance
    - **News Data**: NewsAPI
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Matplotlib, WordCloud
    - **ML Utilities**: scikit-learn
    
    ## 📞 Resources & Guides
    - **Zerodha** - Stock brokerage (₹20 flat brokerage)
    - **Grow** - Portfolio tracking & alerts
    - **SEBI** - Regulatory compliance & registration
    - **NSE** - Market data & trading hours
    - **TradingView** - Technical analysis & charting
    
    ## 🚀 Next Steps
    1. Read TRADING_RULES.md to understand trading principles
    2. Start with Single Company Analysis
    3. Paper trade (practice without real money)
    4. Keep a trading journal
    5. Follow the 1% risk rule strictly
    6. Start with small amounts when ready
    7. Learn continuously and improve
    
    ## 💪 Remember
    > "The best traders aren't the smartest. They're the most disciplined."
    
    Start small, follow rules, learn continuously, and grow steadily.
    
    **Happy Trading! 📈**
    
    ---
    
    Version: 2.0 (Complete Trading Edition)
    Last Updated: January 17, 2026
    Built with ❤️ for aspiring traders and investors
""")
st.caption("📝 FinBERT | NewsAPI | yfinance | Streamlit")
