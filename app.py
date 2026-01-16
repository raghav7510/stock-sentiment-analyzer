import streamlit as st
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Live Stock Sentiment Analyzer", layout="wide")

# Load model with caching
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
    if not API_KEY:
        st.error("❌ API Key not configured")
        return []
    try:
        url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
        r = requests.get(url, timeout=5).json()
        if r.get("status") == "error":
            st.error(f"❌ API Error: {r.get('message')}")
            return []
        articles = r.get("articles", [])
        if not articles:
            st.warning(f"⚠️ No news found for '{company}'")
            return []
        return [a["title"] for a in articles[:5]]
    except requests.exceptions.RequestException as e:
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

st.title("📈 Live Stock Sentiment Analyzer")

col1, col2 = st.columns([3, 1])
with col1:
    company = st.text_input("Enter company name (e.g., Tesla, Infosys, Apple)", placeholder="Type company name...")
with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("🔍 Analyze Live News", use_container_width=True)

if analyze_button and company.strip():
    with st.spinner(f"Fetching news for {company}..."):
        news = get_stock_news(company.strip())
    
    if news:
        st.subheader(f"Results for: {company}")
        
        # Run analysis
        with st.spinner("Analyzing sentiment..."):
            results = [(*analyze(n), n) for n in news]
        
        df = pd.DataFrame(results, columns=["Sentiment","Score","Confidence","Headline"])
        
        # Display individual results
        for i, row in df.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                if row["Sentiment"] == "Positive":
                    st.success(f"🟢 {row['Headline']}")
                elif row["Sentiment"] == "Negative":
                    st.error(f"🔴 {row['Headline']}")
                else:
                    st.info(f"⚪ {row['Headline']}")
            with col2:
                st.metric("Confidence", f"{row['Confidence']}%")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            st.metric("Positive", (df["Sentiment"] == "Positive").sum())
        with col3:
            st.metric("Neutral", (df["Sentiment"] == "Neutral").sum())
        with col4:
            st.metric("Negative", (df["Sentiment"] == "Negative").sum())
        
        # Visualization
        st.subheader("Sentiment Score Trend")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["red" if s == "Negative" else "orange" if s == "Neutral" else "green" for s in df["Sentiment"]]
        ax.bar(range(len(df)), df["Score"], color=colors, alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Article Index")
        ax.set_ylabel("Sentiment Score")
        ax.set_title(f"Sentiment Scores for {company}")
        ax.set_ylim(-1.5, 1.5)
        st.pyplot(fig, use_container_width=True)
        
        # Average sentiment
        avg_score = df["Score"].mean()
        st.subheader("Overall Sentiment")
        if avg_score > 0.3:
            st.success(f"✅ Bullish (Average Score: {avg_score:.2f})")
        elif avg_score < -0.3:
            st.error(f"❌ Bearish (Average Score: {avg_score:.2f})")
        else:
            st.info(f"➡️ Neutral (Average Score: {avg_score:.2f})")

elif analyze_button:
    st.warning("⚠️ Please enter a company name")

st.divider()
st.caption("📝 Powered by FinBERT | Data from NewsAPI")
