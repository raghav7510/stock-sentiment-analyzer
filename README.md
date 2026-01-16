# 📈 Stock Sentiment Analyzer App - Complete Trading System

Your complete **AI-powered stock sentiment analyzer** with real-time trading recommendations, investment strategies, and platform guides!

---

## 🎯 What This App Does

✅ **Real-time sentiment analysis** using FinBERT NLP model  
✅ **Live stock prices** from Yahoo Finance  
✅ **News sentiment** from NewsAPI  
✅ **Trading signals** (BUY/SELL/SHORT/HOLD)  
✅ **Intraday, Swing, Long-term & F&O strategies**  
✅ **Risk management tools** (position sizing, P&L calculator)  
✅ **Platform guides** (Zerodha, Grow, SEBI, NSE, TradingView)  
✅ **Portfolio analysis** with CSV export  
✅ **Technical correlations** (sentiment vs price)  
✅ **Keyword extraction** & word clouds  

---

## 🚀 Quick Start (5 Minutes)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Setup NewsAPI Key
Create `.streamlit/secrets.toml`:
```toml
NEWS_API_KEY = "your_key_from_newsapi.org"
```
Get free key: https://newsapi.org/register

### 3️⃣ Run the App
```bash
streamlit run app.py
```

App opens at: http://localhost:8501

---

## 📊 App Sections

### 1. **Single Company Analysis** 📈
- Real-time sentiment analysis
- Live stock price & chart
- Sentiment vs price correlation
- Trading alerts

### 2. **Stock Watchlist Screener** 📋
- Track multiple stocks
- Compare prices & sentiments
- Download CSV report

### 3. **Company Comparison** 🔄
- Compare 2 companies side-by-side
- Sentiment comparison charts

### 4. **Portfolio Analysis** 💼
- Analyze multiple companies
- Bullish/Bearish/Neutral breakdown

### 5. **Live Stock Analysis** 📈
- Stock price charts (1d-1y)
- Sentiment correlations
- Stock screener
- Real-time alerts

### 6. **Advanced Analysis** 🔬
- Sentiment distribution
- Top keywords extraction
- Risk & market impact
- Company correlations
- Word cloud visualization

### 7. **Trading Strategies** 💹
- **Trade Recommendations** - BUY/SELL/SHORT signals
- **Investment Strategies** - Asset allocation guidance
- **Risk Management** - Position sizing & P&L calc
- **Platform Guides** - Setup tutorials for 5 platforms

---

## 💡 Trading Signals Explained

### Sentiment Score:
```
+0.7 to +1.0 = EXTREME BULLISH (Strong BUY)
+0.3 to +0.7 = BULLISH (Moderate BUY)
-0.3 to +0.3 = NEUTRAL (HOLD/WAIT)
-0.7 to -0.3 = BEARISH (Moderate SELL)
-1.0 to -0.7 = EXTREME BEARISH (Strong SELL/SHORT)
```

### Confidence Levels:
```
>80% = VERY STRONG (Trade full position)
60-80% = STRONG (Trade normal position)
50-60% = MODERATE (Trade half position)
<50% = WEAK (Skip or scalp only)
```

---

## 📚 Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| QUICKSTART.md | Get started in 5 minutes | 5 min |
| TRADING_RULES.md | Complete trading guidelines | 20 min |
| FEATURES.md | All features explained | 30 min |
| SUMMARY.md | Quick feature overview | 10 min |

---

## 🛡️ The 1% Risk Rule (Most Important!)

```
RISK ONLY 1% OF CAPITAL PER TRADE

Example with ₹100,000:
- Risk per trade: 1% = ₹1,000
- Stop loss: 2%
- Position size: ₹50,000
- Loss if SL hit: ₹1,000 (exactly 1%)

You can lose 100 trades and break even!
```

### Must Follow:
✅ Always use stop loss  
✅ Risk only 1% per trade  
✅ Keep trading journal  
✅ Use registered brokers (Zerodha)  
✅ Follow SEBI regulations  
✅ Diversify portfolio  

### Never Do:
❌ Trade without stop loss  
❌ Risk >2% per trade  
❌ Use excessive leverage  
❌ Trade on emotions  
❌ Over-trade (>10/day)  

---

## 📈 Trading Strategy Returns

| Strategy | Time | Return | Risk | For |
|----------|------|--------|------|-----|
| Intraday | 1 day | 2-5% per trade | Very High | Experts |
| Swing | 2-5 days | 5-10% per trade | High | Intermediate |
| F&O | Few days | 10-50% per trade | Very High | Experts |
| Long-term | 1+ year | 12-18% CAGR | Medium | Everyone |
| SIP | 3-5 years | 12-15% CAGR | Low | Beginners |

---

## 🔧 Platform Guides Included

### ✅ **Zerodha** (Stock Brokerage)
- Account setup steps
- ₹20 flat brokerage
- F&O trading guide
- Margin available: 2-4x

### ✅ **Grow** (Portfolio Tracking)
- Real-time tracking
- Price alerts
- Goal monitoring
- Tax planning

### ✅ **SEBI** (Regulations)
- Compliance rules
- Broker verification
- Complaint filing
- Regulations explained

### ✅ **NSE** (Market Data)
- Market hours
- NIFTY 50 tracking
- Data access
- Trading calendar

### ✅ **TradingView** (Technical Analysis)
- Charting tools
- Indicators guide
- Screener usage
- Community insights

---

## 🎓 Example: How to Use the App

### Scenario: You want to trade Apple stock

```
STEP 1: Go to "Single Company Analysis"
STEP 2: Enter ticker: AAPL
STEP 3: App shows:
   - Latest news about Apple
   - Sentiment score: +0.72 (Bullish)
   - Confidence: 85% (Very Strong)
   - Current price: ₹187.50
   - Price chart

STEP 4: Go to "Trade Recommendations"
STEP 5: You see:
   ✅ INTRADAY: BUY momentum, 2% SL, 5% TP
   ✅ SWING: Accumulate, 5% SL, 10% TP
   ✅ LONG-TERM: Strong BUY, 1-year hold
   ✅ F&O: Call spread, 10% SL

STEP 6: Go to "Risk Management"
STEP 7: Enter position size for 1% risk
   Position size: ₹50,000 (if capital is ₹100K, risk is 1%)

STEP 8: Execute on Zerodha (see guide in app)
STEP 9: Track on Grow app
```

---

## 📞 Key Features for Each User Type

### 👨‍💼 **Beginners**
- Start with "Single Company Analysis"
- Read TRADING_RULES.md
- Focus on long-term investing
- Follow the 1% rule strictly
- Use SIP approach

### 👨‍💻 **Intermediate Traders**
- Use "Trade Recommendations" for signals
- Combine sentiment + technical analysis
- Focus on Swing trading (2-5 days)
- Use position sizing calculator
- Keep trading journal

### 🎯 **Advanced Traders**
- Use F&O recommendations
- Combine all analysis types
- Do intraday trading
- Use risk management tools
- Follow platform guides

---

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit |
| NLP Model | FinBERT (HuggingFace) |
| Deep Learning | PyTorch |
| Stock Data | yfinance |
| News Data | NewsAPI |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, WordCloud |
| ML Utils | scikit-learn |

---

## ⚠️ Important Disclaimers

### This App:
✅ Is for EDUCATIONAL purposes  
✅ Helps you LEARN about markets  
✅ Shows SENTIMENT analysis  

### This App is NOT:
❌ Financial advice  
❌ Investment recommendation  
❌ Guaranteed profit  
❌ A substitute for professionals  

### Important Notes:
⚠️ You CAN lose money trading  
⚠️ Past performance ≠ Future results  
⚠️ Consult financial advisor before trading  
⚠️ Use registered brokers only (Zerodha)  
⚠️ Follow all regulations (SEBI)  

---

## 🎯 Recommended Approach

### Portfolio Split:
```
70% Long-term (Index funds SIP, Blue chip stocks)
20% Medium-term (Swing trading 2-5 days)
10% Short-term (Intraday/F&O trading)
```

### Daily Routine:
```
9:15 AM - Check sentiment signals & plan day
12-3 PM - Monitor open positions
5:30 PM - Review & journal trades
```

### Success Formula:
```
SUCCESS = 70% Psychology + 30% Analysis

1. KNOWLEDGE - Learn properly
2. DISCIPLINE - Follow your rules
3. PATIENCE - Wait for good setups
4. RISK MANAGEMENT - Always 1% rule
5. CONSISTENCY - Same approach daily
6. LEARNING - Review trades daily
7. HUMILITY - Accept losses
8. EMOTION CONTROL - Trade logically
```

---

## 🚀 Next Steps

1. Install: `pip install -r requirements.txt`
2. Setup: Add NEWS_API_KEY to `.streamlit/secrets.toml`
3. Run: `streamlit run app.py`
4. Learn: Read TRADING_RULES.md
5. Practice: Paper trade (practice without real money)
6. Trade: Start with small amounts (1% risk)
7. Grow: Increase size as you improve

---

## 📚 Learning Resources

- **Zerodha Varsity** - Free stock market education
- **TradingView** - Free charting & analysis
- **NSE Website** - Official market data
- **SEBI Website** - Regulations & compliance
- **Investopedia** - Financial concepts

---

## 💪 Remember

"The best traders aren't the smartest. They're the most disciplined."

Start small, follow rules, learn continuously, and grow steadily.

**Happy Trading! 📈**

---

Version: 2.0 (Complete Trading Edition)  
Last Updated: January 17, 2026  
Built with ❤️ for aspiring traders and investors
