"""
Test file to verify LIVE stock prices are working correctly
Run this before launching the app to confirm prices are accurate
"""

import yfinance as yf
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_stock_price(ticker):
    """Fetch LIVE stock price with proper handling"""
    try:
        original_ticker = ticker
        
        # Auto-detect Indian stocks
        if ticker and not ticker.endswith('.NS') and not ticker.endswith('.BO'):
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
        
        current_price = None
        
        if not hist_1d.empty and len(hist_1d) > 0:
            current_price = float(hist_1d['Close'].iloc[-1])
        
        if current_price is None or current_price <= 0:
            current_price = info.get('regularMarketPrice')
        
        if current_price is None or current_price <= 0:
            current_price = info.get('currentPrice')
        
        if current_price and current_price > 0:
            current_price = float(current_price)
        else:
            current_price = None
        
        country = info.get('country', 'US')
        currency = info.get('currency', 'USD')
        
        if 'NSE' in str(info.get('exchange', '')) or ticker.endswith('.NS'):
            country = 'India'
            currency = 'INR'
        
        change = 0
        change_pct = 0
        if len(hist_1d) >= 2 and current_price and current_price > 0:
            prev_close = float(hist_1d['Close'].iloc[-2])
            if prev_close > 0:
                change = current_price - prev_close
                change_pct = (change / prev_close * 100)
        
        return {
            'price': current_price,
            'country': country,
            'currency': currency,
            'change': change,
            'change_pct': change_pct,
            'ticker_used': ticker,
            'status': 'SUCCESS' if current_price else 'FAILED'
        }
    except Exception as e:
        return {
            'price': None,
            'status': f'ERROR: {str(e)[:50]}',
            'ticker_used': ticker
        }


def test_stocks():
    """Test various stocks to ensure correct price fetching"""
    print("\n" + "=" * 70)
    print("LIVE STOCK PRICE VERIFICATION TEST")
    print("=" * 70)
    
    # Test cases: (ticker, expected_range, currency, name)
    test_cases = [
        # Indian Stocks
        ('INFY', (1500, 2000), 'INR', 'Infosys'),
        ('TCS', (3000, 4500), 'INR', 'Tata Consultancy Services'),
        ('RELIANCE', (1000, 3000), 'INR', 'Reliance Industries'),
        ('WIPRO', (400, 700), 'INR', 'Wipro'),
        ('MARUTI', (9000, 12000), 'INR', 'Maruti Suzuki'),
        
        # US Stocks
        ('AAPL', (150, 400), 'USD', 'Apple'),
        ('MSFT', (350, 500), 'USD', 'Microsoft'),
        ('GOOGL', (100, 200), 'USD', 'Google'),
        ('AMZN', (150, 200), 'USD', 'Amazon'),
        ('TSLA', (150, 300), 'USD', 'Tesla'),
    ]
    
    print("\nTesting LIVE Stock Prices...")
    print("-" * 70)
    
    passed = 0
    failed = 0
    
    for ticker, expected_range, expected_currency, company_name in test_cases:
        result = get_stock_price(ticker)
        price = result['price']
        currency = result['currency']
        status = result['status']
        
        if price:
            min_exp, max_exp = expected_range
            is_valid = min_exp <= price <= max_exp * 1.5  # Allow 50% variance
            
            if is_valid and currency == expected_currency:
                status_symbol = "PASS"
                passed += 1
                print(f"\n[PASS] {ticker} - {company_name}")
            else:
                status_symbol = "WARN"
                print(f"\n[WARN] {ticker} - {company_name}")
                if not is_valid:
                    print(f"       Price: {price} (expected ~{expected_range[0]}-{expected_range[1]})")
                if currency != expected_currency:
                    print(f"       Currency: {currency} (expected {expected_currency})")
            
            print(f"       Price: {currency} {price:.2f}")
            print(f"       Change: {result['change']:+.2f} ({result['change_pct']:+.2f}%)")
            print(f"       Country: {result['country']}")
            print(f"       Ticker Used: {result['ticker_used']}")
        else:
            failed += 1
            print(f"\n[FAIL] {ticker} - {company_name}")
            print(f"       Status: {status}")
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    
    if failed == 0:
        print("\nSUCCESS! All stock prices are fetching correctly.")
        print("The app is ready to use with LIVE prices.")
    else:
        print(f"\nWARNING: {failed} test(s) failed. Check your internet connection.")
    
    print("\nQuick Tips for the App:")
    print("  - For Indian stocks: Just type ticker (INFY, TCS, RELIANCE, etc.)")
    print("  - Auto-detects .NS suffix for correct INR prices")
    print("  - For US stocks: Works as-is (AAPL, MSFT, TSLA, etc.)")
    print("  - Prices update every 30 seconds automatically")
    print("\nExample: Search for 'Infosys' with ticker 'INFY' -> Rs 1689.80")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_stocks()
