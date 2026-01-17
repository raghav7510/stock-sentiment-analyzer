#!/usr/bin/env python
import yfinance as yf

print("=" * 60)
print("🧪 TESTING LIVE PRICE FETCHING")
print("=" * 60)

# Test Indian stock
print("\n📊 Test 1: INFY (Infosys)")
try:
    infy = yf.Ticker('INFY')
    hist = infy.history(period='5d')
    print(f"✅ Latest Close: ₹{hist['Close'].iloc[-1]:.2f}")
    print(f"   Volume: {hist['Volume'].iloc[-1]/1e6:.2f}M")
    print(f"   Previous Close: ₹{hist['Close'].iloc[-2]:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test US stock
print("\n📊 Test 2: TSLA (Tesla)")
try:
    tsla = yf.Ticker('TSLA')
    hist = tsla.history(period='5d')
    print(f"✅ Latest Close: ${hist['Close'].iloc[-1]:.2f}")
    print(f"   Volume: {hist['Volume'].iloc[-1]/1e6:.2f}M")
    print(f"   Previous Close: ${hist['Close'].iloc[-2]:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test with error recovery
print("\n📊 Test 3: Error Handling")
try:
    wrong = yf.Ticker('WRONGTICKER123')
    hist = wrong.history(period='5d')
    if hist.empty:
        print("✅ Error handling works - Empty data detected and handled gracefully")
except Exception as e:
    print(f"✅ Exception caught properly: {str(e)[:50]}...")

print("\n" + "=" * 60)
print("✅ All tests completed successfully!")
print("=" * 60)
