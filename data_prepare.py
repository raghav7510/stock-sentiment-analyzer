import pandas as pd
import re
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("📊 DATA PREPARATION SCRIPT")
print("="*60)

# Load all datasets with error handling
datasets_info = [
    ("news1.csv/all-data.csv", None, {'header': None, 'names': ['Sentiment', 'Text']}),
    ("news2.csv/stock_data.csv", None, {}),
    ("news3.csv/data.csv", None, {})
]

dfs = []
for i, (path, encoding, kwargs) in enumerate(datasets_info, 1):
    try:
        if not os.path.exists(path):
            print(f"❌ Dataset {i}: File not found - {path}")
            continue
        
        encoding = encoding or 'latin-1'
        df_temp = pd.read_csv(path, encoding=encoding, **kwargs)
        print(f"✓ Dataset {i} loaded: {path}")
        print(f"  Rows: {len(df_temp)}, Columns: {list(df_temp.columns)}")
        dfs.append((df_temp, i))
    except Exception as e:
        print(f"❌ Dataset {i} failed to load: {str(e)}")

if not dfs:
    print("❌ No datasets could be loaded. Exiting.")
    exit(1)

df1, _ = dfs[0] if len(dfs) > 0 else (None, 1)
df2, _ = dfs[1] if len(dfs) > 1 else (None, 2)
df3, _ = dfs[2] if len(dfs) > 2 else (None, 3)

# Standardize column names
if df3 is not None:
    df3 = df3.rename(columns={"Sentence":"Text"})

print("\n" + "="*60)
print("📝 NORMALIZING SENTIMENT LABELS")
print("="*60)

# Normalize sentiment labels
def normalize_sentiment(s):
    s = str(s).lower().strip()
    if s in ["positive", "bullish", "1", "pos"]:
        return "Positive"
    elif s in ["negative", "bearish", "0", "neg"]:
        return "Negative"
    else:
        return "Neutral"

# Apply normalization
all_dfs = [df for df in [df1, df2, df3] if df is not None]
for idx, df in enumerate(all_dfs, 1):
    if 'Sentiment' in df.columns:
        before = df['Sentiment'].nunique()
        df["Sentiment"] = df["Sentiment"].apply(normalize_sentiment)
        after = df['Sentiment'].nunique()
        print(f"✓ Dataset {idx}: Normalized sentiments ({before} unique → {after} unique)")
        print(f"  Distribution: {dict(df['Sentiment'].value_counts())}")
    else:
        print(f"⚠️ Dataset {idx}: No 'Sentiment' column found")

# Combine datasets
print("\n" + "="*60)
print("🔗 COMBINING DATASETS")
print("="*60)
df = pd.concat(all_dfs, ignore_index=True)
print(f"✓ Combined: {len(df)} total records")

# Data quality checks
print("\n" + "="*60)
print("🔍 DATA QUALITY CHECKS")
print("="*60)
print(f"Total records: {len(df)}")
print(f"Missing Text values: {df['Text'].isna().sum()}")
print(f"Missing Sentiment values: {df['Sentiment'].isna().sum()}")
print(f"Empty strings in Text: {(df['Text'] == '').sum()}")

# Remove invalid records
initial_count = len(df)
df = df.dropna(subset=['Text', 'Sentiment'])
df = df[df['Text'].str.strip() != '']
final_count = len(df)
removed = initial_count - final_count
print(f"Removed invalid records: {removed}")
print(f"Records after cleaning: {final_count}")

# Clean text
print("\n" + "="*60)
print("🧹 CLEANING TEXT")
print("="*60)
def clean_text(text):
    try:
        text = str(text).strip()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove special characters but keep spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()
    except:
        return ""

# Apply text cleaning
before_clean = len(df)
df["Text"] = df["Text"].apply(clean_text)
# Remove empty texts after cleaning
df = df[df['Text'].str.len() > 3]
after_clean = len(df)
print(f"Removed empty/short texts: {before_clean - after_clean}")
print(f"✓ Text cleaning complete. Final records: {len(df)}")

# Save combined file
print("\n" + "="*60)
print("💾 SAVING COMBINED DATASET")
print("="*60)
output_file = "combined_financial_news.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")
print(f"  Total records: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print(f"  Sentiment distribution:")
for sentiment, count in df['Sentiment'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"    {sentiment}: {count} ({pct:.1f}%)")

print("\n" + "="*60)
print("✅ DATA PREPARATION COMPLETE")
print("="*60)
