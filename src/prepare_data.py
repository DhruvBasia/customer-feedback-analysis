"""
Step 1 - Data preparation.

Reads the raw public dataset (data/Tweets.csv), cleans it and writes data/feedback_clean.csv.

Run:  python src/prepare_data.py
"""
import pandas as pd

from common import CLEAN_PATH, RAW_PATH, THEME_NAMES, clean_text


def main():
    raw = pd.read_csv(RAW_PATH)
    print(f"Raw rows: {len(raw):,}")

    df = pd.DataFrame({
        "feedback_id": raw["tweet_id"],
        # the timestamp looks like "2015-02-24 11:35:52 -0800"; keep the local date and time
        "created_at": pd.to_datetime(raw["tweet_created"].str.slice(0, 19)),
        "airline": raw["airline"],
        "sentiment": raw["airline_sentiment"],
        "label_confidence": raw["airline_sentiment_confidence"],
        "complaint_theme": raw["negativereason"].map(THEME_NAMES),
        "text": raw["text"],
    })
    # Note: the customer's user name, location and coordinates are deliberately not copied.

    # 1. remove rows without a message or a label
    df = df.dropna(subset=["text", "sentiment"])

    # 2. clean the text
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # 3. remove duplicates: the same message must not appear in both training and test data
    before = len(df)
    df = df.sort_values("created_at").drop_duplicates(subset=["airline", "clean_text"], keep="first")
    print(f"Duplicate messages removed: {before - len(df):,}")

    # 4. simple extra columns used in the analysis
    df["word_count"] = df["clean_text"].str.split().str.len()
    df["hour"] = df["created_at"].dt.hour

    df = df.sort_values("created_at").reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"Clean rows: {len(df):,}  ->  {CLEAN_PATH}")
    print(f"Period: {df['created_at'].min():%d %b %Y} to {df['created_at'].max():%d %b %Y}")
    print(df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
