"""
Shared paths, constants and helper functions used by every script and by the dashboard.
"""
import html
import re
from pathlib import Path

import pandas as pd

# ---------------- paths ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "Tweets.csv"                 # public dataset, as downloaded
CLEAN_PATH = DATA_DIR / "feedback_clean.csv"       # output of prepare_data.py
MODEL_DIR = ROOT / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
THEME_MODEL_PATH = MODEL_DIR / "theme_model.joblib"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# ---------------- constants ----------------
RANDOM_STATE = 42
SENTIMENTS = ["negative", "neutral", "positive"]
SENT_COLORS = {"negative": "#C0392B", "neutral": "#8A8F98", "positive": "#2E8B57"}
BLUE = "#2F6DB5"
UNCLEAR_THEME = "Can't Tell"   # complaints the human labellers could not put in a theme

# The raw dataset has 10 complaint reasons. We tidy the names and merge the two luggage
# reasons, because "Damaged Luggage" alone has too few examples to learn from.
THEME_NAMES = {
    "Customer Service Issue": "Customer Service",
    "Late Flight": "Late Flight",
    "Cancelled Flight": "Cancelled Flight",
    "Lost Luggage": "Luggage Issue",
    "Damaged Luggage": "Luggage Issue",
    "Bad Flight": "Bad Flight Experience",
    "Flight Booking Problems": "Booking Problem",
    "Flight Attendant Complaints": "Flight Attendant",
    "longlines": "Long Lines",
    "Can't Tell": UNCLEAR_THEME,
}


def clean_text(text):
    """Turn a raw customer message into plain lowercase words.

    Steps: decode HTML codes (&amp; -> &), remove links and @mentions, keep the word of a
    hashtag, join contractions (can't -> cant) so that negations stay as one word,
    drop everything that is not a letter or a digit, and collapse extra spaces.
    """
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # links
    text = re.sub(r"@\w+", " ", text)                     # @mentions (also hides who was tagged)
    text = text.lower()
    text = re.sub(r"['’`]", "", text)                # can't -> cant, don't -> dont
    text = re.sub(r"[^a-z0-9 ]+", " ", text)              # punctuation, emojis, '#'
    return re.sub(r"\s+", " ", text).strip()


def load_data():
    """Load the cleaned feedback table created by prepare_data.py."""
    df = pd.read_csv(CLEAN_PATH, parse_dates=["created_at"])
    df["complaint_theme"] = df["complaint_theme"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")
    df["date"] = df["created_at"].dt.normalize()
    return df


def set_chart_style():
    """One consistent look for every saved chart."""
    import matplotlib
    matplotlib.use("Agg")                                 # save to file, no window needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt, sns
