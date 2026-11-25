"""
Customer Feedback Analysis - web application (Flask)

The browser shows one page (templates/index.html). The page asks this server for data through
small web addresses called API routes, and the server answers in JSON.

Run:  python app.py      then open  http://localhost:5000
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (CLEAN_PATH, OUT_DIR, SENTIMENTS, SENTIMENT_MODEL_PATH,  # noqa: E402
                    THEME_MODEL_PATH, UNCLEAR_THEME, clean_text, load_data)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024        # uploaded files: 5 MB at most

for needed in (CLEAN_PATH, SENTIMENT_MODEL_PATH, THEME_MODEL_PATH, OUT_DIR / "metrics.json"):
    if not needed.exists():
        sys.exit(f"Missing {needed.name}. Run prepare_data.py and train_model.py first (see README).")

# Loaded once when the server starts, then reused for every request.
DATA = load_data()
SENT_MODEL = joblib.load(SENTIMENT_MODEL_PATH)["pipeline"]
THEME_MODEL = joblib.load(THEME_MODEL_PATH)["pipeline"]
MAX_BATCH_ROWS = 5000


# ---------------- helpers ----------------
def filtered(airline):
    """The whole dataset, or only one airline's messages."""
    if airline and airline in set(DATA["airline"]):
        return DATA[DATA["airline"] == airline]
    return DATA


def complaints(df):
    """Negative messages that have a clear complaint theme."""
    return df[(df["sentiment"] == "negative") & (df["complaint_theme"] != UNCLEAR_THEME)]


def influential_words(pipeline, cleaned, label, top_n=6):
    """Words in the message that pushed the model towards `label`.

    Logistic Regression has one weight per word per class. A word's influence in this message
    is (its TF-IDF value in the message) x (its weight for the class).
    """
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    vector = tfidf.transform([cleaned])
    weights = clf.coef_[list(clf.classes_).index(label)]
    words = tfidf.get_feature_names_out()
    scores = [(words[i], vector[0, i] * weights[i]) for i in vector.nonzero()[1]]
    scores = sorted((s for s in scores if s[1] > 0), key=lambda s: -s[1])
    return [w for w, _ in scores[:top_n]]


def predict(messages):
    """Sentiment for every message, and a complaint theme for the negative ones."""
    cleaned = [clean_text(m) for m in messages]
    s_proba = SENT_MODEL.predict_proba(cleaned)
    t_proba = THEME_MODEL.predict_proba(cleaned)
    results = []
    for text, clean, sp, tp in zip(messages, cleaned, s_proba, t_proba):
        if not clean:                       # nothing readable, for example only emojis
            results.append({"text": text, "sentiment": None})
            continue
        sentiment = SENT_MODEL.classes_[sp.argmax()]
        item = {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(float(sp.max()), 3),
            "probabilities": {c: round(float(p), 3) for c, p in zip(SENT_MODEL.classes_, sp)},
            "theme": None,
        }
        if sentiment == "negative":
            best = tp.argsort()[::-1][:4]
            item["theme"] = THEME_MODEL.classes_[best[0]]
            item["theme_confidence"] = round(float(tp[best[0]]), 3)
            item["theme_probabilities"] = [
                {"theme": THEME_MODEL.classes_[i], "probability": round(float(tp[i]), 3)} for i in best]
        results.append(item)
    return results, cleaned


# ---------------- page ----------------
@app.route("/")
def home():
    return render_template("index.html", airlines=sorted(DATA["airline"].unique()))


# ---------------- API routes ----------------
@app.route("/api/overview")
def api_overview():
    df = filtered(request.args.get("airline"))
    comp = complaints(df)

    daily = pd.crosstab(df["date"], df["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
    daily = daily[daily.sum(axis=1) >= 30]                       # hide near-empty days
    by_airline = pd.crosstab(df["airline"], df["sentiment"], normalize="index")
    by_airline = by_airline.reindex(columns=SENTIMENTS, fill_value=0).sort_values("negative")
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

    return jsonify({
        "messages": int(len(df)),
        "share": {s: float((df["sentiment"] == s).mean()) for s in SENTIMENTS},
        "top_theme": comp["complaint_theme"].mode().iat[0] if len(comp) else "-",
        "period": [f"{df['date'].min():%d %b %Y}", f"{df['date'].max():%d %b %Y}"],
        "daily": {"dates": [f"{d:%Y-%m-%d}" for d in daily.index],
                  **{s: daily[s].tolist() for s in SENTIMENTS}},
        "by_airline": {"airlines": by_airline.index.tolist(),
                       **{s: by_airline[s].round(4).tolist() for s in SENTIMENTS}},
        "hourly": hourly.tolist(),
    })


@app.route("/api/themes")
def api_themes():
    df = filtered(request.args.get("airline"))
    negative = df[df["sentiment"] == "negative"]
    comp = complaints(df)
    counts = comp["complaint_theme"].value_counts()
    all_comp = complaints(DATA)                                   # the heatmap always shows all airlines
    order = all_comp["complaint_theme"].value_counts().index
    heat = pd.crosstab(all_comp["airline"], all_comp["complaint_theme"], normalize="index")[order]
    return jsonify({
        "total": int(len(comp)),
        "unclear": int(len(negative) - len(comp)),
        "themes": counts.index.tolist(),
        "counts": counts.tolist(),
        "heatmap": {"airlines": heat.index.tolist(), "themes": heat.columns.tolist(),
                    "values": heat.round(4).values.tolist()},
    })


@app.route("/api/messages")
def api_messages():
    comp = complaints(filtered(request.args.get("airline")))
    comp = comp[comp["complaint_theme"] == request.args.get("theme", "")]
    comp = comp.sort_values("label_confidence", ascending=False).head(12)
    return jsonify([{"airline": r.airline, "when": f"{r.created_at:%d %b, %H:%M}", "text": r.text}
                    for r in comp.itertuples()])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:2000]
    if not text:
        return jsonify({"error": "Please type a message first."}), 400
    results, cleaned = predict([text])
    result = results[0]
    if result["sentiment"] is None:
        return jsonify({"error": "The message has no words the model can read. Please type a sentence in English."}), 400
    result["words"] = influential_words(SENT_MODEL, cleaned[0], result["sentiment"])
    return jsonify(result)


@app.route("/api/predict-file", methods=["POST"])
def api_predict_file():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file was uploaded."}), 400
    try:
        table = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "This file could not be read as a CSV."}), 400
    if "text" not in table.columns:
        return jsonify({"error": "The file needs a column named 'text'."}), 400
    messages = table["text"].dropna().astype(str).head(MAX_BATCH_ROWS).tolist()
    if not messages:
        return jsonify({"error": "The 'text' column is empty."}), 400
    results, _ = predict(messages)
    rows = [{"text": r["text"], "sentiment": r["sentiment"] or "unreadable",
             "confidence": r.get("confidence"), "theme": r.get("theme") or ""} for r in results]
    summary = {s: sum(r["sentiment"] == s for r in rows) for s in SENTIMENTS}
    return jsonify({"rows": rows, "summary": summary, "total": len(rows)})


@app.route("/api/model")
def api_model():
    metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(OUT_DIR / "model_comparison.csv")
    comparison = json.loads(comparison.to_json(orient="records"))   # NaN becomes null
    top_words = pd.read_csv(OUT_DIR / "top_words.csv")
    words = {s: top_words.loc[top_words["sentiment"] == s, "word"].tolist() for s in SENTIMENTS}
    return jsonify({"metrics": metrics, "comparison": comparison, "top_words": words})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
