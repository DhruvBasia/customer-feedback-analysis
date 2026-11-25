"""
Step 3 - Model training.

Trains and compares three text classifiers for two tasks:
  A. Sentiment      : negative / neutral / positive          (all messages)
  B. Complaint theme: what a negative message complains about (negative messages only)

The saved models are used by the web page (app.py).

For each task the best model is chosen by 5-fold cross-validation on the training data,
then checked once on a held-out test set that the model has never seen.

Run:  python src/train_model.py
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import (BLUE, FIG_DIR, MODEL_DIR, OUT_DIR, RANDOM_STATE, SENT_COLORS, SENTIMENTS,
                    SENTIMENT_MODEL_PATH, THEME_MODEL_PATH, UNCLEAR_THEME, load_data, set_chart_style)

plt, sns = set_chart_style()


def make_vectorizer():
    """TF-IDF turns each message into numbers: one column per word or two-word phrase.

    Stop words are NOT removed, because words such as "not" and "no" carry the sentiment.
    """
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def make_candidates():
    """The three models that are compared. Each is a pipeline: TF-IDF -> classifier."""
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB(alpha=0.3))]),
        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(C=5, max_iter=2000, class_weight="balanced"))]),
        "Linear SVM": Pipeline([
            ("tfidf", make_vectorizer()),
            # the calibration wrapper lets the SVM give probabilities, which the dashboard shows
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3))]),
    }


def train_task(name, X, y, labels):
    """Split, cross-validate, compare, pick the best model and test it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # baseline: always answer the most common class
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = dummy.predict(X_test)
    rows = [{"model": "Baseline (most common class)", "cv_macro_f1": np.nan,
             "test_accuracy": accuracy_score(y_test, base_pred),
             "test_macro_f1": f1_score(y_test, base_pred, average="macro")}]

    fitted = {}
    for model_name, pipe in make_candidates().items():
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[model_name] = pipe
        rows.append({"model": model_name, "cv_macro_f1": cv_f1,
                     "test_accuracy": accuracy_score(y_test, pred),
                     "test_macro_f1": f1_score(y_test, pred, average="macro")})
        print(f"  [{name}] {model_name:<20} CV macro-F1 {cv_f1:.3f}")

    table = pd.DataFrame(rows).round(5)
    # the choice uses the cross-validation score only, never the test score
    best_name = table.dropna().sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, pred, labels=labels)

    result = {
        "best_model": best_name,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "baseline_accuracy": float(rows[0]["test_accuracy"]),
        "baseline_macro_f1": float(rows[0]["test_macro_f1"]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "per_class": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")}
                      | {"support": int(report[c]["support"])} for c in labels},
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),     # rows = actual class, columns = predicted class
    }
    return best, fitted, table, matrix, result


def plot_comparison(tables):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (title, table) in zip(axes, tables.items()):
        t = table.set_index("model")[["test_accuracy", "test_macro_f1"]]
        t.columns = ["Accuracy", "Macro F1"]
        t.plot.barh(ax=ax, color=[BLUE, "#9DBCE0"], width=0.75)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
        ax.set(title=title, xlim=(0, 1.05), ylabel="")
        ax.invert_yaxis()
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_model_comparison.png")
    plt.close(fig)


def plot_matrix(matrix, labels, title, file_name, size):
    share = matrix / matrix.sum(axis=1, keepdims=True)      # each row adds up to 100%
    text = np.array([[f"{s:.0%}\n({n})" for s, n in zip(rs, rn)] for rs, rn in zip(share, matrix)])
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(share, annot=text, fmt="", cmap="Blues", cbar=False, linewidths=2, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 8}, ax=ax)
    ax.set(title=title, xlabel="Predicted by the model", ylabel="Actual (human label)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.savefig(FIG_DIR / file_name)
    plt.close(fig)


def plot_top_words(lr_pipe, top_n=12):
    """Which words push the Logistic Regression model towards each sentiment."""
    words = lr_pipe.named_steps["tfidf"].get_feature_names_out()
    clf = lr_pipe.named_steps["clf"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    rows = []
    for ax, label in zip(axes, clf.classes_):
        coef = clf.coef_[list(clf.classes_).index(label)]
        top = np.argsort(coef)[-top_n:]
        ax.barh(words[top], coef[top], color=SENT_COLORS[label])
        ax.set(title=f"Words that signal '{label}'", xlabel="Model weight")
        rows += [{"sentiment": label, "word": words[i], "weight": round(float(coef[i]), 3)} for i in top[::-1]]
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_top_words_per_sentiment.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_DIR / "top_words.csv", index=False)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # ---------------- task A: sentiment ----------------
    print("Task A - sentiment")
    s_best, s_fitted, s_table, s_matrix, s_result = train_task(
        "sentiment", df["clean_text"], df["sentiment"], SENTIMENTS)

    # ---------------- task B: complaint theme ----------------
    print("Task B - complaint theme")
    comp = df[(df["sentiment"] == "negative") & (df["complaint_theme"] != UNCLEAR_THEME)]
    themes = comp["complaint_theme"].value_counts().index.tolist()
    t_best, _, t_table, t_matrix, t_result = train_task(
        "theme", comp["clean_text"], comp["complaint_theme"], themes)

    # ---------------- save everything ----------------
    joblib.dump({"pipeline": s_best, "labels": SENTIMENTS, "name": s_result["best_model"]}, SENTIMENT_MODEL_PATH)
    joblib.dump({"pipeline": t_best, "labels": themes, "name": t_result["best_model"]}, THEME_MODEL_PATH)

    s_table.insert(0, "task", "Sentiment")
    t_table.insert(0, "task", "Complaint theme")
    pd.concat([s_table, t_table]).to_csv(OUT_DIR / "model_comparison.csv", index=False)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps({"sentiment": s_result, "theme": t_result}, indent=2), encoding="utf-8")

    plot_comparison({"Sentiment models (test set)": s_table, "Complaint-theme models (test set)": t_table})
    plot_matrix(s_matrix, SENTIMENTS, f"Sentiment confusion matrix - {s_result['best_model']}",
                "08_sentiment_confusion_matrix.png", (5.5, 4.2))
    plot_matrix(t_matrix, themes, f"Complaint-theme confusion matrix - {t_result['best_model']}",
                "09_theme_confusion_matrix.png", (9, 6))
    plot_top_words(s_fitted["Logistic Regression"])

    for title, r in (("SENTIMENT", s_result), ("COMPLAINT THEME", t_result)):
        print(f"\n{title}: best = {r['best_model']}  |  accuracy {r['accuracy']:.3f} "
              f"(baseline {r['baseline_accuracy']:.3f})  |  macro-F1 {r['macro_f1']:.3f}")
        for c, m in r["per_class"].items():
            print(f"  {c:<22} precision {m['precision']:.2f}  recall {m['recall']:.2f}  "
                  f"f1 {m['f1-score']:.2f}  n={m['support']}")


if __name__ == "__main__":
    main()
