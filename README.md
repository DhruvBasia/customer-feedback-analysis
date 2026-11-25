# Customer Feedback Analysis: Sentiment and Complaint Themes

Research Analyst internship project · B.Tech (AI & DS), 5th semester

> **Data note.** This project uses a **public dataset**, *Twitter US Airline Sentiment*
> (originally published by CrowdFlower on Kaggle, licence CC BY-NC-SA 4.0). It is **not**
> Concentrix or client data. The internship gave the problem and the working method; the
> public dataset lets the method be shown without using any confidential information.
> Every number below describes this dataset only.

## 1. Problem statement

Customer-experience companies handle very large volumes of customer messages for their clients:
reviews, support messages and social media posts. A research analyst is expected to read this
feedback and report what customers are unhappy about. Reading thousands of messages by hand is
slow, and two analysts can label the same message differently.

**The problem:** most customer feedback goes unread, so the business does not know, quickly and
consistently, how customers feel or what their main complaints are.

## 2. Objectives

1. Clean and analyse a real set of customer messages.
2. Build a model that labels each message as **negative, neutral or positive**.
3. Build a second model that puts each negative message into a **complaint theme**.
4. Deliver a web page that shows sentiment over time, the top complaint themes, and lets a
   user check a new message or a whole file.

## 3. Dataset

| Item | Value |
|---|---|
| Source | Twitter US Airline Sentiment (CrowdFlower / Kaggle), file `data/Tweets.csv` |
| Raw size | 14,640 customer messages sent to 6 US airlines |
| After cleaning | 14,283 messages (357 duplicates removed) |
| Period | 16 to 24 February 2015 |
| Labels | Sentiment and complaint reason, both assigned by human labellers |

Columns kept in `data/feedback_clean.csv`: `feedback_id`, `created_at`, `airline`, `sentiment`,
`label_confidence`, `complaint_theme`, `text`, `clean_text`, `word_count`, `hour`.
The customers' user names and locations are in the raw file but are not copied to the clean file.

## 4. Methodology

1. **Data preparation** (`src/prepare_data.py`): remove empty rows and duplicates; clean the text
   (remove links and @mentions, lowercase, join contractions such as *can't → cant*, remove
   punctuation); tidy the complaint reasons into 8 themes.
2. **Exploratory data analysis** (`src/eda.py`): sentiment split, sentiment by airline, daily
   trend, complaint themes, themes by airline, message length.
3. **Feature extraction**: TF-IDF on single words and two-word phrases. Stop words are kept,
   because words such as *not* and *no* carry the sentiment.
4. **Modelling** (`src/train_model.py`): Naive Bayes, Logistic Regression and Linear SVM are
   compared for each of the two tasks.
5. **Validation**: 80% training, 20% test, split so both parts have the same class mix. The model
   is chosen by 5-fold cross-validation on the training part only; the test part is used once.
6. **Deployment** (`app.py`): a Flask web server and a web page (HTML, CSS, JavaScript) that
   sends new messages to the saved models and shows the answer, with the words that led to it.

Because 63.5% of messages are negative, accuracy alone is misleading. The main score is
**macro F1**, which gives every class the same weight.

## 5. Results

### What the data shows

- **63.5%** of messages are negative, 20.9% neutral and 15.5% positive.
- The negative share differs a lot by airline: **78%** for US Airways down to **37%** for
  Virgin America.
- **Customer Service** is the top complaint theme (36.5% of themed complaints), followed by
  **Late Flight** (20.9%), **Cancelled Flight** (10.5%) and **Luggage Issue** (10.0%).
- Message volume roughly doubled on 22 and 23 February (about 3,000 a day against about 1,400),
  and almost all of the increase was negative.
- Negative messages are longer: a median of 20 words against 13 for the other two.

### Sentiment model (2,857 test messages)

| Model | CV macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|
| Baseline (always "negative") | – | 63.5% | 0.26 |
| Naive Bayes | 0.608 | 76.8% | 0.64 |
| **Logistic Regression** (selected) | 0.734 | 80.3% | 0.75 |
| Linear SVM | 0.729 | 80.9% | 0.74 |

The model is chosen by the cross-validation score, not the test score, which is why Logistic
Regression is selected although the SVM's test accuracy is 0.6 points higher.

Per class, the selected model reaches F1 0.88 for negative, 0.73 for positive and 0.63 for
neutral. Neutral is the hardest class, because neutral messages are often questions that share
words with complaints.

### Complaint-theme model (1,580 test messages, 8 themes)

| Model | CV macro F1 | Test accuracy | Test macro F1 |
|---|---|---|---|
| Baseline (always "Customer Service") | – | 36.5% | 0.07 |
| Naive Bayes | 0.313 | 57.7% | 0.32 |
| **Logistic Regression** (selected) | 0.586 | 67.5% | 0.58 |
| Linear SVM | 0.547 | 68.5% | 0.55 |

The four large themes are recognised well (F1 0.72 to 0.76). The small themes are weak
(Bad Flight Experience 0.43, Booking Problem 0.48, Flight Attendant 0.49, Long Lines 0.27),
because they have few examples and overlap with Customer Service.

Performance is good for sentiment and moderate for themes. The models are meant to sort
feedback for an analyst, not to replace reading.

Full numbers are in `outputs/model_comparison.csv`, `outputs/metrics.json` and
`outputs/eda_summary.txt`; charts are in `outputs/figures/`.

## 6. Recommendations

1. Treat **customer service response** as the first improvement area; it is the largest theme
   for five of the six airlines.
2. Watch the **daily negative count** as an early-warning signal; the 22 to 23 February spike
   would have been visible within hours.
3. Route complaints by predicted theme (luggage, cancellations, delays) to the team that
   handles them.
4. Send low-confidence predictions to a person; the web page flags these.

## 7. How to run

```bash
pip install -r requirements.txt
python src/prepare_data.py     # clean the raw data
python src/eda.py              # analysis + charts
python src/train_model.py      # train, compare and save the models
python app.py                  # start the web page
```

Then open http://localhost:5000 in a browser. The page works without an internet connection.
The data, models and outputs are already included, so to see the page only the first and the
last command are needed.

## 8. Project structure

```
app.py                    Flask web server (connects the page to the models)
templates/index.html      the web page
static/style.css          page styles
static/app.js             page behaviour and charts
static/vendor/            Plotly chart library (local copy)
requirements.txt
src/common.py             shared paths, text cleaning, chart style
src/prepare_data.py       data cleaning
src/eda.py                exploratory analysis
src/train_model.py        model training and evaluation
data/Tweets.csv           raw public dataset
data/feedback_clean.csv   cleaned dataset
models/                   saved models
outputs/                  metrics, summaries, figures
docs/                     the three PDF guides (and their Markdown sources)
```

## Documents

| File | For |
|---|---|
| `docs/1_Project_Explained_Simply.pdf` | Understanding the project with no NLP background |
| `docs/2_Code_Explained.pdf` | Every file and function, and how they work together |
| `docs/3_Viva_Questions_and_Answers.pdf` | 90 expected questions: project, NLP and machine learning |

## 9. Limitations and future work

- The data covers 9 days, one industry and one channel (Twitter, 2015). The models must be
  retrained before use on other feedback.
- The labels come from human labellers who sometimes disagreed; 1,175 complaints had no clear
  theme and were left out of the theme model.
- TF-IDF ignores word order and sarcasm. A transformer model such as BERT would probably score
  higher, at the cost of being harder to explain and slower to train.
- Each complaint gets one theme, although some messages mention two problems.
- The small themes need more labelled examples.
