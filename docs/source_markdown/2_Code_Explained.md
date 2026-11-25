# The Code Explained

<p class="subtitle">Customer Feedback Analysis: Sentiment and Complaint Themes<br>Every file and every function, and how they work together</p>

This guide walks through the code file by file. For each function it says what goes in, what
comes out, and why it is written that way. Read "The Project Explained Simply" first.

## 1. The big picture

### The files

| File | Job | Language |
|---|---|---|
| `src/common.py` | Shared settings and the text-cleaning function | Python |
| `src/prepare_data.py` | Cleans the raw data | Python |
| `src/eda.py` | Draws the analysis charts | Python |
| `src/train_model.py` | Trains, compares and saves the models | Python |
| `app.py` | The web server; connects the page to the models | Python (Flask) |
| `templates/index.html` | The structure of the web page | HTML |
| `static/style.css` | The look of the web page | CSS |
| `static/app.js` | The behaviour of the web page | JavaScript |

### The order in which things run

<div class="flow">
<span>prepare_data.py</span><span>eda.py</span><span>train_model.py</span><span>app.py</span>
</div>

```
data/Tweets.csv
      |  prepare_data.py
      v
data/feedback_clean.csv
      |-- eda.py ----------> outputs/figures/01 to 06, outputs/eda_summary.txt
      |-- train_model.py --> models/sentiment_model.joblib, models/theme_model.joblib,
      |                      outputs/metrics.json, outputs/model_comparison.csv
      v
app.py  (loads the clean data, the two models and the outputs, then serves the web page)
```

The first three scripts are run once. After that only `app.py` is needed.

### The libraries

| Library | What it is used for |
|---|---|
| pandas | Working with tables: read a CSV, filter, group, count |
| scikit-learn | TF-IDF, the three models, the train/test split, the scores |
| matplotlib, seaborn | Drawing the charts saved as pictures |
| joblib | Saving a trained model to a file and loading it again |
| Flask | The web server |
| Plotly.js | The interactive charts on the web page |

## 2. src/common.py

This file holds everything that more than one script needs. Writing it once avoids copying the
same code into four files.

### Paths

```python
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "Tweets.csv"
```

`__file__` is the location of `common.py`. `.parent.parent` goes up two folders to the project
folder. Every other path is built from `ROOT`, so the project works on any computer, wherever
the folder is placed.

### Constants

| Name | Value | Purpose |
|---|---|---|
| `RANDOM_STATE` | 42 | A fixed seed. Any step that uses randomness gives the same result on every run. |
| `SENTIMENTS` | negative, neutral, positive | The three classes, always in the same order |
| `SENT_COLORS` | red, grey, green | One colour per class, used in every chart |
| `UNCLEAR_THEME` | "Can't Tell" | The label for complaints with no clear theme |
| `THEME_NAMES` | a dictionary | Maps the 10 raw complaint reasons to 8 tidy themes |

`THEME_NAMES` merges "Lost Luggage" and "Damaged Luggage" into "Luggage Issue". Damaged Luggage
had only 74 messages, which is too few for a model to learn from.

### clean_text(text)

**Input:** one raw message. **Output:** the same message as plain lowercase words.

```python
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.lower()
    text = re.sub(r"['’`]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
```

| Line | What it does | Example |
|---|---|---|
| `html.unescape` | Turns HTML codes back into symbols | `&amp;` becomes `&` |
| first `re.sub` | Removes links | `http://t.co/x1` is removed |
| second `re.sub` | Removes @mentions | `@united` is removed |
| `.lower()` | Makes everything lowercase | `LATE` becomes `late` |
| third `re.sub` | Deletes apostrophes | `can't` becomes `cant` |
| fourth `re.sub` | Replaces anything that is not a letter, digit or space | `!!`, emojis and `#` are removed |
| last `re.sub` | Turns several spaces into one | tidy result |

`re.sub(pattern, replacement, text)` is a "find and replace" that uses a pattern. The patterns
are called regular expressions. For example `@\w+` means "an @ sign followed by one or more
letters or digits".

**Why remove @mentions?** The mention is the airline's name. Some airlines receive more negative
messages than others. If the name stayed in the text, the model could learn "this airline name
means negative" and stop reading the real words.

**Why delete the apostrophe and not the whole word?** "can't" must stay one word, "cant". It is
a negation, and negations decide the sentiment.

This function is used in **two** places: when preparing the training data and when a user types
a new message on the web page. Both must be cleaned in exactly the same way, otherwise the model
would see words in a form it never learned.

### load_data()

Reads `feedback_clean.csv` into a pandas table, turns the `created_at` column into real dates,
replaces empty values with empty text, and adds a `date` column (the day without the time).

### set_chart_style()

Sets one look for all saved charts. `matplotlib.use("Agg")` tells matplotlib to save pictures to
files without opening a window.

## 3. src/prepare_data.py

**Input:** `data/Tweets.csv` (14,640 rows). **Output:** `data/feedback_clean.csv` (14,283 rows).

It has one function, `main()`, which does seven things in order.

**1. Read the raw file.** `pd.read_csv(RAW_PATH)` loads the CSV into a table.

**2. Keep only the useful columns, with clear names.**

```python
df = pd.DataFrame({
    "feedback_id": raw["tweet_id"],
    "created_at": pd.to_datetime(raw["tweet_created"].str.slice(0, 19)),
    "airline": raw["airline"],
    "sentiment": raw["airline_sentiment"],
    "label_confidence": raw["airline_sentiment_confidence"],
    "complaint_theme": raw["negativereason"].map(THEME_NAMES),
    "text": raw["text"],
})
```

- `.str.slice(0, 19)` keeps the first 19 characters of the time, for example
  `2015-02-24 11:35:52`, and drops the time-zone part.
- `.map(THEME_NAMES)` replaces each raw reason with its tidy theme name.
- The user name and location columns are **not** copied. They are personal details and the
  analysis does not need them.

**3. Drop incomplete rows.** `df.dropna(subset=["text", "sentiment"])` removes rows with no
message or no label.

**4. Clean every message.** `df["text"].map(clean_text)` runs `clean_text` on each row and
stores the result in a new column, `clean_text`. Messages that become empty are removed.

**5. Remove duplicates.**

```python
df = df.sort_values("created_at").drop_duplicates(subset=["airline", "clean_text"], keep="first")
```

Two rows are duplicates when the airline and the cleaned text are the same. 357 rows were
removed. This matters: if the same message is in the training data and in the test data, the
model has already seen the answer, and the test score would be too high. This problem is called
**data leakage**.

**6. Add two helper columns.** `word_count` (number of words) and `hour` (hour of the day).

**7. Save.** `df.to_csv(CLEAN_PATH, index=False)`.

## 4. src/eda.py

**Input:** the clean data. **Output:** six charts and `outputs/eda_summary.txt`.

### save(fig, name)

A two-line helper that saves a chart into `outputs/figures` and closes it to free memory.

### main()

It first makes three tables: `df` (all messages), `neg` (negative messages only) and `themed`
(negative messages with a clear theme). Then it draws six charts.

| Chart | Key line of code | What the line does |
|---|---|---|
| 01 Sentiment split | `df["sentiment"].value_counts()` | Counts the messages in each class |
| 02 Sentiment by airline | `pd.crosstab(df["airline"], df["sentiment"], normalize="index")` | Counts airline against sentiment, then turns each row into percentages |
| 03 Daily trend | `pd.crosstab(df["date"], df["sentiment"])` | Counts messages per day and sentiment |
| 04 Complaint themes | `themed["complaint_theme"].value_counts()` | Counts each theme |
| 05 Theme by airline | `sns.heatmap(...)` | Draws a coloured grid of percentages |
| 06 Message length | `sns.boxplot(...)` | Compares word counts between the classes |

`normalize="index"` is important. United has 3,786 messages and Virgin America has 494.
Comparing counts would be unfair, so each row is changed to percentages.

The daily chart leaves out days with fewer than 100 messages. 16 February has only 4 messages,
and a point for it would mislead the reader.

The last part builds a list of text lines with the key numbers and writes it to
`eda_summary.txt`.

## 5. src/train_model.py

**Input:** the clean data. **Output:** two saved models, the scores, and four charts.

This is the most important file. It has seven functions.

### make_vectorizer()

```python
return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
```

Creates the tool that turns text into TF-IDF numbers.

| Setting | Meaning | Why |
|---|---|---|
| `ngram_range=(1, 2)` | Use single words and pairs of words | "not good" must be different from "good" |
| `min_df=2` | Ignore a word that appears in only one message | Removes typos and one-off words |
| `sublinear_tf=True` | Count a repeated word as 1 + log(count) | A word used 5 times is not 5 times as important |
| no `stop_words` | Common words are kept | Stop-word lists contain "not" and "no", which carry the sentiment |

### make_candidates()

Returns a dictionary of the three models. Each one is a **Pipeline**:

```python
"Logistic Regression": Pipeline([
    ("tfidf", make_vectorizer()),
    ("clf", LogisticRegression(C=5, max_iter=2000, class_weight="balanced"))]),
```

A Pipeline joins steps into one object. Here: step 1 turns text into numbers, step 2 classifies.
You give it text and it gives you an answer.

| Setting | Meaning |
|---|---|
| `C=5` | How closely the model may fit the training data. A larger value fits more closely. |
| `max_iter=2000` | The largest number of learning steps allowed |
| `class_weight="balanced"` | A mistake on a small class costs more than a mistake on a big class |
| `alpha=0.3` (Naive Bayes) | Smoothing, so a word never seen in a class does not give a probability of zero |
| `CalibratedClassifierCV` (SVM) | Lets the SVM give probabilities, which it cannot do alone |

`class_weight="balanced"` is needed because 63.5% of the messages are negative. Without it the
model would learn to answer "negative" too often.

### train_task(name, X, y, labels)

The main function. It is used twice: once for sentiment and once for complaint themes. `X` is
the cleaned text and `y` is the correct label.

**Step 1: split the data.**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
```

20% is put aside as the test set. `stratify=y` keeps the same share of each class in both parts.

**Step 2: the baseline.** `DummyClassifier(strategy="most_frequent")` always answers the most
common class. Its score is the minimum a real model must beat.

**Step 3: cross-validation.**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()
```

The training data is cut into 5 parts. The model trains on 4 parts and is scored on the fifth.
This is repeated 5 times, with a different part held out each time. The average of the 5 scores
is the model's CV score.

**Step 4: fit and predict.** `pipe.fit(X_train, y_train)` trains the model on all the training
data. `pipe.predict(X_test)` labels the test messages.

**Step 5: choose the best model.**

```python
best_name = table.dropna().sort_values("cv_macro_f1", ascending=False).iloc[0]["model"]
```

The model with the highest **CV score** wins. The test score is not used for choosing. The test
set must stay unseen, so that its score is an honest estimate.

**Step 6: measure the winner.** `classification_report` gives precision, recall and F1 for each
class. `confusion_matrix` counts, for each real class, what the model answered.

**It returns** the best model, all fitted models, the comparison table, the confusion matrix
and a dictionary of results.

<div class="box">
<b>Why the Pipeline prevents leakage.</b> During cross-validation, the TF-IDF vocabulary is
learned again each time, from the 4 training parts only. The held-out part is never seen while
learning. If TF-IDF were fitted once on all the data before splitting, information from the
held-out part would leak into training.
</div>

### plot_comparison, plot_matrix, plot_top_words

Three functions that draw charts.

- `plot_comparison` draws the accuracy and macro F1 of every model as bars.
- `plot_matrix` draws a confusion matrix. The line
  `share = matrix / matrix.sum(axis=1, keepdims=True)` turns each row into percentages.
- `plot_top_words` reads the weights of the Logistic Regression model with `clf.coef_`, sorts
  them with `np.argsort`, and draws the 12 words with the largest weight for each sentiment.

### main()

1. Loads the data.
2. Calls `train_task` for **sentiment**, using all messages.
3. Calls `train_task` for **complaint themes**, using only negative messages that have a clear
   theme (not "Can't Tell").
4. Saves both winning models with `joblib.dump`.
5. Saves the comparison table as CSV and the results as JSON.
6. Draws the four charts.
7. Prints a summary.

## 6. app.py: the web server

The web page and the models live in different places: the page runs in the browser, and the
models run in Python. `app.py` is the bridge between them. It is written with **Flask**.

### How a web application works

1. The browser asks the server for a web address (a **request**).
2. The server runs a Python function and sends back an answer (a **response**).
3. The answer is either a web page (HTML) or data (**JSON**, a simple text format for data).

A web address that returns data is called an **API route**.

### What happens when the server starts

```python
DATA = load_data()
SENT_MODEL = joblib.load(SENTIMENT_MODEL_PATH)["pipeline"]
THEME_MODEL = joblib.load(THEME_MODEL_PATH)["pipeline"]
```

The data and both models are loaded **once**, when the server starts, and kept in memory. Every
later request reuses them, so answers come back in a fraction of a second.

### The routes

A line such as `@app.route("/api/predict", methods=["POST"])` connects a web address to the
function written below it.

| Web address | Function | What it returns |
|---|---|---|
| `/` | `home()` | The web page |
| `/api/overview` | `api_overview()` | Totals, daily counts, sentiment by airline, messages by hour |
| `/api/themes` | `api_themes()` | Theme counts and the airline-by-theme grid |
| `/api/messages` | `api_messages()` | 12 real messages for a chosen theme |
| `/api/predict` | `api_predict()` | The prediction for one typed message |
| `/api/predict-file` | `api_predict_file()` | Predictions for an uploaded CSV file |
| `/api/model` | `api_model()` | The scores and the model comparison |

### The helper functions

**filtered(airline)** returns the messages of one airline, or all messages when no airline is
chosen.

**complaints(df)** returns the negative messages that have a clear theme.

**predict(messages)** is the heart of the application.

```python
cleaned = [clean_text(m) for m in messages]
s_proba = SENT_MODEL.predict_proba(cleaned)
t_proba = THEME_MODEL.predict_proba(cleaned)
```

1. Clean every message with the same `clean_text` used in training.
2. `predict_proba` returns, for each message, one probability per class. For example
   `[0.92, 0.05, 0.03]` means 92% negative, 5% neutral, 3% positive.
3. `sp.argmax()` finds the position of the largest probability. That class is the answer, and
   the probability is the confidence.
4. Only if the answer is "negative", the theme model's top 4 themes are added.
5. If a message has no readable words after cleaning, it is marked as unreadable.

**influential_words(pipeline, cleaned, label)** finds the words that led to the answer.

```python
vector = tfidf.transform([cleaned])
weights = clf.coef_[list(clf.classes_).index(label)]
scores = [(words[i], vector[0, i] * weights[i]) for i in vector.nonzero()[1]]
```

- `vector` holds the TF-IDF score of each word in this message.
- `weights` holds the weight the model learned for each word, for the predicted class.
- A word's influence is its TF-IDF score multiplied by its weight.
- The words are sorted and the top 6 with a positive influence are returned.

### api_predict(), step by step

```python
text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:2000]
if not text:
    return jsonify({"error": "Please type a message first."}), 400
```

1. Read the text that the page sent. Keep at most 2,000 characters.
2. If it is empty, send back an error message with code 400 ("bad request").
3. Call `predict`.
4. Add the influential words.
5. Send the result back as JSON with `jsonify`.

### api_predict_file()

Reads the uploaded file with `pd.read_csv`. It checks three things and sends a clear error if
one fails: the file is a readable CSV, it has a column named `text`, and the column is not
empty. It then predicts up to 5,000 rows and returns the rows with a count of each sentiment.

### Safety checks in the server

- `MAX_CONTENT_LENGTH` limits uploads to 5 MB.
- A typed message is cut at 2,000 characters.
- Errors are returned as short readable messages, not as a crash.
- At start-up the server checks that the data and model files exist and tells the user which
  script to run if they do not.

## 7. The web page

### templates/index.html

The structure of the page: the header with the airline filter, four tab buttons, and four
sections, one per tab. Only one section is visible at a time.

Flask fills in two things before sending the page. `{{ url_for('static', ...) }}` becomes the
address of a file in the `static` folder, and the `{% for a in airlines %}` loop writes one
option per airline into the filter.

### static/style.css

The look of the page: colours, spacing, cards, buttons and tables. The colours are defined once
at the top as variables, for example `--neg: #C0392B`, and reused everywhere.

### static/app.js

The behaviour of the page. The important functions:

| Function | What it does |
|---|---|
| `getJSON(url, options)` | Sends a request to the server and returns the answer. If the server sends an error, it raises it so the page can show it. |
| `el(tag, className, text)` | Creates a page element safely. The text is added with `textContent`, so a message that contains HTML code is shown as text and cannot run. |
| `drawBars(container, rows)` | Draws the horizontal probability bars |
| `showTab(name)` | Shows one tab and hides the others. Loads the tab's data the first time it is opened. |
| `analyse()` | Sends the typed message to `/api/predict` and shows the result |
| `loadOverview()`, `loadThemes()`, `loadMessages()`, `loadModel()` | Fetch the data for a tab and draw its charts and tables |

The key lines of `analyse()`:

```javascript
const r = await getJSON("/api/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
});
$("r-sentiment").textContent = r.sentiment;
```

`fetch` (inside `getJSON`) sends the message to the server. `await` waits for the answer without
freezing the page. The answer `r` is then written into the page.

## 8. One message, from start to finish

Follow the message *"They lost my bag and nobody is helping"* through the whole system.

| Step | Where | What happens |
|---|---|---|
| 1 | Browser | The user types the message and clicks "Analyse message" |
| 2 | `app.js`, `analyse()` | The message is sent to `/api/predict` as JSON |
| 3 | `app.py`, `api_predict()` | The server reads the text and calls `predict` |
| 4 | `common.py`, `clean_text()` | Result: `they lost my bag and nobody is helping` |
| 5 | Pipeline step 1, TF-IDF | The text becomes a long row of numbers; "lost" and "bag" get high scores |
| 6 | Pipeline step 2, Logistic Regression | The word weights are added up for each class; negative gets the highest probability |
| 7 | `app.py`, `predict()` | Because the answer is negative, the theme model runs too: Luggage Issue |
| 8 | `app.py`, `influential_words()` | The words with the most influence are found |
| 9 | `app.py` | Everything is sent back as JSON |
| 10 | `app.js` | The badge, the bars and the words are drawn on the page |

All ten steps take well under one second.

## 9. The saved files

| File | Made by | Contains |
|---|---|---|
| `data/feedback_clean.csv` | `prepare_data.py` | The cleaned messages |
| `models/sentiment_model.joblib` | `train_model.py` | The sentiment pipeline (TF-IDF and Logistic Regression) |
| `models/theme_model.joblib` | `train_model.py` | The complaint-theme pipeline |
| `outputs/metrics.json` | `train_model.py` | Scores of the chosen models, per class, and the confusion matrices |
| `outputs/model_comparison.csv` | `train_model.py` | Scores of all models and the baselines |
| `outputs/top_words.csv` | `train_model.py` | The strongest words for each sentiment |
| `outputs/eda_summary.txt` | `eda.py` | The key numbers of the analysis |
| `outputs/figures/*.png` | `eda.py`, `train_model.py` | Ten charts |
