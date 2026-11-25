# Viva Questions and Answers

<p class="subtitle">Customer Feedback Analysis: Sentiment and Complaint Themes<br>Expected questions on the project, on NLP and on machine learning</p>

The answers are short on purpose, so they can be said aloud. Do not memorise them word for word;
understand the idea and say it in your own words. Part H at the end is a one-page revision sheet.

| Part | Topic | Questions |
|---|---|---|
| A | The project and the internship | 1 to 10 |
| B | The data and the cleaning | 11 to 19 |
| C | TF-IDF and the models used | 20 to 31 |
| D | Results and evaluation | 32 to 42 |
| E | The web page and the code | 43 to 49 |
| F | General NLP | 50 to 72 |
| G | General machine learning | 73 to 90 |
| H | One-page revision sheet | |

## Part A. The project and the internship

#### Q1. Explain your project in a few lines.

Companies get more customer messages than people can read. My project reads each message
automatically, labels it as negative, neutral or positive, and for negative messages finds the
complaint theme. I trained it on 14,640 real customer messages. The sentiment model is 80.3%
accurate. The results are shown on a web page where a new message can be typed and analysed.

#### Q2. What is the problem statement?

Most customer feedback goes unread, so a business does not know quickly and consistently how
customers feel or what their main complaints are.

#### Q3. What are the objectives?

Four: clean and analyse real customer messages; build a sentiment model; build a complaint-theme
model; and deliver a web page that shows the results and analyses new messages.

#### Q4. How is this related to your internship?

I was a Research Analyst at Concentrix, a customer-experience company. The role was about
reviewing information accurately and by fixed quality standards. This project applies the same
work to customer feedback, with machine learning, so thousands of messages can be reviewed in
seconds.

#### Q5. Did you use company data?

No. Company and client data is confidential. I used a public dataset called Twitter US Airline
Sentiment, published by CrowdFlower on Kaggle.

#### Q6. Why did you choose this topic?

Customer feedback is the core of a customer-experience business, and it is text, so it let me
apply NLP, which is a main subject of my branch, AI and Data Science.

#### Q7. Who would use this system?

A research analyst or a customer-support manager. They would see the main complaints each day,
notice a sudden rise in negative messages, and send complaints to the right team.

#### Q8. What is new or useful in your project?

It does two tasks, not one: sentiment and complaint theme. It shows the words that led to each
prediction, so the answer can be explained. And it warns the user when the model is unsure.

#### Q9. What were the main challenges?

Three. The data is unbalanced: 63.5% is negative. Tweets are short and messy. And the complaint
themes overlap, with some themes having very few examples.

#### Q10. What did you learn?

How to clean real text, how to turn text into numbers, how to compare models fairly without
leaking test data, why accuracy can mislead on unbalanced data, and how to connect a model to a
web page.

## Part B. The data and the cleaning

#### Q11. Describe your dataset.

14,640 tweets sent to 6 US airlines between 16 and 24 February 2015. Each one was labelled by
people with a sentiment and, for negative ones, a complaint reason. After cleaning, 14,283
messages remain.

#### Q12. Why this dataset?

It has real customer messages, it has human labels for both of my tasks, and it is widely used,
so my results can be compared with other work.

#### Q13. What cleaning steps did you apply?

I removed links, @mentions, punctuation and emojis; changed the text to lowercase; joined
contractions so "can't" becomes "cant"; and deleted duplicates and empty messages.

#### Q14. Why did you remove duplicates?

If the same message is in both the training and the test data, the model has already seen the
answer, so the test score becomes too high. This is called data leakage. I removed 357
duplicates.

#### Q15. Why did you remove @mentions?

The mention is the airline's name. Some airlines get more negative messages, so the model could
learn "this airline means negative" and ignore the real words. Removing mentions also removes
user names.

#### Q16. Why did you not remove stop words?

Standard stop-word lists include "not", "no" and "never". Removing them turns "not good" into
"good". For sentiment these words are essential, so I kept all words.

#### Q17. Why no stemming or lemmatization?

I tested the simple approach first and it worked well. Word forms can also carry meaning:
"delayed" is a complaint, while "delay" may be a question. With bigrams and 14,000 messages the
model had enough data to learn each form. It can be tried as future work.

#### Q18. Is your data balanced? How did you handle it?

No: 63.5% negative, 20.9% neutral, 15.5% positive. I handled it in three ways:
`class_weight="balanced"` in the models, a stratified split, and macro F1 as the main score.

#### Q19. What are the complaint themes, and why did you change them?

The raw data has 10 reasons. I merged "Lost Luggage" and "Damaged Luggage" into "Luggage Issue",
because Damaged Luggage had only 74 messages. I also left out "Can't Tell" (1,175 messages),
because it is not a real theme. That leaves 8 themes.

## Part C. TF-IDF and the models used

#### Q20. Why must text be converted to numbers?

Machine-learning models work with numbers only. They cannot read letters. So each message must
become a row of numbers, called a feature vector.

#### Q21. What is TF-IDF?

Term Frequency, Inverse Document Frequency. It gives each word in a message a score. The score
is high when the word appears often in that message (TF) and is rare across all messages (IDF).
Common words like "the" get a low score; telling words like "cancelled" get a high one.

#### Q22. Give the formula.

TF-IDF(word, message) = TF × IDF. TF is the number of times the word appears in the message.
IDF = log(N / df), where N is the total number of messages and df is the number of messages that
contain the word. scikit-learn uses a smoothed version: log((1 + N) / (1 + df)) + 1.

#### Q23. What are n-grams? Which did you use?

An n-gram is a group of n words in a row. A unigram is one word and a bigram is two. I used
both, so "not good" is one feature and is not confused with "good".

#### Q24. What do min_df=2 and sublinear_tf=True mean?

`min_df=2` ignores words found in only one message, which are mostly typos. `sublinear_tf`
replaces the count with 1 + log(count), so a word repeated five times does not count five times
as much.

#### Q25. Why TF-IDF and not word embeddings or BERT?

TF-IDF is simple, fast and works well on short texts with a small dataset. I can explain every
number in it and show which words drive each prediction. BERT would probably score a few points
higher but needs more computing power and is harder to explain. It is my future work.

#### Q26. Which models did you compare, and why these?

Naive Bayes, Logistic Regression and Linear SVM. They are the three standard models for text
classification with TF-IDF. They work well with many features and train in seconds.

#### Q27. How does Naive Bayes work?

It uses Bayes' theorem. For each class it multiplies the probability of the class by the
probability of each word given that class, and picks the class with the highest result. It is
called "naive" because it assumes the words are independent of each other.

#### Q28. How does Logistic Regression work?

It learns a weight for every word for every class. For a message, it multiplies each word's
TF-IDF score by its weight and adds them up, giving one total per class. The softmax function
turns the totals into probabilities that add up to 1. The highest probability is the answer.

#### Q29. How does a Linear SVM work?

It finds the boundary between classes that leaves the widest gap, called the margin. The
messages closest to the boundary are called support vectors. An SVM gives a distance, not a
probability, so I wrapped it in a calibration step to get probabilities.

#### Q30. Why was Logistic Regression selected?

It had the best cross-validation macro F1 on the training data: 0.734, against 0.729 for the SVM
and 0.608 for Naive Bayes. It also gives probabilities directly, and its word weights make the
predictions easy to explain.

#### Q31. The SVM has slightly higher test accuracy. Why did you not choose it?

The model must be chosen without looking at the test set, otherwise the test score is no longer
an honest estimate. I chose by cross-validation, where Logistic Regression is best. The test
difference is tiny, 80.3% against 80.9%, and Logistic Regression has the better test macro F1,
0.75 against 0.74.

## Part D. Results and evaluation

#### Q32. What are your results?

Sentiment: 80.3% accuracy and 0.75 macro F1, against a 63.5% baseline. Complaint themes: 67.5%
accuracy and 0.58 macro F1, against a 36.5% baseline.

#### Q33. What is a baseline?

The score of the simplest possible guess: always answering the most common class. A model is
only useful if it beats the baseline. Mine beats it by 16.8 points for sentiment and 31.1 points
for themes.

#### Q34. Define accuracy, precision, recall and F1.

Accuracy: the share of all predictions that are correct. Precision: of the messages predicted as
a class, how many really belong to it. Recall: of the messages that really belong to a class, how
many were found. F1: the harmonic mean of precision and recall, 2PR / (P + R).

#### Q35. Why macro F1 and not accuracy?

The classes are unbalanced. A model that always answers "negative" gets 63.5% accuracy but is
useless; its macro F1 is only 0.26. Macro F1 averages the F1 of each class with equal weight, so
the small classes count as much as the large one.

#### Q36. What is a confusion matrix? What does yours show?

A table with the real classes as rows and the predicted classes as columns. The diagonal is the
correct predictions. Mine shows that 87% of negative messages are found. Most errors are
between negative and neutral, in both directions. Negative and positive are rarely confused.

#### Q37. Which class is hardest, and why?

Neutral, with F1 0.63. Neutral messages are often questions, such as "can I change my flight?",
and they use the same words as complaints.

#### Q38. Why is the theme model weaker?

It chooses between 8 classes instead of 3, it has less data (7,899 messages), and the themes
overlap. Small themes have few examples: Long Lines has only 177. The four big themes score
0.72 to 0.76; the small ones 0.27 to 0.49.

#### Q39. How did you split the data?

80% for training and 20% for testing, stratified, with a fixed random seed. That is 11,426
training and 2,857 test messages for sentiment.

#### Q40. What is cross-validation and why did you use it?

The training data is cut into 5 parts. The model trains on 4 and is scored on the fifth, 5
times. The average is more reliable than one split. I used it to choose the model without
touching the test set.

#### Q41. Is your model overfitting?

No. The cross-validation score (0.734) and the test score (0.748) are close. An overfitted model
would score much lower on unseen data.

#### Q42. Is 80% accuracy good enough?

For summarising thousands of messages, yes: totals and trends are reliable. For acting on one
single message, a person should check. The page warns when confidence is below 50%. Also, the
human labellers themselves did not always agree, so 100% is not possible on this data.

## Part E. The web page and the code

#### Q43. How is the web page built?

The server is written in Python with Flask. The page is HTML, CSS and JavaScript. The page asks
the server for data through API routes, and the server answers in JSON. The charts are drawn
with Plotly.

#### Q44. What is Flask? Why did you use it?

Flask is a small Python web framework. I used it because the models are in Python, so the same
language loads the model and serves the page, and Flask needs very little code.

#### Q45. What is an API? What is JSON?

An API is a set of web addresses that a program can call to get data or ask for an action. JSON
is a simple text format for data, made of names and values, that both Python and JavaScript
understand.

#### Q46. What happens when I click "Analyse message"?

JavaScript sends the text to `/api/predict`. The server cleans the text, the pipeline turns it
into TF-IDF numbers and Logistic Regression gives a probability for each class. The server sends
back the sentiment, the confidence, the theme and the key words, and JavaScript shows them.

#### Q47. How do you show "words that led to this result"?

For each word in the message, I multiply its TF-IDF score by the weight the model learned for
the predicted class. The words with the largest result are shown.

#### Q48. How is the model saved and loaded?

With joblib. `joblib.dump` writes the trained pipeline to a file, and `joblib.load` reads it
back. The server loads the models once at start-up, so it does not retrain on every request.

#### Q49. What is a Pipeline and why did you use it?

A Pipeline joins TF-IDF and the classifier into one object. It prevents data leakage during
cross-validation, and it means the saved model accepts text directly.

## Part F. General NLP

#### Q50. What is NLP?

Natural Language Processing is the field of AI that makes computers able to understand, analyse
and generate human language.

#### Q51. Give some applications of NLP.

Sentiment analysis, spam detection, machine translation, chatbots, search engines, text
summarisation, speech recognition and autocomplete.

#### Q52. What is sentiment analysis? What types are there?

Finding the opinion or feeling in a text. Types: polarity (positive, negative, neutral),
fine-grained (1 to 5 stars), emotion detection (angry, happy, sad) and aspect-based (the
sentiment about each feature, for example "food good, service bad").

#### Q53. What is a corpus, a document and a vocabulary?

A corpus is the whole collection of texts. A document is one text; in my project, one message.
The vocabulary is the set of all different words in the corpus.

#### Q54. What is tokenization?

Splitting text into small units called tokens, usually words. "flight was late" becomes
["flight", "was", "late"]. In my project TfidfVectorizer does this.

#### Q55. What are stop words?

Very common words such as "the", "is" and "and" that usually carry little meaning. They are
often removed. I kept them because negation words matter for sentiment.

#### Q56. What is the difference between stemming and lemmatization?

Both reduce a word to a base form. Stemming cuts the ending by rule and can give non-words:
"studies" becomes "studi". Lemmatization uses a dictionary and gives a real word: "studies"
becomes "study", "better" becomes "good". Stemming is faster; lemmatization is more accurate.

#### Q57. What is Bag of Words?

A way to represent text by counting how many times each vocabulary word appears, ignoring grammar
and word order. Each message becomes a row of counts.

#### Q58. What is the difference between Bag of Words and TF-IDF?

Bag of Words uses raw counts, so common words get large values. TF-IDF multiplies the count by
how rare the word is, so common words are pushed down and telling words are pushed up.

#### Q59. What are the limits of Bag of Words and TF-IDF?

They ignore word order and context, they do not know that "good" and "great" are similar, and
they produce very large, sparse vectors (mostly zeros).

#### Q60. What is a sparse matrix?

A table where almost all values are zero. A TF-IDF table has one column per word, and a single
message uses only a few of them. It is stored in a special compact form to save memory.

#### Q61. What are word embeddings?

Dense vectors, usually 100 to 300 numbers, that represent a word's meaning. Words with similar
meanings get similar vectors, so "good" and "great" are close together.

#### Q62. What is Word2Vec?

A method that learns word embeddings from a large amount of text by predicting a word from its
neighbours (CBOW) or the neighbours from a word (Skip-gram). GloVe and FastText are similar
methods.

#### Q63. What is the difference between TF-IDF and word embeddings?

TF-IDF is based on counts; each word is a separate column and no meaning is captured. Embeddings
are learned; each word is a short dense vector that captures meaning and similarity.

#### Q64. What are RNN and LSTM?

A Recurrent Neural Network reads a sentence word by word and keeps a memory of what it has read,
so it uses word order. A plain RNN forgets long sentences. An LSTM (Long Short-Term Memory) adds
gates that decide what to keep and what to forget, so it remembers longer.

#### Q65. What is a transformer?

A neural network design from 2017 that reads all the words at once, not one by one, and uses
attention to work out which words relate to each other. It is faster to train and better with
long texts. BERT and GPT are transformers.

#### Q66. What is attention?

A mechanism that lets the model, for each word, look at all the other words and give more weight
to the important ones. In "the bag that I checked in was lost", attention links "lost" to "bag".

#### Q67. What is BERT?

Bidirectional Encoder Representations from Transformers, by Google. It is pre-trained on a huge
amount of text and reads each word using the words on both its left and its right. It can then
be fine-tuned for a task such as sentiment analysis with a small labelled dataset.

#### Q68. What is the difference between BERT and GPT?

Both are transformers. BERT is an encoder that reads in both directions and is used to
understand text: classification, question answering. GPT is a decoder that reads left to right
and is used to generate text.

#### Q69. What is a large language model (LLM)?

A very large transformer trained on huge amounts of text to predict the next word. Examples are
GPT, Claude and Gemini. They can do many language tasks without task-specific training.

#### Q70. What are pre-training and fine-tuning?

Pre-training: a model learns general language from a huge unlabelled text collection.
Fine-tuning: the pre-trained model is trained a little more on a small labelled dataset for one
task. This is called transfer learning.

#### Q71. What is the difference between lexicon-based and machine-learning sentiment analysis?

Lexicon-based uses a ready dictionary of positive and negative words and adds up their scores;
VADER and TextBlob work this way and need no training data. Machine learning learns from
labelled examples, as in my project, and adapts to the domain.

#### Q72. Why is sentiment analysis difficult?

Sarcasm ("great, another delay"), negation ("not bad"), mixed opinions in one message, slang and
spelling mistakes, and meaning that depends on context.

## Part G. General machine learning

#### Q73. What is machine learning?

A way for computers to learn rules from data instead of being given the rules by a programmer.

#### Q74. What is the difference between supervised and unsupervised learning?

Supervised learning uses data with labels, to predict the label for new data; my project is
supervised. Unsupervised learning uses data without labels, to find groups or patterns, for
example clustering.

#### Q75. What is the difference between classification and regression?

Classification predicts a category, such as negative or positive. Regression predicts a number,
such as a price.

#### Q76. Is your problem binary or multi-class?

Multi-class. Sentiment has 3 classes and complaint theme has 8. Each message gets exactly one
class.

#### Q77. Why is it called Logistic "Regression" if it does classification?

It first computes a number, as linear regression does, and then passes it through a function
(sigmoid for two classes, softmax for more) that turns it into a probability. The name comes
from that first step.

#### Q78. What are the sigmoid and softmax functions?

Sigmoid turns any number into a value between 0 and 1: 1 / (1 + e^-x). It is used for two
classes. Softmax turns several numbers into probabilities that add up to 1. It is used for more
than two classes.

#### Q79. What is Bayes' theorem?

P(A | B) = P(B | A) × P(A) / P(B). In my case: the probability of a class given the words equals
the probability of the words given the class, times the probability of the class, divided by the
probability of the words.

#### Q80. What is Laplace smoothing?

If a word never appeared in a class during training, its probability would be zero and would
cancel the whole product in Naive Bayes. Smoothing adds a small value (alpha) to every count so
nothing is zero.

#### Q81. What are overfitting and underfitting?

Overfitting: the model memorises the training data, including noise, and does badly on new data.
Underfitting: the model is too simple and does badly on both. Signs of overfitting are a high
training score and a much lower test score.

#### Q82. How do you prevent overfitting?

More data, a simpler model, regularization, cross-validation, and removing rare features (I used
`min_df=2`).

#### Q83. What is regularization? What is C?

Regularization adds a penalty for large weights, which keeps the model simple. In scikit-learn,
C is the inverse of the regularization strength: a small C means strong regularization.

#### Q84. What is the bias-variance trade-off?

Bias is error from a model that is too simple. Variance is error from a model that is too
sensitive to its training data. Reducing one usually raises the other; the goal is the balance
that gives the lowest total error.

#### Q85. What is a hyperparameter?

A setting chosen before training, not learned from the data. Examples in my project: C,
`ngram_range`, `min_df` and alpha.

#### Q86. What is the difference between training, validation and test data?

Training data is used to learn. Validation data is used to choose between models or settings.
Test data is used once at the end for the final score. In my project cross-validation plays the
role of the validation data.

#### Q87. What is data leakage?

When information that would not be available in real use gets into training, so the scores look
better than they are. Examples: duplicates across train and test, or fitting TF-IDF on all data
before splitting.

#### Q88. How can class imbalance be handled?

Class weights (what I used), oversampling the small class (for example SMOTE), undersampling the
large class, collecting more data, and using suitable scores such as macro F1.

#### Q89. When is precision more important, and when recall?

Precision matters when a false alarm is costly, for example marking a real email as spam. Recall
matters when a miss is costly, for example missing a disease or an angry customer. For
complaints, recall on the negative class is important.

#### Q90. What other models could be used for text classification?

Decision Tree, Random Forest, XGBoost, k-Nearest Neighbours, and neural networks such as CNN,
LSTM and BERT. Tree models are usually weaker than linear models on TF-IDF features, because the
data has very many sparse columns.

## Part H. One-page revision sheet

### Numbers

| Fact | Number |
|---|---|
| Raw messages / after cleaning | 14,640 / 14,283 |
| Duplicates removed | 357 |
| Airlines / period | 6 / 16 to 24 February 2015 |
| Negative / neutral / positive | 63.5% / 20.9% / 15.5% |
| Training / test messages (sentiment) | 11,426 / 2,857 |
| Sentiment: accuracy / macro F1 / baseline | 80.3% / 0.75 / 63.5% |
| Sentiment F1 per class | negative 0.88, positive 0.73, neutral 0.63 |
| Themes: accuracy / macro F1 / baseline | 67.5% / 0.58 / 36.5% |
| Number of themes | 8 |
| Top complaint theme | Customer Service, 36.5% |
| Most / least negative airline | US Airways 78% / Virgin America 37% |

### The pipeline in one line

Raw text → clean → TF-IDF (unigrams and bigrams) → Logistic Regression → sentiment → (if
negative) complaint theme → web page.

### Five sentences to remember

1. "I kept stop words because 'not' and 'no' decide the sentiment."
2. "I used macro F1 because 63.5% of the data is negative, so accuracy alone misleads."
3. "I chose the model by cross-validation, never by the test set."
4. "The Pipeline prevents data leakage and lets the saved model accept text directly."
5. "The model is a tool to sort feedback; a person still checks the uncertain cases."

### If you do not know an answer

Say: "I have not studied that in detail, but my understanding is..." and give the basic idea.
An honest partial answer is better than a guess stated as a fact.
