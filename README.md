# 🎯 Sentiment Analysis of YouTube Comments  
> Predicting Positive vs. Negative Reactions Using NLP and ML Models

## 📘 Overview  
This project builds a binary sentiment classification model to detect **positive** and **negative** YouTube comments, using over **63,000** labeled examples. The goal is to maximize **F1-score** for negative comments, which tend to be underrepresented. Labeling was performed using a hybrid strategy combining **VADER**, **Flair**, and **HuggingFace** zero-shot sentiment models.

> 🎵 Dataset source: Comments from *Justin Bieber – Baby ft. Ludacris* on YouTube  
> 💡 Built during the **Data Science Immersive Bootcamp at General Assembly (2023)**

---

## 📂 Data  

| Feature     | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| `sequence`  | text   | Cleaned comment or reply (post-processed and normalized)                    |
| `label`     | int    | Target label (0 = negative, 1 = positive)                                    |

- Initial dataset: 99,941 raw comments  
- Final modeling set: ~63,000 labeled and preprocessed rows  

---

## 🧪 Methodology

### ✍️ Labeling Strategy
- Labeled 100 samples manually  
- Ran predictions with **VADER**, **Flair**, and **HuggingFace (zero-shot)**  
- Took the **mean probability** of all 3 models to assign final binary labels

### 🔧 Data Preprocessing
| Step                                      | Description                                                   |
|-------------------------------------------|---------------------------------------------------------------|
| Removed non-English text                  | Using langdetect                                               |
| Expanded contractions & removed URLs      | Standard NLP text cleaning                                     |
| Removed emojis, punctuation, numerics     | Cleaned text for modeling consistency                          |
| Applied both **stemming** and **lemmatization** | For model comparisons                                         |
| Removed stopwords and NA rows             | Ensured clean, dense token distribution                        |

---

## 📊 Exploratory Data Analysis (EDA)

| Type                        | Action                                                    |
|----------------------------|-----------------------------------------------------------|
| Token Distribution         | Top 10 tokens, bigrams, trigrams                          |
| POS Tagging                | Top adjectives using spaCy                                |
| Word Count Distribution    | Insights on text length vs. sentiment                     |

---

## 🤖 Modeling & Results

### ✅ Evaluation Metric  
> **Primary metric:** F1-score for negative class (label = 0)  

### 🧮 Vectorizers Used  
- **CountVectorizer**: 15,000 max features, 1–3 n-grams  
- **TF-IDF**: 6,000 features, 1–2 n-grams  
- **Word2Vec** embeddings (Gensim)  

### 🧠 Model Benchmarks

#### 🔹 Naive Bayes  
| Preprocessing | Vectorizer        | F1 Test |
|---------------|-------------------|---------|
| Stemming      | Count (1–3)       | 0.61341 |
| Lemmatization | Count (1–3)       | 0.61526 |
| Stemming      | TF-IDF (1–2)      | 0.57242 |
| Lemmatization | TF-IDF (1–2)      | 0.56727 |

#### 🔹 Logistic Regression  
| Preprocessing | Vectorizer  | F1 Test | Notes                |
|---------------|-------------|---------|----------------------|
| Lemmatization | Count (1–3) | **0.65556** | Best overall model   |
| Word2Vec      | -           | 0.60733 | Lower than expected  |

#### 🔹 Tree-Based & Ensemble  
| Model                         | Vectorizer | F1 Test  |
|-------------------------------|------------|----------|
| Random Forest (Depth 5)       | Word2Vec   | 0.55768  |
| Hist Gradient Boosting        | Word2Vec   | 0.61250  |
| Stacking (LR + RF + XGB)      | Word2Vec   | 0.62219  |
| Random Forest (Depth 50)      | Count      | 0.60019  |
| Gradient Boosting (Depth 100) | Count      | 0.60653  |
| Stacking Ensemble             | Count      | 0.65228  |

---

## ✅ Final Model  
**Logistic Regression + CountVectorizer (1–3-gram, 15,000 features) + Lemmatization**  
- Cross-val F1: 0.64863  
- Train F1: 0.73867  
- **Test F1: 0.65556** (Negative Class)

---

## 🧠 Recommendations

- Introduce a third sentiment label: **Neutral**, for better separation  
- Explore enhanced Word2Vec + Doc2Vec + Transformer-based embeddings  
- Use **SMOTE** or focal loss for class imbalance  
- Add sarcasm detection and emotion classification  

---

## 🔍 Tools & Libraries

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- NLP: NLTK, spaCy, Flair, VADER, HuggingFace Transformers
- Modeling: Logistic Regression, Naive Bayes, Random Forest, XGBoost

---

## 📁 Files

- `project_3_data_collection.ipynb`: Scraping and labeling  
- `project_3_cleaning_eda_modeling.ipynb`: Preprocessing, EDA, modeling  
- `README.md`: Documentation  

---

## 📚 References

1. https://www.youtube.com/watch?v=kffacxfA7G4  
2. https://www.nytimes.com/2020/11/19/learning/what-students-are-saying-about-cancel-culture-friendly-celebrity-battles-and-finding-escape.html  
3. https://www.scmp.com/magazines/style/celebrity/article/3204356/14-celebrities-who-got-cancelled-2022-elon-musks-twitter-mess-and-kanye-wests-controversial-comments  
4. https://pub.towardsai.net/textblob-vs-vader-for-sentiment-analysis-using-python-76883d40f9ae  
5. https://medium.com/@AmyGrabNGoInfo/sentiment-analysis-hugging-face-zero-shot-model-vs-flair-pre-trained-model-57047452225d  
6. https://medium.com/@chyun55555/unsupervised-sentiment-analyis-with-sentiwordnet-and-vader-in-python-a519660198be  
7. https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values  
8. https://towardsdatascience.com/stop-using-smote-to-treat-class-imbalance-take-this-intuitive-approach-instead-9cb822b8dc45  

---

**Author:** Wes Lee  
🔗 [LinkedIn](https://www.linkedin.com/in/wes-lee) · 💻 Portfolio available upon request  
📜 License: MIT

---

