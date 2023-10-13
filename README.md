**Problem statement:** 

Build a machine learning classification model that can predict whether a comment on a Youtube video is either positive or negative. This will be a binary classification problem with 2 class labels which are either 'Positive' or 'Negative' for positive comments and negative comments respectively.

**Acceptance performance metric:**
Maximizing the f1-score for negative comments

**Notebooks**

project_3_data_collection
project_3_cleaning_eda_modeling

**README Overview**

1. Data
2. Approach
3. Models Performance
4. Conclusion
5. References


# 1. Data:

A csv file containing raw comments and replies from Justin Bieber - Baby ft. Ludacris on Youtube). The data will be the comments and replies to a youtube music video by singer/popstar Justin Beiber. The modelling process will be carried out with the assistance of a pre-trained zero-shot deep-learning language model from Hugging Face, a custom sentiment analysis model named Flair trained on IMDB data, and VADER, a lexicon-based sentiment analysis library, in the labeling of about 63,000 comments.


**Summary of dataframe that is worked on:**

|  Feature |  Type  |    Dataset   |                                                                       Description                                                                      |
|:--------:|:------:|:------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
| sequence | object | final_df.csv |       Preprocessed text data of comments and replies that can be considered 'clean' for downstream modeling. Has not yet been stem or lemmatized.      |
|   label  |  int64 | final_df.csv | Target label for classification model to make a prediction on. Labels are either 1 or 0, where 1 is a 'positive' comment and 0 is a 'negative' comment |





# 2. Approach:

**Data Inspection Summary:**

|          Method          |                 Observation                  |       Action Taken     |
|:------------------------:|:--------------------------------------------:|:----------------------:|
|        Check shape       | There are 99941 rows of comments/replies |           ---          |
|  Check for duplicates    |        There are 13854 duplicated rows       |  Drop duplicated rows  |
| Check for null/na values |              There is 1 na value             | Drop row with na value |

**Summary on Data Cleaning for Sentiment Analysis:**

|                              Action Taken                             |
|:---------------------------------------------------------------------:|
|                  Remove non-english comments/replies                  |
| Remove html tags, url links, single characters, extra spaces and tabs |
|                          Expand contractions                          |
|                         Drop 'Comments' column                        |
|          Replace empty strings with NA and drop all NA values         |
|               Save df as english_df.csv as a checkpoint               |


**Summary on Sentiment Analysis:**

|                                              Action Taken                                             |
|:-----------------------------------------------------------------------------------------------------:|
| Randomly Sampled 100 rows of comments/replies and hand-labeled as either Positive (1) or Negative (0) |
|      Predicted Target Labels on 100 rows of hand-labeled data using Vader, Flair and Hugging Face     |
|    Decided on using the mean of all 3 methods to preditct the target label for the entire dataframe   |
|                       Save combined dataframe as combined_df.csv as a checkpoint                      | 


**Summary on Data Cleaning for Downstream EDA and Modeling:**

|                        Action Taken                       |
|:---------------------------------------------------------:|
|    Text with emojis replaced by textual representations   |
|   Text with punctuation and non-Roman characters removed  |
| Text with numeric digits replaced by word representations |
|                Text converted to lowercase                |
|                Text with stopwords removed                |
|       Replace empty strings with NA and drop NA rows      |
|        Save combined_df as final_df as a checkpoint       |

**Summary on Exploratory Data Analysis:**

|                Action Taken For Visualizations                |
|:-------------------------------------------------------------:|
|      Frequency distribution of the top 10 tokens (words)      |
|                         Top 10 2-grams                        |
|                         Top 10 3-grams                        |
| Top 10 most common parts-of-speech (POS) for adjectives (ADJ) |
|                  Distribution of word counts                  |

# 3. Models Performance:

**Interim Summary Part 1:**

| No. | Preprocessing (if any) |     Vectorizer    | Vectorizer Max Features | Vectorizer N-gram Range |          Model          | Cross-val f1-score | Train f1-score | Test f1-score |
|:---:|:----------------------:|:-----------------:|:-----------------------:|:-----------------------:|:-----------------------:|:------------------:|:--------------:|:-------------:|
|  1A |    Snowball Stemmer    |  Count Vectorizer |          15000          |          (1, 3)         | Multinomial Naive Bayes |       0.61112      |     0.65971    |    0.61341    |
|  1B |   Word Net Lemmatizer  |  Count Vectorizer |          15000          |          (1, 3)         | Multinomial Naive Bayes |       0.60807      |     0.65896    |    0.61526    |
|  2A |    Snowball Stemmer    | TF-IDF Vectorizer |           6000          |          (1, 2)         | Multinomial Naive Bayes |       0.56397      |     0.61512    |    0.57242    |
|  2B |   Word Net Lemmatizer  | TF-IDF Vectorizer |           6000          |          (1, 2)         | Multinomial Naive Bayes |       0.56166      |     0.61202    |    0.56727    |

**Interim Summary Part 2:**

| No. | Preprocessing (if any) |    Vectorizer    | Vectorizer Max Features | Vectorizer N-gram Range |          Model          | Log_reg C Hyperparameter | Cross-val f1-score | Train f1-score | Test f1-score |
|:---:|:----------------------:|:----------------:|:-----------------------:|:-----------------------:|:-----------------------:|:------------------------:|:------------------:|:--------------:|:-------------:|
|  1B |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         | Multinomial Naive Bayes |            NA            |       0.60807      |     0.65896    |    0.61526    |
|  3A |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         |   Logistic Regression   |           0.25           |       0.64863      |     0.73867    |    0.65556    |
|  3B |        See Notes       |     Word2Vec     |        See Notes        |        See Notes        |   Logistic Regression   |            1            |       0.61310      |     0.62333    |    0.60733    |

**Interim Summary Part 3:**

| No. | Vectorizer |                     Model                     | Log_reg C Hyperparameter | Max Depth | Cross-val f1-score | Train f1-score | Test f1-score |
|:---:|:----------:|:---------------------------------------------:|:------------------------:|:---------:|:------------------:|:--------------:|:-------------:|
|  3B |  Word2Vec  |              Logistic Regression              |             1            |     NA    |       0.61310      |     0.62333    |    0.60733    |
|  4A |  Word2Vec  |            Random Forest Classifier           |            NA            |     5     |       0.57045      |     0.58741    |    0.55768    |
|  4B |  Word2Vec  | Histogram-based Gradient Boosting Classifier |            NA             |     5     |       0.61225      |     0.70951    |    0.61250    |
|  4C |  Word2Vec  |              Stacking Classifier              |         See Notes        | See Notes |         NA         |     0.73709    |    0.62219    |


**Interim Summary Part 4:**

| No. | Preprocessing (if any) |    Vectorizer    | Vectorizer Max Features | Vectorizer N-gram Range |             Model            | Max Depth | Log_reg C Hyperparameter | Cross-val f1-score | Train f1-score | Test f1-score |
|:---:|:----------------------:|:----------------:|:-----------------------:|:-----------------------:|:----------------------------:|:---------:|:------------------------:|:------------------:|:--------------:|:-------------:|
|  3A |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         |      Logistic Regression     |     NA    |           0.25           |       0.64863      |     0.73867    |    0.65556    |
|  5A |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         |   Random Forest Classifier   |     50    |            NA            |       0.60028      |     0.70352    |    0.60019    |
|  5B |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         | Gradient Boosting Classifier |    100    |            NA            |       0.58446      |     0.88657    |    0.60653    |
|  5C |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         |      Stacking Classifier     | See Notes |         See Notes        |         NA         |     0.79157    |    0.65228    |

**Final Summary:**

| No. | Preprocessing (if any) |    Vectorizer    | Vectorizer Max Features | Vectorizer N-gram Range |        Model        | Max Depth | Log_reg C Hyperparameter | Cross-val f1-score | Train f1-score | Test f1-score |
|:---:|:----------------------:|:----------------:|:-----------------------:|:-----------------------:|:-------------------:|:---------:|:------------------------:|:------------------:|:--------------:|:-------------:|
|  3A |   Word Net Lemmatizer  | Count Vectorizer |          15000          |          (1, 3)         | Logistic Regression |     NA    |           0.25           |       0.64863      |     0.73867    |    0.65556    |
|  4C |           NA           | Word2Vec         |            NA           |            NA           | Stacking Classifier | See Notes |         See Notes        |         NA         |     0.73709    |    0.62219    |


# 4. Conclusion:

From the evaluated models, Logistic Regression with non-word embedding (model 3A) emerges as the most proficient performer. It exhibits superior test results and reduced overfitting compared to the other models, leading to its selection as the ultimate choice for classification to address the problem statement.

Create a Third Label ('Neutral') During Sentiment Analysis. If this improves the model's performance, this ties back into better addressing the problem statement which is a machine learning classification model that can predict whether a comment on a Youtube video is either positive or negative.

Further Preprocessing of Text Before Vectorizing with a Word2Vec Model. Theoretically, the performance of the Word2Vec model relative to CountVectorizer or TF-IDF Vectorizer should be superior and should be explored as a recommendation to create a better classifier to better address the problem statement.

# 5. References

1. https://www.youtube.com/watch?v=kffacxfA7G4
2. https://www.nytimes.com/2020/11/19/learning/what-students-are-saying-about-cancel-culture-friendly-celebrity-battles-and-finding-escape.html
3. https://www.scmp.com/magazines/style/celebrity/article/3204356/14-celebrities-who-got-cancelled-2022-elon-musks-twitter-mess-and-kanye-wests-controversial-comments
4. https://pub.towardsai.net/textblob-vs-vader-for-sentiment-analysis-using-python-76883d40f9ae
5. https://medium.com/@AmyGrabNGoInfo/sentiment-analysis-hugging-face-zero-shot-model-vs-flair-pre-trained-model-57047452225d
6. https://medium.com/@chyun55555/unsupervised-sentiment-analyis-with-sentiwordnet-and-vader-in-python-a519660198be
7. https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values
8. https://towardsdatascience.com/stop-using-smote-to-treat-class-imbalance-take-this-intuitive-approach-instead-9cb822b8dc45