# Natural Language Processing with Disaster Tweets

## 📌 Project Goal

The goal of this project is to build a machine learning model that can automatically classify tweets as either:

	•	Disaster (1) → The tweet reports a real disaster or emergency.
 
	•	Non-disaster (0) → The tweet is unrelated to a real disaster.

This work is inspired by a Kaggle competition and uses a dataset of 10,000 labeled tweets.

⸻

## 📈 Business Context & Real-World Application

In disaster management, speed and accuracy of information are critical. Social media platforms like Twitter are often the first channels where people report emergencies in real-time — sometimes faster than official news outlets.

However, tweets can be ambiguous or metaphorical, making it hard for automated systems to distinguish between genuine emergencies and unrelated chatter.

By developing an NLP-based classification model, emergency services, news agencies, and humanitarian organizations can:

	•	Quickly identify and prioritize real disaster reports for faster response.
	•	Reduce misinformation by filtering out irrelevant tweets.
	•	Automate large-scale monitoring of social media during crisis events.
	•	Support decision-making in allocating rescue resources efficiently.

This project demonstrates how natural language processing and machine learning can be integrated into real-time disaster response systems, potentially saving lives by reducing the time between an incident and emergency action.

⸻

## 📂 Project Structure

**Step 1: Import Libraries**

**All the necessary libraries to run this project on Python are stated below-**

import numpy as np  #for numerical operations
import pandas as pd  #for data manipulation
import random  #for shuffling the data
import nltk
import string
import re  # or handling regular expressions

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from nltk.stem import WordNetLemmatizer  #for lemmatizing words
from nltk.corpus import stopwords  #for stop word removal
from nltk.tokenize import word_tokenize  #for tokenizing sentences into words
nltk.download('punkt_tab')  #Downloads the 'punkt' tokenizer table used for tokenization of text into sentences or words
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

**Downloading necessary NLTK resources**

nltk.download('stopwords')  # List of common stop words in English
nltk.download('punkt')  # Pre-trained tokenizer models
nltk.download('wordnet')  # WordNet lemmatizer dataset

**Libraries for text feature extraction and model training**

from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text into numerical features (TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer  # Convert text into numerical features (Count Vectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # Logistic regression for classification
from sklearn.svm import LinearSVC  # Support Vector Machines for classification

**Libraries for model evaluation**

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # For model evaluation metrics
from sklearn.model_selection import KFold, cross_val_score  # For cross-validation

**Step 2: Load & Preview Data**

The Sentence Polarity dataset contains 10,662 sentences — 5,331 positive and 5,331 negative. The dataset has 2 columns
	**•	text:** the text of the sentence
	**•	target:** sentiment indicator (0 = negative, 1 = positive)
 
 **Data can be found in Kaggle** (https://www.kaggle.com/c/nlp-getting-started/overview)

**Step 3: Exploratory Data Analysis (EDA) Summary**

    •	NO missing values
	
	•	Class Balance — Non-disaster tweets: 57%, disaster tweets: 42.9%.
 
	•	Tweet Length — Disaster tweets are generally longer, often providing more context; very short tweets (<20 chars) are usually non-disaster and ambiguous.
 
	•	Word Clouds — Disaster tweets frequently include terms like fire, flood, storm, death, bomb, and location names (California, Hiroshima), indicating urgency and human impact. Common noise tokens (t, co, amp, u, https) require cleaning.
 
	•	Hashtags & Links — Most tweets lack hashtags or mentions, but ~50% contain links, which may correlate with news/disaster reports.
 
	•	Overlapping Words — 34 of the top 50 words are common across both classes, mostly stopwords and generic terms; only 16 are unique, showing the need for stopword removal, TF-IDF weighting, and contextual features (n-grams, embeddings).

 📌 Implication: Strong preprocessing (URL removal, stopword filtering, lemmatization) and engineered features (tweet length, presence of hashtags/links) are key for improving classification.

 
**Step 4: Data Preprocessing**

Text Preprocessing

	•	Tools Used: NLTK for tokenization, lemmatization, POS tagging, and stopword removal.
 
	•	Steps:
 
	    1.	Resource Setup — Downloaded NLTK stopwords, POS tags, and WordNet data.
	 
	    2.	POS Mapping — Mapped NLTK POS tags to WordNet tags for precise lemmatization.
	 
	    3.	Cleaning Function (preprocess_tweet) —
	 
	       •	Converted text to lowercase.
	       •	Removed URLs, mentions, hashtags, HTML tags, numbers, and punctuation.
	       •	Tokenized text, removed stopwords, and filtered short/non-alphabetic words.
	       •	Lemmatized tokens using POS tags for better normalization.
	       •	Rejoined tokens into a clean string.
		
	    4.	Applied to Dataset — All tweets transformed into clean, normalized text.
	    5. Word Clouds after preprocessing: Disaster tweets feature urgent, event-specific terms (fire, flood, storm, death), while non-disaster tweets focus on casual, everyday vocabulary (new, love, day, make), providing clear thematic separation for classification.

📌 Why Important: Cleans noisy raw tweets, removes irrelevant tokens, and creates high-quality input for vectorization and modeling, improving both accuracy and interpretability.

**Step 5: Feature Extraction (TF-IDF Vectorization)** 

	•	Purpose: Convert cleaned text into numeric feature vectors for machine learning models.
	•	Method Used: Applied vectorization techniques (e.g., TF-IDF, CountVectorizer) to represent tweets based on:
	•	Word frequency and
	•	Importance in context (down-weighting common words).
	•	Why Important:
	•	Enables ML algorithms like Logistic Regression, Naive Bayes, and SVM to process text data.
	•	Preserves meaningful patterns while reducing noise.
	•	Produces scalable, sparse matrices for efficient computation.
	•	Impact: Provides the numerical foundation for training accurate and interpretable NLP classification models.

**Step 6: Modeling & Evaluation Summary (Pre-Tuning)** 

	•	Naive Bayes
	•	Logistic Regression
	•	Linear SVC
	•	Random Forest
 
Four models were tested using CountVectorizer and TF-IDF representations: Naive Bayes, Logistic Regression, Linear SVC, and Random Forest.

Key Findings:

	•	TF-IDF consistently outperformed CountVectorizer across all models, providing better term discrimination.
	•	Naive Bayes (TF-IDF) achieved the highest accuracy (0.82), performing especially well on non-disaster tweets, though recall for disaster tweets was slightly lower (69%).
	•	Logistic Regression (TF-IDF) delivered balanced precision and recall for both classes, slightly trailing Naive Bayes in accuracy.
	•	Linear SVC showed a notable drop in accuracy with CountVectorizer but was competitive with TF-IDF.
	•	Random Forest performed consistently but lagged slightly behind linear models.

Best Overall Choice (Pre-Tuning): ✅ Naive Bayes with TF-IDF — best accuracy and solid class balance.

This dataset is **imbalanced** (more non-disaster tweets than disaster tweets), so we evaluate models primarily using **F1-score** — especially for the disaster class (label 1) — instead of accuracy alone.

| Model                | Vectorizer | F1-score (Class 1) | Macro Avg F1 | Notes |
|----------------------|------------|--------------------|--------------|-------|
| **Naive Bayes**      | **TF-IDF** | **0.76**           | **0.80**     | Best overall — high precision, decent recall |
| Logistic Regression  | TF-IDF     | 0.76               | 0.80         | Balanced precision & recall |
| Linear SVC           | TF-IDF     | 0.76               | 0.79         | Slightly higher recall, lower precision |
| Random Forest        | TF-IDF     | 0.75               | 0.79         | Lower recall for disasters |
| Naive Bayes          | Count      | 0.77               | 0.80         | Strongest in CountVectorizer group |
| Others (Count)       | Count      | ≤0.75               | ≤0.79        | Lower performance than TF-IDF |

**✅ Key Takeaways**

- **Best Overall Model:** Naive Bayes with TF-IDF (highest disaster F1-score, balanced performance)
  
- **Why Not Accuracy?** With imbalanced data, accuracy can be misleading — a model predicting mostly non-disasters could still score high accuracy.
  
- **Metric Focus:** F1-score for disaster class + macro average gives a fairer evaluation.
  
- **Next Steps:**
  
  - Hyperparameter tuning for Naive Bayes & Logistic Regression.
  - Use class weights or oversampling to improve disaster recall.
 
**Step 7: Hyperparameter Tuning** 

After testing multiple baseline models, **Logistic Regression** was further optimized using **GridSearchCV** to improve performance, especially for the minority class (Disaster tweets).

**Best Parameters Found:**
python
{'C': 1, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'liblinear'}

**Key Results (TF-IDF Vectorizer, ngram_range=(1,2), max_features=5000):**

	•	Cross-Validation Accuracy: 0.7946
	•	Test Accuracy: 0.8175 ✅ (higher than CV — good sign of generalization)
	•	F1-score (Disaster class): 0.77 → Improved from ~0.76 (baseline)
	•	False Positives: 92 (Non-disaster misclassified as disaster)
	•	False Negatives: 186 (Disaster misclassified as non-disaster)

**Why It Improved:**

	•	C=1 → Balanced regularization, avoiding over/underfitting.
	•	liblinear solver → Efficient for small-to-medium datasets, supports L2 penalty.
	•	Bigrams (1,2) → Captured important disaster-related phrases like “fire outbreak” and “flood warning”.
	•	Feature limit (5000) → Reduced noise and focused on most informative tokens.

**Step 8: Conclusion**

	•	Best Overall Model: Naive Bayes (TF-IDF) remains slightly ahead in recall for disasters.
	•	Best Tuned Model: Logistic Regression after hyperparameter tuning — competitive performance with improved precision and interpretability.
	•	Real-World Impact: Both models are suitable for production in disaster tweet detection systems, with Logistic Regression being easier to explain to stakeholders.
 
	•	Next Steps:
 
	•	Experiment with ensemble methods (e.g., Voting Classifier combining Naive Bayes + Logistic Regression).
	•	Try deep learning approaches (e.g., LSTM, BERT) for richer context understanding.
