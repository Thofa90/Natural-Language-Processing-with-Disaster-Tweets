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

All the necessary libraries to run this project on Python are stated below-

import numpy as np  #for numerical operations
import pandas as pd  # for data manipulation
import random  # for shuffling the data
import nltk
import re  # for handling regular expressions

from nltk.stem import WordNetLemmatizer  # for lemmatizing words
from nltk.corpus import stopwords  # for stop word removal
from nltk.tokenize import word_tokenize  # for tokenizing sentences into words
nltk.download('punkt_tab')  # Downloads the 'punkt' tokenizer table used for tokenization of text into sentences or words

Downloading necessary NLTK resources

nltk.download('stopwords')  # List of common stop words in English
nltk.download('punkt')  # Pre-trained tokenizer models
nltk.download('wordnet')  # WordNet lemmatizer dataset

Libraries for text feature extraction and model training

from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text into numerical features (TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer  # Convert text into numerical features (Count Vectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # Logistic regression for classification
from sklearn.svm import LinearSVC  # Support Vector Machines for classification

Libraries for model evaluation

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # For model evaluation metrics
from sklearn.model_selection import KFold, cross_val_score  # For cross-validation

**Step 2: Load & Preview Data**

**Step 3: Exploratory Data Analysis (EDA)**

	•	Data info & null value check
	•	Target class distribution
	•	Tweet length analysis
	•	Frequent word analysis & word clouds
	•	Hashtag, mention, and link frequency
	•	Examples of ambiguous tweets
	•	Overlapping words between classes
 
**Step 4: Data Preprocessing**

**Step 5: Feature Extraction (TF-IDF Vectorization)**  

**Step 6: Model Training & Evaluation** 

	•	Naive Bayes
	•	Logistic Regression
	•	Linear SVC
	•	Random Forest
 
**Step 7: Hyperparameter Tuning** 

**Step 8: Conclusion**

