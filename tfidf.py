import pandas as pd
import pickle as pk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class naive_bayes:
    review = pd.read_csv('reviews (1).csv')
    review = review.rename(columns={'text': 'review'}, inplace=False)
    review.head()
    
    X = review.review
    y = review.polarity
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=False)
    
    # Fit the vectorizer on the training data
    X_train_transformed = vectorizer.fit_transform(X_train)
    
    # Print vocabulary
    print(vectorizer.vocabulary_)
    
    # Transform test data using the fitted vectorizer
    X_test_transformed = vectorizer.transform(X_test)
    
    # Create and train the Naive Bayes classifier
    naivebayes = MultinomialNB()
    naivebayes.fit(X_train_transformed, y_train)
