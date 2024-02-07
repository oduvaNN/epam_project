import nltk
import pandas as pd
import string
import re
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
nltk.download('wordnet')

parent = Path(__file__).parent
data_path = os.path.join(parent, "data/raw")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')
test_data = pd.read_csv(os.path.join(data_path, "test.csv"), sep=',')

def remove_irrelevant_chars(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  
    return text

def tokenize_text(text):
    return word_tokenize(text)

def filter_stop_words(tokens):
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]
    return filtered_tokens

def stem_tokens(tokens):
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def lemmatize_tokens(tokens):
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams

def print_most_frequent_ngrams(data, column, sentiment, n, top_n=20):
    sentiment_ngrams = data[data['sentiment'] == sentiment][column]
    all_ngrams = [ng for sublist in sentiment_ngrams for ng in sublist]
    ngram_freq = Counter(all_ngrams)
    print(f"Top {top_n} {n}-grams in {column} for {sentiment} reviews:")
    for ng, freq in ngram_freq.most_common(top_n):
        print(f"{ng}: {freq} times")

def preprocess_data(data):
    # Apply preprocessing steps
    data['cleaned_review'] = data['review'].apply(remove_irrelevant_chars)
    data['tokenized_review'] = data['cleaned_review'].apply(tokenize_text)
    data['filtered_review'] = data['tokenized_review'].apply(filter_stop_words)
    data['stemmed_review'] = data['filtered_review'].apply(stem_tokens)
    data['lemmatized_review'] = data['filtered_review'].apply(lemmatize_tokens)
    data['final_review'] = data['lemmatized_review'].apply(lambda tokens: [token for token in tokens if len(token) > 2])
    return data

def vectorize_and_train(train_data, test_data, vectorizer_type='count'):
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer()
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid vectorizer_type. Choose 'count' or 'tfidf'.")

    X_train = vectorizer.fit_transform(train_data['final_review'].apply(lambda x: ' '.join(x)))
    X_test = vectorizer.transform(test_data['final_review'].apply(lambda x: ' '.join(x)))

    # Train logistic regression model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, train_data['sentiment'])

    # Predict on test_data
    y_pred = lr.predict(X_test)

    # Print classification report
    print(f"Classification Report for {vectorizer_type.capitalize()}Vectorizer:")
    print(classification_report(test_data['sentiment'], y_pred))

if __name__ == '__main__':

    stemmer = SnowballStemmer('english')
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Preprocess train_data
    train_data = preprocess_data(train_data)

    # Preprocess test_data
    test_data = preprocess_data(test_data)
    # Call the function for CountVectorizer
    vectorize_and_train(train_data, test_data, vectorizer_type='count')

    # Call the function for TfidfVectorizer
    vectorize_and_train(train_data, test_data, vectorizer_type='tfidf')

    #------------------------------------------------------------------
    #this is for eda
    train_data['bigrams'] = train_data['final_review'].apply(lambda x: generate_ngrams(x, 2))
    train_data['threegrams'] = train_data['final_review'].apply(lambda x: generate_ngrams(x, 3))

    print_most_frequent_ngrams(train_data, 'bigrams', 'positive', 2)
    print_most_frequent_ngrams(train_data, 'threegrams', 'positive', 3)
    print_most_frequent_ngrams(train_data, 'bigrams', 'negative', 2)
    print_most_frequent_ngrams(train_data, 'threegrams', 'negative', 3)

    #analysis for eda
    '''Positive reviews tend to focus on specific elements such as effects, character development, and narrative quality.
    Negative reviews often mention disappointment with aspects like plot coherence, acting, and overall production value.
    Certain phrases like "worst movie ever" and "best movie ever" are highly polarizing and frequently appear in negative and positive reviews, respectively.
    The mention of "new york city" in both positive and negative reviews suggests the importance of setting or context in shaping viewer opinions.'''
    #------------------------------------------------------------------
    

"""For CountVectorizer:
The precision, recall, and F1-score for the 'negative' class are all 0.89, indicating balanced performance in correctly identifying negative sentiment.
The macro average and weighted average scores for precision, recall, and F1-score are consistent, suggesting that the model performs consistently across both classes.
For TfidfVectorizer:
The model performs slightly better for the 'positive' class compared to the 'negative' class, as seen from the precision, recall, and F1-score metrics for both classes.
The weighted average F1-score is 0.90, indicating a good balance between precision and recall across both classes."""