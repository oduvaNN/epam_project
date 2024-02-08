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
from tqdm import tqdm
nltk.download('wordnet')
nltk.download('stopwords')
parent = Path(__file__).parent
data_path = os.path.join(parent, "../data/raw")
processed_path = os.path.join(parent, "../data/processed")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')
test_data = pd.read_csv(os.path.join(data_path, "test.csv"), sep=',')

vector_size = 512

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
    print("removed irrelevant characters")
    tqdm.pandas()
    print("tokenizing", flush=True)
    data['tokenized_review'] = data['cleaned_review'].progress_apply(tokenize_text)
    print("filtering reviews", flush=True)
    data['filtered_review'] = data['tokenized_review'].progress_apply(filter_stop_words)
    print("stemming reviews", flush=True)
    data['stemmed_review'] = data['filtered_review'].progress_apply(stem_tokens)
    print("lemmatizing reviews", flush=True)
    data['lemmatized_review'] = data['filtered_review'].progress_apply(lemmatize_tokens)
    print("removing all words that are less than 2 characters", flush=True)
    data['final_review'] = data['lemmatized_review'].apply(lambda tokens: [token for token in tokens if len(token) > 2])
    print("made final preprocessed version", flush=True)
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
    train_data[['final_review', 'sentiment']].to_csv(os.path.join(processed_path, "train.csv"))
    test_data[['final_review', 'sentiment']].to_csv(os.path.join(processed_path, "test.csv"))
    train_data[['final_review', 'sentiment']].to_pickle(os.path.join(processed_path, "train.pkl"))
    test_data[['final_review', 'sentiment']].to_pickle(os.path.join(processed_path, "test.pkl"))
    # Call the function for CountVectorizer
    vectorize_and_train(train_data, test_data, vectorizer_type='count')

    # Call the function for TfidfVectorizer
    vectorize_and_train(train_data, test_data, vectorizer_type='tfidf')

    #------------------------------------------------------------------

    

"""For CountVectorizer:
The precision, recall, and F1-score for the 'negative' class are all 0.89, indicating balanced performance in correctly identifying negative sentiment.
The macro average and weighted average scores for precision, recall, and F1-score are consistent, suggesting that the model performs consistently across both classes.
For TfidfVectorizer:
The model performs slightly better for the 'positive' class compared to the 'negative' class, as seen from the precision, recall, and F1-score metrics for both classes.
The weighted average F1-score is 0.90, indicating a good balance between precision and recall across both classes."""