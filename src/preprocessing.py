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

nltk.download('wordnet')

parent = Path(__file__).parent
data_path = os.path.join(parent, "data/raw")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')

def remove_irrelevant_chars(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove numbers
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

if __name__ == '__main__':
    train_data['cleaned_review'] = train_data['review'].apply(remove_irrelevant_chars)

    train_data['tokenized_review'] = train_data['cleaned_review'].apply(tokenize_text)

    stop_words = set(stopwords.words('english'))

    train_data['filtered_review'] = train_data['tokenized_review'].apply(filter_stop_words)

    stemmer = SnowballStemmer('english')
    wordnet_lemmatizer = WordNetLemmatizer()

    train_data['stemmed_review'] = train_data['filtered_review'].apply(stem_tokens)
    train_data['lemmatized_review'] = train_data['filtered_review'].apply(lemmatize_tokens)

    train_data['final_review'] = train_data['lemmatized_review'].apply(lambda tokens: [token for token in tokens if len(token) > 2])

    for i in range(5):
        print(f"Original Review:\n{train_data['review'][i]}\n")
        print(f"Filtered Review:\n{train_data['filtered_review'][i]}\n")
        print(f"Stemmed Review:\n{train_data['stemmed_review'][i]}\n")
        print(f"Lemmatized Review:\n{train_data['lemmatized_review'][i]}\n")
        print(f"Final Review:\n{train_data['final_review'][i]}\n")
        print("="*50)
