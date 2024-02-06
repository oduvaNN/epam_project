from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import os
from nltk.stem.snowball import SnowballStemmer
from src.util import get_wordnet_pos

# Load the data
parent = Path(__file__).parent
data_path = os.path.join(parent, "data/raw")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')
train_data['tokenized_review'] = train_data['review'].apply(lambda x: word_tokenize(x))


def tokenize_and_filter_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return filtered_tokens


def stematize_words(stemmer, text):
    words = word_tokenize(text)
    return [stemmer.stem(word) for word in words]


def lemmatize_words(lemmatizer, text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]


if __name__ == '__main__':
    # Display the updated DataFrame with the tokenized column
    print(train_data[['review', 'tokenized_review']].head())

    stop_words = set(stopwords.words('english'))

    train_data['filtered_review'] = train_data['review'].apply(tokenize_and_filter_stopwords)

    # Display the updated DataFrame with the filtered review column
    print(train_data[['review', 'filtered_review']].head())