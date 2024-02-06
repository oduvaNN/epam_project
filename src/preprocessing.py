import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from pathlib import Path
import os

# Load the data
parent = Path(__file__).parent
data_path = os.path.join(parent, "data/raw")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')
train_data['tokenized_review'] = train_data['review'].apply(lambda x: word_tokenize(x))


def tokenize_and_filter_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return filtered_tokens


if __name__ == '__main__':
    # Display the updated DataFrame with the tokenized column
    print(train_data[['review', 'tokenized_review']].head())

    stop_words = set(stopwords.words('english'))

    train_data['filtered_review'] = train_data['review'].apply(tokenize_and_filter_stopwords)

    # Display the updated DataFrame with the filtered review column
    print(train_data[['review', 'filtered_review']].head())