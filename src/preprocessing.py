import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Load the data
train_data = pd.read_csv("C:/Users/user/Desktop/homework/project/train.csv", sep=',')

train_data['tokenized_review'] = train_data['review'].apply(lambda x: word_tokenize(x))

# Display the updated DataFrame with the tokenized column
print(train_data[['review', 'tokenized_review']].head())

stop_words = set(stopwords.words('english'))

def tokenize_and_filter_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return filtered_tokens

train_data['filtered_review'] = train_data['review'].apply(tokenize_and_filter_stopwords)

# Display the updated DataFrame with the filtered review column
print(train_data[['review', 'filtered_review']].head())