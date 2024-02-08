import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from pathlib import Path
import os
from wordcloud import WordCloud
from collections import Counter
from src.preprocessing import print_most_frequent_ngrams, generate_ngrams

nltk.download('punkt')
nltk.download('stopwords')

parent = Path(__file__).parent
data_path = os.path.join(parent, "data/processed")

train_data = pd.read_csv(os.path.join(data_path, "train.csv"), sep=',')
test_data = pd.read_csv(os.path.join(data_path, "test.csv"), sep=',')

def generate_word_cloud_with_frequencies(text, title):
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)

    # Get word frequencies from the WordCloud object
    word_freq = Counter(wordcloud.words_)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Display the words and their frequencies
    print("Top words and their frequencies:")
    for word, freq in word_freq.most_common(20):
        print(f"{word}: {freq}")

def get_word_frequencies(text):
    words = text.split()
    word_freq = Counter(words)
    return word_freq



if __name__ == '__main__':
    print("Train Data Shape:", train_data.shape)
    print("\nTrain Data Info:")
    print(train_data.info())

    # Check basic statistics of numerical columns
    print(train_data.describe())

    '''The sentiment distribution is balanced with 20,000 positive and likely 20,000 negative reviews.
    This balance is advantageous for training a sentiment analysis model.
    No missing values are observed in either the "review" or "sentiment" columns, indicating good data quality.'''

    # Check the distribution of sentiments in the training set
    sns.countplot(x='sentiment', data=train_data)
    plt.title('Distribution of Sentiments in Training Set')
    plt.show()
    print(train_data['sentiment'].describe())

    '''The standard deviation suggests moderate variability in review lengths, reflecting the diversity of reviews.'''

    # Analyze the length of reviews
    train_data['review_length'] = train_data['review'].apply(len)

    # Distribution of review lengths
    sns.histplot(data=train_data, x='review_length', bins=30, kde=True)
    plt.title('Distribution of Review Lengths')
    plt.show()
    print(train_data['review_length'].describe())

    # Visualize a few positive and negative reviews
    positive_reviews = train_data[train_data['sentiment'] == 'positive']['review']
    negative_reviews = train_data[train_data['sentiment'] == 'negative']['review']


    # Function to generate and plot word cloud

    # Generate a word cloud for positive reviews
    positive_text = ' '.join(positive_reviews)
    generate_word_cloud_with_frequencies(positive_text, 'Word Cloud for Positive Reviews')
    '''
    The top words align with positive sentiments associated with movie enjoyment, good content, and positive expressions.
    
    Words like "character," "story," and "show" highlight the importance of well-developed characters and engaging narratives in positive reviews.
    
    Some words, like "one," "see," and "time," are neutral and may not directly convey sentiment. Contextual analysis is crucial.
    
    Positive expressions like "great," "love," and "good" contribute to the overall positive sentiment in the word cloud.'''
    # Generate a word cloud for negative reviews
    negative_text = ' '.join(negative_reviews)
    generate_word_cloud_with_frequencies(negative_text, 'Word Cloud for Negative Reviews')
    '''
    The top words align with negative sentiments associated with dissatisfaction with movies, characters, and aspects of the content.
    
    Words like "one," "make," and "even" are neutral and may be used in both positive and negative contexts. 
    
    The presence of "good" and "bad" suggests an expression of opinions regarding the quality of content in negative reviews.
    
    Some words, such as "character," "see," and "time," are present in both positive and negative word clouds. '''

    # Function to get word frequencies

    # Get word frequencies for positive reviews
    positive_word_freq = get_word_frequencies(positive_text)

    # Get word frequencies for negative reviews
    negative_word_freq = get_word_frequencies(negative_text)

    # Display the top words and their frequencies
    print("Top 10 words in Positive Reviews:")
    print(positive_word_freq.most_common(20))

    print("\nTop 10 words in Negative Reviews:")
    print(negative_word_freq.most_common(20))

    train_data['bigrams'] = train_data['final_review'].apply(lambda x: generate_ngrams(x, 2))
    train_data['threegrams'] = train_data['final_review'].apply(lambda x: generate_ngrams(x, 3))

    print_most_frequent_ngrams(train_data, 'bigrams', 'positive', 2)
    print_most_frequent_ngrams(train_data, 'threegrams', 'positive', 3)
    print_most_frequent_ngrams(train_data, 'bigrams', 'negative', 2)
    print_most_frequent_ngrams(train_data, 'threegrams', 'negative', 3)

    # analysis for eda
    '''Positive reviews tend to focus on specific elements such as effects, character development, and narrative quality.
    Negative reviews often mention disappointment with aspects like plot coherence, acting, and overall production value.
    Certain phrases like "worst movie ever" and "best movie ever" are highly polarizing and frequently appear in negative and positive reviews, respectively.
    The mention of "new york city" in both positive and negative reviews suggests the importance of setting or context in shaping viewer opinions.'''
    # ------------------------------------------------------------------



