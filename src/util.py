from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def count_vectorize(text):
    vectorizer = CountVectorizer()
    text = vectorizer.fit_transform(text)
    return text