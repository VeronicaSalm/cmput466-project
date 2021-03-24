# --------------------------------------------------
# twitter_util.py
#
# Utility functions for working with the twitter dataset.
# --------------------------------------------------

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string, nltk, csv, os, re

# Project-wide constants, file paths, etc.
import settings


# You might get an error with nltk.
# It can be resoloved by the following lines of code:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def download_twitter():
    pass


def load_data_twitter():
    pass


def tokenize_twitter(text, remove_stopwords=False):
    '''
    Tokenize a given text. Also removes URLs and
    has the option to remove stop words.

    Arguments:
        - text (string): The text to tokenize.
        - remove_stopwords (boolean): Flag for if we should remove stopwords or not.

    Return Values:
        - (list): The tokenized text.
    '''

    # First, use a simple regex to remove the URLs. Then tokenize the text.
    # We remove URLs here as it'll be more difficult to do this when we normalize.
    text = re.sub(r"http\S+", "", text)
    tokens = word_tokenize(text)

    # Handle stopwords if needed.
    if remove_stopwords:
        s_words = set(stopwords.words('english'))
        tokens = list(filter(lambda x: x not in s_words, tokens))

    return tokens


def normalize_twitter(tokens):
    '''
    Normalize a list of tokens.
    We do case folding, remove punctuation, emojis, and lemmatization.
    Hashtags will be preserved without the '#' character,
    and same with mentions and the '@' character.

    Arguments:
        - tokens (list): The tokens to normalize.

    Return Values:
        - (list): The normalized tokens.

    Notes:
        - This function keeps duplicates and numbers. We may or may not want to
          change this.
    '''

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if (token not in string.punctuation) and (token.encode("ascii", "ignore").decode())]

