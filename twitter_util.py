# --------------------------------------------------
# twitter_util.py
#
# Utility functions for working with the twitter dataset.
# --------------------------------------------------

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import json
import string, nltk, csv, os, re, sys

# Project-wide constants, file paths, etc.
import settings


# You might get an error with nltk.
# It can be resoloved by the following lines of code:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def download_twitter():
    pass


def load_data_twitter(twitter_dir):
    '''
    Load in the Twitter data from the json files.

    Note that since the Twitter data is unlabelled, None is used as
    the label for every item.

    Arguments:
        - twitter_dir (string): the path to a directory containing the twitter jsonl files

    Return Values:
        - train (list): List of training files with dummy labels.
        - test (list): Empty list, as there is no test data for the Twitter dataset
        - classes (list): Empty list, as there are no class labels
    '''
    if settings.DEBUG: print('Loading in the twitter dataset.')
    if twitter_dir == None:
        raise Exception("Must specify the path to the twitter directory")

    # First make sure the files exist on the system.
    if not os.path.exists(twitter_dir) or not os.path.isdir(twitter_dir):
        raise Exception('Can not load in training data, files do not exist.')

    classes, train, test  = [], [], []
    
    # Read in the training data next
    for f in sorted(os.listdir(twitter_dir)):
        fpath = os.path.join(twitter_dir, f)
        
        if settings.DEBUG: print(f"Loading {fpath}")
        with open(fpath, "r") as json_file:
            line = json_file.readline()
            while line:
                d = json.loads(line)
                tweetID = d["id"]
                text = d["full_text"]
                date = d["created_at"]
                
                # store the tweetID and date in case we need them later
                train.append([None, text, tweetID, date])

                # get the next tweet
                line = json_file.readline()
            
        break

    return (train, test, classes)


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

def normalize_twitter_porter(tokens):
    """
    Ensure all tokens are normalized.
    Our normalization consists of:
        - Case folding
        - Split on '|'
        - Remove all punctuation

    Arguments:
        tokens: a list of tokens

    Returns:
        result: a list of normalized tokens
    """
    stemmer = PorterStemmer()

    result = []
    punctuation_table = str.maketrans('', '', string.punctuation + "–—−—”“’‘,")  # https://stackoverflow.com/a/34294398

    # make hashtags special
    del punctuation_table[ord("#")]

    for t in tokens:
        t = t.lower()
        # remove punctuation and convert token to lowercase
        t = t.translate(punctuation_table)
        if not t:
            continue
        else:
            t = stemmer.stem(t)
            result.append(t)

    return result

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

