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

# Read the stop list from stop_list.txt
fobj = open(settings.STOP_LIST, 'r')
stop_list = set([l.strip() for l in fobj.readlines()])
if settings.DEBUG:
    print(f"Loaded stop list: {sorted(list(stop_list))}")


def download_twitter(path='./TwitterDataset'):
    '''
    Downloads the twitter dataset from the git repository:
        https://github.com/VeronicaSalm/TwitterDataset

    Arguments:
        - path (string): an absolute or relative path to the directory where the
                         Twitter repository should be downloaded to, defaults to
                         the current directory '.'
    '''
    os.system(f"git clone https://github.com/VeronicaSalm/TwitterDataset {path}")


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

    return (train, test, classes)


def tokenize_twitter(text, remove_stopwords=True):
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

    # remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    # Handle stopwords if needed.
    if remove_stopwords:
        s_words = set([s.lower() for s in stopwords.words('english')])
        tokens = list(filter(lambda x: x.lower() not in s_words, tokens))

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
    valid = []
    for token in tokens:
        if token.lower() not in stop_list:
            valid.append(token)
    return [lemmatizer.lemmatize(token.lower()) for token in valid if (token not in string.punctuation) and (token.encode("ascii", "ignore").decode())]

