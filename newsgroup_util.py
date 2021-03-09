# --------------------------------------------------
# newsgroup_util.py
#
# Utility functions for working with the newsgroup dataset.
# --------------------------------------------------

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string, nltk, csv, os

# Project-wide constants, file paths, etc.
import settings


# You might get an error with nltk.
# It can be resoloved by the following lines of code:
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def download_newsgroup():
    '''
    Download the newsgroup data to be later loaded.
    Note that we get the file paths from the settings file,
    so they don't need to be passed as arguments.
    '''

    if settings.DEBUG: print('Downloading newsgroup dataset.')

    # Download the training and test datasets.
    train = fetch_20newsgroups(data_home=settings.NEWSGROUP_DIR, subset='train')
    test = fetch_20newsgroups(data_home=settings.NEWSGROUP_DIR, subset='test')

    # Write the class names tsv first.
    with open(settings.NEWSGROUP_CLASSES, 'w') as f:
        fout = csv.writer(f, delimiter='\t')
        fout.writerow(train.target_names)
    
    # Next write the training data to file.
    # Each line is the true label and the document, with tabs removed.
    with open(settings.NEWSGROUP_TRAIN, 'w') as f:
        fout = csv.writer(f, delimiter='\t')
        
        for i in range(len(train.data)):
            target = train.target_names[train.target[i]]
            fout.writerow([target, train.data[i].replace('\t', ' ')])

    # Lastly, write the test data to it's file.
    with open(settings.NEWSGROUP_TEST, 'w') as f:
        fout = csv.writer(f, delimiter='\t')
        
        for i in range(len(test.data)):
            target = test.target_names[test.target[i]]
            fout.writerow([target, test.data[i].replace('\t', ' ')])


def load_data_newsgroup():
    '''
    Load in the newsgroup data from the tsv files.
    Note that we get the file paths from the settings file,
    so they don't need to be passed as arguments.

    Return Values:
        - train (list): List of training files with true labels.
        - test (list): List of test files with true labels.
        - classes (list): List of all class names.
    '''

    if settings.DEBUG: print('Loading in the newsgroup dataset.')

    # First make sure the files exist on the system.
    if not os.path.exists(settings.NEWSGROUP_TRAIN) \
            or not os.path.exists(settings.NEWSGROUP_TEST) \
            or not os.path.exists(settings.NEWSGROUP_CLASSES):
        raise Exception('Can not load in training or test data, files do not exist.')
    
    # Read in the class names first.
    with open(settings.NEWSGROUP_CLASSES, 'r') as f:
        classes = list(csv.reader(f, delimiter="\t"))[0]
        
    # Read in the training and test data next.
    train, test = [], []

    with open(settings.NEWSGROUP_TRAIN, 'r') as f:
        fin = csv.reader(f, delimiter="\t")

        for doc in fin:
            train.append(doc)
    
    with open(settings.NEWSGROUP_TEST, 'r') as f:
        fin = csv.reader(f, delimiter="\t")

        for doc in fin:
            test.append(doc)

    return (train, test, classes)


def tokenize_newsgroup(text, remove_stopwords=False):
    '''
    Tokenize a given text.

    Arguments:
        - text (string): The text to tokenize.
        - remove_stopwords (boolean): Flag for if we should remove stopwords or not.
    
    Return Values:
        - (list): The tokenized text.
    '''

    # First tokenize the text.
    tokens = word_tokenize(text)
    
    # Handle stopwords if needed.
    if remove_stopwords:
        s_words = set(stopwords.words('english'))
        tokens = list(filter(lambda x: x not in s_words, tokens))
    
    return tokens

def normalize_newsgroup(tokens):
    '''
    Normalize a list of tokens.

    Arguments:
        - tokens (list): The tokens to normalize.
    
    Return Values:
        - (list): The normalized tokens.
    '''

    # Initialize the lemmatizer.
    lemmatizer = WordNetLemmatizer()

    # Filter out tokens that are punctuation too.
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token not in string.punctuation]
