from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os
import nltk

# You might get an error with nltk
# It can be resoloved by the following commands
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def tokenize(text, rmStopWords=False):
    # tokenize a text, remove punctuations
    # and remove stopwords on demand (default: No)
    # Argument:a text (string)
    # Return value: a list of tokens
    # to be discussed: does numbers contribute to the subject?
    # to be discussed: do we keep duplicate words? (set)
    allTokens = word_tokenize(text.lower())
    if rmStopWords:
        stopWords = set(stopwords.words('english'))
        filteredTokens = [w for w in allTokens if (
            not (w in stopWords)) and not ((w in string.punctuation))]
        return filteredTokens
    else:     # rmStopWrods = False
        allTokens = [w for w in allTokens if not w in string.punctuation]
        return allTokens


def normalize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
