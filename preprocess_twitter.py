from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk
import re


# You might get an error with nltk
# It can be resoloved by the following commands
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def tokenize(text, rmStopWords=False):
    """
    tokenize a text, remove URLs
    and remove stopwords on demand (default: No)
    Argument: a text (string)
    Return value: a list of tokens
    """
    text = re.sub(r"http\S+", "", text)  # remove URLs
    allTokens = word_tokenize(text)
    if rmStopWords:
        stopWords = set(stopwords.words('english'))
        filteredTokens = [token for token in allTokens if token not in stopWords]
        return filteredTokens
    else:     # rmStopWrods = False
        return allTokens


def normalize(tokens):
    """
    case folding, remove punctuations and emojs
    and lemmatization
    Hashtags will be preserved but without "#" sign
    Mentions will be preserved but without "@" sign
    Argument: a list of tokens
    Return value: a list of tokens after normalization
    Note: this function keeps duplicates and numbers
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if (token not in string.punctuation) and (token.encode("ascii", "ignore").decode())]
