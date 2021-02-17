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

class management:

    def __init__(self):

        self.__all_data = []
        self.__path = None
        self.__classified_data = {}
        self.__data_class = []
        self.__class_space = []

    def loadIn(self, path):
        self.__path = path
        self.__class_space = os.listdir(path)
        for each in self.__class_space:
            self.__classified_data[each] = []
            file_path = os.path.join(path, each)

            for f in os.listdir(file_path):
                name = os.path.join(file_path, f)
                opening = open(name, "rb")
                temp = tokenize(opening.read().decode(
                    "utf-8", errors="ignore"))
                temp = normalize(temp)

                self.__all_data.append(temp)
                self.__classified_data[each].append(temp)
                self.__data_class.append(each)


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
