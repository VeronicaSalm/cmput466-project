# Temporary file with code for calculating log likelihood

# I just copy pasted all the imports i use, do not suggest copy pasting this too.
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import sklearn
import matplotlib.pyplot as plt

from newsgroup_util import get_confusion_matrix, get_accuracy
from DataManager import DataManager
import settings

import nltk

# The main import for accuracy.
from sklearn.metrics import accuracy_score

def run_nmf(data, alpha=1.34, beta_loss='kullback-leibler', l1_ratio=0.66, solver='mu', num_iterations=1000, num_components=20):
    """
    Run NMF on the given data. Defaults to the training set if no data is specified.
    Arguments:
        data: Data to run on. Same format as received from get_all_data, get_all_folds, etc.
        alpha (float): The normalization constant for NMF.
        beta_loss (string): The beta-loss function to use.
        l1_ratio (float): The ratio in which NMF uses L1 regularization vs. L2.
        solver (string): Numerical solver to use for NMF.
        num_iterations (int): The number of iterations to run for.
        num_components (int): The number of components or topics NMF generates.
    
    Returns:
        A list of lists, the j'th element of the i'th list is the probability that the i'th document belongs to topic j. (i.e. the weighting of topic j)
    """

    # First, make sure the given beta-loss and numerical solver are valid.
    # Also make sure the L1 ratio is in [0, 1].
    if beta_loss not in ['kullback-leibler', 'frobenius', 'itakura-saito']:
        msg = 'Invalid beta-loss function given for NMF.\n'
        options = "', '".join(['kullback-leibler', 'frobenius', 'itakura-saito'])
        msg += f"Please use one of: '{options}'."
        raise Exception(msg)
    
    if solver not in ['mu', 'cd']:
        msg = 'Invalid numerical solver given for NMF.\n'
        msg += "Please use one of: 'mu', 'cd'."
        raise Exception(msg)
    
    if l1_ratio < 0 or l1_ratio > 1:
        raise Exception('Invalid L1 ration given for NMF.\nPlease make sure l1_ratio is in the range [0, 1].')

    if settings.DEBUG: print('Vectorizing the data for NMF.')
    tfidf_vect = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=10000,
        stop_words='english'
    )

    vectorized_data = tfidf_vect.fit_transform([x[1] for x in data])

    if settings.DEBUG: print('Running NMF.')
    nmf_model = NMF(
        init='nndsvda',
        n_components=num_components,
        random_state=1,
        beta_loss=beta_loss,
        solver=solver,
        max_iter=num_iterations,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
    
    nmf_model.fit(vectorized_data)
    
    return nmf_model


# Initialize the data manager class.
dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
dm.load_data()

# Vectorize the data.
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=2,
    max_features=10000,
    stop_words='english'
)

data = dm.get_all_data(True)
validate = tfidf_vectorizer.fit_transform(x[1] for x in data)


# Get nmf model.
print('Training NMF.')
nmf = run_nmf(dm.get_all_data())

# Run nmf on test data.
topicnames = ["Topic" + str(i) for i in range(nmf.n_components)]
docnames = ["Doc" + str(i) for i in range(validate.shape[0])]

df_document_topic = pd.DataFrame(np.round(nmf.transform(validate), 2), columns=topicnames, index=docnames)
dominant_topics = np.argmax(df_document_topic.values, axis=1)

predicted = [dm.get_classes()[i] for i in dominant_topics]
real = [x[0] for x in dm.get_all_data(True)]

# log likelihood
print(accuracy_score(predicted, real))