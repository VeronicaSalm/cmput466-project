
from sklearn.decomposition import LatentDirichletAllocation
import settings
import numpy as np
import pandas as pd
import re
import nltk

from DataManager import DataManager
from newsgroup_util import get_confusion_matrix, get_accuracy

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools

import sklearn
import matplotlib.pyplot as plt


def get_data():
    dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
    print("Loading data...")
    dm.load_data()
    dm.divide_into_folds(10)
    # ignore labels, just grab the strings

    vectorizer = CountVectorizer(analyzer='word',
                                min_df=10,                        # minimum df
                                stop_words='english',             # remove stop words
                                lowercase=True,
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )
    all_data = dm.get_all_fold_data() + dm.get_all_validation_data()
    vectorized = vectorizer.fit_transform(x[1] for x in all_data)
    train, validate = vectorized[:len(dm.get_all_fold_data())], vectorized[len(dm.get_all_fold_data()):]

    return train, validate, dm


def run_LDA(train, validate, doc_topic_prior=None, topic_word_prior=None,  learning_decay=0.7, learning_offset=10, batch_size=128, num_iterations=10):
    
    print("Running LDA...")
    lda_model = LatentDirichletAllocation(n_components=20,
                                          doc_topic_prior=doc_topic_prior,
                                          topic_word_prior=topic_word_prior,
                                          learning_method='online',
                                          learning_decay=learning_decay,
                                          learning_offset=learning_offset,
                                          max_iter=num_iterations,
                                          batch_size=batch_size,
                                          )
    lda_model.fit(train)
    
    return lda_model


def show_topics(vectorizer, lda_model, n_words=20):
    '''
    List feature words for each topics

    Arguments:
        - vectorizer (Class): vectorizr of LDA class.
        - lda_model (sklearn.lda): A instance of trained data

    Return Values:
        - (list): The feature keywords for each topic
    '''
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def change_color(val):
    '''
    helper function to change the color of the display for result

    Arguments:
        - val(int): a system setting value used to decide the color for the text

    Return Values:
        - (string): a format string
    '''
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def bold(val):
    '''
    helper function to set the result bold 

   Arguments:
        - val(int): a system setting value used to decide the font weight  

    Return Values:
        - (string): a format string 
    '''
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# best doc_topic_prior = 0.5
# best topic_word_prior = 0.1
# best learning_decay = 0.4
# best learning_offset = 5
# best batch_size = 135
train, validate, dm = get_data()
lda_model = run_LDA(train, validate, doc_topic_prior=0.5, topic_word_prior=0.1, learning_decay=0.4, learning_offset=5, batch_size=135)

topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
processed_data = [each[1] for each in dm.get_all_fold_data()]
docnames = ["Doc" + str(i) for i in range(len(processed_data))]

df_document_topic = pd.DataFrame(
    np.round(lda_model.transform(train), 2), columns=topicnames, index=docnames)
dominant_topic = np.argmax(df_document_topic.values, axis=1)

real_topics = [x[0] for x in dm.get_all_fold_data()]

conf_mat = get_confusion_matrix(dominant_topic, real_topics)
print(get_accuracy(conf_mat))