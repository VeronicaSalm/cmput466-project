
from sklearn.decomposition import LatentDirichletAllocation
import settings
import numpy as np
import pandas as pd
import re
import nltk
import json
from DataManager import DataManager
from newsgroup_util import get_confusion_matrix, assign_topics

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools

import sklearn
import matplotlib.pyplot as plt


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


# load in the data with dataManager class
dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
dm.load_data()
processed_data = [each[1] for each in dm.get_all_data()]


vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,                        # minimum df
                             stop_words='english',             # remove stop words
                             lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             )

data_vectorized = vectorizer.fit_transform(processed_data)

# Materialize data
data_dense = data_vectorized.todense()


lda_model = LatentDirichletAllocation(n_components=20,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every=-1,       # compute perplexity every n iters, default: Don't

                                      )
out_put = lda_model.fit_transform(data_vectorized)

LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                          evaluate_every=-1, learning_decay=0.7,
                          learning_method='online', learning_offset=10.0,
                          max_doc_update_iter=100, max_iter=10, mean_change_tol=0.001,
                          n_components=20, n_jobs=-1, perp_tol=0.1,
                          random_state=100, topic_word_prior=None,
                          total_samples=1000000.0, verbose=0)

# Log Likelyhood
print("Log Likelihood: ", lda_model.score(data_vectorized))

out_put = lda_model.transform(data_vectorized)

# visulize the result in text
topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]


docnames = ["Doc" + str(i) for i in range(len(processed_data))]


df_document_topic = pd.DataFrame(
    np.round(out_put, 2), columns=topicnames, index=docnames)


dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic


real_topics = [x[0] for x in dm.get_all_data()]


conf_mat = get_confusion_matrix(dominant_topic, real_topics)

acc_num = 0
acc_denom = 0
for key in conf_mat:
    acc_denom += conf_mat[key]
    if key[0] == key[1]:
        acc_num += conf_mat[key]
print("accuracy =", acc_num/acc_denom)

total_prec = 0
total_recall = 0

for real_topic in set(real_topics):
    numerator = 0
    prec_denom = 0
    recall_denom = 0
    for key in conf_mat:
        if key[0] == real_topic == key[1]:
            numerator += conf_mat[key]
            prec_denom += conf_mat[key]
            recall_denom += conf_mat[key]
        elif key[0] == real_topic:
            prec_denom += conf_mat[key]
        elif key[1] == real_topic:
            recall_denom += conf_mat[key]
    if prec_denom != 0:
        print(real_topic, "precision =", numerator/prec_denom)
        total_prec += numerator/prec_denom
    if recall_denom != 0:
        print(real_topic, "recall =", numerator/recall_denom)
        total_recall += numerator/recall_denom

print("avg precision =", total_prec/20)
print("avg recall =", total_recall/20)


df_document_topics = df_document_topic.head(
    100).style.applymap(change_color).applymap(bold)


topic_keywords = show_topics(
    vectorizer=vectorizer, lda_model=lda_model, n_words=15)

# Topic - Keywords Dataframe
TopicNumberToTopicName = assign_topics(dominant_topic, real_topics)
frequent_words = pd.DataFrame(topic_keywords)
frequent_words.columns = ['Word ' + str(i)
                          for i in range(frequent_words.shape[1])]
frequent_words.index = [TopicNumberToTopicName[i]
                        for i in range(frequent_words.shape[0])]
print(frequent_words)

# Define Search Param
search_params = {'n_components': [10, 15, 20,
                                  25, 30], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)


GridSearchCV(cv=None, error_score='raise',
             estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                                                 evaluate_every=-1, learning_decay=0.7, learning_method=None,
                                                 learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
                                                 mean_change_tol=0.001, n_components=10, n_jobs=1,
                                                 n_components=None, perp_tol=0.1, random_state=None,
                                                 topic_word_prior=None, total_samples=1000000.0, verbose=0),
             n_jobs=1,
             param_grid={'n_topics': [10, 15, 20, 25, 30],
                         'learning_decay': [0.5, 0.7, 0.9]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
             scoring=None, verbose=0)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# Get Log Likelyhoods from Grid Search Output
n_topics = [10, 15, 20, 25, 30]
log_likelyhoods_5 = [round(gscore.mean_validation_score)
                     for gscore in model.cv_results_ if gscore.parameters['learning_decay'] == 0.5]
log_likelyhoods_7 = [round(gscore.mean_validation_score)
                     for gscore in model.cv_results_ if gscore.parameters['learning_decay'] == 0.7]
log_likelyhoods_9 = [round(gscore.mean_validation_score)
                     for gscore in model.cv_results_ if gscore.parameters['learning_decay'] == 0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()

# Output Json file of top n=10 words in each topic for Intruder detection
words = {}
for i in range(frequent_words.shape[0]):
    words[i] = topic_keywords[i].tolist()[:10]
with open("topic_words_lda.json", 'w') as json_file:
    json.dump(words, json_file)