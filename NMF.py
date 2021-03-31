# Testing NMF

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

from DataManager import DataManager
import settings

def show_topics(vectorizer, nmf_model, n_words = 20):
    """
    List feature words for each topic

    Arguments:
        - vectorizer (Class): vectorizer of NMF class
        - nmf_model (sklearn.nmf): an instance of trained data
        - n_words: number of words to extract for each topic
    
    Return Values:
        - (list): Theh feature keywords for each topic
    """
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in nmf_model.components_:
        topic_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(topic_keyword_locs))
    return topic_keywords


def get_confusion_matrix(generated_topics, real_topics):
    '''
    Gives a confusion matrix for a set of generated topics. 
    Generated based on assigning real topics using the assign_topics function.
    Arguments:
        - generated_topics (list): The generated topics for each document. 
        generated_topics[i] is the generated topic of the i'th document in the corpus.
        - real_topics (list): The real topics for each document, as labelled in the dataset.
        real_topics[i] is the real topic of the i'th document in the corpus.
        The dimensions of generated_topics and real_topics should match.
    
    Returns:
        The confusion matrix out, represented as a dict mapping tuples to counts.
        out[(topicA, topicB)] is the number of documents predicted as topicA that are actually topicB
    '''
    assigned_topics = assign_topics(generated_topics, real_topics)

    out = dict()
    for real_topic in set(real_topics):
        for real_topic2 in set(real_topics):
            out[(real_topic, real_topic2)] = 0

    for generated_topic, real_topic in zip(generated_topics, real_topics):
        out[(assigned_topics[generated_topic], real_topic)] += 1
    
    return out


def assign_topics(generated_topics, real_topics):
    '''
    Assign real names in a "heuristic" manner to auto generated topics.
    For each generated topic, assign it to the real topic that is most often assigned to it, throwing out real topics that we have aalready assigned. 
    Arguments:
        - generated_topics (list): The generated topics for each document. 
        generated_topics[i] is the generated topic of the i'th document in the corpus.
        - real_topics (list): The real topics for each document, as labelled in the dataset.
        real_topics[i] is the real topic of the i'th document in the corpus.
        The dimensions of generated_topics and real_topics should match.
    
    Returns:
        A dict mapping values in generated_topics to values in real_topics.
    '''
    out = dict()
    used_real_topics = set()
    for topic in set(generated_topics):
        real_topic_count = dict()
        for real_topic in set(real_topics):
            real_topic_count[real_topic] = 0
        for generated_topic, real_topic in zip(generated_topics, real_topics):
            if generated_topic == topic:
                real_topic_count[real_topic] += 1
        
        # sort in descending order by value
        sorted_counts = sorted(real_topic_count.items(), key = lambda x: -x[1])
        
        for real_topic, count in sorted_counts:
            if real_topic not in used_real_topics:
                # if we haven't used this topic, then use it and mark it so we don't use it again
                out[topic] = real_topic 
                used_real_topics.add(real_topic)
                break
        
    return out

n_feat = 20000

dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
dm.load_data()

train_data = [doc[1] for doc in dm.get_all_data()]
test_data = [doc[1] for doc in dm.get_all_data(True)]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_feat,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(train_data)

nmf = NMF(n_components=20, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=0.1,
          l1_ratio=0.5).fit(tfidf)

tfidf_new = tfidf_vectorizer.transform(test_data)
predicted = nmf.transform(tfidf_new)

#print(get_confusion_matrix(predicted, dm.get_data_classes(True)))

dominant_topic = np.argmax(predicted, axis=1)
#df_document_topic['dominant_topic'] = dominant_topic

print(dominant_topic)

real_topics = [x[0] for x in dm.get_all_data(True)]

print(real_topics)

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


# Topic - Keywords DataFrame
TopicNumberToTopicName = assign_topics(dominant_topic, real_topics)
topic_keywords = show_topics(vectorizer=tfidf_vectorizer, nmf_model=nmf, n_words=15)
frequent_words = pd.DataFrame(topic_keywords)
frequent_words.columns = ['Word ' + str(i)
                          for i in range(frequent_words.shape[1])]
frequent_words.index = [TopicNumberToTopicName[i]
                        for i in range(frequent_words.shape[0])]
print(frequent_words)

