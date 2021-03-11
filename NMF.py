# Testing NMF

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups

from DataManager import DataManager
import settings

n_feat = 1000

dm = DataManager(settings.NEWSGROUP_DIR, 'newsgroup')
dm.load_data()

train_data = [doc[1] for doc in dm.get_all_data()]
test_data = [doc[1] for doc in dm.get_all_data(True)]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_feat,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(train_data)

nmf = NMF(n_components=20, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

tfidf_new = tfidf_vectorizer.transform(test_data)
X_new = nmf.transform(tfidf_new)
predicted = [np.argsort(doc)[::-1][0] for doc in X_new]

cnt = len(test_data)
correct = 0

labels = dm.get_classes()

for i in range(len(test_data)):
    if labels[predicted[i]] == dm.get_label(i):
        correct += 1
    else:
        print('predicted:', labels[predicted[i]], 'true:', dm.get_label(i))

print(correct / cnt)
