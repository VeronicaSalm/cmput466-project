from sklearn.datasets import fetch_20newsgroups

from settings import *

# Maybe use this instead?
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html#sklearn.datasets.fetch_20newsgroups_vectorized

newsgroups_train = fetch_20newsgroups(data_home=NEWSGROUP_PATH, subset='train', categories=['alt.atheism', 'sci.space'])

print(newsgroups_train)