# --------------------------------------------------
# DataManager.py
#
# Data manager class, for unifying the interface for
# the topic modeling algorithms, regardless of the dataset.
# --------------------------------------------------

import os, sys, csv
import random
import json

# sklearn
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import numpy as np

import newsgroup_util
import twitter_util

# Project-wide constants, file paths, etc.
import settings


# Update the field size limit, as often the files will be too large.
csv.field_size_limit(sys.maxsize)

class DataManager:
    '''
    Class for loading in and managing data.
    '''

    def __init__(self, data_dir, dataset, remove_stopwords=False):
        '''
        Initialize the data manager class.

        Arguments:
            - data_dir (string): Absolute path where our data is stored.
            - dataset (string): The dataset we want to work with. Must be either 'newsgroup' or 'twitter'.
            - remove_stopwords (boolean): Flag for if we should remove stopwords or not.
        '''

        # First, raise an exception if the given dataset is invalid.
        if dataset not in ['newsgroup', 'twitter']:
            raise Exception('Invalid dataset given. Options are: newsgroup, twitter')

        # Each document takes up one row containing its terms.
        # It also has the true label attached, which is possibly null.
        # For example, [(label, ["token1", "token2", ...]), (label, ["token1", ...]), ...].
        self.__train, self.__test = [], []
        self.__classified_train, self.__classified_test = {}, {}
        self.__classes = [] # All class names.
        self.__dir = data_dir
        self.__dataset = dataset
        self.__rm_stop = remove_stopwords # Flag for if we should remove stopwords or not.

        # Set the appropriate function pointers for loading, normalizing, tokenizing, etc.
        # Also just set any other member variables we need for our specific dataset.
        if dataset == 'newsgroup':
            self.__download = newsgroup_util.download_newsgroup
            self.__load_data = newsgroup_util.load_data_newsgroup
            self.__tokenize = newsgroup_util.tokenize_newsgroup
            self.__normalize = newsgroup_util.normalize_newsgroup

            self.__train_file = settings.NEWSGROUP_TRAIN
            self.__test_file = settings.NEWSGROUP_TEST
            self.__class_file = settings.NEWSGROUP_CLASSES
        elif dataset == 'twitter':
            self.__download = twitter_util.download_twitter
            self.__load_data = twitter_util.load_data_twitter
            self.__tokenize = twitter_util.tokenize_twitter
            self.__normalize = twitter_util.normalize_twitter

        # There are no folds until self.divide_into_folds(k) is called
        self.__folds = None
        self.__num_folds = None
        self.__validation = None

        # Printed whenever a fold methods are called before divide_into_folds
        self.__no_folds_exception_msg = "You cannot call a fold-related method as divide_into_folds has not yet been called on this instance."


    def load_data(self, tweet_cache, download=False, download_path=None):
        '''
        Interface for loading in the dataset.
        Defaults to the specified function pointer from when the class is initialized.

        Arguments:
            - tweet_cache (string): path to the tweet cache
            - download (boolean): Flag for if we should force re-downloading of the data.
            - download_path: For Twitter dataset only, specifies the download path
        '''

        # Download the data if necessary.
        if self.__dataset == "twitter" and download:
            if download_path == None:
                raise Exception("Please specify a path to which the Twitter data should be downloaded.")
            self.__download(download_path)
        elif self.__dataset == "newsgroups" and download:
            if not os.path.exists(self.__train_file) \
                    or not os.path.exists(self.__test_file) \
                    or not os.path.exists(self.__class_file):

                self.__download()


        # Load in the data however the specific dataset needs to be done,
        # and get the training and test data, and possibly empty list of classes.
        if self.__dataset == "twitter":
            # use the path provided for the twitter dataset
            self.__train, self.__test, self.__classes = self.__load_data(self.__dir)
        else:
            self.__train, self.__test, self.__classes = self.__load_data()


        # We now want to tokenize and normalize our data.
        # Loop through the training and test data and update each document.
        try:
            # Caching tweets avoids having to re-normalize them every execution
            cache_file = open(tweet_cache, 'r+')
        except:
            cache_file = open(tweet_cache, 'w+')

        # Extract all tweets from the cache
        for i in range(len(self.__train)):
            if i and i%10000 == 0:
                print(i)
            cached = self.get_cached_tweet(cache_file, self.__train[i][2])
            if cached:
                self.__train[i][1] = cached
            else:
                self.__train[i][1] = ' '.join(self.__normalize(self.__tokenize(self.__train[i][1])))
                self.cache_tweet(cache_file, self.__train[i][2], self.__train[i][1])
        cache_file.close()

        # Extract all test tweets (only for newsgroups)
        for i in range(len(self.__test)):
            self.__test[i][1] = ' '.join(self.__normalize(self.__tokenize(self.__test[i][1])))

        if settings.DEBUG: print('Finished tokenizing and normalizing the training and test data.')

        # For ease of reference, we are going to organize the data by class.
        # This is only for newsgroups, the labelled dataset.
        if self.__dataset == "newsgroups":
            self.__classified_train = { c: [] for c in self.__classes }
            self.__classified_test = { c: [] for c in self.__classes }

            for doc in self.__train:
                self.__classified_train[doc[0]].append(doc[1])

            for doc in self.__test:
                self.__classified_test[doc[0]].append(doc[1])

        if settings.DEBUG: print('Finished loading in the dataset.')


    def load_cached_tweets(self, cache_file):
        '''
        Loads all cached tweets from cache_file into the __tweet_cache dictionary.

        Arguments:
            cache_file (file): File to read from.
        '''

        self.__tweet_cache = dict()
        for line in cache_file.readlines():
            tweet_id, content = line.split()[0], ' '.join(line.split()[1:])
            self.__tweet_cache[tweet_id] = content


    def get_cached_tweet(self, cache_file, tweet_id):
        '''
        Gets a cached tweet based on tweet id. The resulting tweet is already normalized / tokenized.

        Arguments:
            - cache_file (file): File to read from.
            - tweet_id  (int): The id of the tweet you want to retrieve.
        '''

        try:
            if str(tweet_id) in self.__tweet_cache:
                return self.__tweet_cache[str(tweet_id)]
            else:
                return None
        except:
            self.load_cached_tweets(cache_file)
            return self.get_cached_tweet(cache_file, tweet_id)


    def cache_tweet(self, cache_file, tweet_id, content):
        '''
        Gets a cached tweet based on tweet id. The resulting tweet is already normalized / tokenized.

        Arguments:
            - cache_file (file): File to write to.
            - tweet_id  (int): The id of the tweet you want to cache.
            - content (string): The content of the tweet you want to cache. No newlines please!
        '''

        cache_file.write(str(tweet_id) + " " + content + '\n')


    # Below are all the getter methods for retrieving data.
    def get_label(self, i, test=False):
        '''
        Given an index i into the dataset, return that indice's label.

        Arguments:
            - i (integer): Index of the document in the dataset.
            - test (boolean): Flag for if it should come from the training or test data.

        Return Values:
            - (string): The label at index i.
        '''
        return (self.__test[i][0] if test else self.__train[i][0])


    def get_text(self, i, test=False):
        '''
        Given an index i into the dataset, return that indice's text.

        Arguments:
            - i (integer): Index of the document in the dataset.
            - test (boolean): Flag for if it should come from the training or test data.

        Return Values:
            - (string): The text at index i.
        '''

        return (self.__test[i][1] if test else self.__train[i][1])


    def get_all_data(self, test=False):
        '''
        Get all of either the training data or test data.

        Arguments:
            - test (boolean): Flag for if it should come from the training or test data.

        Return Values:
            - (list): The list of documents. Each document has it's true label and text as a tuple.
        '''

        return (self.__test if test else self.__train)


    def get_data_by_class(self, class_name, test=False):
        '''
        Return all the data of given certain class.
        This will be useful to calculate the conditional probability, i.e. P(words | class).

        Arguments:
            - class_name (string): The class name we want to retrieve data of.
            - test (boolean): Flag for if it should come from the training or test data.
        '''

        return (self.__classified_test[class_name] if test else self.__classified_train[class_name])


    def get_data_classes(self, test=False):
        '''
        Get a list of all classes of all documents in a list. Used to test accruracy.

        Arguments:
            - test (boolean): Flag for if it should come from the training or test data.

        Return Values:
            - (list): The list of classes/labels for each text. Order is preserved.
        '''

        return ([i[0] for i in self.__test] if test else [i[0] for i in self.__train])


    def get_classes(self):
        '''
        Get all existing classes in this dataset.

        Return Values:
            - (list): The list of unique classes/labels that we've seen.
        '''

        return self.__classes


    def get_all_fold_data(self):
        """
        Get all data (text and labels) from the num_folds-1 folds that
        represent the training data. Excludes the validation set.

        Returns:
            (list): the list of training data points from all folds except validation.
        """
        if self.__num_folds == None:
            raise Exception(self.__no_folds_exception_msg)
        data = []
        for f in range(self.__num_folds):
            if f == self.__validation:
                # skip the validation set
                continue
            # if this is not the validation fold, extract all its training data
            data.extend([self.__train[self.__folds[f][i]] for i in range(len(self.__folds[f]))])
        return data


    def get_all_validation_data(self):
        """
        Get all data (text and labels) from the validation fold.

        Returns:
            (list): the list of training data points from the validation fold.
        """
        if self.__num_folds == None:
            raise Exception(self.__no_folds_exception_msg)

        f = self.__validation
        return [self.__train[self.__folds[f][i]] for i in range(len(self.__folds[f]))]


    def set_validation(self, idx):
        """
        Set the validation set to the fold indicated by idx.

        Arguments:
            - idx (int): 0 <= idx < self.__num_folds, the index of the new validation fold
        """
        if self.__num_folds == None:
            raise Exception(self.__no_folds_exception_msg)
        if not (0 <= idx < self.__num_folds):
            raise ValueError("The validation set index must be in the range [0, k), where k is the number of folds.")
        self.__validation = idx


    def get_num_folds(self):
        """
        Returns the number of folds.
        """
        if self.__num_folds == None:
            raise Exception(self.__no_folds_exception_msg)
        return self.__num_folds


    def __partition(self, lst, n):
        """
        Partition the list lst into n roughly equal parts.

        Arguments:
            lst (list): the list to partition
            n (int): n >= 2, the number of parts to divide lst into

        Returns:
            (list): the list of lists representing the partitions
        """
        # Ensure that the number of parts is at least 2
        assert(n >= 2)
        # From: https://stackoverflow.com/questions/3352737/how-to-randomly-partition-a-list-into-n-nearly-equal-parts
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]


    def divide_into_folds(self, k, mode=settings.ROUND_ROBIN):
        """
        Divide the training data into k folds.

        Arguments:
            k (int): k >= 2, the number of folds to create
            mode (int): one of three possible modes,
                ROUND_ROBIN (0): assign points to folds in round-robin order
                RANDOM (1): randomly assign training points to folds
                EVEN_SPLIT (2): divide the training data into partitions using the
                                existing order of the training data to create the
                                split points
                All three modes attempt to divide the datapoints as evenly as possible.
        """
        if mode not in {settings.ROUND_ROBIN, settings.RANDOM, settings.EVEN_SPLIT}:
            exception_msg = "Invalid mode value provided for division of training data into folds for cross validation.\n"
            exception_msg += f"Please use one of ROUND_ROBIN={settings.ROUND_ROBIN}, RANDOM={settings.RANDOM}, or EVEN_SPLIT={settings.EVEN_SPLIT}."
            raise Exception(exception_msg)

        # we have k folds, and currently the first fold is our validation set
        self.__num_folds = k
        self.__validation = 0

        # folds are stored as a list of lists
        # the ith fold is a list of indices to training elements
        # this is done rather than shuffling the documents themselves, as it's likely faster
        # not to store and pass around copies of long documents
        if mode == settings.ROUND_ROBIN:
            # assign the ith datapoint to fold i%k
            self.__folds = [[] for i in range(k)]
            for i in range(len(self.__train)):
                self.__folds[i%k].append(i)
        elif mode == settings.RANDOM:
            # First shuffle the list randomly, then partition
            indices = list(range(len(self.__train)))
            random.shuffle(indices)
            self.__folds = self.__partition(indices, k)
        elif mode == settings.EVEN_SPLIT:
            # simply partition the list without shuffling
            indices = list(range(len(self.__train)))
            self.__folds = self.__partition(indices, k)


    def run_lda(self, data=None, doc_topic_prior=0.5, topic_word_prior=0.1, learning_decay=0.4, learning_offset=5, batch_size=135, num_iterations=10, num_components=20):
        """
        Run LDA on the given data. Defaults to the training set if no data is specified.

        Arguments:
            data: Data to run on. Same format as received from get_all_data, get_all_folds, etc.
            doc_topic_prior (float): The doc_topic_prior hyperparam.
            topic_word_prior (float): The topic_word_prior hyperparam.
            learning_offset (float): The learning_offset hyperparam.
            batch_size (int): The batch_size hyperparam.
            num_iterations (int): The number of iterations to run for.
            num_components (int): The number of components or topics LDA generates.

        Returns:
            A list of lists, the j'th element of the i'th list is the probability that the i'th document belongs to topic j. (i.e. the weighting of topic j)
        """
        # run LDA on the given data. defaults to the training set.

        if not data: data = self.get_all_data()

        if settings.DEBUG: print("Vectorizing Data...")
        count_vect = CountVectorizer(
            min_df=2,
            max_features=10000,
            stop_words='english'
        )
        vectorized_data = count_vect.fit_transform([x[1] for x in data])

        if settings.DEBUG: print("Running LDA...")
        lda_model = LatentDirichletAllocation(
            n_components=num_components,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method='online',
            learning_decay=learning_decay,
            learning_offset=learning_offset,
            max_iter=num_iterations,
            batch_size=batch_size,
        )
        lda_model.fit(vectorized_data)
        return lda_model.transform(vectorized_data), lda_model, count_vect


    def run_nmf(self, data=None, alpha=1.34, beta_loss='kullback-leibler', l1_ratio=0.66, solver='mu', num_iterations=1000, num_components=20):
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

        # run NMF on the given data. defaults to the training set.
        if not data: data = self.get_all_data()

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

        return nmf_model.transform(vectorized_data), nmf_model, tfidf_vect


    def get_top_words_per_topic(self, model, vectorizer, n_top_words):
        '''
        Extracts the top words from each topic.

        Arguments:
            - model: the LDA or NMF model
            - vectorizer: the vectorizer (count or tf-idf) used when training the model
            - n_top_words (int): the number of words from each topic to return.

        Returns:
            - topic_words (dict): a dictionary mapping each topic id to a list of its top
                                  words, in decreasing order of probability.
        '''
        # https://stackoverflow.com/questions/44208501/getting-topic-word-distribution-from-lda-in-scikit-learn
        vocab = vectorizer.get_feature_names()
        topic_words = {}
        for topic, comp in enumerate(model.components_):
            word_idx = np.argsort(comp)[::-1][:n_top_words]
            topic_words[topic] = [vocab[i] for i in word_idx]
        return topic_words

      
    def save_words_as_json(self, words, path):
        '''
        Stores the dictionary of topic words to the json file represented by path.
        The main purpose of this function is to create the input data for intruder
        detection.

        Arguments:
            - words (dict): the dictionary mapping topic ids to a list of the most
                            probable words for that topic
            - path (string): the path where the resulting json file should be stored
        '''
        with open(path, 'w+') as json_file:
            print(json.dumps(words, indent=4), file=json_file)
