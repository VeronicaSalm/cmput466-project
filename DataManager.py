# --------------------------------------------------
# DataManager.py
#
# Data manager class, for unifying the interface for
# the topic modeling algorithms, regardless of the dataset.
# --------------------------------------------------

import os, sys, csv
import random

import newsgroup_util

# Project-wide constants, file paths, etc.
import settings


# Update the field size limit, as often the files will be too large.
csv.field_size_limit(sys.maxsize)

# Modes for fold creation
ROUND_ROBIN = 0
RANDOM = 1
EVEN_SPLIT = 2

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
            # TODO: Incorporate downloading and loading functions.
            # self.__download = twitter_util.download_twitter
            # self.__load_data = twitter_util.load_data_twitter
            self.__tokenize = twitter_util.tokenize_twitter
            self.__normalize = twitter_util.normalize_twitter

            # self.__train_file = settings.TWITTER_TRAIN
            # self.__test_file = settings.TWITTER_TEST
            # self.__class_file = settings.TWITTER_CLASSES

        # There are no folds until self.divide_into_folds(k) is called
        self.__folds = None
        self.__num_folds = None
        self.__validation = None


    def load_data(self, download=False):
        '''
        Interface for loading in the dataset.
        Defaults to the specified function pointer from when the class is initialized.

        Arguments:
            - download (boolean): Flag for if we should force re-downloading of the data.
        '''

        # Download the data if necessary.
        if download or not os.path.exists(self.__train_file) \
                or not os.path.exists(self.__test_file) \
                or not os.path.exists(self.__class_file):

            self.__download()

        # Load in the data however the specific dataset needs to be done,
        # and get the training and test data, and possibly empty list of classes.
        self.__train, self.__test, self.__classes = self.__load_data()

        # We now want to tokenize and normalize our data.
        # Loop through the training and test data and update each document.
        for i in range(len(self.__train)):
            self.__train[i][1] = ' '.join(self.__normalize(self.__tokenize(self.__train[i][1])))

        for i in range(len(self.__test)):
            self.__test[i][1] = ' '.join(self.__normalize(self.__tokenize(self.__test[i][1])))

        if settings.DEBUG: print('Finished tokenizing and normalizing the training and test data.')

        # For ease of reference, we are going to organize the data by class.
        self.__classified_train = { c: [] for c in self.__classes }
        self.__classified_test = { c: [] for c in self.__classes }

        for doc in self.__train:
            self.__classified_train[doc[0]].append(doc[1])

        for doc in self.__test:
            self.__classified_test[doc[0]].append(doc[1])

        if settings.DEBUG: print('Finished loading in the dataset.')


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
            - class_name (string): The class name we want to retrieve Gdata of.
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
            raise Exception("You cannot call a fold-related method as divide_into_folds has not yet been called on this instance.")
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
            raise Exception("You cannot call a fold-related method as divide_into_folds has not yet been called on this instance.")

        f = self.__validation
        return [self.__train[self.__folds[f][i]] for i in range(len(self.__folds[f]))]


    def set_validation(self, idx):
        """
        Set the validation set to the fold indicated by idx.

        Arguments:
            - idx (int): 0 <= idx < self.__num_folds, the index of the new validation fold
        """
        if self.__num_folds == None:
            raise Exception("You cannot call a fold-related method as divide_into_folds has not yet been called on this instance.")
        if not (0 <= idx < self.__num_folds):
            raise ValueError("The validation set index must be in the range [0, k), where k is the number of folds.")
        self.__validation = idx

    def get_num_folds(self):
        """
        Returns the number of folds.
        """
        return self.__num_folds

    def _partition(self, lst, n):
        """
        Partition the list lst into n roughly equal parts.

        Arguments:
            lst (list): the list to partition
            n (int): n >= 2, the number of parts to divide lst into

        Returns:
            (list): the list of lists representing the partitions
        """
        # From: https://stackoverflow.com/questions/3352737/how-to-randomly-partition-a-list-into-n-nearly-equal-parts
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

    def divide_into_folds(self, k, mode=ROUND_ROBIN):
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
        if mode not in {ROUND_ROBIN, RANDOM, EVEN_SPLIT}:
            print("Error: Invalid mode value provided for division of training data into folds for cross validation.")
            print(f"Please use one of ROUND_ROBIN={ROUND_ROBIN}, RANDOM={RANDOM}, or EVEN_SPLIT={EVEN_SPLIT}.")
            sys.exit()

        # we have k folds, and currently the first fold is our validation set
        self.__num_folds = k
        self.__validation = 0

        # folds are stored as a list of lists
        # the ith fold is a list of indices to training elements
        # this is done rather than shuffling the documents themselves, as it's likely faster
        # not to store and pass around copies of long documents
        if mode == ROUND_ROBIN:
            # assign the ith datapoint to fold i%k
            self.__folds = [[] for i in range(k)]
            for i in range(len(self.__train)):
                self.__folds[i%k].append(i)
        elif mode == RANDOM:
            # First shuffle the list randomly, then partition
            indices = list(range(len(self.__train)))
            random.shuffle(indices)
            self.__folds = self._partition(indices, k)
        elif mode == EVEN_SPLIT:
            # simply partition the list without shuffling
            indices = list(range(len(self.__train)))
            self.__folds = self._partition(indices, k)

    def set_dummy_data(self):
        # TODO: remove this before making PR
        self.__train = [(1, chr(i)) for i in range(ord("A"), ord("Z")+1)]
        self.__test = [(1, chr(i)) for i in range(ord("a"), ord("z")+1)]
