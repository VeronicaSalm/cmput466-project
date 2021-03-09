# --------------------------------------------------
# DataManager.py
#
# Data manager class, for unifying the interface for
# the topic modeling algorithms, regardless of the dataset.
# --------------------------------------------------

import os, sys, csv

import newsgroup_util

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
            # TODO: Implement normalizing and tokenizing functions for twitter dataset.
            pass


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
            self.__train[i][1] = self.__normalize(self.__tokenize(self.__train[i][1]))
        
        for i in range(len(self.__test)):
            self.__test[i][1] = self.__normalize(self.__tokenize(self.__test[i][1]))

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
