import preprocess_newsgroups  # tokenazilation normalization code
import os
import sys
import csv


class DataManager:

    def __init__(self, path):

        # each document takes up one row containing its terms. e.g. [["toekn1", "token2......"], ["token1"....]]
        self.__all_data = []
        self.__path = path  # path to read in raw datas
        # dictionary keys = class, values = all documents within that class
        self.__classified_data = {}
        # class info according to each documents in self.__all_data. e.g ["sports", "political"......]
        self.__data_class = []
        self.__class_space = []  # all classes's name

    def loadIn(self):
        '''
        Need to be implemented: check if data are raw or already processed and stored in the disc, if so load from disc 
        LoadIn data into class attribute with proper processing 
        Input: None
        Return: None
        SideEffect: change class attribute 
        '''
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

    def get_all(self):
        '''
        Return all data list 
        '''
        return self.__all_data

    def get_data_by_class(self, class_name):
        '''
        Return all the data of given certain class.
        This will be useful to calculate the conditional probability e.g P(words | class )
        '''
        return self.__classified_data[class_name]

    def write_to(self, path):
        '''
        Save everything into disc

        To be implemented 
        '''
        file_name = "all_data"
        f_name = os.path.join(path, file_name)
        out = csv.writer(f, delimiter='\t')
        with open(f_name, "w") as f:

            for i in range(size(self.__all_data)):
                # write into csv: each row has 2 cols, first is the class of this doc, second is tokens of this doc
                written = [self.__data_class[i], self.__all_data[i]]
                out.writerow(written)

    def get_data_class(self):
        '''
        return Y, AKA. classes of all documents in a list. Used to test accruracy  
        '''
        return self.__data_class

    def get_all_classes(self):
        '''
        return all existing classes in this dataset
        '''
        return self.__class_space
