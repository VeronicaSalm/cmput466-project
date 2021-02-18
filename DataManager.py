import preprocess_newsgroups  # tokenazilation normalization code
import os
import sys
import csv


class DataManager:

    def __init__(self, path, load_path=None):

        # each document takes up one row containing its terms. e.g. [["toekn1", "token2......"], ["token1"....]]
        self.__all_data = []
        self.__path = path  # path to read in raw datas
        # dictionary keys = class, values = all documents within that class
        self.__classified_data = {}
        # class info according to each documents in self.__all_data. e.g ["sports", "political"......]
        self.__data_class = []
        self.__class_space = []  # all classes's name

        # if load_path is provided, load data from existing tsv files
        if (load_path != None):
            tsv_file = open(load_path, 'r')  # read file
            read_tsv = csv.reader(tsv_file, delimiter="\t")

            for each_row in read_tsv:

                self.__data_class.append(each_row[0])  # class is first col
                self.__all_data.append(eval(each_row[1]))

                if each_row[0] in self.__classified_data.keys():
                    self.__classified_data[each_row[0]].append(
                        eval(each_row[1]))
                else:
                    self.__classified_data[each_row[0]] = [eval(each_row[1])]
            self.__class_space = list(
                set(self.__data_class))  # remove duplicates

    def loadIn(self):
        '''
        Need to be implemented: check if data are raw or already processed and stored in the disc, if so load from disc 
        LoadIn data into class attribute with proper processing 
        Input: None
        Return: None
        SideEffect: change class attribute 
        '''
        self.__class_space = os.listdir(self.__path)
        for each in self.__class_space:
            self.__classified_data[each] = []
            file_path = os.path.join(self.__path, each)

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
        Save everything into disc as tsv file
        '''
        file_name = "all_data"
        f_name = os.path.join(path, file_name)

        with open(f_name, "w") as f:
            out = csv.writer(f, delimiter='\t')

            for i in range(len(self.__all_data)):
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
