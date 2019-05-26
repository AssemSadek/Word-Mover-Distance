#http://mlg.ucd.ie/datasets/bbc.html
import glob
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import download
from nltk.tokenize import RegexpTokenizer
download('stopwords')

def preprocess_document(document, vocabulary):
    """preprocess a given document to return a list of the significant words in the content of the document.
    
    Arguments:
        document {[string]} -- [content of a document]
        vocabulary {[dictionnary]} -- [list of available words in a pretrained vocabulary]
    
    Returns:
        document [List of strings] -- [List of the words of the document]
    """
    document = document.lower()
    def check_number_existant(word):
        numbers = ['0','1','2','3','4','5','6','7','8','9']
        for c in word:
            if c in numbers:
                return True

        return False

    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')

    document = tokenizer.tokenize(document)
    document = [w for w in document if w not in stop_words]
    document = [w for w in document if not check_number_existant(w)]
    document = [w for w in document if w in vocabulary]
    return document

random.seed(4)

"""This class serves as a data loader to get and preprocess the raw data of the BBCSport public dataset
"""
class BBCSportDataLoader():

    def __init__(self, raw_data_folder, vocabulary):
        """initialize the dataset by crawling the raw folder of BBCSport files and preprocess the file contents.
        
        Arguments:
            raw_data_folder {[string]} -- [Path to the raw dataset folder]
            vocabulary {[dictionary]} -- [list of available words in a pretrained vocabulary]
        """
        self.vocabulary = vocabulary
        self.raw_data_folder = raw_data_folder
        self.classes = ['athletics', 'cricket', 'football', 'rugby', 'tennis']
        self.dataset_x = []
        self.dataset_y = []

        for i, c in enumerate(self.classes):
            self.crawl_class_folder(c,i)

        content_class = list(zip(self.dataset_x, self.dataset_y))
        random.shuffle(content_class)
        self.dataset_x, self.dataset_y = zip(*content_class)

    def crawl_class_folder(self, class_name, class_id):
        """crawl the folder of a certain category/class
        
        Arguments:
            class_name {[string]} -- [class/category of documents, also refer to as a subfolder]
            class_id {[int]} -- [class label]
        """
        file_paths = [f for f in glob.glob(self.raw_data_folder + class_name + "/*.txt")]
        documents = []

        for filename in file_paths:

            with open(filename, 'r') as f:
                content = f.read()
                documents.append(preprocess_document(content, self.vocabulary))

        self.dataset_x += documents
        self.dataset_y += [class_id] * len(documents)


    def train_test_split(self, data_portion = 1, test_precentage = 0.2):
        """Specify the portion of the data that will be used for as actual dataset, also 
        specify a percentage of this actual dataset to be used as testing data and the remaining
        as training data.
        
        Keyword Arguments:
            data_portion {int} -- [portion percentage of the actual dataset] (default: {1})
            test_precentage {float} -- [percentage of the test data from the actual data] (default: {0.2})
        
        Returns:
            self.x_train [List of list of strings] -- [List of training documents]
            self.y_train [numpy array of integers] -- [vector of labels for the training documents]
            self.x_test [List of list of strings] -- [List of test documents]
            self.y_test [numpy array of integers] -- [vector of labels for the test documents]
        """
        current_dataset_size =  int(data_portion * len(self.dataset_x))
        self.current_dataset_x = self.dataset_x[:current_dataset_size]
        self.current_dataset_y = self.dataset_y[:current_dataset_size]

        train_size = int((1 - test_precentage) * len(self.current_dataset_x))
        self.x_train = self.current_dataset_x[0:train_size]
        self.y_train = self.current_dataset_y[0:train_size]
        self.x_test = self.current_dataset_x[train_size:]
        self.y_test = self.current_dataset_y[train_size:]

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        return self.x_train, self.y_train, self.x_test, self.y_test
