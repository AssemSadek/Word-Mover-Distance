import numpy as np
from utils import timeit

"""
This class is dedicated for the training curriculum of the K-nearest neigbor algorithm for Word Mover Distance, it was inspired from the 
classical k-nearest neigbor classifier.
"""

class kNNClassifier():

    def __init__(self, WMD_model):
        """Initialize the classifier with a word mover distance model object that already
           has a pretrained Word2Vec model
        
        Arguments:
            WMD_model {[WordMoverDistance Object]} -- [an Object that has all the functionalities and features of Word Mover Distance and its variations]
        """
        self.WMD_model = WMD_model

    def train(self, x_train, y_train):
        """Stored the training data that will be used at prediction time.
        
        Arguments:
            x_train {[list of list of strings]} -- [training data of documents]
            y_train {[numpy array of int]} -- [classification for each document in the training data]
        """
        self.x_train = x_train
        self.y_train = y_train

    @timeit
    def predict(self, x, k = 1, m = None, algorithm = 'prefetch_and_prune'):
        """Predict the class of each of the query documents given 
        
        Arguments:
            x {[list of list of strings]} -- [list of documents to be tested]
        
        Keyword Arguments:
            k {int} -- [the number of nearest numbers that should be returned] (default: {1})
            m {[type]} -- [the total length of the document that could be scanned] (default: {None})
            algorithm {str} -- [the algorithm that should be run to get the nearest neighbor: it could be prefetch_and_prune, wmd or rwmd] (default: {'prefetch_and_prune'})
        
        Raises:
            ValueError: [When the algorithm argument is not given or not as expected]
        
        Returns:
            y_pred [numpy array of int] -- [classification of the document to be tested]
            kNN_indices [List of list of integer] -- [nearest neigbors indices for each of the query document]
            kNN_docs [List of list of documents] -- [nearest neigbors documents for each of the query document]
            kNN_distances [List of list of float] -- [the distance between the nearest neighbors documents with respect to the corresponding query]
        """
        test_size = len(x)
        kNN_indices = np.zeros((test_size, k), dtype=int)
        kNN_distances = np.zeros((test_size, k))
        y_pred = np.zeros(test_size)
        kNN_docs = []

        if algorithm == 'wmd':
            for i, query in enumerate(x):
                kNN_indices[i], kNN_distances[i] = self.WMD_model.kNN_exhaustive_WMD(query, self.x_train, k = k)
        elif algorithm == 'rwmd':
            for i, query in enumerate(x):
                kNN_indices[i], kNN_distances[i] = self.WMD_model.kNN_RWMD(query, self.x_train, k = k)
        elif algorithm =='prefetch_and_prune':
            for i, query in enumerate(x):
                kNN_indices[i], kNN_distances[i] = self.WMD_model.kNN_prefetch_and_prune(query, self.x_train, k = k, m = m)
        else:
            raise ValueError("kNN algorithm is not correct or not given")


        for i, kNN_ind in enumerate(kNN_indices):
            nearest_labels = self.y_train[kNN_ind]
            y_pred[i] = np.argmax(np.bincount(nearest_labels))
            kNN_docs.append([self.x_train[i] for i in kNN_ind[0:k]])

        return y_pred, kNN_indices, kNN_docs, kNN_distances
