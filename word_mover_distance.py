import numpy as np
from gensim.models import KeyedVectors

"""
This class is dedicated for all word mover distance functions
provided in the original paper
"""
class WordMoverDistance():

    def __init__(self, pretrained_w2v_path, normalize = True):
        """Intialize an object from WordMoverDistance linked to a pretrained model,
        which enable to use Word Mover distance and its variations.
        
        Arguments:
            pretrained_w2v_path {[string]} -- [a path to the pretrained Word2Vec model]
        
        Keyword Arguments:
            normalize {bool} -- [normalize all the vectors in the pretrained model or not ] (default: {True})
        """
        self.w2v_model = KeyedVectors.load_word2vec_format(pretrained_w2v_path, binary=True)
        self.w2v_model.init_sims(replace=normalize)

    def calculate_nBoW(self, document):
        """Calculate the per word frequency in the document
        
        Arguments:
            document {[List of strings]} -- [list contains all the words that represent the content of a document]
        
        Returns:
            [dictionnary] -- [dictionnary with words as keys and frequency as a value]
        """
        dic = {}
        length = len(document)
        for word in document:
            if word not in dic.keys():
                dic[word] = 1
            else:
                dic[word] += 1

        for k,v in dic.items():
            dic[k] /= length

        return dic

    def calculate_nBoW_all(self, documents):
        """calculates the per word frequency for each document separately
        
        Arguments:
            documents {[List of list of strings]} -- [list of documents, each contains a list of all the words that represent the content of a document]
        
        Returns:
            [list of dictionnaries] -- [dictionnaries with words as keys and frequency as a value]
        """
        docs_nBoW = []
        for d in documents:
            docs_nBoW.append(self.calculate_nBoW(d))

        return docs_nBoW

    def calculate_centroid(self, document):
        """calculate the centroid of each document by a weighted sum of the word vectors that represent the document content
        
        Arguments:
            document {[list of strings]} -- [list of documents, each contains a list of all the words that represent the content of a document]
        
        Returns:
            [numpy array of float] -- [a vector represent the centroid of all the words in a document]
        """
        doc_nBoW = self.calculate_nBoW(document)
        centroid = np.zeros_like(self.w2v_model[list(doc_nBoW.keys())[0]])

        for k,v in doc_nBoW.items():
            centroid += v * self.w2v_model[k]

        return centroid

    def calculate_L2_Word2Vec(self, word_1, word_2):
        """calculate the L2 norm of the difference between two words vectors
        
        Arguments:
            word_1 {[string]} -- [a word]
            word_2 {[string]} -- [a word]
        
        Returns:
            [float] -- [norm of the distance between two words vectors]
        """
        word_1_vec = self.w2v_model[word_1]
        word_2_vec = self.w2v_model[word_2]

        return np.linalg.norm(word_1_vec - word_2_vec)

    def get_min_distance(self, word, doc_nBoW):
        """Get the minimum distance between a word and a group of words in a document
        
        Arguments:
            word {[string]} -- [a word]
            doc_nBoW {[dictionnaries]} -- [represent the words and its frequency in a document]
        
        Returns:
            [distance] -- [minimum distance between a group of calculated distances]
        """
        min_distance = self.calculate_L2_Word2Vec(word, list(doc_nBoW.keys())[0])

        for k, v in doc_nBoW.items():
            distance = self.calculate_L2_Word2Vec(word, k)

            if distance < min_distance:
                min_distance = distance

        return min_distance


    def WMD(self, doc_1, doc_2):
        """calculate the word mover distance between two documents
        
        Arguments:
            doc_1 {[list of strings]} -- [Content of a document]
            doc_2 {[list of strings]} -- [Content of a document]
        
        Returns:
            [float] -- [word mover distance between two documents]
        """
        return self.w2v_model.wmdistance(doc_1, doc_2)


    def WCD(self, doc_1, doc_2):
        """calculate the word centroid distance between two documents
        
        Arguments:
            doc_1 {[list of strings]} -- [Content of a document]
            doc_2 {[list of strings]} -- [Content of a document]
        
        Returns:
            [float] -- [word centroid distance between two documents]
        """
        centroid_1 = self.calculate_centroid(doc_1)
        centroid_2 = self.calculate_centroid(doc_2)

        return np.linalg.norm(centroid_1 - centroid_2)

    def RWMD_one_constraint(self, doc_1, doc_2):
        """calculate the relaxed word mover distance between two documents with respect to one constraint
        
        Arguments:
            doc_1 {[list of strings]} -- [Content of a document]
            doc_2 {[list of strings]} -- [Content of a document]
        
        Returns:
            [float] -- [relaxed word mover distance between two documents with respect to one constraint]
        """
        doc_1_nBoW = self.calculate_nBoW(doc_1)
        doc_2_nBoW = self.calculate_nBoW(doc_2)
        sum_T = 0

        for k_1,v_1 in doc_1_nBoW.items():
            sum_T += v_1 * self.get_min_distance(k_1, doc_2_nBoW)

        return sum_T

    def RWMD_one_constraint_nBoW(self, doc_1_nBoW, doc_2_nBoW):
        """calculate the relaxed word mover distance between two documents with respect to one constraint
        
        Arguments:
            doc_1_nBoW {[dictionary]} -- [word-frequency pairs of a document]
            doc_2_nBoW {[dictionary]} -- [word-frequency pairs of a document]
        
        Returns:
            [float] -- [relaxed word mover distance between two documents with respect to one constraint]
        """
        sum_T = 0

        for k_1,v_1 in doc_1_nBoW.items():
            sum_T += v_1 * self.get_min_distance(k_1, doc_2_nBoW)

        return sum_T

    def RWMD(self, doc_1, doc_2):
        """calculate the relaxed word centroid distance between two documents
        
        Arguments:
            doc_1 {[list of strings]} -- [Content of a document]
            doc_2 {[list of strings]} -- [Content of a document]
        
        Returns:
            [float] -- [relaxed word word distance between two documents]
        """
        doc_1_nBoW = self.calculate_nBoW(doc_1)
        doc_2_nBoW = self.calculate_nBoW(doc_2)
        L_1 = self.RWMD_one_constraint_nBoW(doc_1_nBoW, doc_2_nBoW)
        L_2 = self.RWMD_one_constraint_nBoW(doc_2_nBoW, doc_1_nBoW)

        return max(L_1,L_2)


    def RWMD_nBoW(self, doc_1_nBoW, doc_2_nBoW):
        """calculate the relaxed word mover distance between two documents
        
        Arguments:
            doc_1_nBoW {[dictionary]} -- [word-frequency pairs of a document]
            doc_2_nBoW {[dictionary]} -- [word-frequency pairs of a document]
        
        Returns:
            [float] -- [relaxed word mover distance between two documents]
        """
        L_1 = self.RWMD_one_constraint_nBoW(doc_1_nBoW, doc_2_nBoW)
        L_2 = self.RWMD_one_constraint_nBoW(doc_2_nBoW, doc_1_nBoW)

        return max(L_1,L_2)

    def kNN_exhaustive_WMD(self, query, docs, k = None):
        """get the k-nearest documents to a query document by calculating the word mover distance between
            the query and the reference documents and then compare.
        
        Arguments:
            query {[list of strings]} -- [document]
            docs {[list of strings]} -- [document]
        
        Keyword Arguments:
            k {[int]} -- [the number of nearest documents that should be returned] (default: {None})
        
        Returns:
            kNN_indices [type] -- [the indices of the nearest document with respect to their place in the given training data]
            k_WMDs [list of float] -- [list of Word mover distances between the nearest documents and the query]
        """
        if k is None or k > len(docs):
            k = len(docs)

        WMDs = [self.WMD(query,d) for d in docs]

        indices_sorted = np.argsort(WMDs)

        kNN_indices = [i for i in indices_sorted[:k]]

        k_WMDs = [WMDs[i] for i in kNN_indices]

        return kNN_indices, k_WMDs

    def kNN_RWMD(self, query, docs, k = None):
        """get the k-nearest documents to a query document by calculating the relaxed word mover distance between
            the query and the reference documents and then compare.
        
        Arguments:
            query {[list of strings]} -- [document]
            docs {[list of strings]} -- [document]
        
        Keyword Arguments:
            k {[int]} -- [the number of nearest documents that should be returned] (default: {None})
        
        Returns:
            kNN_indices [type] -- [the indices of the nearest document with respect to their place in the given training data]
            k_RWMDs [list of float] -- [list of Relaxed Word mover distances between the nearest documents and the query]
        """

        query_nBoW = self.calculate_nBoW(query)
        docs_nBoW = self.calculate_nBoW_all(docs)

        if k is None or k > len(docs):
            k = len(docs)

        RWMDs = [self.RWMD_nBoW(query_nBoW,d) for d in docs_nBoW]

        indices_sorted = np.argsort(RWMDs)

        kNN_indices = [i for i in indices_sorted[:k]]

        k_RWMDs = [RWMDs[i] for i in kNN_indices]

        return kNN_indices, k_RWMDs

    def kNN_prefetch_and_prune(self, query, docs, k = None, m = None):
        """get the k-nearest documents by implementing the prefetch and prune algorithm mentionned 
           in the original paper.
        
        Arguments:
            query {[list of strings]} -- [document]
            docs {[list of strings]} -- [document]
        
        Keyword Arguments:
            k {[int]} -- [the number of nearest documents that should be returned] (default: {None})
            m {[int]} -- [the total length of the document that could be scanned] (default: {None})
        Returns:
            current_kNN_indices [type] -- [the indices of the nearest document with respect to their place in the given training data]
            k_WMDs [list of float] -- [list of Word mover distances between the nearest documents and the query]
        """

        if k is None or k > len(docs):
            k = len(docs)
            m = len(docs)

        if m is None or m > len(docs):
            m = len(docs)

        WCDs = []

        for d in docs:
            WCDs.append(self.WCD(query,d))

        indices_sorted = np.argsort(WCDs)
        current_kNN_indices = [i for i in indices_sorted[:k]]

        k_WMDs = [self.WMD(query,docs[i]) for i in current_kNN_indices]

        if m > k:

            for j,ind in enumerate(indices_sorted[k+1:m]):
                temp_rwmd = self.RWMD(query, docs[ind])
                # last_rwmd = self.RWMD(query, docs[nearest_k_docs_indices[-1]])
                if temp_rwmd < k_WMDs[-1]:
                    temp_wmd = self.WMD(query,docs[ind])

                    if temp_wmd < k_WMDs[-1]:
                        k_WMDs[-1] = temp_wmd
                        current_kNN_indices[-1] = ind

                        k_indices_sorted = np.argsort(k_WMDs)
                        kNN_indices = [current_kNN_indices[i] for i in k_indices_sorted]
                        current_kNN_indices = kNN_indices
                        k_WMDs = np.sort(k_WMDs)


        return current_kNN_indices, k_WMDs
