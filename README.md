# Word Mover Distance
This code is an implementation of the Word Mover Distance and its variations described in its [original paper](http://proceedings.mlr.press/v37/kusnerb15.pdf)

The functions was tested over the public dataset BBCSport News. To download it, please refer to this [link](http://mlg.ucd.ie/datasets/bbc.html)

The raw version of this dataset will be needed, in order to reproduce the same experiment in the given notebook file.

I will explain the main component of this expirement, and then the analysis of the result.

### Class Word Mover Distance
In this core class, the variations of the Word Mover Distance and the given prefetch and prune algorithm provided in the paper are implemented, in addition to some other helpers functons that could be found in the same python file of the class.

#### Important methods
In this section I will explain the important methods used for document matching and document classifications, in addition to the complexity for each method.

To know how to use these functions, please refer to the detailed inline documentation provided in source code word_mover_distance.py

##### WMD(doc_1,doc_2)
This function is just a wrapper for the provided wmdistance(doc_1, doc_2) function in gensim package

The complexity of this function will be similar to the mentioned in the paper due to solving the linear programming optimization problem which is $O(p^3log(p))$

##### WCD(doc_1,doc_2)
In this function, the word centroid distance is calculated by using these steps:
1. calculate the nBoW of each document to get a dictionary that pair every unique word in the document with its frequency $d_i$ within the document. (complexity = 2L, where L is length of the document)
2. calculate the centroid for every document alone, This is done by a weighted sum over all the word embedding vectors of each words in the document, the weight correspond to the mentioned frequency $d_i$ in the calculated nBoW. (complexity = dp where d is the length of every vector and p is the number of every unique element in the document)
3. After having the two documents centroids, we calculate the L2 norm of the difference between the two centroids vectors. (complexity = d)

The total exact complexity is 2L + dp + d. Thus $O(dp)$
##### RWMD_one_constraint(doc_1,doc_2)
In this function, the relaxed version of Word Mover Distance is calculated by using one constraint only which is corresponding to doc_1.
1. calculate again the nBoW of each document.
2. for every unique word in document 1, find the minimum distance with the second document by calculating the distance with every unique word in document 2 and then multiply it with $d_i$ (complexity = $p^2$)
3. accumulating all the weighted minimum distances to get the RWMD

It's easy to demonstrate that the complexity is $O(p^2)$
#####RWMD(doc_1,doc_2)
In this function we used the previous function to get the final relaxed word mover distance:
1. calculate RWMD with constraint on doc_1
2. calculate RWMD with constraint on doc_2
3. take the maximum distance
   
The complexity is the same as with one constraint.
#####kNN_exhaustive_WMD(query,docs, k)
Since we now have a distance function between two documents which is the word mover distance, we can use it to calculate the normal kNN algorithm by replacing the normal L2 euclidean distance function in the classical kNN algorithm by wmd(doc_1, doc_2) function

The complexity of this algorithm will be $O(N* p^3log(p))$, where N is the number of the training set, since we will calculate WMD N times.
#####kNN_RWMD(query,docs, k)
Same as the previous algorithm but we will replace the wmd function with rwmd function. Thus, the complexity will drop to $O(N*p^2)$
#####kNN_prefetch_and_prune(query,docs, k, m)
With the description of the algorithm in the paper, the implementation is as follow:
1. calculate WCD for all documents in the training set with the query document. (complexity = N*dp)
2. if m == k, stop the algorithm and return the WMD minimum distances with the corresponding indices that represent the nearest documents in the training sets.
3. if m > k, continue on checking if we can replace some of the nearest documents by other documents from the k+1 to m documents. This is done by:
   - calculate RWMD for every outter documents (the one from k+1 to m)
   - if RWMD < WMD[$k^{th}$ doc], calculate WMD of this document
   - if WMD < WMD[$k^{th}$ doc], replace it with this element and resort again the new kNN documents

The complexity of this algorithm:
- Best case scenario, when m = k : $O(N*dp)$
- Worst case scenario, when m = N and every time get into the final step of calculating WMD and replacing documents: $O(N*p^3log(p))$, like the exhaustive WMD


###BBCSport Data Loader
This class is helpful to load, process and manage the raw dataset of BBC sport which was used to run some experiments on the mentioned algorithms

The most important function is train_test_split. A detailed documentation will be found in the source code of the class.
###kNN Classifier
this class is a wrapper for the kNN algorithms provided by WordMoverDistance class. It use mainlly to function:
- train(x_train, y_train): This is like any kNN classifier's train function, just store the training data.
- predict(x_test, k, m, algorithm): In this function we choose which kNN algorithm from WordMoverDistance to run on the x_test


###Experiment results and interpretations

I run these experiments on my modest computer, thus I didn't have the luxury to run it in parallel on multiple core. It took a huge time to compute kNN on all the data. Therefore, I decide to run it on a portions of the dataset. 

All experiment holds k=3

####Experiment 1:
dataset size= 100 documents
training size = 80 documents
test size = 20 documents
|Algorithm| m       | Accuracy           | Time (seconds)  |speedup
|---------| ------------- |:-------------:| -----:|-----:|
|Exhaustive WMD| not applicable | 0.9 | 723.75| - |
|RWMD| not applicable |   0.85   | 519.79 | 1.4x|
|prefetch and prune| k (WCD) | 0.85 | 37.56| 19x|
|prefetch and prune| 2k |0.85 | 73.01 |9x|
|prefetch and prune| 4k |0.85 | 163.0 |4x|
|prefetch and prune| 8k |0.9 | 334.45 |2x|
|prefetch and prune| n (exact) | 0.9|  747.79| 1x|

####Experiment 2:
dataset size= 184 documents (25% of the original dataset)
training size = 147 documents
test size = 37 documents
|Algorithm| m       | Accuracy           | Time  |speedup
|---------| ------------- |:-------------:| -----:|-----:|
|Exhaustive WMD| not applicable | 0.91 | 2345.71| - |
|RWMD| not applicable |   0.91   |  1735.45|1.35x|
|prefetch and prune| k (WCD) |0.81 | 67.47|34x|
|prefetch and prune| 2k | 0.83 | 131.83|18x|
|prefetch and prune| 4k | 0.89| 306.3|7x|
|prefetch and prune| 8k | 0.83| 631.541|4x|
|prefetch and prune| n (exact) | 0.89 | 2427.41|1x|


####Experiment 3 (not completed yet):
dataset size= 737 documents (100% of the original dataset)
training size = 589 documents
test size = 148 documents

In this experiment I compare the results with the RWMD version

|Algorithm| m       | Accuracy           | Time  |speedup
|---------| ------------- |:-------------:| -----:|-----:|
|RWMD| not applicable |   0.95   |  28344.9|1x|
|prefetch and prune| k (WCD) |0.91 |414.5|68x|
|prefetch and prune| 2k | 0.91 | 730.74|38x|
|prefetch and prune| 4k | 0.91|1486.96|19x|
|prefetch and prune| 8k |0.94|2844.32|10x|
|prefetch and prune| n (exact) | - |-|-|
|Exhaustive WMD| not applicable | 0.91 | -| - |


####Interpretation
We can see a tradeoff between accuracy and the speedup. I believe that my implementation could be more optimized. specially for the exact version of the prefetch and prune.
