import time
import numpy as np


def timeit(method):
    """decorator function that give the elapsed time of a given decorated function

    Arguments:
        method {[function]}- [the decorated function]

    Returns:
        [timed] -- [the elapsed time]
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        # if 'log_time' in kw:
        #     name = kw.get('log_name', method.__name__.upper())
        #     kw['log_time'][name] = int((te - ts) * 1000)
        # else:
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


def calculate_accuracy(y_pred, y_test):
    """Calculate the accuracy of a predicted vector compared to
    te ground truth vector

    Arguments:
        y__pred {[numpy array of int]} -- [predicted vector]
        y_test {[numpy array of int]} -- [ground truth vector ]

    Returns:
        accuracy [float] -- [the accuracy of the predicted vector]
    """
    test_size = y_test.shape[0]
    num_correct = np.sum(y_pred == y_test)
    accuracy = float(num_correct) / test_size
    return accuracy
