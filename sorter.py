from debugg import debug

from typing import List
from scipy.spatial.distance import euclidean

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# ===========================================



def score(element, parameters):
    """
        Returns the euclidian distance between values of the row and the parameter
    """
    
    return euclidean(element, parameters)


def normalize_list(list):
    """
        Normalize the list using scikit-learn
    """
    return scaler.fit_transform(list).tolist()


def myorder(list, parameters = None):
    """
        Given a list and parameters, returns the list ordered
    """

    debug("\t\t\t Ordenando uma lista")

    if (parameters == None):
        parameters = [1]*len(list[0])
        # default parameters are 111111

    list.sort(key=lambda x: score(x, parameters))