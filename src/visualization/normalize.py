import numpy as np

def minmax_norm_0_1(data):
    """
    Min-max normalization of data to a range of [0, 1]. 

    :param data: The arry to normalize.
    
    :param return: The normalized array. 
    """

    data = (data - np.min(data)) / (np.max(data)- np.min(data))

    return data


def minmax_norm_minus1_1(data):
    """
    Min-max normalization of data to a range of [-1, 1]. 

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.min(data)) / (np.max(data)- np.min(data))
    data = data * 2 - 1

    return data

def minmax_binning(data):
    """
    Min-max normalization of data to a range of [0, 255] and then it bins it to 256 bins.

    :param data: The arry to normalize.

    :param return: The normalized array. 
    """

    data = (data - np.min(data)) / (np.max(data)- np.min(data))
    data = np.uint8(np.round(data * 255, decimals=0))

    return data