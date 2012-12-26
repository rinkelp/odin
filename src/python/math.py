
"""
math.py

Various mathematical functions and operations.
"""

import numpy as np
from random import randrange, seed



def rand_pairs(numItems,numPairs):
	seed()
	i = 0
	pairs = []
	while i < numPairs:
		ind1 = randrange(numItems)
		ind2 = ind1
		while ind2 == ind1:
			ind2 = randrange(numItems)
		pair = [ind1,ind2]
		if pairs.count(pair) == 0:
			pairs.append(pair)
			i += 1
	return pairs


def fft_acf(data):
    '''
    Return the autocorrelation of a 1D array using the FFT
    Note: the result is normalized
    
    Parameters
    ----------
    data : ndarray, float, 1D
        Data to autocorrelate
        
    Returns
    -------
    acf : ndarray, float, 1D
        The autocorrelation function
    '''
    data = data - np.mean(data)
    result = signal.fftconvolve(data, data[::-1])
    result = result[result.size / 2:] 
    acf = result / result[0]
    return acf
    
    
