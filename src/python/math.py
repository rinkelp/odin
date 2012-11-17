
"""
math.py

Various mathematical functions and operations.
"""


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