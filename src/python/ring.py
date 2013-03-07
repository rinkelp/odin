from scipy.special import legendre
from math import cos,sin,pi, acos,asin,sqrt
from numpy import isnan,zeros,real,transpose,array,average
from numpy.random import randint
from numpy.ma import conjugate
from scipy import fftpack
from random import seed,randrange
#import stats
import pylab as plt
import numpy as np

def q2theta(q,qres,wavelen):
  theta = asin( q*qres*wavelen  / 4./pi)
  return theta

def dePolarize(I,t,outOfPlane):
  """ polarization correction I:intensity vs phi; t: theta; outOfPlane : % polarize"""
  N = len(I)
  Ic = np.copy(I)
  SinTheta = sin(t)
  for i in xrange(N):
    phi = float(i)*2*pi/float(N)
    SinPhi = sin(phi)
    CosPhi= cos(phi)
    correction = outOfPlane*(1-SinTheta*SinTheta*CosPhi*CosPhi)
    correction += (1-outOfPlane)*(1-SinTheta*SinTheta*SinPhi*SinPhi)
    Ic[i] = Ic[i] / correction
  return Ic

def aveNo0(ar):
	"""
	return the average of the non-zero terms in list a

	Parameters
	-----------
	
	ar : list
	    list which we will calculate the average over
	
	
	Returns
	--------
	ave : float
	    average of non-zero terms
	"""

	ave = 0
	num = 0
	for i in ar:
		if i > 0:
			ave += i
			num += 1
	if num > 0:
		ave = ave / float(num)
		return ave
	else:
		return ave

def stdNo0(ar,arave):
	"""
	return the standard deviation of the non-zero terms in list ar, which has average
	(average over non-zeros terms) arave.

	Parameters
	-----------
	
	ar : list
	    list which we will calculate the standard deviation of
	
	arave : float
	    average of non-zero terms
	
	Returns
	--------

	stdave : float
	    standard deviation of non-zero terms
	"""
	stdave = 0
	num = 0
	for i in ar:
		if i > 0:
			val = i - arave
			stdave += sqrt( val*val )
			num += 1
	if num > 0:
		stdave = stdave / float(num)
		return stdave
	else:
		return stdave

def find_thresh(ar,n1,n2):
	""" 
	finds thresholds for a set of array elements
	
	Paramters:
	-----------

	ar : list
	    a list of array elements (usually floats) to be trimmed

	n1 : float 
	    the lower threshold will be this many
	    standard deviations BELOW the mean

	n2 : float 
	    the upper threshold will be this many
	    standard deviations ABOVE the mean

	Returns:
	-----------
	
	lower threshold, upper threshold : float, float
	    These are passed to the function trim.
	    Only array elements bounded by these values will be kept.
	
	"""

	ave = aveNo0(ar)
	std = stdNo0(ar,ave)
	lower_threshold = ave - n1*std
	higher_threshold = ave + n2*std
	return lower_threshold,higher_threshold	

def trim(ar,stdl,stdh):
  """
  Trims the vector ar according to lower and higher thresholds

  Parameters
  ----------

  ar : list or 1D numpy array
      list to be trimmed (usually float values)

  stdl : float
      lower bound
    
  stdh : float
      upper bound

  Returns
  -------
  
  copy of original array but trimmed according to higher/lower
  
  """
  lower,higher = find_thresh(ar,stdl,stdh)

  arc = np.copy(ar)
  for i in xrange(len(arc)):
    if arc[i] < lower or arc[i] > higher:
      arc[i] = 0
  return arc

def trim_in_place(ar,stdl,stdh):
  """
  Trims the vector ar according to lower and higher thresholds

  Parameters
  ----------

  ar : list or 1D numpy array
      list to be trimmed (usually float values)

  stdl : float
      lower bound
    
  stdh : float
      upper bound

  Returns
  -------
  
  copy of original array but trimmed according to higher/lower
  
  """
  lower,higher = find_thresh(ar,stdl,stdh)
  for i in xrange(len(ar)):
    if ar[i] < lower or ar[i] > higher:
      ar[i] = 0

def moving_trim(ar,bins_per_seg,stdl,stdh):
  arc = np.copy(ar)
  for i in xrange(len(ar)):
    trim_in_place( arc[i*bins_per_seg:(i+1)*bins_per_seg:1], stdl, stdh  )
  return arc

def getFit(ar,deg):
	"""
	Fits a polynomial of degree deg to non-zeros elements of ar

	Parameters
	----------
	
	ar : list
	    list of floats that a polynomial wil be fit to
	
	deg : int
	    degree of the fitted polynomial

	Returns
	-------
	
	fit : list
	    These are the fitted values on the same domain as ar


	"""
        arx,ary = [],[]
        for i in ar:
		if i > 0:
			arx.append( ar.index(i)  )
			ary.append( i )
        coef = np.polynomial.polynomial.polyfit(arx,ary,deg)
        
	fit = []
	for x in xrange( len(ar) ) :
                poly = 0
		for i in xrange(deg+1):
                        poly += coef[i]*pow(x,i)
                fit.append(poly)
        return fit



def randPairs(numItems,numPairs):
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


def Cl (c,Lvals = range(50)):
	cor_l = zeros((len(Lvals),2))
	N     = len(c[:,0])
	for l in xrange(len(Lvals)):
		Pl     = legendre(Lvals[l])
		cl_sum = 0
		for i in xrange(N):
			cl_sum += c[i,1] * Pl( c[i,0] )
		cor_l[l,1] = 2*pi*(2.*Lvals[l] + 1) * cl_sum	
		cor_l[l,0] = Lvals[l]
	return cor_l

def histPsi(c,q1,q2=0,wave=0.7293,Npsi=100): #cpsi,n_cpsi,t1,t2):
	if q2==0:
		q2 = q1
	t1     = pi/2 + asin( q1*wave  / 4./pi)
	t2     = pi/2 + asin( q2*wave  / 4./pi)
	
	dpsi   = [-1  + 2. / Npsi / 2. + i*2./(Npsi)  for i in xrange(Npsi)]

	cpsi   = zeros(Npsi)
	n_cpsi = zeros(Npsi)
	N      = len(c)
	for i in xrange(N):
		cosPsi =  cos(t1)*cos(t2) + sin(t1)*sin(t2)*\
			  cos(i*2.*pi/float(N)) 
#		convert cosPsi to an integer between 0 and NPsi
		cosPsi1 = int( ( 1 + cosPsi) *float(Npsi) / 2. )
		if cosPsi1 == Npsi:
			continue
		cosPsi2 = int( ( 1 - cosPsi) *float(Npsi) / 2. )
		cpsi[cosPsi1]   += c[i]
		n_cpsi[cosPsi1] += 1.
		cpsi[cosPsi2]   += c[i]
		n_cpsi[cosPsi2] += 1.
	cpsi              = cpsi / n_cpsi
	cpsi[isnan(cpsi)] = 0
	return transpose( array( (dpsi,cpsi) ) )

def correlate_ring_brute(I1,I2): 
    	"""
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta. This is a brute-force method and requires order N**2 iterations.
    
        Parameters
        ----------
        I1 : numpy.ndarray
            ring1
    
        q2 : float or numpy.ndarray
            ring2 
        Returns
        -------
        cor : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        
        """
       
	'''
	if type(I1) != type(I2):
    	    print "Arguments must both be an instance of the same type..."
    	    print "Exiting function..."
    	    return 0
 
    	if not isinstance(I1,np.ndarray):
    	    print "The arguments are not of type 'numpy.ndarray'."
    	    print "Exiting function..."
    	    return 0
	
	n_phi = I1.shape
	if n_phi != I2.shape:
    	    print "Both rings must have the same number of pixels."
    	    print "Exiting function..."
	'''	
	n_phi    = I1.shape[0]

    	#I1mean   = I1.mean()
    	#I2mean   = I2.mean()
	
        I1mean = I1[I1>0].mean()#    = aveNo0(I1)
        I2mean = I2[I2>0].mean() #   = aveNo0(I2)

        I1std    = (I1-I1mean).std() # might use as norm factors in future
        I2std    = (I2-I2mean).std()
	
    	#norm     = n_phi*I1std*I2std
    	norm     = n_phi*I1mean*I2mean

        cor      = zeros((n_phi, 2))
        cor[:,0] = range(n_phi)

    	for phi in xrange(n_phi):
    	    for i in xrange(n_phi):
    		j=i+phi
    		if j>= n_phi: 
    		    j=j-n_phi
    		cor[phi,1]+= (I1[i]-I1mean)*(I2[j]-I2mean)/norm
        
    	return cor

def correlate_ring_elite(I1,I2): 
    	"""
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta. This is a elite method and requires order N*logN iterations.

	Use this method on simulation data only!
    
        Parameters
        ----------
        I1 : numpy.ndarray
            ring1
    
        q2 : float or numpy.ndarray
            ring2 
        Returns
        -------
        cor : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        
        """
       
	'''
	if type(I1) != type(I2):
    	    print "Arguments must both be an instance of the same type..."
    	    print "Exiting function..."
    	    return 0
 
    	if not isinstance(I1,np.ndarray):
    	    print "The arguments are not of type 'numpy.ndarray'."
    	    print "Exiting function..."
    	    return 0
	
	n_phi = I1.shape
	if n_phi != I2.shape:
    	    print "Both rings must have the same number of pixels."
    	    print "Exiting function..."
	'''	
	n_phi    = I1.shape[0]

    	I1mean   = I1.mean()
    	I2mean   = I2.mean()
	
        I1std    = (I1-I1mean).std() # might use as norm factors in future
        I2std    = (I2-I2mean).std()
	
    	#norm     = n_phi*I1std*I2std
    	norm     = n_phi*I1mean*I2mean

        cor      = zeros((n_phi, 2))
        cor[:,0] = range(n_phi)
	
	ff1      = fftpack.fft(I1-I1mean)
	ff2      = fftpack.fft(I2-I2mean)
        cor[:,1] = real(fftpack.ifft( conjugate(ff1) * ff2 )) / norm
	
    	return cor

def correlate_ring_elite2(I1,I2): 
    	"""
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta. This is a elite method and requires order N*logN iterations.

	Use this method on simulation data only!
    
        Parameters
        ----------
        I1 : numpy.ndarray
            ring1
    
        q2 : float or numpy.ndarray
            ring2 
        Returns
        -------
        cor : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        
        """
       
	'''
	if type(I1) != type(I2):
    	    print "Arguments must both be an instance of the same type..."
    	    print "Exiting function..."
    	    return 0
 
    	if not isinstance(I1,np.ndarray):
    	    print "The arguments are not of type 'numpy.ndarray'."
    	    print "Exiting function..."
    	    return 0
	
	n_phi = I1.shape
	if n_phi != I2.shape:
    	    print "Both rings must have the same number of pixels."
    	    print "Exiting function..."
	'''	
	n_phi    = I1.shape[0]

    	I1mean   = I1.mean()
    	I2mean   = I2.mean()
	
        I1std    = (I1-I1mean).std() # might use as norm factors in future
        I2std    = (I2-I2mean).std()
	
    	norm     = n_phi*I1std*I2std
#    	norm     = n_phi*I1mean*I2mean

        cor      = zeros((n_phi, 2))
        cor[:,0] = range(n_phi)
	
	ff1      = fftpack.fft(I1)
	ff2      = fftpack.fft(I2)
        cor[:,1] = real(fftpack.ifft( conjugate(ff1) * ff2 ))# / norm

	return cor
	
def correlate_ring_elite3(I1,I2,I1mean,I2mean): 
    	"""
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta. This is a elite method and requires order N*logN iterations.

	Use this method on simulation data only!
    
        Parameters
        ----------
        I1 : numpy.ndarray
            ring1
    
        q2 : float or numpy.ndarray
            ring2 
        Returns
        -------
        cor : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        
        """
       
	'''
	if type(I1) != type(I2):
    	    print "Arguments must both be an instance of the same type..."
    	    print "Exiting function..."
    	    return 0
 
    	if not isinstance(I1,np.ndarray):
    	    print "The arguments are not of type 'numpy.ndarray'."
    	    print "Exiting function..."
    	    return 0
	
	n_phi = I1.shape
	if n_phi != I2.shape:
    	    print "Both rings must have the same number of pixels."
    	    print "Exiting function..."
	'''	
	n_phi    = I1.shape[0]

    	#I1mean   = I1.mean()
    	#I2mean   = I2.mean()
	
        #I1std    = (I1-I1mean).std() # might use as norm factors in future
        #I2std    = (I2-I2mean).std()
	
    	#norm     = n_phi*I1std*I2std
#    	norm     = n_phi*I1mean*I2mean

        cor      = zeros((n_phi, 2))
        cor[:,0] = range(n_phi)
	
	ff1      = fftpack.fft(I1-I1mean)
	ff2      = fftpack.fft(I2-I2mean)
        cor[:,1] = real(fftpack.ifft( conjugate(ff1) * ff2 ))# / norm

	return cor

def correlate_ring_brute3(I1,I2,I1mean,I2mean): 
    	"""
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta. This is a elite method and requires order N*logN iterations.

	Use this method on simulation data only!
    
        Parameters
        ----------
        I1 : numpy.ndarray
            ring1
    
        q2 : float or numpy.ndarray
            ring2 
        Returns
        -------
        cor : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        
        """
       
	'''
	if type(I1) != type(I2):
    	    print "Arguments must both be an instance of the same type..."
    	    print "Exiting function..."
    	    return 0
 
    	if not isinstance(I1,np.ndarray):
    	    print "The arguments are not of type 'numpy.ndarray'."
    	    print "Exiting function..."
    	    return 0
	
	n_phi = I1.shape
	if n_phi != I2.shape:
    	    print "Both rings must have the same number of pixels."
    	    print "Exiting function..."
	'''	
	n_phi    = I1.shape[0]

    	#I1mean   = I1.mean()
    	#I2mean   = I2.mean()
	
        #I1std    = (I1-I1mean).std() # might use as norm factors in future
        #I2std    = (I2-I2mean).std()
	
    	#norm     = n_phi*I1std*I2std
#    	norm     = n_phi*I1mean*I2mean

        cor      = zeros((n_phi, 2))
        cor[:,0] = range(n_phi)
    	
        for phi in xrange(n_phi):
    	    for i in xrange(n_phi):
    		j=i+phi
    		if j>= n_phi: 
    		    j=j-n_phi
    		cor[phi,1]+= (I1[i]-I1mean[i])*(I2[j]-I2mean[j])/norm
	
	return cor
	
	
def intra_SM(ringsA,ringsB,num_cors=0):
  if num_cors == 0:
    num_cors = ringsA.shape[1]
  intra = zeros( (ringsA.shape[0],2) )
  for i in xrange(num_cors):
    intra += correlate_ring_elite2(ringsA[:,i],ringsB[:,i])
  return intra[:,1]

def intra(ringsA,ringsB,num_cors=0):
  if num_cors == 0:
    num_cors = ringsA.shape[1]
  intra = zeros( (ringsA.shape[0],2) )
  aveA = average(ringsA,axis=1)
  aveB = average(ringsB,axis=1)
  for i in xrange(num_cors):
    intra += correlate_ring_brute(ringsA[:,i],ringsB[:,i],aveA,aveB)
  return intra[:,1]

def inter(ringsA,ringsB,num_cors = 0):
  if num_cors == 0:
    num_cors = ringsA.shape[1]
  inter = zeros( (ringsA.shape[0],2) )
  pairs = stats.randPairs(ringsA.shape[1],num_cors)
  aveA = average(ringsA,axis=1)
  aveB = average(ringsB,axis=1)
  for i,j in pairs:
    inter += correlate_ring_elite3(ringsA[:,i],ringsB[:,j],aveA,aveB)
  return inter[:,1]

def compare_delta(cor1,name1,cor2,name2,title='Delta correlators'):
  if title != 'Delta correlators':
    title = 'Delta correlators;\n' + title
  plt.xlabel("Delta (0-2PI)",fontsize=24)
  plt.ylabel("C(q,q,Delta)",fontsize=24)
  plt.suptitle(title,fontsize=24)
  plt.plot(cor1,linewidth=2,label=name1)
  plt.plot(cor2,linewidth=2,label=name2)
  plt.legend()
  plt.show()

def compare_kam(cor1,name1,cor2,name2,title='Kam correlators'):
  if title != 'Kam correlators':
    title = 'Kam correlators;\n' + title
  plt.xlabel("cos[Psi]",fontsize=24)
  plt.ylabel("C(q,q,cos[Psi])",fontsize=24)
  plt.suptitle(title,fontsize=24)
  plt.plot(cor1[:,0],cor1[:,1],linewidth=2,label=name1)
  plt.plot(cor2[:,0],cor2[:,1],linewidth=2,label=name2)
  plt.legend()
  plt.show()

def compare_legendre(leg1,name1,leg2,name2,\
    title='Legendre projections of Kam correlators'):
  if title != 'Legendre projections of Kam correlators':
    title = 'Legendre projections of Kam correlators;\n' + title
  plt.xlabel("Legendre coefficient",fontsize=24)
  plt.ylabel("magnitude of coefficient",fontsize=24)
  plt.suptitle(title,fontsize=24)
  plt.plot(leg1[:,0],leg1[:,1],'co',label=name1)
  plt.plot(leg2[:,0],leg2[:,1],'rd',label=name2)
  plt.plot(leg1[:,0],leg1[:,1],'c')
  plt.plot(leg2[:,0],leg2[:,1],'r')
  plt.legend()
  plt.show()

'''
def plot_I_profile(ring_file):
  f    = h5py.File(ring_file,'r')  
  r    = f['rings']
  keys = r.keys()
  qres = r[keys.pop(0)][0]

  qvals = []
  aves  = []

  for key in keys:
    q = float(key.strip().split('_')[-1])*qres
    dat = np.array(r[key])
    ave = np.average( np.sum( dat,axis=0) )
    qvals.append(q)
    aves.append(ave)

  f.close()

  plt.plot(qvals,aves,linewidth=2)
  plt.xlabel(r'$q \,\AA^{-1}$',fontsize=24)
  plt.ylabel('average intensity',fontsize=24)
  plt.suptitle('Scattering Profile for '+ring_file,fontsize=24)
  plt.show()
'''
  
