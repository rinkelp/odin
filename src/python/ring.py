import h5py
import numpy as np
import pylab as plt

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
  
