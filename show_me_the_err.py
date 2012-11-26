import numpy as np
from odin import xray
from mdtraj import trajectory

lyz = trajectory.load('3LYZ.pdb')
d = xray.Detector.generic()
shot = xray.Shot.simulate(lyz, 512*2, d)

intensities = shot.intensities # these are the GPU-generated values

print "Found %d values > 1e7 (abnormally large)" % len(np.where(intensities > 1e7)[0])
print "%d total intensities" % len(intensities)
print intensities
