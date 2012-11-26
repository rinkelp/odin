from odin import xray
from mdtraj import trajectory

lyz = trajectory.load('3LYZ.pdb')
d = xray.Detector.generic()
shot = xray.Shot.simulate(lyz, 1000, d)

intensities = shot.intensities # these are the GPU-generated values

print "Found %d values > 1e6 (abnormally large)" % len(np.where(intensities > 1e6)[0])
print "Found %d values < 0 (``impossible``)" % len(np.where(intensities > 0.0)[0])

print intensities