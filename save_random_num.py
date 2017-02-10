import numpy as np
import h5py

def randomize():
	# \/ \/ \/ PUT LABELS BACK HERE 
	count = 767
	permutation = np.random.permutation(count)
	print permutation[:20]

	f2 = h5py.File('permutation.hdf5', 'w')
	dset = f2.create_dataset('permutation', data=permutation)
	f2.close()

randomize()