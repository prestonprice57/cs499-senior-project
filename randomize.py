import h5py
import numpy as np
# f = h5py.File("/Users/prestonprice/Documents/cs499/kaggle/train/ALB.hdf5", "r")
# data = (f['dataset'][0]*255)+(255/2)


# print data[0]
# img = Image.fromarray(data[0], 'RGB')
# img.save('my.png')
# img.show()
# data[1] = data[1]
# print data[1][0,880:990]
# from matplotlib import pyplot as plt
# plt.imshow(data[1])
# plt.show()

train_folders = ['/Users/prestonprice/Documents/cs499/kaggle/train/ALB.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/BET.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/DOL.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/LAG.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/NoF.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/OTHER.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/SHARK.hdf5', '/Users/prestonprice/Documents/cs499/kaggle/train/YFT.hdf5']
test_folders = ['/Users/prestonprice/Documents/cs499/kaggle/test/test.hdf5']

img_scale=0.5
img_width = int(1280*img_scale)
img_height=int(720*img_scale)
num_classes=8

def get_data(folders):
	num_items = 1000
	data = np.zeros((len(folders)*num_items, img_height, img_width, 3))
	labels = np.zeros((len(folders)*num_items, num_classes))
	count = 0

	for (i, file) in enumerate(folders):
		f = h5py.File(file, "r")
		label = file.split('/')[-1].split('.')[0]
		print "getting " + label + " data. i is " + str(i)

		length = len(f['dataset'])
		if length < num_items:
			data[count:count+length] = f['dataset'][:,:,:]
			labels[count:count+length,i] = 1
			count += length
		else:
			data[count:count+num_items] = f['dataset'][:num_items,:,:]
			labels[count:count+num_items,i] = 1
			count += num_items
		f.close()

	data = data[:count,:,:,:]
	labels = labels[:count,:]

	return count, data, labels

def randomize(folders):
	count, data, labels = get_data(folders)
	permutation = np.random.permutation(count)
	print permutation[:20]
	shuffled_data = data[permutation,:,:,:]
	shuffled_labels = labels[permutation,:]
	num_items = len(shuffled_data)

	print('Saving randomized random_data%d' % int(num_items))
	filename = 'random_data' + str(num_items) + '.hdf5'
	f = h5py.File(filename, "w")
	dset = f.create_dataset("dataset", data=shuffled_data)
	f.close()

	print('Saving randomized random_labels%d' % num_items)
	filename = 'random_labels' + str(num_items) + '.hdf5'
	f2 = h5py.File(filename, "w")
	dset = f2.create_dataset("labels", data=shuffled_labels)
	f2.close()

	return shuffled_data , shuffled_labels

shuffled_data, shuffled_labels = randomize(train_folders)
print shuffled_labels[:20]



