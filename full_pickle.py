import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import os
import h5py

train_folders = ['/Users/prestonprice/Documents/cs499/kaggle/train/BET', '/Users/prestonprice/Documents/cs499/kaggle/train/ALB', '/Users/prestonprice/Documents/cs499/kaggle/train/DOL', '/Users/prestonprice/Documents/cs499/kaggle/train/LAG', '/Users/prestonprice/Documents/cs499/kaggle/train/NoF', '/Users/prestonprice/Documents/cs499/kaggle/train/OTHER', '/Users/prestonprice/Documents/cs499/kaggle/train/SHARK', '/Users/prestonprice/Documents/cs499/kaggle/train/YFT']
test_folders = ['/Users/prestonprice/Documents/cs499/kaggle/test/ALB', '/Users/prestonprice/Documents/cs499/kaggle/test/BET', '/Users/prestonprice/Documents/cs499/kaggle/test/DOL', '/Users/prestonprice/Documents/cs499/kaggle/test/LAG', '/Users/prestonprice/Documents/cs499/kaggle/test/NoF', '/Users/prestonprice/Documents/cs499/kaggle/test/OTHER', '/Users/prestonprice/Documents/cs499/kaggle/test/SHARK', '/Users/prestonprice/Documents/cs499/kaggle/test/YFT']

img_width = 1280
img_height=720

def load_fish(folder):
	image_files = os.listdir(folder)
	num_data = 500
	data = np.ndarray(shape=(num_data, img_height, img_width, 3), dtype=np.float32)

	num_images = 0 
	for image in image_files[:num_data]:
		image_file = os.path.join(folder, image)
		print(image_file)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - 255 / 2) / 255
			# image_data = ndimage.imread(image_file, mode='RGB').astype(float)
			# image_data = ndimage.imread(image_file).astype(float) / 255

			# POTENTIAL CHANGE (now I am just cropping off the right and bottom)
			image_data = image_data[:img_height,:img_width,:]

			if image_data.shape != (img_height, img_width, 3):
				image_data = imresize(image_data, round(img_height/float(image_data.shape[0]), 3)+.001)
				# POTENTIAL CHANGE (now I am just cropping off the right and bottom)
				image_data = image_data[:img_height,:img_width,:]

				if image_data.shape != (img_height, img_width, 3):
					print('Skipping image of size: %s' % str(image_data.shape))
					continue
			data[num_images, :, :] = image_data
			num_images = num_images + 1
		except IOError as e:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

	print('Full dataset tensor:', data.shape)
	return data

def pickle_fish(data_folders, force=False):
	dataset_names = []
	for folder in data_folders:
		print(folder)
		set_filename = folder + '.hdf5'
		dataset_names.append(set_filename)

		if os.path.exists(set_filename) and not force:
			# You may override by setting force=True.
			print('%s already present - Skipping adding data.' % set_filename)
		else:
			print('Pickling %s.' % set_filename)
			dataset = load_fish(folder)

			f = h5py.File(set_filename, "w")
			dset = f.create_dataset("dataset", data=dataset)

			f.close()

	return dataset_names

pickle_fish(train_folders)
# pickle_fish(['/Users/prestonprice/Documents/cs499/kaggle/test'])
