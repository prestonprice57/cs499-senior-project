import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import os
import h5py
from PIL import Image

user_name="ec2-user"

# /home/ec2-user
train_folders = ['/Users/prestonprice/Documents/cs499/kaggle/train/BET', '/Users/prestonprice/Documents/cs499/kaggle/train/ALB', '/Users/prestonprice/Documents/cs499/kaggle/train/DOL', '/Users/prestonprice/Documents/cs499/kaggle/train/LAG', '/Users/prestonprice/Documents/cs499/kaggle/train/NoF', '/Users/prestonprice/Documents/cs499/kaggle/train/OTHER', '/Users/prestonprice/Documents/cs499/kaggle/train/SHARK', '/Users/prestonprice/Documents/cs499/kaggle/train/YFT']
test_folders = ['/Users/prestonprice/Documents/cs499/kaggle/test']

img_scale=0.5
img_width = int(1280*img_scale)
img_height=int(720*img_scale)


def load_fish(folder):
	image_files = os.listdir(folder)
	num_data = 50
	data = np.ndarray(shape=(len(image_files), img_height, img_width, 3), dtype=np.float32)

	num_images = 0 
	for image in image_files:
		image_file = os.path.join(folder, image)
		print(image_file)
		try:
			# image_data = (ndimage.imread(image_file).astype(float) - 255 / 2) / 255
			# image_data = ndimage.imread(image_file, mode='RGB').astype(float)
			image_data = ndimage.imread(image_file).astype(float)
			image_data = imresize(image_data, img_scale)

			# POTENTIAL CHANGE (now I am just cropping off the right and bottom)
			image_data = image_data[:img_height,:img_width,:]

			if image_data.shape != (img_height, img_width, 3):
				image_data = imresize(image_data, round(img_height/float(image_data.shape[0]), 3)+.001)
				# POTENTIAL CHANGE (now I am just cropping off the right and bottom)
				image_data = image_data[:img_height,:img_width,:]

				if image_data.shape != (img_height, img_width, 3):
					print('Skipping image of size: %s' % str(image_data.shape))
					continue

			image_data = image_data/255.0

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
# pickle_fish(test_folders)
