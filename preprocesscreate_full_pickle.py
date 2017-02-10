import numpy as np
import os

folders = ['~/Documents/cs499/kaggle/train/ALB', '~/Documents/cs499/kaggle/train/BET', '~/Documents/cs499/kaggle/train/DOL', '~/Documents/cs499/kaggle/train/LAG', '~/Documents/cs499/kaggle/train/NoF', '~/Documents/cs499/kaggle/train/OTHER', '~/Documents/cs499/kaggle/train/SHARK', '~/Documents/cs499/kaggle/train/YFT']
img_width = 1280
img_height=720

def load_fish(folder):
	image_files = os.listdir(folder)
	data = np.ndarray(shape=(len(image_files), img_width, img_height), dtype=np.float32)

	print(folder)
	num_images = 0 
	for image in image_files:
		image_file = os.path.join(folder, image)
		try:
			image_data = (ndimage.imread(image_file).astype(float) - 255 / 2) / 255

			image_data = image_data[:img_width,:img_height]
			if image_data.shape != (img_width, img_height):
				raise Exception('Unexpected image shape: %s' % str(image_data.shape))
			dataset[num_images, :, :] = image_data
			num_images = num_images + 1
		except:
			print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

load_fish(folders[0])
