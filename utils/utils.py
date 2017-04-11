import os
import numpy as np
import shutil

def split_train_val(total_dir, 
					train_dir,
					val_dir,
					split=0.8,
					method="COPY",
					seed=None):
	if seed is not None:
		np.random.seed(seed)
	nbr_train_samples = 0
	nbr_val_samples = 0

	is_copy = True
	if method == "MOVE":
		is_copy = False
	elif method == "COPY":
		is_copy = True
	else:
		raise ValueError()

	for subdir in os.list_dir(total_dir):
		total_subdir = os.path.join(total_dir, subdir)
		train_subdir = os.path.join(train_dir, subdir)
		val_subdir = os.path.join(val_dir, subdir)
		if not os.path.exists(train_subdir):
			os.mkdir(train_subdir)
		if not os.path.exists(val_subdir):
			os.mkdir(val_subdir)

		total_images = os.listdir(train_subdir)

		nbr_train = int(len(total_images) * split)
		np.random.shuffle(total_images)

		train_images = total_images[:nbr_train]
		val_images = total_images[nbr_train:]

		for img in train_images:
			source = os.path.join(total_subdir, img)
			target = os.path.join(train_subdir, img)
			if method = 1:
				shutil.copy(source, target)
			else:
				shuffle.move(source, target)
			nbr_train_samples += 1

		for img in val_images:
			source = os.path.join(total_subdir, img)
			target = os.path.join(val_subdir, img)
			if is_copy:
				shutil.copy(source, target)
			else:
				shuffle.move(source, target)
			nbr_val_samples += 1

	print('Finish splitting train and val images!')
	print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
