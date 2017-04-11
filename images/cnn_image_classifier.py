from __future__ import print_function

import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import ..utils.utils

class CNNImageClassifier:
	def __init__(self):
		# image define
		self.image_width = 299
		self.image_height = 299
		self.image_channel = 3
		# data define
		self.data_dir = "image/"
		self.categories = ["category1", "category2"]
		self.batch_size = 32
		self.test_data_dir=""
		self.output_file=""
		# model define
		self.model = None
		self.learning_rate = 0.0001
		self.nbr_epochs = 25
		self.model_file = "models/weights.h5"
	
	def _define_image_generator(self, data_dir):
		# this is the augmentation configuration we will use
		image_datagen = ImageDataGenerator(
				rescale=1./255,
				shear_range=0.1,
				zoom_range=0.1,
				rotation_range=10.,
				width_shift_range=0.1,
				height_shift_range=0.1,
				horizontal_flip=True)

		image_generator = image_datagen.flow_from_directory(
				data_dir,
				target_size = (self.image_width, self.image_height),
				batch_size = self.batch_size,
				shuffle = True,
				classes = self.categories,
				class_mode = 'categorical')
		nbr_samples = sum([len(os.listdir(os.path.join(data_dir, sub_dir))) \
				for sub_dir in os.listdir(self.data_dir)])
		return (image_generator, nbr_samples)
		
	def _load_data(self):
		utils.split_train_val(self.data_dir, self.train_dir, self.val_dir, self.split)
		train_generator, nbr_train_samples = self._define_image_generator(self.train_dir)
		val_generator, nbr_val_samples = self._define_image_generator(self.val_dir)
		return (train_generator, nbr_train_samples, val_generator, nbr_val_samples)
	
	def _define_model(self):
		from keras.applications.inception_v3 import InceptionV3
		InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
							input_tensor=None, input_shape=input_shape)
		# Note that the preprocessing of InceptionV3 is:
		# (x / 255 - 0.5) x 2
		
		print('Adding Average Pooling Layer and Softmax Output Layer ...')
		output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
		output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
		output = Flatten(name='flatten')(output)
		output = Dense(8, activation='softmax', name='predictions')(output)
		
		self.model = Model(InceptionV3_notop.input, output)
		self.model.summary()
		
	def train(self):
		train_generator, nbr_train_samples, \
		val_generator, nbr_val_samples = self._load_data()
		# compile model
		self._define_model()
		optimizer = SGD(lr = self.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
		self.model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
		
		# autosave best Model
		best_model = ModelCheckpoint(self.model_file, monitor='val_acc', verbose = 1, save_best_only = True)

		self.model.fit_generator(
				train_generator,
				samples_per_epoch = nbr_train_samples,
				nb_epoch = self.nbr_epochs,
				validation_data = val_generator,
				nb_val_samples = nbr_val_samples,
				callbacks = [best_model])
				
		# call best model
		self.model = None
		self.model = load_model(self.model_file)
		loss, acc = self.model.evaluate_generator(
				val_generator,
				val_samples=nbr_val_samples,
				nb_worker=4)
		print("train finish: loss-%f, acc-%f" % (loss, acc))

	def predict(self):
		# test data generator for prediction
		test_datagen = ImageDataGenerator(rescale=1./255)

		test_generator = test_datagen.flow_from_directory(
		        self.test_data_dir,
		        target_size=(self.image_width, self.image_height),
		        batch_size=self.batch_size,
		        shuffle = False, # Important !!!
		        classes = None,
		        class_mode = None)
		nbr_test_samples = len(os.listdir(self.test_data_dir))
		test_image_list = test_generator.filenames

		print('Loading model and weights from training process ...')
		model = load_model(self.model_file)

		print('Begin to predict for testing data ...')
		predictions = model.predict_generator(test_generator, nbr_test_samples)

		np.savetxt(self.output_file, predictions)


		print('Begin to write submission file ..')
		f_submit = open(self.output_file, 'w')
		f_submit.write('image,' + ",".join(self.categories) + "\n")
		for i, image_name in enumerate(test_image_list):
		    pred = ['%.6f' % p for p in predictions[i, :]]
		    if i % 100 == 0:
		        print('{} / {}'.format(i, nbr_test_samples))
		    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

		f_submit.close()

		print('Submission file successfully generated!')