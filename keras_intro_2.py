from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
# We will use the cifar10 data. It contains 50,000 images which contain one
# of 10 images, all labeled 0-9. The images are stored in 3D Numpy arrays.
# Two of the dimensions deal with pixels' position, the other dimension deals
# with color (which uses 0-255 RGB).
# We will only use images marked 0 (airplanes) and 1 (cars)

(x_train_old, y_train_old), (x_test_old, y_test_old) = keras.datasets.cifar10.load_data()
# We have both train and test data to better evaluate our model.
# While training a neural network, we only let the model see the training data;
# When evaluating the neural network, we test it using the testing data--the
# data it has never seen.

# For ML (machine learning), it helps to standardize all data between values of
# 0 and 1.
# We do that here:
x_train_old = x_train_old / 255.0
x_test_old = x_test_old / 255.0

# This code here will filter the images so we only have airplanes and cars
####################################
x_train = []
y_train = []
x_test = []
y_test = []
for i, label in enumerate(y_train_old):

	if label == 0 or label == 1:
		x_train.append(x_train_old[i])
		y_train.append(label)

for i, label in enumerate(y_test_old):
	if label == 0 or label == 1:
		x_test.append(x_test_old[i])
		y_test.append(label)


# convert to Numpy arrays for convenience
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
####################################

# This is to view a sample of the images:
for x in x_train[:4]:
	plt.imshow(x)
	plt.show()


# We must "flatten" our data to feed it into a standard densely connected model.
# Note that each image is a three dimensional array of shape (32, 32, 3)
# The entire data, then, is shape (num_images, 32, 32, 3).
# To "flatten" the data, we reshape our Numpy array data to be
# (num_images, 32 * 32 * 3)

# '-1' has Numpy automatically pick the correct dimension to facilitate the
# other arguments
x_train_flat = x_train.reshape(-1, 32*32*3)
x_test_flat = x_test.reshape(-1,32*32*3)

# To build the model, we will again use the 'Sequential' model.
# Thsi model's architecture is a densely connected model with relu activations.
# for the output, we use sigmoid because it insures all model oututs will be
# between 0 and 1.

model = keras.models.Sequential()

model.add(keras.Input(shape = (x_train_flat.shape)))

model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(32, activation = 'relu'))
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
	metrics = ['accuracy'])

model.fit(x_train_flat, y_train, epochs = 10)
model.evaluate(x_test_flat, y_test)


# Here, will will look at some examples of the models performance.
print("--TESTING MODEL--")
for i, label in enumerate(y_test):
	pred = model.predict(x_test_flat[i: i + 1])

	print("This image is actually a", "plane." if label == 0 else "car.")
	print("The model predicted it to be a", "plane." if pred < 0.5 else "car.")
	plt.imshow(x_test[i])
	plt.show()
	print("-#-$-#-$-#-$-#-")

# To save our model so that we could import into another program, we would use
# "model.save('path/to/location')". To then import the model into code, we would
# use "keras.models.load_model('path/to/model')". 