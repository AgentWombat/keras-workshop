from tensorflow import keras # Same as "import tensorflow.keras as keras"

# Keras is a high-level API for Tensorflow. It allows for rapid creation of
# neural networks.

# First, we will use Keras to perform linear regression on a "toy" dataset.
# Then, we will use Keras to create a model for regognizing whether an image
# is of a human or a horse.

# The Sequential model in Keras facilitates creation of standard
# feed forward networks (that is, data passes from one layer to the next
# until the model generates an output). We will use the Sequential model in
# this workshop.


# LINEAR REGRESSION
###############################################
# Array with numbers 0 to 100
x = [i for i in range(500)]

# y = 0.5x + 0
y = [0.5*i + 0 for i in x]

# Keras implements the Sequential model as an object
model = keras.models.Sequential()

# Here, we add layers to the model.
# This adds an 'Input' layer to our model with a specific shape.
# '(1,)' means that the model takes one number as an input.
# Generally, a shape of (a,b,c,d,e...) means the model, as input, recieves
# inputs of shape (a,b,c,d,e...). This shape here functions identically to
# both Tensorflow and Numpy arrays. 
model.add(keras.Input(shape = (1,)))

# Next, we will add one 'Dense' layer to the model.
# Each node in a 'Dense' layer takes on a value which is a linear combination
# of the previous layer's outputs.
# The argument '1' specifies how many nodes are in the layer.
# For linear regression, we only want one node.
model.add(keras.layers.Dense(1))

# Having built our model, we must now compile it.
# Compiling the model defines how it trains.
# The two most important things to define are the loss function and the
# optimizer. The loss function determines how the models performance is
# evaluated and the optimzer determines how its parameters are updated.

# For almost any case, default to using the "adam" optimizer.
# For linear regression, use "mean_squared_error" as the loss function.
model.compile(loss = "mean_squared_error", optimizer = 'adam')

# Now, we train our model with our data; we "fit" the model to the dataset.
# epochs determines how many times the dataset is considered for updating
# parameters.

model.fit(x,y, epochs = 200)

print("The model predicted these weight values:",model.weights)
print("And the models prediction for input [4,5,60,100,11,12] is",
	model.predict([4,5,60,100,11,12]))

# See part two for an example of image classification with Keras.