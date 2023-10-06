# fetching the mnist dataset.
mnist = tf.keras.datasets.mnist
# using the inbuilt function to split the dataset into training and testing.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalizing the values from 0-255 to 0-1 of pixels
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# fetching the model
model = tf.keras.models.Sequential()
# adding the layers to the model

# it will convert the grid data into a column of 784 rows so basically
# will flatten it.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# it will add the next layer with 128 neurons, relu(rectified linear unit)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# final layer with softmax, basically a function which gives confidence for each of the 10 possible outputs.
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
model.save('Digit_recog.model')
