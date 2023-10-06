import os
import cv2  # for computer vision to deal with loading and processing images
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # for machine learning

# To work on the model uncomment the below part.

# # fetching the mnist dataset.
# mnist = tf.keras.datasets.mnist
# # using the inbuilt function to split the dataset into training and testing.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # normalizing the values from 0-255 to 0-1 of pixels
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # fetching the model
# model = tf.keras.models.Sequential()
# # adding the layers to the model

# # it will convert the grid data into a column of 784 rows so basically
# # will flatten it.
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # it will add the next layer with 128 neurons, relu(rectified linear unit)
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # final layer with softmax, basically a function which gives confidence for each of the 10 possible outputs.
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=4)
# model.save('Digit_recog.model')

model = tf.keras.models.load_model('Digit_recog.model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

image_num = 1
while os.path.isfile(f"test/dig{image_num}.png"):
    try:
        img = cv2.imread(f"test/dig{image_num}.png")[:, :, 0]
        # invert it because by default it is white on black and not black on white.
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        # the output will be a confidence value for each of the digits, so printing the digit with max confidence.
        print(f"This digit is probably: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception:
        print("Error!")
    finally:
        image_num += 1
