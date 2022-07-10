import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt


(x_input, x_output), (test_input, test_output) = mnist.load_data()

#plt.imshow(x_input[0], cmap='gray_r')
#plt.show()

digits, width, height = x_input.shape
# flatten and convert grayscale vals to 0..1
x_input = x_input.reshape(digits, width * height).astype('float32') /  255
test_input = test_input.reshape(test_input.shape[0], width * height).astype('float32') /  255

#x_output = keras.utils.to_categorical(x_output, 10)
x_output = np.array([np.array([
    bool(x & 1 << 4),
    bool(x & 1 << 3),
    bool(x & 1 << 2),
    bool(x & 1)], dtype=np.float32) for x in x_output])
#test_output = keras.utils.to_categorical(test_output, 10)

model = keras.Sequential()
model.add(keras.layers.Dense(units=15, activation='sigmoid', input_shape=(width * height,)))
model.add(keras.layers.Dense(units=10, activation='sigmoid'))
model.add(keras.layers.Dense(units=4, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

def fit(epochs):
    model.fit(x_input, x_output, epochs=epochs, verbose=True, validation_split=0.01)
