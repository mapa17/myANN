import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from pudb import set_trace as st

def mnist(path):
    """
    Load the specified mnist dataset in its csv format returning a tuple containing (label, np.array(28,28))
    """
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(',')

            # Label is a vector with one element per class
            label = [0.0] * 10
            label[int(data[0])] = 1.0 

            # The data are images of 28x28 pixels
            image_array = np.asfarray(data[1:]).reshape((28, 28))
            # Normalize the pictures 
            image_array = image_array / 255.0

            #plt.imshow(image_array, cmap='Greys', interpolation='None')
            yield (image_array, label)

def mnist4keras(path):
    loader = mnist(path)
    data = []
    labels = []
    for d, l in loader:
        data.append(d.flatten())
        labels.append(l)
    return np.asarray(data), np.asarray(labels)

#d, l = mnist4keras('data/mnist_test_10.csv')
x_train, y_train = mnist4keras('data/mnist_train.csv')
x_test, y_test = mnist4keras('data/mnist_test.csv')

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(100, activation='relu', input_dim=28*28))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5,
          batch_size=10)
score = model.evaluate(x_test, y_test, batch_size=10)
print('Test score %f' % score[1])