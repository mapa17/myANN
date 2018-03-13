import numpy as np
from pudb import set_trace as st
from scipy.special import expit
import matplotlib.pyplot as plt

class myANN(object):
    def __init__(self, nNeurons, learning_rate):
        # Check if we got a list of #Neurons in each layer
        if len(nNeurons) < 3:
            raise ValueError('Network needs at least 3 layers!')

        self.learning_rate = learning_rate

        # Generate a list of weight matrixes, initializing them with random(uniform) 
        self.W = []
        for N1, N2 in zip(nNeurons[:-1], nNeurons[1:]):
            self.W.append(np.random.normal(0.0, pow(N1, -0.5), (N2, N1)))


    def input_process(self, inputs, labels=True):
        """
        Return a list of pairs of input data (input, label)
        If there is no label (i.e query than the label is set to None)
        """
        if not labels:
            labels = [None]*len(inputs)
            inputs = zip(inputs, labels)

        results = []
        for input, label in inputs:
            # All inputs are row vectors
            input = input.flatten()
            input = np.reshape(input, (1, len(input)))
            try:
                label = np.reshape(label, (1, len(label)))
            except TypeError:
                label = None
            results.append((input, label))
    
        return results


    def _backpropagation(self, label, layer_results):
        error = np.transpose(label - layer_results[-1])
        dWs = [None]*len(self.W)
        # Iterate from back to front
        for l in list(range(len(self.W)))[::-1]:
            W = self.W[l]
            output = layer_results[l]

            # Calculate the weight update
            S = 1.0 / (1.0 + np.exp(-np.dot(W, output.T)))

            # Update weights based on the scaled prediction error of this layer
            dW = np.dot(error * S*(1.0-S), output)
            W += self.learning_rate * dW
            dWs[l] = dW

            # Propagate the error making it the output of the previous layer
            error = np.dot(W.T, error)
        
        return dWs
        


    def train(self, inputs):
        dWs = []
        for input, label in inputs:
            prediction = self._inference(input, full_output=True)
            dWs.append(self._backpropagation(label, prediction[1]))
        return dWs
    

    def _inference(self, input, full_output=False):
        S = input
        # Include the 'output' of the input nodes
        outs = [S]
        for W in self.W:
            #S = np.dot(S, W.T)
            # Process row vectors
            S = np.dot(S, W.T)
            #S = scipy.special.expit(S) 
            S = 1.0 / (1.0 + np.exp(-S))
            outs.append(S)
        
        if full_output:
            return S, outs
        else:
            return S

    def query(self, inputs, full_output=False):
        prediction = []
        for input, label in inputs:
            prediction.append(self._inference(input, full_output=full_output))    
        return prediction

def mnist(path):
    """
    Load the specified mnist dataset in its csv format returning a tuple containing (label, np.array(28,28))
    """
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(',')

            # Label is a vector with one element per class
            label = [0.01] * 10
            label[int(data[0])] = 0.99

            # The data are images of 28x28 pixels
            image_array = np.asfarray(data[1:]).reshape((28, 28))
            # Normalize all values between [0.01, 0.99]
            #image_array = ((image_array) / 255.0 * 0.98) + 0.01
            image_array = ((image_array) / 255.0 * 0.98) + 0.01

            #plt.imshow(image_array, cmap='Greys', interpolation='None')
            yield (image_array, label)

#mnist_test_data = mnist('data/mnist_test_10.csv')
#t1 = next(mnist_test_data)
NN = myANN([784, 100, 10], learning_rate=0.1)

# Training
print('Loading training data set ...\n')
training_data = mnist('./data/mnist_train.csv')
#training_data = mnist('./data/mnist_test_10.csv')
td_all = [x for x in training_data]
training_data = NN.input_process(td_all)
for idx, td in enumerate(training_data):
    dWs = NN.train([td])
    change = [dW.sum() for dW in dWs[0]]
    print("Training %d dW = %s" % (idx, change))

# Testing
print('Loading test data set ...\n')
data = mnist('./data/mnist_test.csv')
#training_data = mnist('./data/mnist_test_10.csv')
d_all = [x for x in data]
pro_data = NN.input_process(d_all)
confusion_matrix = np.zeros((10,10))
for idx, td in enumerate(pro_data):
    image, label = td
    S = NN._inference(image)
    p = np.argmax(S)
    r = np.argmax(label)
    confusion_matrix[p, r] += 1
    if p == r:
        print("[%d] Success %d == %d" % (idx, p, r))
    else:
        print("[%d] Failed %d != %d" % (idx, p, r))

correct = confusion_matrix.diagonal().sum()
alldata = confusion_matrix.sum()
print(confusion_matrix)
print('\nCorrect classified (%d/%d), Acc %f' % (correct, alldata, correct/alldata))

