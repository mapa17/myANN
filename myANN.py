import numpy as np
from pudb import set_trace as st
from scipy.special import expit

class myANN(object):
    def __init__(self, nNeurons, learning_rate):
        # Check if we got a list of #Neurons in each layer
        if len(nNeurons) < 3:
            raise ValueError('Network needs at least 3 layers!')

        self.learning_rate = learning_rate

        # Generate a list of wheight matrixes, initializing them with random(uniform) 
        self.W = []
        for N1, N2 in zip(nNeurons[:-1], nNeurons[1:]):
            self.W.append(np.random.uniform(0.01, 1.0 - 0.01, size=(N2, N1)))

    def _input_process(self, inputs, labels=None):
        if labels is None:
            labels = [None]*len(inputs)

        results = []
        for input, label in zip(inputs, labels):
            # All inputs are row vectors
            input = np.reshape(input, (1, len(input)))
            try:
                label = np.reshape(label, (1, len(label)))
            except TypeError:
                label = None
            results.append((input, label))
    
        return results

    #def _calculate_error(truth, prediction):
    #    return (truth-prediction)*(truth-prediction)

    def _backpropagation(self, label, layer_results):
        # Iterate from back to front
        error = label - layer_results[-1] 
        st()
        for W, output in zip(self.W[::-1], layer_results[0:-1][::-1]):

            # Calculate the weight update
            #S = 1.0 / (1.0 + np.exp(-np.dot(output, W.T)))
            S = 1.0 / (1.0 + np.exp(-np.dot(W, output.T)))
            #S = 1.0 / (1.0 + np.exp(-np.dot(output, W)))
            #dW = np.dot(output, -error * S*(1-S))
            #dW = np.dot(output, error * S*(1-S))
            dW = np.dot(error * S*(1-S), output)

            W += self.learning_rate * dW

            # Propagate the error making it the output of the previous layer
            #error = np.dot(error.T, W)
            #error = np.dot(W, error)
            error = np.dot(W.T, error)


    def train(self, inputs, labels):
        for input, label in zip(inputs, labels):
            prediction = self.query([input], full_output=True)
            self._backpropagation(label, prediction[0][1])
    
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
        processed_inputs = self._input_process(inputs)
        prediction = []
        for input, label in processed_inputs:
            prediction.append(self._inference(input, full_output=full_output))    
            
        return prediction

NN = myANN([2, 3, 1], learning_rate=0.1)
NN.W
NN.query([[0.5, 0.5]])
old = NN.W
NN.train([[0.1, 0.9]], [0.1])
NN.W
print('Change W:')
NN.W - old