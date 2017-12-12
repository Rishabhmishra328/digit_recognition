import cPickle
import gzip
import numpy as np
from progress.bar import Bar

no_layer = 2
layer_size = [784, 10]
learning_rate = 2.0
biases = np.random.randn(10,1)
weights = np.random.randn(10, 784)

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def train():
    #loadong mnist data
    training_data, validation_data, test_data = load_data_wrapper()
    net_delta_bias = 0
    net_delta_weights = 0
    training_pbar = Bar('Training Progress', max = len(training_data)*50)
    for i in range(50):
        for x,y in training_data:
            training_pbar.next()
            activation = sigmoid(np.dot(weights, x) + biases)
            cost = activation - y
            delta = cost * sigmoid_prime(activation)
            net_delta_bias += delta
            net_delta_weights += np.dot(delta, x.transpose())

    return (net_delta_weights,net_delta_bias, len(training_data))

def evaluate():
    #loading mnist data
    training_data, validation_data, test_data = load_data_wrapper()
    accuracy = .0
    test_pbar = Bar('Test Progress', max = len(test_data))
    for x,y in test_data:
        test_pbar.next()
        activation = sigmoid(np.dot(weights, x) + biases)
        prediction = np.argmax(activation)
        if(prediction ==  y):
            accuracy += 1
    return (accuracy/len(test_data))


        
nw,nb,size = train()

biases_correction = learning_rate*nb/size
for bc,b in zip(biases_correction,biases):
    b = np.asarray(b-bc, dtype = np.float32)

weights_correction = learning_rate*nw/size
for wc,w in zip(weights_correction,weights):
    w = np.asarray(w-wc, dtype = np.float32)

acc = evaluate()
print('\nAccuracy : ' + str(acc*100) + '%')



