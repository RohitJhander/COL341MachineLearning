import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne

def load_dataset(trainfilename,testfilename):
	train_data = np.loadtxt(trainfilename, dtype=np.uint8, delimiter=",", skiprows=1)
	X_train = train_data[:,1:]
	X_train = X_train.reshape(-1,1,28,28)
	y_train = train_data[:,0]
	X_test = np.loadtxt(testfilename, dtype=np.uint8,delimiter=",", skiprows=1)
	X_train, X_val = X_train[:-10000], X_train[-10000:]
	y_train, y_val = y_train[:-10000], y_train[-10000:]
	print X_train.shape
	print y_train.shape
	return  X_train/np.float32(256), y_train, X_val/np.float32(256), y_val, X_test/ np.float32(256)

def build_cnn(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=500):

    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test = load_dataset("./train.csv","./test.csv")
   
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = build_cnn(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
  
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01,momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("Validation accuracy",val_acc / val_batches * 100)

    np.savez('parameters.npz', *lasagne.layers.get_all_param_values(network))

    test_prediction = lasagne.layers.get_output(network,deterministic=True)
    predict_fn = theano.function([input_var],T.argmax(test_prediction,axis=1))
    X_test = X_test.reshape(-1,1,28,28)
    preds = predict_fn(X_test)

    subm = np.empty((len(preds),2))
    subm[:,0] = np.arange(1,len(preds)+1)
    subm[:,1] = preds

    np.savetxt('result.csv',subm,fmt='%d',delimiter= ',',header='ImageId,Label',comments='')

main(400)