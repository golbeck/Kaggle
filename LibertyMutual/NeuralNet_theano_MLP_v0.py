import os
import sys
import time
import datetime

import cPickle

import numpy as np
import pandas as pd
import theano
import theano.tensor as T


####################################################################################
####################################################################################
####################################################################################
def load_data():
    ####################################################################################
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    X_in=np.loadtxt("X_train_pca_v2.gz",delimiter=",")
    n=X_in.shape[0]
    Y_in=np.loadtxt("Y_train.gz",delimiter=",")
    if len(Y_in.shape)==1:
        m=1
    else:
        m=Y_in.shape[1]

    Y_in=Y_in.reshape((n,m))

    #shuffle train set
    rng_state = np.random.get_state()
    #randomly permuate the features and outputs using the same shuffle for each epoch
    np.random.shuffle(X_in)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_in)        

    frac=0.95
    n_train_set=int(frac*n)
    train_set=(X_in[range(n_train_set),:],Y_in[range(n_train_set),:])
    train_set_x, train_set_y = shared_dataset(train_set)

    valid_set=(X_in[range(n_train_set,n),:],Y_in[range(n_train_set,n),:])
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    #normalize to intensities in [0,1]
    test_in=np.loadtxt("X_test_pca_v2.gz",delimiter=",")
    test_set_x=theano.shared(np.asarray(test_in,dtype=theano.config.floatX),
                                 borrow=True)
    rval = [(train_set_x, train_set_y),(valid_set_x, valid_set_y), test_set_x]
    return rval

####################################################################################
####################################################################################
####################################################################################
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    ####################################################################################
    ####################################################################################
    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        if W is None:
            W_values = np.zeros((n_in, n_out),dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.y_pred = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

####################################################################################
####################################################################################
####################################################################################
class HiddenLayer(object):
    ####################################################################################
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),
                dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
####################################################################################
####################################################################################
####################################################################################
class MLP(object):
    """convolutional neural network """
    ####################################################################################
    def __init__(self, rng, input, batch_size, n_in,n_hidden, n_out, activation):


        self.layers=[]
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, n_kerns[0], 12, 12)

        #image inputs
        next_layer_input=input
        count_layer=0
        for n_layer in range(len(n_hidden)):
            count_layer +=1

            #hidden layer without dropout, and weights adjusted by dropout probability
            next_layer=HiddenLayer(
                rng,
                input=next_layer_input,
                n_in=n_in,
                n_out=n_hidden[n_layer],
                activation=activation
            )
            self.layers.append(next_layer)
            next_layer_input=next_layer.output
            #update input dimension
            n_in=n_hidden[n_layer]


        count_layer +=1

        # classify the values of the fully-connected sigmoidal layer
        logRegressionLayer = LogisticRegression(
                input=next_layer_input, 
                n_in=n_in, 
                n_out=n_out,
        )

        self.layers.append(logRegressionLayer)
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = np.array([abs(self.layers[i].W).sum() for i in range(count_layer)]).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = np.array([(self.layers[i].W ** 2).sum() for i in range(count_layer)]).sum()

        # the parameters of the model are the parameters of the two layer it is
        # made out of

        #list of lists [[W,b],[W,b],...]
        self.params = [self.layers[i].params for i in range(count_layer)]
        #single flat list [W,b,W,b,...]
        self.params = [item for sublist in self.params for item in sublist]
        # self.params = sum(self.params,[])

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        self.y_out = self.layers[-1].y_pred

        # loss function uses
        self.loss = lambda y: self.mse(self.y)

    ####################################################################################
    def predict_y(self):
        #predictions for validation and test set
        return self.y_out

    ####################################################################################
    def mse(self, y):
        # error between output and target
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_out', self.y_out.type))

        return T.mean((self.y_out - y) ** 2)

####################################################################################
####################################################################################
####################################################################################
class TrainMLP(object):
    """
        builds a MLP and trains the network
    """
    ####################################################################################
    def __init__(self, n_in=5, rng=None, n_hidden=np.array([50,50]), n_out=1, 
                 learning_rate=0.1, rate_adj=0.40, n_epochs=100, L1_reg=0.00, 
                 L2_reg=0.00, learning_rate_decay=0.40,
                 activation='tanh',final_momentum=0.99, initial_momentum=0.5,
                 momentum_epochs=400.0,batch_size=100,patience_init=70):

        #initialize the inputs (tunable parameters) and activations
        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

        #np array [width,height] pixel dims
        self.n_in = n_in
        #number of neurons in each hidden layer
        self.n_hidden = n_hidden
        #number of classes
        self.n_out = int(n_out)

        self.learning_rate = float(learning_rate)
        self.rate_adj=float(rate_adj)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_epochs = int(momentum_epochs)
        self.batch_size=int(batch_size)
        self.patience_init=patience_init

        #build the network
        self.ready()
    ####################################################################################
    def ready(self):
        #builds the network given the tunable parameters and activation
        # input 
        self.x = T.matrix('x')
        # target 
        self.y = T.matrix('y')

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError


        self.MLP = MLP(rng=self.rng, input=self.x, batch_size=self.batch_size,n_in=self.n_in,
                        n_hidden=self.n_hidden, n_out=self.n_out,activation=activation)

    ####################################################################################
    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.MLP.params:
            param.set_value(i.next().get_value(borrow=True))



    ####################################################################################
    def model_trained(self,path):
        """ load saved parameters from .zip file specified in 'path' and predict model on test set
        """

        #load parameters saved from training
        save_file = open(path)
        weights=cPickle.load(save_file)
        save_file.close()
        #set value of MLP params using trained parameters
        self._set_weights(weights)

        #data used for the predictions
        datasets = load_data()
        test_set_x = datasets[2]
        ind = T.lscalar('ind')   

        #output probabilities
        y_fit = theano.function(
            inputs=[ind],
            outputs=self.MLP.predict_y(),
            givens={
                self.x: test_set_x[0:ind]
                }
        )

        #for test set predictions, update batch size to the full set
        n_test_set=test_set_x.get_value(borrow=True).shape[0]
        y=np.array(y_fit(n_test_set))
        return y

    ####################################################################################
    def fit(self,path,validation_frequency=10):
        """ Fit model
        validation_frequency : int
            in terms of number of epochs
        """

        datasets = load_data()

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x = datasets[2]
        #number of observations
        n_train = train_set_x.get_value(borrow=True).shape[0]
        n_valid = valid_set_x.get_value(borrow=True).shape[0]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        # index to a case
        index = T.lscalar('index')    
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        # momentum
        mom = T.scalar('mom', dtype=theano.config.floatX)  

        #the cost function used for grad descent
        cost = (
            self.MLP.mse(self.y)
            + self.L1_reg * self.MLP.L1
            + self.L2_reg * self.MLP.L2_sqr
        )

        #given training data, compute the error
        compute_train_error = theano.function(
                inputs=[index],
                outputs=self.MLP.mse(self.y),
                givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
            )

        #given training data, compute the error
        # compute_valid_error = theano.function(
        #         inputs=[index],
        #         outputs=self.MLP.mse(self.y),
        #         givens={
        #             self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
        #             self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        #             }
        #     )

         #given training data, compute the error
        compute_valid_error = theano.function(
                inputs=[index],
                outputs=self.MLP.mse(self.y),
                givens={
                    self.x: valid_set_x[0:index],
                    self.y: valid_set_y[0:index]
                    }
            )

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.MLP.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        #apply the updates using the gradients and previous updates
        updates=[]
        for param, gparam in zip(self.MLP.params, gparams):
            #update from last iteration
            weight_update = self.MLP.updates[param]
            #current update with momentum
            upd = mom * weight_update - l_r * gparam
            #update the weight parameters and their grad descent updates
            updates.append((self.MLP.updates[param],upd))
            updates.append((param,param + upd))

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(
                inputs=[index, l_r, mom],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                    self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
                    }
        )
        # compiling a Theano function that predicts the classes for a set of inputs
        predict_model = theano.function(
            inputs=[index],
            outputs=self.MLP.predict_y(),
            givens={
                self.x: test_set_x[0:index]
                }
        )


        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        #initial error is set to 100%
        best_valid_loss = np.inf
        #start clock
        start_time = time.clock()
        epoch = 0

        tol=0.005
        improvement_threshold=1.0
        patience=self.patience_init
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / self.batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / self.batch_size
        #train the network over epochs
        done_looping = False

        while (epoch < self.n_epochs) and (not done_looping):
            epoch = epoch + 1
            for idx in xrange(n_train_batches):
                effective_momentum = self.initial_momentum + (self.final_momentum - self.initial_momentum)*epoch/self.momentum_epochs
                example_cost = train_model(idx, self.learning_rate,effective_momentum)

            if (epoch % validation_frequency == 0):
                # compute loss on validation set
                # valid_losses = [compute_valid_error(i) for i in xrange(n_valid_batches)]
                # this_valid_loss = np.mean(valid_losses)

                this_valid_loss=compute_valid_error(n_valid)

                # if we got the best validation score until now
                if this_valid_loss < best_valid_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_valid_loss < best_valid_loss *
                        improvement_threshold
                    ):
                        #only adjust patience if above the min number of epochs (patience_init)
                        if epoch>=self.patience_init:
                            patience += validation_frequency
                        #save parameters if best performance
                        save_file = open(path, 'wb')  # this will overwrite current contents
                        cPickle.dump(self.MLP.params, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL and it triggers much more efficient storage than np's default
                        save_file.close()

                    best_valid_loss = this_valid_loss
                else:
                    #if no improvement seen, adjust learning rate
                    self.learning_rate*=self.rate_adj

                print(
                    'epoch %i, validation error %f, best validation error %f, learning rate %f, patience %i' %
                    (
                        epoch,
                        this_valid_loss,
                        best_valid_loss,
                        self.learning_rate,
                        patience
                    )
                )

            if epoch>patience-1:
                done_looping = True
                break

            self.learning_rate *= self.learning_rate_decay

        # compute loss on training set
        train_losses = [compute_train_error(i) for i in xrange(n_train_batches)]
        this_train_loss = np.mean(train_losses)
        print(
                'training error %f' %this_train_loss
            )

        end_time = time.clock()
        print ('The code ran for %.2fm' % ((end_time - start_time) / 60.))


        #load parameters for best validation set performance
        save_file = open(path)
        weights=cPickle.load(save_file)
        save_file.close()
        #set value of MLP params 
        self._set_weights(weights)
        #for test set predictions, update batch size to the full set
        n_test_set=test_set_x.get_value(borrow=True).shape[0]
        y_test=np.array(predict_model(n_test_set))
        print y_test.shape
        return y_test
####################################################################################
####################################################################################
####################################################################################
def test_MLP():
    """ Test MLP. """
    n_hidden = np.array([800,400,200])
    n_in = 95
    n_out = 1
    learning_rate=0.1
    rate_adj=0.999
    learning_rate_decay=0.999
    batch_size=50
    final_momentum=0.99
    initial_momentum=0.50
    momentum_epochs=100.0
    n_epochs=1000
    L1_reg=0.0
    L2_reg=0.0
    patience_init=30

    rng = np.random.RandomState(2479)
    np.random.seed(0)

    model = TrainMLP(n_in=n_in, rng=rng, n_hidden=n_hidden, n_out=n_out, learning_rate=learning_rate, 
                 rate_adj=rate_adj, n_epochs=n_epochs, L1_reg=L1_reg, L2_reg=L2_reg, learning_rate_decay=learning_rate_decay,
                 activation='sigmoid',final_momentum=final_momentum, initial_momentum=initial_momentum,
                 momentum_epochs=momentum_epochs,batch_size=batch_size,patience_init=patience_init)

    path='params_MLP.zip'
    temp=model.fit(path=path,validation_frequency=5)
    # temp=model.model_trained(path=path)
    return temp
####################################################################################
####################################################################################
####################################################################################
####################################################################################
if __name__ == "__main__":
    pwd_temp=os.getcwd()
    dir1='/home/sgolbeck/workspace/Kaggle/LibertyMutual'
    # dir1='/home/golbeck/Workspace/Kaggle/LibertyMutual'
    dir1=dir1+'/data' 
    if pwd_temp!=dir1:
        os.chdir(dir1)
    y_test=test_MLP()
    print y_test
    df=pd.DataFrame(y_test)
    df.columns=['Hazard']
    indices=np.loadtxt("X_test_indices.gz",delimiter=",").astype('int32')
    df.insert(loc=0,column='Id',value=indices)
    # np.savetxt("MLP_predictions_Theano.csv.gz", df, delimiter=",")
    df.to_csv("MLP_predictions_Theano_pca.csv",sep=",",index=False)