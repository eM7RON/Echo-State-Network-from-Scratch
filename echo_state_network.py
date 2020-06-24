import numpy as np


class EchoStateNetwork:   
    '''
    
    This is a simple implementation of an echo state network that does not include
    feedback. Only the output neurons 'Wout' are trained using a
    batch, 'offline' style by solving a generalized least-square problem.
    
    Args:
        X         - array-like, input-data, shape [-1, 1] or [1, ]
        n_in      - int, the number features
        n_neuron  - int, the size of the reservoir
        sparsity  - float:the fraction of neurons in the reservoir to empty space
                    a value of 0.2 will set a random uniform distribution of 80%
                    to zero.                    
        specR     - float: the spectral radius that will be imposed on the reservoir
        leak_rate - float, between 0 and 1 the leak rate of the neurons
        
    Attributes:
        Win      - 2D ndarray: the input neurons: shape(Nr, 2)
        Wr       - 2D ndarray: the reservoir: shape(Nr, Nr)
        Wout     - 2D ndarray: the output neurons: shape(2+ Nr,)
        X        - 2D ndarray: with the harvested network states as rows: shape(len(train_data, 2+Nr))
        x        - 0D ndarray: previous state of network
        targets  - 0D ndarray: the forecast targets
        Y        - 0D ndarray: forecasts
        
    '''
    
    def __init__(self, n_in=10, n_neuron=500, sparsity=0.3, specR=0.7, leak_rate=1, random_state=117):
        
        np.random.seed(random_state) # seed random
        
        #self.a = leak_rate
        self.n_in = n_in
        self.n_neuron = n_neuron # number of neurons in the reservoir
        self.Win = np.random.uniform(-1, 1, (n_neuron, 1 + n_in)) # Input neurons
        Wr = np.random.uniform(-1, 1, (n_neuron, n_neuron)) # Reservoir
        Wr[np.random.rand(n_neuron, n_neuron) > sparsity] = 0 # Sets random array elements to zero to make array sparse
        self.Wr = Wr * specR / max(abs(np.linalg.eigvals(Wr))) # Imposes desired spectral radius
           
    def fit(self, X, y):
        
        XK = np.zeros([X.shape[0], 1 + self.n_in + self.n_neuron]) # Design matrix - collects network states
        self.x = np.zeros(self.n_neuron) # Initial state of the network
        
        # The below loop will harvest states of the network

        for i in range(X.shape[0]):
            u = np.append(1, X[i])
            self.x = self.tanh(self.Wr.dot(self.x) + self.Win.dot(u))
            XK[i] = np.append(u, self.x) # Bias and u also collected in X with synapse x
            
        self.Wout = np.linalg.pinv(XK).dot(y) # Train output neurons with linear regression.
                                              # As Wout is a 0-dimensional numpy array it does not require
                                              # transposing as this will be handled by broadcasting.     
        return
            
    def predict(self, X):
        
        # The below loop will make predictions of targets using test_data

        self.Y = np.zeros([X.shape[0]])
        for i in range(X.shape[0]):
            u = np.append(1, X[i])
            self.x = self.tanh(self.Wr.dot(self.x) + self.Win.dot(u))
            y = np.sign(self.Wout.dot(np.append(u, self.x))) # forecast
            self.Y[i] = y
        return self.Y
    
    def tanh(self, x):
        '''
        Calculates the hyperbolic tangent of x
        '''
        return ( np.exp(2 * x) - 1 ) / ( np.exp(2 * x) + 1 )
    
    def sigmoid(self, x):
        '''
        Calculates the sigmoid of x
        '''
        return 1 / (1 + np.exp(-x))