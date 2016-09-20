import numpy as np
from scipy import optimize
# basically our data
# hours sleep , hours studied, datatyle = float
trainX = np.array(([3,5], [5,1], [10,2], [6, 1.5]), dtype = float)
trainY = np.array(([75], [82], [93], [70]), dtype = float)

testX = np.array(([5,8], [4.5,1],[9,2.5],[6,2]), dtype = float)
testY = np.array(([70], [89],[85],[75]), dtype=float)

# supervised because we have inputs and output

# scaling so we have same 'units'
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100.

testX = testX/np.amax(testX, axis=0)
testY = testY/100.

# sigmoid activation functions

# new complete class, with changes:
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters
        # we have hours of sleep and hours studied
        self.inputLayerSize = 2
        # we have our final grade prediction
        self.outputLayerSize = 1
        # we decided to have 3 nodes (neurons)
        self.hiddenLayerSize = 3

        '''
        IN =W1> n1, n2, n3 \
        					=> yHat
        IN =W2> n1, n2, n3 /
        '''
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given x,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
	def __init__(self, N):
		# local reference to neural network
		self.N = N

	def costFunctionWrapper(self, params, x, y):
		self.N.setParams(params)
		cost = self.N.costFunction(x , y)
		grad = self.N.computeGradients(x, y)
		return cost, grad

	def callBackF(self,params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.x, self.y))
		self.testJ.append(self.N.costFunction(self.testX, self.testY))

	def train(self, trainX , trainY, testX, testY):
		# make internal variable for callback function:
		self.x = trainX
		self.y = trainY

		self.testX = testX
		self.testY = testY

		# make empyt list to store costs
		self.J = []
		self.testJ = []

		params0 = self.N.getParams()

		options = {'maxiter':200, 'disp': True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, \
			jac = True, method='BFGS', args=(trainX,trainY), \
			options = options, callback=self.callBackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res


nn = Neural_Network()
T = trainer(nn)
T.train(trainX,trainY,testX,testY)

print nn.forward(testX)




'''
Part 1: Data + Architecture
Part 2: Forward Propagation
Part 3: Gradient Descent
Part 4: Backpropagation
Part 5: Numerical Gradient Checking
Part 6: Training
Part 7: Overfitting, Testing, and Regularization
'''















