import numpy as np
from scipy import optimize

# input is number of hours studied, on facebook, eating, and sleeping
trainX = np.array(([1, 3, 2, 8],[2, 1, 1,10],[5, 0.5, 1.5,7],[3, 1, 3, 8]), dtype=float)
# output is expected test mark
trainY = np.array(([80],[85],[92],[74]),dtype=float)


testX = np.array(([3,3,4,6],[2,0,1,10],[5,5,1,5],[4.5,2.5,2,7.5]), dtype=float)
testY = np.array(([70],[89],[85],[75]), dtype = float)


testX = testX/np.amax(testX, axis=0)
testY = testY/np.amax(testY, axis=0)

trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100.

class Neural_Network(object):
	def __init__(self, Lambda = 0):
		# we have num hours of workout and num calories
		self.inputLayerSize = 4
		self.outputLayerSize = 1
		self.firstLayerSize = 3
		self.secondLayerSize = 3

		self.W1 = np.random.randn(self.inputLayerSize,self.firstLayerSize)
		self.W2 = np.random.randn(self.firstLayerSize,self.secondLayerSize)
		self.W3 = np.random.randn(self.secondLayerSize,self.outputLayerSize)

		self.Lambda = Lambda

	# the propagation
	def forward(self, x):
		# synapse
		self.z2 = np.dot(x, self.W1)
		# node
		self.a2 = self.sigmoid(self.z2)
		# synapse
		self.z3 = np.dot(self.a2, self.W2)
		# node
		self.a3 = self.sigmoid(self.z3)
		# synapse
		self.z4 = np.dot(self.a3, self.W3)
		# node (last node)
		yHat = self.sigmoid(self.z4)
		return yHat

	def costFunction(self, x, y):
		self.yHat = self.forward(x)
		J = 0.5*sum((y-self.yHat)**2)/x.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2)+np.sum(self.W3**2))
		return J

	def costFunctionPrime(self, x, y):
		self.yHat = self.forward(x)

		delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
		dJdW3 = np.dot(self.a3.T, delta4) + self.Lambda*self.W3

		delta3 = np.dot(delta4, self.W3.T) * self.sigmoidPrime(self.z3)
		dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2

		delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(x.T, delta2) + self.Lambda*self.W1

		return dJdW1, dJdW2, dJdW3

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		# derivative of sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)

	def getParams(self):
        # get W1, W2 and W3 Rolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
		return params
    
	def setParams(self, params):
		#Set W1 and W2 using single parameter vector:
		W1_start = 0
		W1_end = self.firstLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], \
			(self.inputLayerSize, self.firstLayerSize))

		W2_end = W1_end + self.firstLayerSize*self.secondLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], \
			(self.firstLayerSize, self.secondLayerSize))

		W3_end = W2_end + self.secondLayerSize*self.outputLayerSize
		self.W3 = np.reshape(params[W2_end:W3_end], \
			(self.secondLayerSize, self.outputLayerSize))

	def computeGradients(self, x, y):
		dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(x,y)
		return np.concatenate((dJdW1.ravel(),dJdW2.ravel(),dJdW3.ravel()))

class trainer(object):
	def __init__(self, N):
		# local reference to neural network
		self.N = N

	def costFunctionWrapper(self, params, x, y):
		self.N.setParams(params)
		cost = self.N.costFunction(x,y)
		grad = self.N.computeGradients(x,y)
		return cost, grad

	def callBackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.x, self.y))
		self.testJ.append(self.N.costFunction(self.testX, self.testY))

	def train(self, trainX, trainY, testX, testY):
		self.x = trainX
		self.y = trainY

		self.testX = testX
		self.testY = testY

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


study = raw_input('How many hours did you study?: ')
facebook = raw_input('How many hours were you on facebook?: ')
eat = raw_input('How many hours did you spend eating?: ')
sleep = raw_input('How many hours did you sleep?: ')

testInput = np.array(([study,facebook,eat,sleep]),dtype=float)

print "your next test score will be: " + str(nn.forward(testInput))


















