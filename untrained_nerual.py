import numpy as np
from scipy import optimize

# input is number of hours exercized, and number of calories eaten
x = np.array(([1, 2000],[2,2200],[1.5,1900]), dtype=float)
# output is weight difference from last day
y = np.array(([-1],[0.5],[-1.5]),dtype=float)

x = x/np.amax(x, axis=0)
y = y/100.

class Neural_Network(object):
	def __init__(self, Lambda = 0):
		# we have num hours of workout and num calories
		self.inputLayerSize = 2
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

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

nn = Neural_Network()
print nn.forward(x)

