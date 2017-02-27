import numpy as np
import random

class Neuron():
	def __init__(self):
		self.weight = random.uniform(-1, 1)

	def propagate(self,_input):
		_input = _input / 100.0
		print "adjusted input: " + str(_input)
		linear = np.dot(_input, self.weight)
		# node
		print "linear part: " + str(linear)
		non_linear = self.sigmoid(linear)

		return non_linear

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

myNeuron = Neuron()
userin = raw_input("what would you like to propagate? (0 to 100) ")
try:
	print myNeuron.propagate(int(userin))
except:
	print "wrong type lmao."
