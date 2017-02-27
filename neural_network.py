import numpy as np
print "hello world"

class BackPropagationNetwork:
	# lmao!

	# class variables

	layer_count = 0
	shape = None
	weights = []

	# class methods

	def __init__(self, layer_size):

		self.layer_count = len(layer_size)
		self.shape = layer_size

		# input and output from previous run
		self._layerInput = []
		self._layerOuput = []

		# create the weight arrays
		for [11,12] in zip(layer_size[ : -1], layer_size[1 : ]):
			self.weight.append(np.random.normal(scale = 0.1, size = (12, 11+1)))

# if run as a script, creating a test object

if __name__ == "__main__":
	bpn = BackPropagationNetwork([2,2,1])
	print bpn.shape