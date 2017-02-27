import numpy as np

def word-tokenize

class recurrent_network(self):
	def __init__(self):
		self.x1 = np.random.randn(1)

		self.x2 = np.random.randn(1)
		self.c2 = np.random.randn(1)
		self.h2 = np.random.randn(1)

		self.x3 = np.random.randn(1)
		self.c3 = np.random.randn(1)
		self.h3 = np.random.randn(1)

		self.layersize = 3


	def sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def sigmooid_prime(self, x):
		return (-exp(-x)) / (1 + np.exp(-x)) ** 2

	def forward(self, cin, hin):
		concat_1 = self.x1 + hin
		sig_1 = sigmoid(concat_1)
		t_1 = np.tanh(sig_1) * sig_1

		concat_2 = self.h1 + self.x2
		sig_2 = sigmoid(concat_2)
		t_2 = np.tanh(sig_2) * sig_2

		concat_3 = self.h2 + self.x3
		sig_3 = sigmoid(concat_3)
		t_3 = np.tanh(sig_3) * sig_3

		self.c = (sig * cin) + t
		self.h = sig * np.tanh(t)

		return (self.h1 + self.h2 + self.h3)

	def cost_function(self, cin, hin, trainout):
		harray = forward(cin, hin)
		return = harray - trainout














