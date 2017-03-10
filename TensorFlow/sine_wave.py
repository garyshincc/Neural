import matplotlib.pyplot as plt
import numpy as np
import time

period = 8000
freq = 2
sample = 8000
amp = 1
x = np.arange(0, sample)
sin = amp * np.sin(2 * np.pi * freq * x / period)

plt.plot(x, sin)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


def mutate(child, period, freq, amp):
	pass

class Sine():

	def __init__(self):
		self.period = 10000
		self.freq = 2
		self.amp = 1
		self.x = np.arange(0, 10000)

	def draw(self):
		wf = (2 * np.pi * self.freq * self.x / self.period) 
		y = self.amp * np.sin(wf)
		plt.plot(self.x, y)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()
		plt.ioff()

	def clear(self):
		plt.clf()

mywave = Sine()
for i in range(3):
	mywave.draw()
	mywave.clear()
	time.sleep(1)
	print i