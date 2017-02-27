import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([1.], tf.float32)
b = tf.Variable([1.], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)

linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)
# training data
x_train = [1,3,5,7]
y_train = [2,3,4,5]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print ("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
result = (sess.run(linear_model, {x:[1,2,3,4]}))
result2 = sess.run(linear_model, {x:[1,3,5,7]})

for i in result:
	print i
for i in result2:
	print i