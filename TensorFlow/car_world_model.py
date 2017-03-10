import numpy as np
import tensorflow as tf

# Model parameters
weight = tf.Variable([1.], tf.float32)

b = tf.Variable([1.], tf.float32)
# Model input and output
acceleration = tf.placeholder(tf.float32, (1,5))

linear_model = weight * acceleration + b
# linear_model = car_accel * weight + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)
# training data
x_train = [[-127,-62,0,62,127]]

y_train = [43]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10000):
  sess.run(train, {acceleration:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([weight, b, loss], {acceleration:x_train, y:y_train})
print ("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

results = sess.run(linear_model, {acceleration:[[-127,-67,0,67,127]]})
print (results)






