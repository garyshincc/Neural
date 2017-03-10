import numpy as np
import tensorflow as tf

# Model parameters
weight1 = tf.Variable([1.], tf.float32)
weight2 = tf.Variable([1.], tf.float32)
b = tf.Variable([1.], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)

linear_model = weight2 * (weight1 * x + b)
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [2.5,3.5,4.5,5.5]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_w1, curr_w2, curr_b, curr_loss  = sess.run([weight1, weight2, b, loss], {x:x_train, y:y_train})
print ("Adam optimizer running y = w2 * (w1 * x + b)")
print ("weight1: %s weight2: %s bias: %s loss: %s"%(curr_w1, curr_w2, curr_b, curr_loss))
results = sess.run(linear_model, {x:[1,2,3,4]})

print ("expected: " + str(2.5) + " " + str(3.5) + " " + str(4.5) + " " + str(5.5))
print (results)