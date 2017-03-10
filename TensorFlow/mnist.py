import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# datatype float 32 bytes
# x isnt a specific value. a placeholder
# will be a list of 28x28 2-d matrix
x = tf.placeholder(tf.float32, [None,784])
# batch size NONE = not actually none, just unspecified.
# weights is a 784 x 10 matrix,
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
synapse = tf.matmul(x, weights)

# y is the predicted probability distribution
# squared sum error is used for 'continuous' regression cost function
# softmax and cross entropy is used for discrete mutually exclusive results cost function
y = tf.nn.softmax(synapse + biases)

# y_ is the true distribution
# again, batch size NONE = not actually none.
y_ = tf.placeholder(tf.float32, [None, 10])

# measurement of how inefficient our prediction is
# more like get_mean lmao, and get_sum
reduction_sum = tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1])
cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
cross_entropy = tf.reduce_mean(cross_entropy_with_logits)

# use the grad descent descent optimizer
#trainer = tf.train.AdamOptimizer(0.001) adam with slower learning rate
trainer = tf.train.GradientDescentOptimizer(0.5)
train_step = trainer.minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


for _ in xrange(1000):
	batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch[0], y_:batch[1]})



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))














