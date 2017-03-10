import tensorflow as tf

x = tf.placeholder("float", [4,2])
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [[0,1],[1,2],[2,3],[3,4]]})
    print (result)




    1 1 1 
    0 1 0 == 2
    0 1 0 

    3x3 = [9,10]
    00100 00000
    01234 56789