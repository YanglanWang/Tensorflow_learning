import tensorflow as tf
import numpy as np
# a=np.arange(6).reshape(2,3).astype(float)
# x=tf.placeholder(tf.float32,[2,3])
# y=tf.reduce_mean(x)
# with tf.Session() as sess:
#     print(sess.run(y,feed_dict={x:a}))

# a=tf.constant([0,0,0,0],tf.float32)
# b=tf.constant([1,2,3,4],tf.float32)
# result1=tf.nn.softmax(a)
# result2=tf.nn.softmax(b)
# sess=tf.Session()
# print(sess.run(result1))
# print(sess.run(result2))

w=tf.Variable(1.0)
ema=tf.train.ExponentialMovingAverage(0.9)
update=tf.assign_add(w,1.0)
with tf.control_dependencies([update]):
    ema_op=ema.apply([w])
ema_val=ema.average(w)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run(ema_val))
