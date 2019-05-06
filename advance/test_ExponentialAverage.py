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

# #instance1

# w=tf.Variable(1.0)
# ema=tf.train.ExponentialMovingAverage(0.9)
# update=tf.assign_add(w,1.0)
# with tf.control_dependencies([update]):
#     ema_op=ema.apply([w])
# ema_val=ema.average(w)
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for i in range(3):
#         sess.run(ema_op)
#         print(sess.run(ema_val))


#instance2

# v1 = tf.Variable(0, dtype=tf.float32)   # 定义一个变量，初始值为0
# step = tf.Variable(0, trainable=False)  # step为迭代轮数变量，控制衰减率
# ema = tf.train.ExponentialMovingAverage(0.99, step)  # 初始设定衰减率为0.99
# maintain_averages_op = ema.apply([v1])                 # 更新列表中的变量
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()        # 初始化所有变量
# sess.run(init_op)
# print(sess.run([v1, ema.average(v1)]))                # 输出初始化后变量v1的值和v1的滑动平均值
# sess.run(tf.assign(v1, 5))                            # 更新v1的值
# sess.run(maintain_averages_op)                        # 更新v1的滑动平均值
# print(sess.run([v1, ema.average(v1)]))
# sess.run(tf.assign(step, 10000))                      # 更新迭代轮转数step
# sess.run(tf.assign(v1, 10))
# sess.run(maintain_averages_op)
# print(sess.run([v1, ema.average(v1)]))
#                                                       # 再次更新滑动平均值，
# sess.run(maintain_averages_op)
# print(sess.run([v1, ema.average(v1)]))
#                                                       # 更新v1的值为15
# sess.run(tf.assign(v1, 15))
#
# sess.run(maintain_averages_op)
# print(sess.run([v1, ema.average(v1)]))


ema = tf.train.ExponentialMovingAverage(decay=0.5)
ema_apply_op=ema.apply([fc_mean,fc_var])
with tf.control_dependencies([ema_apply_op]):
    tf.identity(fc_mean),tf.identity(fc_var)