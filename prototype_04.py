
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#make sure tensorflow using GPU!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 50
learning_rate = 0.1
num_steps = 1000
display_step = 100

fp = os.path.join('csv', 'DataSet_01.csv')
data = pd.read_csv(fp, skiprows=[0], usecols=[0, 1, 2, 3], header=None)
dataShape = data.shape
numSmp = round(dataShape[0] * 0.8)
remSmp = dataShape[0] - numSmp
dataToTrain = data.sample(n=numSmp, random_state=35489)
dataToTest = data.drop(dataToTrain.index)

dataToTrain_feature = pd.DataFrame.drop(dataToTrain,columns=[0,1])
dataToTrain_lable = pd.DataFrame.drop(dataToTrain,columns=[0,2,3])

dataToTest_feature = pd.DataFrame.drop(dataToTest,columns=[0,1])
dataToTest_lable = pd.DataFrame.drop(dataToTest,columns=[0,2,3])

dttn_f_a = np.array(dataToTrain_feature.values,'float32')
dttn_l_a = np.array(dataToTrain_lable.values,'float32')

dtts_f_a = np.array(dataToTest_feature.values,'float32')
dtts_l_a = np.array(dataToTest_lable.values,'float32')



X = tf.placeholder(dtype='float32',shape=[2],name='X')
Y = tf.placeholder(dtype='float32',shape=[1], name='Y')
w = tf.Variable(tf.zeros([2]), name='weights')
b = tf.Variable(tf.zeros([1]), name='bias')
pred = tf.add(tf.multiply(X,w),b)
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * numSmp)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(num_steps):
        for (x, y) in zip(dttn_f_a,dttn_l_a):
            sess.run(optimizer,feed_dict={'X:0':x,'Y:0':y})
            if (step + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={'X:0':x,'Y:0':y})
                print("Step:","%04d".format(step + 1), "cost=", "{:.9f}".format(c),
                      "W=", sess.run(w), "b=", sess.run(b))
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={'X:0':x, 'Y:0':y})
    print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')