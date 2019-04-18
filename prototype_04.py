import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#make sure tensorflow using GPU!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
graph1 = tf.Graph()
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

#print(dttn_l_a.shape)
datasetToTrain = tf.data.Dataset.from_tensor_slices((dttn_f_a, dttn_l_a))
datasetToTrain = datasetToTrain.repeat()
datasetToTrain = datasetToTrain.batch(batch_size)
datasetToTrain = datasetToTrain.prefetch(batch_size)

iterator = tf.data.make_one_shot_iterator(datasetToTrain)




with graph1.as_default():
    X = tf.placeholder(dtype='float32',shape=[None,2],name='X')
    Y = tf.placeholder(dtype='float32',shape=[None,1], name='Y')
    w = tf.Variable(tf.zeros([2]), name='weights')
    b = tf.Variable(tf.zeros([1]), name='bias')
    pred = tf.add(tf.multiply(X,w),b)
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * numSmp)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session(graph1) as sess:
        sess.run(init)

        for step in range(num_steps):
            for (x, y) in zip(dttn_f_a,dttn_l_a):
                sess.run(optimizer, feed_dict={'X' : x , 'Y' : y})