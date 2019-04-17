import os
import tensorflow as tf

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#make sure tensorflow using GPU!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 50
learning_rate = 0.1

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


datasetToTrain = tf.data.Dataset.from_tensor_slices((dttn_f_a, dttn_l_a))
datasetToTrain = datasetToTrain.repeat().datasetToTrain.batch(batch_size).datasetToTrain.prefetch(batch_size)

iterator = tf.data.make_one_shot_iterator(datasetToTrain)


w = tf.Variable(tf.zeros([None,2]),name= 'weights')
b = tf.Variable(tf.zeros([2]),name='bias')

def logistic_regression(inputs):
    return tf.matmul(inputs, w) + bgit

def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))

def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = optimizer.minimize(loss=loss_fn(logistic_regression()))