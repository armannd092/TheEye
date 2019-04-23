import os
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

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


datasetToTrain = tf.data.Dataset.from_tensor_slices((dttn_f_a, dttn_l_a))
datasetToTrain = datasetToTrain.repeat()
datasetToTrain = datasetToTrain.batch(batch_size)
datasetToTrain = datasetToTrain.prefetch(batch_size)

iterator = tf.data.make_one_shot_iterator(datasetToTrain)


w = tf.Variable(tf.zeros([2]),name= 'weights')
b = tf.Variable(tf.zeros([1]),name='bias')

def logistic_regression(inputs):
    return tf.add(tf.multiply(inputs, w) ,b)

def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))

def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#loss = loss_fn(logistic_regression(dttn_f_a),dttn_f_a,dttn_l_a)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(loss_fn)

av_lss = 0.
av_acc = 0.
for step in range(num_steps):
    d = iterator.get_next()
    x_batch = d[0]
    y_batch = d[1]

    batch_loss = loss_fn(logistic_regression,x_batch,y_batch)
    av_lss += batch_loss
    batch_acc = accuracy_fn(logistic_regression,x_batch,y_batch)
    av_acc += batch_acc
    if step == 0:
        # Display the initial cost, before optimizing
        print("Initial loss= {:.9f}".format(av_lss))
    optimizer.apply_gradients(grad(logistic_regression, x_batch, y_batch))

    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            av_lss /= display_step
            av_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(av_lss), " accuracy=",
              "{:.4f}".format(av_acc))
        av_lss = 0.
        av_acc = 0.