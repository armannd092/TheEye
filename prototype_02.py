import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#make sure tensorflow using GPU!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config= tf.ConfigProto(log_device_placement=True)
learning_rate = 0.1
num_steps = 500
batch_size = 10
display_step = 10

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
datasetToTest = tf.data.Dataset.from_tensor_slices((dtts_f_a, dtts_l_a))

datasetToTrain = datasetToTrain.repeat()
datasetToTrain = datasetToTrain.batch(batch_size)
datasetToTrain = datasetToTrain.prefetch(batch_size)

datasetToTest = datasetToTest.repeat()
datasetToTest = datasetToTest.batch(batch_size)
datasetToTest = datasetToTest.prefetch(batch_size)

iterator = datasetToTrain.make_one_shot_iterator()
it_int = iterator.make_initializer(datasetToTrain)
val_iterator = datasetToTest.make_initializable_iterator()
#print(dataToTrain_lable.shape)
#make the neural network model
graph1 = tf.Graph()
num_input = 2
num_output = 1
architecture = [num_input,3,num_output]


def weights(architecture):
    weight = []
    #architecture.remove(architecture[0])
    for i in range(len(architecture)-1):
        wl = tf.Variable(tf.random_normal([architecture[i], architecture[i+1]]))
        weight.append(wl)
    return weight
def baieses(architecture):
    baies = []
    for i in range(1,len(architecture)):
        bl = tf.Variable(tf.random_normal([architecture[i]]))
        baies.append(bl)
    return baies
def neuralNet(x,architecture,weights,baies):
    global layer
    layer = tf.add(tf.matmul(x, weights[0]), baies[0])
    for i in range(1,len(architecture)-1):

        layer = tf.add(tf.matmul(layer, weights[i]), baies[i])
    return layer

with graph1.as_default():

    X = tf.placeholder("float32", [None, num_input],name='X')
    Y = tf.placeholder("float32", [None, num_output],name='Y')
    logistic = neuralNet(X, architecture, weights(architecture), baieses(architecture))
    prediction = tf.nn.sigmoid(logistic)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logistic, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainOp = optimizer.minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

with tf.Session(graph=graph1,config=config) as sess:
    sess.run(init)
    sess.run(it_int)
    batch_x, batch_y = iterator.get_next()
    for step in range(1, num_steps + 1):
        sess.run(trainOp,{'X:0':batch_x,'Y:0':batch_y})

        if step % display_step == 0 or step == 1:

            loss, acc = sess.run([loss_op, accuracy],{'X:0':batch_x,'Y:0':batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    sess.close()