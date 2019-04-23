import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#make sure tensorflow using GPU!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config= tf.ConfigProto(log_device_placement=True)
learning_rate = 0.1
num_steps = 9000
batch_size = 10
display_step = 10

fp = os.path.join('csv', 'DataSet_02.csv')
data = pd.read_csv(fp, skiprows=[0], usecols=[0, 2, 3, 4], header=None)

def cleaner(data):
    data = pd.DataFrame.drop(data,columns=[0])
    indz = []
    for index,row in data.iterrows():
        check = 0
        for i in row:
            check+=i
        if check == 0:
            indz.append(index)
    data = pd.DataFrame.drop(data,index=indz)
    return data

data = cleaner(data)
dataShape = data.shape
numSmp = round(dataShape[0] * 0.8)
while numSmp % batch_size != 0:
    numSmp+=1

remSmp = dataShape[0] - numSmp
dataToTrain = data.sample(n=numSmp, random_state=35489)
dataToTest = data.drop(dataToTrain.index)

dataToTrain_feature = pd.DataFrame.drop(dataToTrain,columns=[2])
dataToTrain_lable = pd.DataFrame.drop(dataToTrain,columns=[3,4])

dataToTest_feature = pd.DataFrame.drop(dataToTest,columns=[2])
dataToTest_lable = pd.DataFrame.drop(dataToTest,columns=[3,4])

dttn_f_a = np.array(dataToTrain_feature.values,'float32')
dttn_l_a = np.array(dataToTrain_lable.values,'float32')

dtts_f_a = np.array(dataToTest_feature.values,'float32')
dtts_l_a = np.array(dataToTest_lable.values,'float32')

#print(np.shape(dttn_f_a))
dataBatch_x = np.vsplit(dttn_f_a,[batch_size])
dataBatch_y = np.vsplit(dttn_l_a,[batch_size])



#print(dataToTrain_lable.shape)
#make the neural network model
graph1 = tf.Graph()
num_input = 2
num_output = 1
architecture = [num_input,3,num_output]


def weights(architecture):
    weight = []
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
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logistic, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    trainOp = optimizer.minimize(loss_op)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

with tf.Session(graph=graph1,config=config) as sess:
    sess.run(init)
    #sess.run(it_int)
    for batch_x, batch_y in zip(dataBatch_x, dataBatch_y):
        for step in range(1, num_steps + 1):
            sess.run(trainOp,{X:batch_x,Y:batch_y})
            if (step+1) % display_step == 0:
                loss, acc = sess.run([loss_op, accuracy],{X:batch_x,Y:batch_y})
                print("Step " , str(step+1) , ", Minibatch Loss= " ,
                      "{:.4f}".format(loss) , ", Training Accuracy= " ,
                      "{:.3f}".format(acc))
    sess.close()