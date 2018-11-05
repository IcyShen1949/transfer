max_size = 20
seqlen = 100
spix = 2000
data_path = "./data/"
import os
import numpy as np
import random
import sys
import h5py
import ljqpy
import random

from keras import backend as K
from keras.engine.topology import Layer

# class MyLayer(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)
#         print(output_dim)
#
#     def build(self, input_shape):
#         #print(input_shape)
#         input_dim = input_shape[1]
#         output_dim = int(self.output_dim)
#         initial_weight_value = np.random.random((output_dim, output_dim))
#         #t=np.array([0.5])
#         #self.W = K.variable(initial_weight_value)
#         self.wtf = K.variable(initial_weight_value, name = 'wtf')
#         self.trainable_weights = [self.wtf]
#
#     def call(self, x, mask = None):
#         #print(x.shape)
#         #print(x.shape.eval({x :x}))
#         #l = len(x[0])
#         #print(l)
#         #print(x.eval())
#         #print(np.array(x))
#         print(1)
#         input_left = x[:, :, :60]
#         input_right = x[:, :, 60:]
#         #print(input_left.shape)
#         #print(1)
#         #output_sum = []
#         #for k, line in enumerate(input_left):
#             #output_sum.append(self.W * input_left[k] + input_right[k])
#         print(2)
#         return (K.dot(input_left, self.wtf) + input_right)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[1], self.output_dim)

# class MyLayer_1(Layer):
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer_1, self).__init__(**kwargs)
#         print(output_dim)
#
#     def build(self, input_shape):
#         input_dim = input_shape[1]
#         output_dim = int(self.output_dim)
#         initial_weight_value = np.random.random((output_dim, output_dim))
#
#         self.wtf = K.variable(initial_weight_value, name = 'wtf')
#         self.trainable_weights = [self.wtf]
#
#     def call(self, x, mask = None):
#         input_left = x[:, :100]
#         input_right = x[:, 100:]
#         return (K.dot(input_left, self.wtf) + input_right)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.output_dim)


class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
        # print(output_dim)

    def build(self, input_shape):
        output_dim = int(self.output_dim)
        self.wtf= self.add_weight(name = "wtf", shape = (output_dim, output_dim), initializer='uniform', trainable=True)
        self.den = self.add_weight(name="den", shape=(output_dim, output_dim * 2), initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)
        # initial_weight_value = np.random.random((output_dim, output_dim))
        #t=np.array([0.5])
        #self.W = K.variable(initial_weight_value)
        # self.wtf = K.variable(initial_weight_value, name = 'wtf')
        # self.trainable_weights = [self.wtf]
    def call(self, x, mask = None):
        input_left = x[:, :, :60]
        input_right = x[:, :, 60:]
        return K.dot((K.dot(input_left, self.wtf) + input_right), self.den)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

class MyLayer_1(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer_1, self).__init__(**kwargs)

    def build(self, input_shape):
        output_dim = int(self.output_dim)
        self.wtf = self.add_weight(name="wtf", shape=(output_dim, output_dim), initializer='uniform', trainable=True)
        self.den = self.add_weight(name="den", shape=(output_dim, output_dim * 2), initializer='uniform', trainable=True)
        super(MyLayer_1, self).build(input_shape)

    def call(self, x, mask = None):
        input_left = x[:, :100]
        input_right = x[:, 100:]
        return K.dot((K.dot(input_left, self.wtf) + input_right), self.den)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

import keras
from keras.models import Model
from keras.layers import core, Dense, Activation, Dropout, TimeDistributed, LSTM, Embedding, Input, merge

from keras.preprocessing import sequence

from keras.models import Sequential



def preRead():
    global id2ch, ch2id, id2tg, vocabulary_size
    id2ch = ljqpy.LoadList(data_path  + 'id2ch.txt')
    ch2id = {v:k for k,v in enumerate(id2ch)}
    id2tg = []
    vocabulary_size = len(id2ch)
    print(vocabulary_size)

def getX(inputFile):
    hfile = h5py.File(inputFile,'r+')
    X = hfile['data'][:]
    hfile.close()
    X = X[:][:]
    return X

def getY(inputFile):
    hfile = h5py.File(inputFile,'r+')
    YY = hfile['data'][:]
    hfile.close()
    YY = YY[:][:]
    Y = []
    for k, line in enumerate(YY):
        tem = []
        for kk, tag in enumerate(line):
            temTag = [0] * 45
            if tag != 0:
                temTag[tag] = 1
            tem.append(temTag)
            del temTag
        Y.append(tem)
        del tem
    Y = np.array(Y)
    return Y


def getTrainingData():
    global left_X, left_Y, right_X, right_Y
    left_X = getX(data_path  + 'movie_TrainX.h5')
    #for line in left_X:
        #print(line)
    right_X = getX(data_path  + 'tweet_TrainX.h5')
    left_Y = getX(data_path  + 'movie_TrainY.h5')
    right_Y = getX(data_path  + 'tweet_TrainY.h5')

def showAcc(model, testX, testY):
    testZZ=model.predict(testX)
    # with open("result.txt","w",encoding="utf-8") as writer:
    #     for i in range(len(testX)):
    #         for wordIndex in testX[i]:
    #             writer.write("{} ".format(id2ch[wordIndex-1]))
    #         writer.write(str(testY[i])+" "+str(testZZ[i])+"\n")
    #     writer.close()
    #
    #print(testZZ[1])
    testZ = []

    for line in testZZ:
        maxx = 0
        ans = 0
        ll = line[0]
        # '''
        # for k, tag in enumerate(line):
        #     if tag > maxx:
        #        maxx = tag
        #        ans = k
        # '''
        if ll > 0.5: ans = 1
        else: ans = 0
        testZ.append(ans)
    # '''
    # for line in testZZ:
    #     tem = []
    #     for k, tag in enumerate(line):
    #         maxTag = 0
    #         maxScore = 0
    #         for kk, score in enumerate(tag):
    #             if maxScore < score:
    #                 maxScore = score
    #                 maxTag = kk
    #         tem.append(maxTag)
    #     testZ.append(tem)
    # '''
    TP = 0
    PP = 0
    RR = 0
    for i in range(len(testY)):
        PP += 1
        wtf = testY[i][0]
        # '''
        # for k, tag in enumerate(testY[i]):
        #     if tag == 1 and testZ[i] == k:
        # '''
        if testZ[i] == wtf:
            TP += 1
    print('TP/PP', TP, PP)
    prec = TP / PP
    print('P : ', prec)
    return prec
    


preRead()
getTrainingData()
# print(len(left_X))
# print(len(left_Y))
# print(len(right_X))
# print(len(right_Y))

#left_X, right_X, left_Y, right_Y are training data
#left means open domain
#right means specific domain

model = Sequential()


# model.add(Embedding(input_dim = 5, output_dim = 1, input_length = seqlen))
# model.add(LSTM(output_dim = 50, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(LSTM(output_dim = 45, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(TimeDistributed(Dense(5)))
# model.add(Activation('softmax'))
# model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')

#model for baseline1

vector_dim = 60
model.add(Embedding(input_dim = vocabulary_size + 3, output_dim = vector_dim, input_length = seqlen, mask_zero = True, name = 'embed'))
model.add(LSTM(output_dim = 100, return_sequences=False, name = 'lstm_aa'))
model.add(Dropout(0.2))
#model.add(LSTM(output_dim = 45, return_sequences=True, name = 'lstm_bb'))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='Adagrad')
# model.summary()
# print(len(left_X))
# print(int(len(left_X) * 0.1))
vali = 2000
testX = left_X[: spix]
testY = left_Y[: spix]

#the training process


maxA = 0
if os.path.exists('model_m2t.h5'): 
    model.load_weights('model_m2t.h5')
    showAcc(model, testX, testY)
    spix = 500
    vali = 500
    showAcc(model, right_X[:vali], right_Y[:vali])
else:
    for k in range(0, 1):
        print(k)
        #print(left_X[:5])
        #print(left_Y[:5])
        model.fit(left_X[spix : -vali], left_Y[spix : -vali], batch_size = 32, nb_epoch = 1, verbose = 1, validation_data = (left_X[-vali:], left_Y[-vali:]))
        temA = showAcc(model, testX, testY)
        #model.fit(left_X[:spix], _Y[:30000], batch_size = 32, nb_epoch = 1, verbose =1 , validation_data = (right_X[-5000:], right_Y[-5000:]))

        #temA = showAcc(model, right_X[30000: -5000], right_Y[30000: -5000])
        if temA > maxA:
                maxA = temA
                model.save_weights('model.h5', overwrite = True)
#model.fit(left_X[spix:], left_Y[spix:], batch_size = 32, nb_epoch = 1, verbose = 1)
#testX = left_X[: spix]
#testY = left_Y[: spix]
#testZ = model.predict(testX)
#print(testZ.shape)
#showAcc(model, testX, testY)

model.load_weights('model.h5')
freeze_embed = model.get_layer('embed').get_weights()
freeze_lstm_aa = model.get_layer('lstm_aa').get_weights()
#freeze_lstm_bb = model.get_layer('lstm_bb').get_weights()

del model

from keras.models import Model

input_layer = Input(shape = (seqlen,), dtype = 'int32')
left_embed = Embedding(output_dim = vector_dim, input_dim = vocabulary_size + 3, input_length = seqlen, name = 'left_embed')(input_layer)
left_lstm_aa = LSTM(output_dim = 100, return_sequences = False, name = 'left_lstm_aa')(left_embed)
left_drop_aa = Dropout(0.2)(left_lstm_aa)
#left_lstm_bb = LSTM(output_dim = 45, return_sequences = True, name = 'left_lstm_bb')(left_drop_aa)

right_embed = Embedding(output_dim = vector_dim, input_dim = vocabulary_size + 3, input_length = seqlen)(input_layer)

right_concat_embed = merge([left_embed, right_embed], mode = 'concat')
#model.add(LambdaMerge([model0, model1], lambda inputs: p0*inputs[0]+p1*inputs[1]))
#right_merge_embed = merge([left_embed, right_embed], mode = lambda x: p0 * x[0] + x[1], output_shape = lambda x: x[1])
right_merge_embed = MyLayer(output_dim = vector_dim)(right_concat_embed)

right_lstm_aa = LSTM(output_dim = 100, return_sequences = False)(right_merge_embed)
right_drop_aa = Dropout(0.2)(right_lstm_aa)

right_concat_aa = merge([left_lstm_aa, right_drop_aa], mode = 'concat') #??
#right_merge_aa = LambdaMerge([left_lstm_aa, right_drop_aa], lambda inputs: p0 * inputs[0] + inputs[1])
#right_merge_aa = MyLayer(inputs = [left_lstm_aa, right_drop_aa])
right_merge_aa = MyLayer_1(output_dim = 100)(right_concat_aa)

#right_lstm_bb = LSTM(output_dim = 45, return_sequences = True)(right_merge_aa)
#right_drop_bb = Dropout(0.2)(right_lstm_bb)

#right_concat_bb = merge([left_lstm_bb, right_drop_bb], mode = 'concat') #??
#right_merge_bb = LambdaMerge([left_lstm_bb, right_drop_bb], lambda inputs: p0 * inputs[0] + inputs[1])
#right_merge_bb = MyLayer(inputs = [left_lstm_bb, right_drop_bb])
#right_merge_bb = MyLayer(output_dim = 45)(right_concat_bb)

right_dense = Dense(1)(right_merge_aa)
right_activation = Activation('sigmoid')(right_dense)

# , mask_zero = True
#model.summary()

model = Model(input = [input_layer], output = [right_activation])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='Adagrad')



testxx = []
testyy = []
devxx = []
devyy = []
trainxx = []
trainyy = []
print(right_X.shape)
right_X = list(right_X)
right_Y = list(right_Y)
print(len(right_X))
for k, line in enumerate(right_X):
    if k < 500: continue
    xx = random.randint(0,100)
    if xx < 90:
        trainxx.append(line)
        trainyy.append(right_Y[k])
        #testxx.append(line)
        #testyy.append(right_Y[k])
    elif xx > 90:
        devxx.append(line)
        devyy.append(right_Y[k])
    #else:
        #trainxx.append(line)
        #trainyy.append(right_Y[k])

#testxx = np.array(testxx)
#testyy = np.array(testyy)
testxx = right_X[:500]
testyy = right_Y[:500]
testxx = np.array(testxx)
testyy = np.array(testyy)
devxx = np.array(devxx)
devyy = np.array(devyy)
trainxx = np.array(trainxx)
trainyy = np.array(trainyy)
print("testxx.shape is ", testxx.shape)
print("testyy.shape is ", testyy.shape)
print("devxx.shape", devxx.shape)
print("devyy.shape", devyy.shape)
print("trainxx.shape", trainxx.shape)
print(trainyy.shape)
f = h5py.File('testxx.h5', 'w')
dset = f.create_dataset('data', data = testxx)
f.close()
f= h5py.File('testyy.h5', 'w')
dset = f.create_dataset('data', data = testyy)
f.close
f = h5py.File('devxx.h5', 'w')
dset = f.create_dataset('data', data = devxx)
f.close()
f = h5py.File('devyy.h5', 'w')
dset = f.create_dataset('data', data = devyy)
f.close()
f = h5py.File('trainxx.h5', 'w')
dset = f.create_dataset('data', data = trainxx)
f.close()
f = h5py.File('trainyy.h5', 'w')
dset = f.create_dataset('data', data = trainyy)
f.close()

spix = 30000
vali = 30000

#testX = right_X[spix: -vali]
#testY = right_Y[spix: -vali]
for i in range(0, 20):
    model.get_layer('left_embed').set_weights(freeze_embed)
    model.get_layer('left_lstm_aa').set_weights(freeze_lstm_aa)
    #model.get_layer('left_lstm_bb').set_weights(freeze_lstm_bb)
    #model.fit(right_X[: spix], right_Y[: spix], batch_size=32, nb_epoch=1, verbose = 1, validation_data = (right_X[-vali:], right_Y[-vali:]))
    model.fit(trainxx, trainyy, batch_size = 10000, nb_epoch = 1, verbose = 1, validation_data = (devxx, devyy))
    showAcc(model, testxx, testyy)
#testX = right_X[: spix]
#testY = right_Y[: spix]
#testZ = model.predict(testX)
#print(testZ.shape)
#showAcc(model, testX, testY)

