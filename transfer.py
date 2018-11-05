import os
import numpy as np
import random
import sys
import h5py
import ljqpy
import random
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.models import Model, Sequential
from keras.layers import core, Dense, Activation, Dropout, TimeDistributed, LSTM, Embedding, Input, merge
from keras.preprocessing import sequence

from tqdm import tqdm

max_size = 20
seqlen = 50
spix = 2000
vali = 200
batch_size = 10000
Episodes = 10
vector_dim = 60
data_path= "./data/"
model_path = "./model/"
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

def preRead():
    global id2ch, ch2id, id2tg, vocabulary_size
    id2ch = ljqpy.LoadList(data_path + 'id2ch.txt')
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
    left_X = getX(data_path + 'movieX.h5')
    #for line in left_X:
        #print(line)
    right_X = getX(data_path + 'tweetX.h5')
    left_Y = getX(data_path + 'movieY.h5')
    right_Y = getX(data_path + 'tweetY.h5')

def showAcc(model, testX, testY):
    testZZ=model.predict(testX)
    #
    # with open("result.txt","w",encoding="utf-8") as writer:
    #     for i in range(len(testX)):
    #         for wordIndex in testX[i]:
    #             writer.write("{} ".format(id2ch[wordIndex-1]))
    #         writer.write(str(testY[i])+" "+str(testZZ[i])+"\n")
    #     writer.close()


    #print(testZZ[1])
    testZ = []

    for line in testZZ:
        maxx = 0
        ans = 0
        ll = line[0]
        # for k, tag in enumerate(line):
        #     if tag > maxx:
        #        maxx = tag
        #        ans = k
        if ll > 0.5: ans = 1
        else: ans = 0
        testZ.append(ans)
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

    TP = 0
    PP = 0
    RR = 0
    for i in range(len(testY)):
        PP += 1
        wtf = testY[i][0]


        # for k, tag in enumerate(testY[i]):
        #     if tag == 1 and testZ[i] == k:

        if testZ[i] == wtf:
            TP += 1
    print('TP/PP', TP, PP)
    prec = TP / PP
    print('P : ', prec)
    return prec

def build_model(vocabulary_size, vector_dim, seqlen):
    model = Sequential()
    embed = Embedding(input_dim=vocabulary_size + 3, output_dim=vector_dim, input_length=seqlen, mask_zero=True,
                        name='embed')
    model.add(embed )
    lstm_ss = LSTM(output_dim=100, return_sequences=False, name='lstm_aa')
    model.add(lstm_ss)
    dropout = Dropout(0.2)
    model.add(dropout)
    # model.add(LSTM(output_dim = 45, return_sequences=True, name = 'lstm_bb'))
    # model.add(Dropout(0.2))
    den = Dense(1)
    model.add(den)
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    # model.summary()
    return model

def train_baseline(train_X, train_Y, valid_X, valid_Y, test_X, test_Y):
    model = build_model(vocabulary_size, vector_dim, seqlen)
    if os.path.exists(model_path + 'model_baseline.h5'):
        model.load_weights(model_path + 'model_baseline.h5')
        showAcc(model, test_X, test_Y)
        # spix = 500
        # vali = 500
        # showAcc(model, right_X[:vali], right_Y[:vali])
    else:
        maxA = 0
        for k in tqdm(range(Episodes)):
            model.fit(train_X, train_Y,
                      batch_size = batch_size , nb_epoch = 1, verbose = 0, validation_data = (valid_X, valid_Y))
            temA = showAcc(model, test_X, test_Y)
            #temA = showAcc(model, right_X[30000: -5000], right_Y[30000: -5000])
            if temA > maxA:
                maxA = temA
                model.save_weights(model_path + 'model_baseline'+'.h5', overwrite = True)
        print("the final acc of the model is %f" % maxA)

def write_data(file_name, data):
    f = h5py.File(file_name, 'w')
    dset = f.create_dataset('data', data = data)
    f.close()

def get_targetData(right_X, right_Y):
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
    testxx = np.array(right_X[:500])
    testyy = np.array(right_Y[:500])
    devxx = np.array(devxx)
    devyy = np.array(devyy)
    trainxx = np.array(trainxx)
    trainyy = np.array(trainyy)
    return testxx, testyy, devxx, devyy, trainxx, trainyy
# model.add(Embedding(input_dim = 5, output_dim = 1, input_length = seqlen))
# model.add(LSTM(output_dim = 50, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(LSTM(output_dim = 45, return_sequences = True))
# model.add(Dropout(0.2))
# model.add(TimeDistributed(Dense(5)))
# model.add(Activation('softmax'))
# model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')


#model for baseline1
def build_tranfer_model():
    input_layer = Input(shape = (seqlen,), dtype = 'int32')
    left_embed = Embedding(output_dim = vector_dim, input_dim = vocabulary_size + 3,
                           input_length = seqlen, name = 'left_embed')(input_layer)
    left_lstm_aa = LSTM(output_dim = 100, return_sequences = False, name = 'left_lstm_aa')(left_embed)#100
    left_drop_aa = Dropout(0.2)(left_lstm_aa)


    #left_lstm_bb = LSTM(output_dim = 45, return_sequences = True, name = 'left_lstm_bb')(left_drop_aa)
    right_embed = Embedding(output_dim = vector_dim, input_dim = vocabulary_size + 3, input_length = seqlen)(input_layer)

    right_concat_embed = merge([left_embed, right_embed], mode = 'concat')

    #model.add(LambdaMerge([model0, model1], lambda inputs: p0*inputs[0]+p1*inputs[1]))
    #right_merge_embed = merge([left_embed, right_embed], mode = lambda x: p0 * x[0] + x[1], output_shape = lambda x: x[1])
    right_merge_embed = MyLayer(output_dim = vector_dim)(right_concat_embed)
    right_lstm_aa = LSTM(output_dim = 100, return_sequences = False)(right_merge_embed )#100
    right_drop_aa = Dropout(0.2)(right_lstm_aa)

    right_concat_aa = merge([left_lstm_aa, right_drop_aa], mode = 'concat') #??
    #right_merge_aa = LambdaMerge([left_lstm_aa, right_drop_aa], lambda inputs: p0 * inputs[0] + inputs[1])
    #right_merge_aa = MyLayer(inputs = [left_lstm_aa, right_drop_aa])
    right_merge_aa = MyLayer_1(output_dim = 100 )(right_concat_aa)#100
    #right_lstm_bb = LSTM(output_dim = 45, return_sequences = True)(right_merge_aa)
    #right_drop_bb = Dropout(0.2)(right_lstm_bb)

    #right_concat_bb = merge([left_lstm_bb, right_drop_bb], mode = 'concat') #??
    #right_merge_bb = LambdaMerge([left_lstm_bb, right_drop_bb], lambda inputs: p0 * inputs[0] + inputs[1])
    #right_merge_bb = MyLayer(inputs = [left_lstm_bb, right_drop_bb])
    #right_merge_bb = MyLayer(output_dim = 45)(right_concat_bb)
    right_dense = Dense(1)(right_merge_aa )
    right_activation = Activation('sigmoid')(right_dense)
    # , mask_zero = True
    #model.summary()
    model = Model(input = [input_layer], output = [right_activation])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    return model

def train_transfer():
    model_before = build_model(vocabulary_size, vector_dim, seqlen)
    model_before.load_weights(model_path + "model_baseline.h5")
    freeze_embed = model_before.get_layer('embed').get_weights()
    freeze_lstm_aa = model_before.get_layer('lstm_aa').get_weights()
    model = build_tranfer_model()
    testxx, testyy, devxx, devyy, trainxx, trainyy = get_targetData(right_X, right_Y)
    maxAcc = 0
    for i in tqdm(range(Episodes)):
        model.get_layer('left_embed').set_weights(freeze_embed)
        model.get_layer('left_lstm_aa').set_weights(freeze_lstm_aa)
        model.fit(trainxx, trainyy, batch_size=batch_size, epochs=1, verbose=0, validation_data=(devxx, devyy))

        _ = showAcc(model, testxx, testyy)
        if _ > maxAcc:
            maxAcc = _
            model.save_weights(model_path + 'model' + "trasfer" + '.h5', overwrite=True)
    print("the final acc of the model is %f" % (maxAcc))


if __name__ == "__main__":
    preRead()
    getTrainingData()
    strainX = left_X[spix: -vali]
    strainY = left_Y[spix: -vali]
    svalidationX = left_X[-vali:]
    svalidationY = left_Y[-vali:]
    stestX = left_X[: spix]
    stestY = left_Y[: spix]


    train_baseline(strainX, strainY, svalidationX, svalidationY, stestX, stestY)
    train_transfer()
    #the training process

    #model.fit(left_X[spix:], left_Y[spix:], batch_size = 32, nb_epoch = 1, verbose = 1)
    #testX = left_X[: spix]
    #testY = left_Y[: spix]
    #testZ = model.predict(testX)
    #print(testZ.shape)
    #showAcc(model, testX, testY)

    #testX = right_X[: spix]
    #testY = right_Y[: spix]
    #testZ = model.predict(testX)
    #print(testZ.shape)
    #showAcc(model, testX, testY)
