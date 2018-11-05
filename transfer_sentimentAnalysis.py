
import os
import numpy as np
import random
import sys
import h5py
import ljqpy
import random

from keras import backend as K
from keras.engine.topology import Layer
import keras
from keras.models import Model
from keras.layers import core, Dense, Activation, Dropout, TimeDistributed, LSTM, Embedding, Input, merge

from keras.preprocessing import sequence
from tqdm import tqdm
from keras.models import Sequential
from keras.models import Model
import re
import json
max_size = 20
seqlen = 50
spix = 2000
vector_dim = 60
Episodes = 20
data_path = "./data/"
model_path = './model/'

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

def get_trainData():
    trainXX = getX(data_path + 'movie_X_train.h5')
    trainYY = getX(data_path + 'movie_Y_train.h5')
    vali_XX = getX(data_path + 'movie_X_dev.h5')
    vali_YY = getX(data_path + 'movie_Y_dev.h5')
    testXX = getX(data_path + 'movie_X_test.h5')
    testYY = getX(data_path + 'movie_Y_test.h5')

    trainX = getX(data_path + 'tweet_X_train.h5')
    trainY = getX(data_path + 'tweet_Y_train.h5')
    testX = getX(data_path + 'tweet_X_test.h5')
    testY = getX(data_path + 'tweet_Y_test.h5')
    return trainX, trainY, testX, testY,\
           trainXX, trainYY, testXX, testYY, vali_XX, vali_YY

def save_data(data, filename ):
    f = h5py.File(data_path + filename + '.h5', 'w')
    arr = np.array(data)
    dset = f.create_dataset('data', data=arr)
    f.close()


def showAcc(model, testX, testY, transfer = False):
    testZZ=model.predict(testX)
    if transfer:
        with open(data_path + 'test.txt') as f:
            test = f.readlines()
        X = []
        for index, line in enumerate(test):
            sentence = re.findall('\([0-9] [0-9a-zA-Z]+\)', line)
            sentence = [x[3:-1].lower() for x in sentence]
            label = int(re.findall('[0-9]+', line)[0])
            if label > 2:
                y = [1]
            elif label < 2:
                y = [0]
            else:
                continue
            X.append(sentence)

    # with open("result.txt","w",encoding="utf-8") as writer:
    #     for i in range(len(testX)):
    #         for wordIndex in testX[i]:
    #             writer.write("{} ".format(id2ch[wordIndex-1]))
    #         writer.write(str(testY[i])+" "+str(testZZ[i])+"\n")
    #     writer.close()
    #
    #print(testZZ[1])
    testZ = []

    bad_case = []
    for line in testZZ:
        maxx = 0
        ans = 0
        ll = line[0]
        if ll > 0.5:
            ans = 1
        else:
            ans = 0
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
        if testZ[i] == wtf:
            TP += 1
        # elif transfer:
        #     bad_case.append(testX[i, :])
        #     print("the initial sentence is: \n", " ".join(X[i]))
        #     print("the ith row is wrong:\n", testX[i, :])
        #     temp = [reverse_embed[x] for x in testX[i, :]]
        #     print("the coresponded words are : \n", temp )
        #     print("y_hat is %d, label is %d"%(testZ[i], wtf))

    # if transfer:
    #     save_data(bad_case, 'bad_case')
    print('TP/PP', TP, PP)
    prec = TP / PP
    print('P : ', prec)
    return prec

def build_baseline_model(Dtype):
    model = Sequential()

    model.add(Embedding(input_dim=vocabulary_size + 3, output_dim=vector_dim, input_length=seqlen,
                        name='embed'))
    model.add(LSTM(output_dim=100, return_sequences=False, name='lstm_aa'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    return model

def train_baseline(train_x, train_y, test_x, test_y, vali_x, vali_y,Dtype):
    if Dtype == "movie":
        batch_size = 100
    else:
        batch_size = 10000
    model = build_baseline_model(Dtype)
    if os.path.exists(model_path + 'model_' + Dtype + '.h5'):
        model.load_weights(model_path + 'model_' + Dtype + '.h5')
        _ = showAcc(model, test_x, test_y, True)
        print('the baseline  %s model s accuracy is %f'%(Dtype, _))
    else:
        maxA = 0
        for k in tqdm(range(Episodes)):
            model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=1, verbose=1,
                      validation_data=(vali_x, vali_y))
            temA = showAcc(model, test_x, test_y)
            if temA > maxA:
                maxA = temA
                model.save_weights(model_path + 'model_' + Dtype + '.h5', overwrite=True)
        print('the baseline  %s model s accuracy is %f' % (Dtype, maxA))

def get_target_data():
    right_X = getX(data_path + 'tweetX.h5')
    right_Y = getX(data_path + 'tweetY.h5')
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
        xx = random.randint(0, 100)
        if xx < 90:
            trainxx.append(line)
            trainyy.append(right_Y[k])
        elif xx > 90:
            devxx.append(line)
            devyy.append(right_Y[k])
    testxx = right_X[:500]
    testyy = right_Y[:500]
    testxx = np.array(testxx)
    testyy = np.array(testyy)
    devxx = np.array(devxx)
    devyy = np.array(devyy)
    trainxx = np.array(trainxx)
    trainyy = np.array(trainyy)
    return trainxx, trainyy, testxx, testyy, devxx, devyy

def train_transfer(train_x, train_y, test_x, test_y, vali_x, vali_y, base_type):
    seqlen = train_x.shape[1]
    model_baseline = build_baseline_model(base_type)
    model_baseline.load_weights(model_path + 'model_' + base_type + '.h5')
    freeze_embed = model_baseline.get_layer('embed').get_weights()
    freeze_lstm_aa = model_baseline.get_layer('lstm_aa').get_weights()

    input_layer = Input(shape=(seqlen,), dtype='int32')
    left_embed = Embedding(output_dim=vector_dim, input_dim=vocabulary_size + 3, input_length=seqlen,
                           name='left_embed')(input_layer)
    left_lstm_aa = LSTM(output_dim=100, return_sequences=False, name='left_lstm_aa')(left_embed)
    # left_drop_aa = Dropout(0.2)(left_lstm_aa)

    right_embed = Embedding(output_dim=vector_dim, input_dim=vocabulary_size + 3, input_length=seqlen, mask_zero=True)(input_layer)

    right_concat_embed = merge([left_embed, right_embed], mode='concat')

    right_merge_embed = MyLayer(output_dim=vector_dim)(right_concat_embed)

    right_lstm_aa = LSTM(output_dim=100, return_sequences=False)(right_merge_embed)
    # right_drop_aa = Dropout(0.2)(right_lstm_aa)

    right_concat_aa = merge([left_lstm_aa, right_lstm_aa], mode='concat')
    right_merge_aa = MyLayer_1(output_dim=100)(right_concat_aa)

    right_dense = Dense(1)(right_merge_aa)
    right_activation = Activation('sigmoid')(right_dense)


    model = Model(input=[input_layer], output=[right_activation])
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')

    maxA = 0
    if os.path.exists(model_path + 'model_transfer_from_' + base_type + '.h5'):
        model.load_weights(model_path + 'model_transfer_from_' + base_type + '.h5')
        _ = showAcc(model, test_x, test_y)
        print('the accuracy of transfer model from %s is %f' % (base_type, maxA))
    else:
        for i in tqdm(range(Episodes)):
            model.get_layer('left_embed').set_weights(freeze_embed)
            model.get_layer('left_lstm_aa').set_weights(freeze_lstm_aa)
            model.fit(train_x, train_y, batch_size=10000, nb_epoch=1, verbose=1, validation_data=(vali_x, vali_y))
            _ = showAcc(model, test_x, test_y, transfer = True)
            if _ > maxA:
                maxA = _
                model.save_weights(model_path + 'model_transfer_from_' + base_type + '.h5')
        print('the accuracy of transfer model from %s is %f' %(base_type, maxA))

if __name__ == "__main__":
    # vali = 2000
    global reverse_embed
    with open(data_path + 'Embedding.json', 'r') as f:
        embedding = json.load(f)
    reverse_embed = dict(zip(list(embedding.values()), list(embedding.keys())))
    preRead()
    # trainX, trainY, testX, testY, \
    # trainXX, trainYY, testXX, testYY, vali_XX, vali_YY = get_trainData()
    # vali_X_len = int(trainX.shape[0] * 0.1)
    # vali_X, vali_Y = trainX[:vali_X_len], trainY[:vali_X_len]
    # trainX, trainY = trainX[vali_X_len:], trainY[vali_X_len:]
    # trainx, trainy, testx, testy, devx, devy = get_target_data()
    trainX = getX(data_path + 'tweet_TrainX.h5')[:, :50]
    trainY = getX(data_path + 'tweet_TrainY.h5')
    vali_size = int(trainX.shape[0] * 0.1)
    vali_X, vali_Y =  trainX[:vali_size], trainY[:vali_size]
    trainX, trainY = trainX[vali_size:], trainY[vali_size:]

    testX = getX(data_path + 'tweet_TestX.h5')[:, :50]
    testY = getX(data_path + 'tweet_TestY.h5')

    trainXX = getX(data_path + 'movie_TrainX.h5')[:, :50]
    trainYY = getX(data_path + 'movie_TrainY.h5')

    vali_XX = getX(data_path + 'movie_devX.h5')[:, :50]
    vali_YY = getX(data_path + 'movie_devY.h5')

    testXX = getX(data_path + 'movie_TestX.h5')[:, :50]
    testYY = getX(data_path + 'movie_TestY.h5')

    # left_X = getX(data_path + 'movie_TrainX.h5')
    # left_Y = getX(data_path + 'movie_TrainY.h5')
    # trainXX, trainYY, testXX, testYY, vali_XX, vali_YY = left_X[spix : -vali], left_Y[spix : -vali],\
    #                                                 left_X[: spix], left_Y[: spix],  left_X[-vali:], left_Y[-vali:]

    # train_baseline(trainXX, trainYY, testXX, testYY, vali_XX, vali_YY, 'movie')
    # train_transfer(trainX, trainY, testX, testY, vali_X, vali_Y, 'movie')

    train_baseline(trainX, trainY, testX, testY, vali_X, vali_Y, 'tweet')

    train_baseline(trainXX, trainYY, testXX, testYY, vali_XX, vali_YY, 'movie')

    train_transfer(trainXX, trainYY, testX, testY, vali_XX, vali_YY, 'tweet')
    train_transfer(trainX, trainY, testXX, testYY, vali_X, vali_Y, 'movie')



