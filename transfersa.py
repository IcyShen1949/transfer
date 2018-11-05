import os
import numpy as np
import random
import sys
import h5py
import ljqpy
import time
import sys
from keras import backend as K
from keras.engine.topology import Layer
from models import AttentionSeq2Seq, SimpleSeq2Seq, Seq2Seq
from custom_recurrents import AttentionDecoder
from keras.layers import TimeDistributed, Bidirectional, Lambda
# from CustomLSTM import LSTMCell1,LSTMCell2, LSTMCell_Left, TransferLSTM, KerasLSTMCell, KerasLSTMCell_left, KerasLSTMCell_right, KerasLSTMCell_right_concatenate
from myLSTM import KerasLSTMCell_right_concatenate,KerasLSTMCell_left
# from myLSTM import KerasLSTMCell_right_concatenate,KerasLSTMCell_left
import keras
from keras.models import Model
from keras.layers import core, Dense, Activation, Dropout, TimeDistributed, LSTM, Embedding, Input, concatenate,  Flatten

from keras.preprocessing import sequence

from keras.models import Sequential
from keras.models import Model
from recurrentshop import RecurrentSequential, cells
from tqdm import tqdm


max_size = 20
seqlen = 40
tweet_spix = 400
movie_spix = 1800
spix = 2000
vali = 200
batch_size = 10
Episodes = 20
vector_dim = 60
data_path= "./data/"
model_path = "./model/"
dim = 45
def preRead():
    id2ch = ljqpy.LoadList(data_path + 'id2ch.txt')
    ch2id = {v:k for k,v in enumerate(id2ch)}
    id2tg = []
    vocabulary_size = len(id2ch)
    print("vocabulary_size is %f " % vocabulary_size)
    return id2ch, ch2id, id2tg, vocabulary_size


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

    trainXX = np.vstack((trainXX, vali_XX))
    trainYY = np.vstack((trainYY, vali_YY))
    testXX = getX(data_path + 'movie_X_test.h5')
    testYY = getX(data_path + 'movie_Y_test.h5')

    trainX = getX(data_path + 'tweet_X_train.h5')
    trainY = getX(data_path + 'tweet_Y_train.h5')
    testX = getX(data_path + 'tweet_X_test.h5')
    testY = getX(data_path + 'tweet_Y_test.h5')
    return trainXX, trainYY, testXX, testYY,\
           trainX, trainY, testX, testY

def get_trainingData():
    left_X = getX(data_path + 'movieX.h5')
    right_X = getX(data_path + 'tweetX.h5')
    left_Y = getX(data_path + 'movieY.h5')
    right_Y = getX(data_path + 'tweetY.h5')


    return left_X, left_Y, right_X, right_Y


def showAcc(model, testX, testY, thr = 0.5):
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
        if ll > thr: ans = 1
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


def flatten(tensor):
    import numpy as np
    return np.concatenate(tensor, 2)


def build_baseline_model():
    input_layer = Input(shape=(seqlen,), dtype='int32')
    embed = Embedding(input_dim=vocabulary_size + 3, output_dim=dim, input_length=seqlen,
                      name='embed')(input_layer)

    rnn = RecurrentSequential(return_sequences=True, name="left_rnn")
    rnn.add(KerasLSTMCell_left(dim, input_dim=dim))
    # rnn_y11=LSTM(45,return_sequences=False)(embed)
    rnn_y1 = rnn(embed)
    rnn_y1 = Dropout(0.2)(rnn_y1)
    rnn_y11 = Lambda(lambda x: x[:, -1:, :dim])(rnn_y1)
    rnn_y11 = Flatten()(rnn_y11)

    densed = Dense(1, activation="sigmoid")(rnn_y11)
    # activationed = Activation('softmax')(densed)

    model = Model(inputs=[input_layer], outputs=[densed])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    # input_layer = Input(shape=(seqlen,), dtype='int32')
    # embed = Embedding(output_dim=dim, input_dim=vocabulary_size + 3, input_length=seqlen,
    #                   name='embed')(input_layer)
    # rnn = RecurrentSequential(return_sequences=True, name="left_rnn")
    # rnn.add(KerasLSTMCell_left(dim, input_dim=dim))
    # rnn_y1 = rnn(embed)
    # rnn_y11 = Lambda(lambda x: x[:, -1:, :dim])(rnn_y1)
    # rnn_y11 = Flatten()(rnn_y11)
    # densed = Dense(1, activation="sigmoid")(rnn_y11)
    # model = Model(inputs=[input_layer], outputs=[densed])
    # model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    # input_layer = Input(shape=(seqlen,), dtype='int32')
    # embed = Embedding(output_dim=45, input_dim=vocabulary_size + 2, input_length=seqlen, mask_zero = True,
    #                   name='embed')(input_layer)
    # rnn = RecurrentSequential(return_sequences=False, name="left_rnn")
    # rnn.add(KerasLSTMCell_left(45, input_dim=45))
    # rnn_y1 = rnn(embed)
    # densed = Dense(1, activation="sigmoid")(rnn_y1)
    # model = Model(inputs=[input_layer], outputs=[densed])
    # model.compile(loss='binary_crossentropy', optimizer='Adagrad')

    # input_layer = Input(shape = (seqlen, ), dtype = 'int32')
    # embed = Embedding(output_dim=vector_dim, input_dim=vocabulary_size + 3,
    #                   input_length=seqlen, mask_zero=True , name = "embed")(input_layer)
    # rnn = RecurrentSequential(return_sequences=False, name="left_rnn")
    # rnn.add(KerasLSTMCell_left(vector_dim, input_dim=vector_dim))
    # rnn_y1 = rnn(embed)
    #
    # densed = Dense(1)(rnn_y1 )
    # activation = Activation('sigmoid')(densed)
    #
    # model = Model(inputs=[input_layer], outputs=[activation])
    # model.compile(loss='binary_crossentropy', optimizer = "Adagrad")
    return model


def train_baseline(Dtype, train_X, train_Y, test_X, test_Y, obiTestx, objTesty, batchsize = 100):
    batch_size = 100
    model = build_baseline_model()
    if (os.path.exists(model_path + "model_baseline" + Dtype + ".h5")) and \
            (os.path.exists(model_path + 'model_baseline_2_' + Dtype + ".h5")):
        model.load_weights(model_path + "model_baseline" + Dtype + ".h5")
        showAcc(model, test_X, test_Y, )
        model2 = build_baseline_model()
        model2.load_weights(model_path + "model_baseline_2_" + Dtype + ".h5")
        showAcc(model, obiTestx, objTesty)

    else:
        maxA = 0
        max_obj = 0.0
        if (Dtype == "m2t_2") or (Dtype == "t2m_2"):
            batch_size = batch_size * 100
        for eposide in tqdm(range(Episodes)):
            model.fit(train_X, train_Y, batch_size = batchsize, nb_epoch=1, verbose=0)
            temA = showAcc(model, test_X, test_Y)
            obj = showAcc(model, obiTestx, objTesty)

            if temA > maxA:
                maxA = temA
                model.save_weights(model_path + "model_baseline" + Dtype + ".h5")
            if max_obj < obj:
                max_obj = obj
                model.save_weights(os.path.join(model_path, 'model_baseline_2_' + Dtype + '.h5'),
                                   overwrite=True)
        print("the final acc of baseline_1_ " + Dtype + "is %f" % (maxA))
        print("the final acc of baseline_2_ " + Dtype + "is %f" % (max_obj))


def get_targetData(right_X, right_Y):
    testxx = []
    testyy = []
    devxx = []
    devyy = []
    trainxx = []
    trainyy = []
    print("right_X's shape is ", right_X.shape)
    right_X = list(right_X)
    right_Y = list(right_Y)

    for k, line in enumerate(right_X):
        if k < tweet_spix:
            continue
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
    testxx = np.array(right_X[:tweet_spix])
    testyy = np.array(right_Y[:tweet_spix])
    devxx = np.array(devxx)
    devyy = np.array(devyy)
    trainxx = np.array(trainxx)
    trainyy = np.array(trainyy)
    return testxx, testyy, devxx, devyy, trainxx, trainyy


def build_transfer_model():
    # right(specific domain) part
    # input_layer = Input(shape=(seqlen,), dtype='int32')
    # left_embed = Embedding(output_dim=45, input_dim=vocabulary_size + 2, input_length=seqlen, name='left_embed',
    #                        trainable=False)(input_layer)
    # rnn = RecurrentSequential(return_sequences=True, name="left_rnn")
    # rnn.add(KerasLSTMCell_left(45, input_dim=45))
    # left_rnn_y11 = rnn(left_embed)
    # right_embed = Embedding(output_dim=45, input_dim=vocabulary_size + 2, input_length=seqlen, name='right_embed')(
    #     input_layer)
    #
    # rnn_transfer = RecurrentSequential(return_sequences=True, name="right_rnn")
    # rnn_transfer.add(KerasLSTMCell_right_concatenate(45, input_dim=45 * 4))
    #
    # right_rnn_y11 = rnn_transfer(concatenate([left_rnn_y11, right_embed]))
    #
    # right_dense = TimeDistributed(Dense(1))(right_rnn_y11)
    # reduce_dimension = Lambda(lambda x: x[:, :, -1])(right_dense)
    # predictions = Dense(1)(reduce_dimension)
    # right_activation = Activation('sigmoid')(predictions)
    #
    # model = Model(inputs=[input_layer], outputs=[right_activation])
    # # model.summary()
    # model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    input_layer = Input(shape=(seqlen,), dtype='int32')
    left_embed = Embedding(output_dim=dim, input_dim=vocabulary_size + 3, input_length=seqlen, name='left_embed',
                           trainable=False)(input_layer)
    rnn = RecurrentSequential(return_sequences=True, name="left_rnn")
    rnn.add(KerasLSTMCell_left(dim, input_dim=dim))

    left_rnn_y11 = rnn(left_embed)
    left_rnn_y11 = Dropout(0.8)(left_rnn_y11)

    right_embed = Embedding(output_dim=dim, input_dim=vocabulary_size + 2, input_length=seqlen, name='right_embed')(
        input_layer)

    rnn_transfer = RecurrentSequential(return_sequences=True, name="right_rnn")
    rnn_transfer.add(KerasLSTMCell_right_concatenate(dim, input_dim=dim * 4))

    right_rnn_y11 = rnn_transfer(concatenate([left_rnn_y11, right_embed]))
    right_rnn_y11 = Dropout(0.2)(right_rnn_y11)

    right_dense = TimeDistributed(Dense(1))(right_rnn_y11)

    reduce_dimension = Lambda(lambda x: x[:, :, -1])(right_dense)
    predictions = Dense(1)(reduce_dimension)
    right_activation = Activation('sigmoid')(predictions)

    model = Model(inputs=[input_layer], outputs=[right_activation])
    model.compile(loss='binary_crossentropy', optimizer='Adagrad')


    # input_layer = Input(shape=(seqlen,), dtype='int32')
    # left_embed = Embedding(output_dim=dim, input_dim=vocabulary_size + 3, input_length=seqlen, name='left_embed',
    #                        trainable=False)(input_layer)
    # rnn = RecurrentSequential(return_sequences=True, name="left_rnn")
    # rnn.add(KerasLSTMCell_left(dim, input_dim=dim))
    #
    # left_rnn_y11 = rnn(left_embed)
    # left_rnn_y11 = Dropout(0.2)(left_rnn_y11)
    #
    # right_embed = Embedding(output_dim=dim, input_dim=vocabulary_size + 2, input_length=seqlen, name='right_embed')(
    #     input_layer)
    #
    # rnn_transfer = RecurrentSequential(return_sequences=True, name="right_rnn")
    # rnn_transfer.add(KerasLSTMCell_right_concatenate(dim, input_dim=dim * 4))
    #
    # right_rnn_y11 = rnn_transfer(concatenate([left_rnn_y11, right_embed]))
    # right_rnn_y11 = Dropout(0.2)(right_rnn_y11)
    #
    # right_dense = TimeDistributed(Dense(1))(right_rnn_y11)
    #
    # reduce_dimension = Lambda(lambda x: x[:, :, -1])(right_dense)
    # predictions = Dense(1)(reduce_dimension)
    # right_activation = Activation('sigmoid')(predictions)
    #
    # model = Model(inputs=[input_layer], outputs=[right_activation])
    # # model.summary()
    # model.compile(loss='binary_crossentropy', optimizer='Adagrad')
    return model


def train_ours(trainXX, trainYY, testXX, testYY, Dtype, batchsize = 100):
    print("train our model......")
    model_before = build_baseline_model()
    model_before.load_weights(os.path.join(model_path, 'model_baseline'+ Dtype + '.h5'))
    freeze_embed = model_before.get_layer('embed').get_weights()
    freeze_rnn = model_before.get_layer('left_rnn').get_weights()

    model = build_transfer_model()
    model.get_layer('left_embed').set_weights(freeze_embed)
    model.get_layer('left_rnn').set_weights(freeze_rnn)
    max_acc = 0.0
    for i in tqdm(range(Episodes)):
        model.get_layer('left_embed').set_weights(freeze_embed)
        model.get_layer('left_rnn').set_weights(freeze_rnn)

        model.fit(trainXX, trainYY, batch_size= batchsize, epochs=1, verbose=0)
        _ = showAcc(model, testXX, testYY)
        if max_acc < _:
            max_acc = _
            model.save_weights(os.path.join(model_path, 'model_ours'+Dtype+'.h5'),
                               overwrite=True)


    print("the final acc of our model is %f" % (max_acc))


if __name__ == "__main__":
    global vocabulary_size
    id2ch, ch2id, id2tg, vocabulary_size = preRead()
    trainXX, trainYY, testXX, testYY, \
    trainX, trainY, testX, testY = get_trainData()

    print("train the first baseline model")
    train_baseline('t2m_1', trainXX[:, :seqlen], trainYY, testXX[:, :seqlen], testYY, testX[:, :seqlen], testY)


    print("train the second baseline model")
    train_baseline('t2m_2', trainX[:, :seqlen], trainY, testX[:, :seqlen], testY, testXX[:, :seqlen], testYY,batchsize = 10000)

    train_ours(trainXX[:, :seqlen], trainYY, testXX[:, :seqlen], testYY, Dtype='t2m_2')
    train_ours(trainX[:, :seqlen], trainY, testX[:, :seqlen], testY, Dtype='t2m_1', batchsize = 10000)

