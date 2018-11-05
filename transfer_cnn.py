max_size = 20
seqlen = 100
spix = 2000

import os
import numpy as np
import random
import sys
import h5py
import ljqpy
import random

from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer, self).__init__(**kwargs)
		print(output_dim)
	
	def build(self, input_shape):
		#print(input_shape)
		input_dim = input_shape[1]
		output_dim = int(self.output_dim)
		initial_weight_value = np.random.random((output_dim, output_dim))
		#t=np.array([0.5])
		#self.W = K.variable(initial_weight_value)
		self.wtf = K.variable(initial_weight_value, name = 'wtf')
		self.trainable_weights = [self.wtf]
	
	def call(self, x, mask = None):
		#print(x.shape)
		#print(x.shape.eval({x :x}))
		#l = len(x[0])
		#print(l)
		#print(x.eval())
		#print(np.array(x))
		print(1)
		input_left = x[:, :, :300]
		input_right = x[:, :, 300:]
		#print(input_left.shape)
		#print(1)
		#output_sum = []
		#for k, line in enumerate(input_left):
			#output_sum.append(self.W * input_left[k] + input_right[k])
		print(2)
		return (K.dot(input_left, self.wtf) + input_right)
	
	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], self.output_dim)

class MyLayer_1(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(MyLayer_1, self).__init__(**kwargs)
		print(output_dim)
	
	def build(self, input_shape):
		#print(input_shape)
		input_dim = input_shape[1]
		output_dim = int(self.output_dim)
		initial_weight_value = np.random.random((output_dim, output_dim))
		
		self.wtf = K.variable(initial_weight_value, name = 'wtf')
		self.trainable_weights = [self.wtf]
	
	def call(self, x, mask = None):
		input_left = x[:, :, :6]
		input_right = x[:, :, 6:]
		return (K.dot(input_left, self.wtf) + input_right)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], self.output_dim)

class MyLayer_2(Layer):
	def __init__(self, output_dim, **kwargs):
		self.output_dim= output_dim
		super(MyLayer_2, self).__init__(**kwargs)
	
	def build(self, input_shape):
		input_dim = input_shape[1]
		output_dim = int(self.output_dim)
		initial_weight_value = np.random.random((output_dim, output_dim))
		self.wtf = K.variable(initial_weight_value, name = 'wtf')
		self.trainable_weights = [self.wtf]

	def call(self, x, mask = None):
		
		input_left = x[:, :, :1]
		input_right = x[:, :, 1:]
		return (K.dot(input_left, self.wtf) + input_right)
	
	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], self.output_dim)
	

import keras
from keras.models import Model
from keras.layers import core, Dense, Activation, Dropout, TimeDistributed, LSTM, Embedding, Input, Merge, convolutional, pooling
from keras.engine.topology import merge
from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.regularizers import WeightRegularizer, ActivityRegularizer, l2, activity_l2


def preRead():
    global id2ch, ch2id, id2tg, vocabulary_size
    id2ch = ljqpy.LoadList('id2ch_w2v.txt')
    ch2id = {v:k for k,v in enumerate(id2ch)}
    id2tg = []
    vocabulary_size = len(id2ch)
    print('vocabulary: %d' % (vocabulary_size))

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
        '''
        for kk, tag in enumerate(line):
            temTag = [0] * 2
            if tag != 0:
                temTag[tag] = 1
            tem.append(temTag)
            del temTag
        '''
        tag = line[0]
        tem = [0] * 2
        tem[tag] = 1
        Y.append(tem)
        del tem
    Y = np.array(Y)
    return Y

def getXX(fileName):
	w2v = []
	with open(fileName, 'r', encoding = 'utf-8') as fin:
		for line in fin:
			tem = []
			s = line.strip()
			ss = s.split(' ')
			for num in ss:
				tem.append(eval(num))
			w2v.append(tem)
			del(tem)
	return w2v


def getTrainingData():
    global movieX, movie_X_train, movie_Y_train, movie_X_test, movie_Y_test, movie_X_dev, movie_Y_dev, w2v, tweetX_train, tweetY_train, tweet_X_test, tweet_Y_test, vocabulary_size
    #left_X = getX('movieX.h5')
    #for line in left_X:
        #print(line)
    #ght_X = getX('tweetX.h5')
    #left_Y = getY('movieY.h5')
    #right_Y = getY('tweetY.h5')
    movie_X_train = getX('movie_X_train.h5')
    movie_Y_train = getY('movie_Y_train.h5')
    movie_X_test = getX('movie_X_test.h5')
    movie_Y_test = getY('movie_Y_test.h5')
    movie_X_dev = getX('movie_X_dev.h5')
    movie_Y_dev = getY('movie_Y_dev.h5')
    tweetX_train = getX('tweet_X_train.h5')
    tweetY_train = getY('tweet_Y_train.h5')
    tweet_X_test = getX('tweet_X_test.h5')
    tweet_Y_test = getY('tweet_Y_test.h5')
    movieX = getX('movieX.h5')
    #w2v = getXX('GoogleNews-vectors-negative300.txt')
    w2v = getX('w2v.h5')
    vocabulary_size = len(w2v)
    w2v = [w2v]
    #print(left_Y)

def showAcc(model, testX, testY, trh = 0.5):
    testZZ=model.predict(testX)

    print('aaaaaaaaaaaaaa')
    '''
    with open("result.txt","w",encoding="utf-8") as writer:
        for i in range(len(testX)):
            for wordIndex in testX[i]:
                writer.write("{} ".format(id2ch[wordIndex-1]))
            writer.write(str(testY[i])+" "+str(testZZ[i])+"\n")
        writer.close()
    '''
    #print(testZZ[-10:])
    #print(testY[-10:])
    testZ = []
    '''
    for line in testZZ:
        maxx = 0
        ans = 0
        ll = line[0]
    '''
    '''
        for k, tag in enumerate(line):
            if tag > maxx:
               maxx = tag
               ans = k
    '''
    '''
        if ll > trh: ans = 1
        else: ans = 0
        testZ.append(ans)
    '''
    for line in testZZ:
        maxScore = 0
        maxTag = 0
        for tag, tagScore in enumerate(line):
            if tagScore > maxScore:
                maxScore = tagScore
                maxTag = tag
        testZ.append(maxTag)
    
    TP = 0
    PP = 0
    RR = 0
    for i in range(len(testY)):
        PP += 1
        #wtf = testY[i][0]
        '''
        for k, tag in enumerate(testY[i]):
            if tag == 1 and testZ[i] == k:
        '''
        if testY[i][testZ[i]] == 1: TP += 1
        '''
        if testZ[i] == wtf:
            TP += 1
        '''
    print('TP/PP', TP, PP)
    prec = TP / PP
    print('P : ', prec)
    return prec
    


preRead()
getTrainingData()


from keras.models import Model

#vocabulary_size = 18907
vocabulary_size= 139626
vector_dim = 300

input_layer = Input(shape = (seqlen,) , dtype = 'int32')
embed_1 = Embedding(output_dim = vector_dim, input_dim = vocabulary_size, input_length = seqlen, name = 'embed_1', weights = w2v)(input_layer)
embed_2 = Embedding(output_dim = vector_dim, input_dim = vocabulary_size, input_length = seqlen, name = 'embed_2', weights = w2v)(input_layer)
cnn_layer_1 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_1')(embed_1)
cnn_layer_2 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_2')(embed_1)
cnn_layer_3 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_3')(embed_1)
cnn_layer_4 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_4')(embed_2)
cnn_layer_5 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_5')(embed_2)
cnn_layer_6 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'cnn_layer_6')(embed_2)

dropout_layer_1 = Dropout(0.5)(cnn_layer_1)
dropout_layer_2 = Dropout(0.5)(cnn_layer_2)
dropout_layer_3 = Dropout(0.5)(cnn_layer_3)
dropout_layer_4 = Dropout(0.5)(cnn_layer_4)
dropout_layer_5 = Dropout(0.5)(cnn_layer_5)
dropout_layer_6 = Dropout(0.5)(cnn_layer_6)

dense_layer_1 = Dense(1, name = 'dense_layer_1')(dropout_layer_1)
dense_layer_2 = Dense(1, name = 'dense_layer_2')(dropout_layer_2)
dense_layer_3 = Dense(1, name = 'dense_layer_3')(dropout_layer_3)
dense_layer_4 = Dense(1, name = 'dense_layer_4')(dropout_layer_4)
dense_layer_5 = Dense(1, name = 'dense_layer_5')(dropout_layer_5)
dense_layer_6 = Dense(1, name = 'dense_layer_6')(dropout_layer_6)
'''
dense_layer_4 = Dense(1)(dropout_layer_4)
dense_layer_5 = Dense(1)(dropout_layer_5)
dense_layer_6 = Dense(1)(dropout_layer_6)
'''

pool_layer_1 = pooling.MaxPooling1D(pool_length = seqlen - 2)(dense_layer_1)
pool_layer_2 = pooling.MaxPooling1D(pool_length = seqlen - 3)(dense_layer_2)
pool_layer_3 = pooling.MaxPooling1D(pool_length = seqlen - 4)(dense_layer_3)
pool_layer_4 = pooling.MaxPooling1D(pool_length = seqlen - 2)(dense_layer_4)
pool_layer_5 = pooling.MaxPooling1D(pool_length = seqlen - 3)(dense_layer_5)
pool_layer_6 = pooling.MaxPooling1D(pool_length = seqlen - 4)(dense_layer_6)

'''
pool_layer_1 = pooling.MaxPooling1D(pool_length = seqlen - 2)(dropout_layer_1)
pool_layer_2 = pooling.MaxPooling1D(pool_length = seqlen - 3)(dropout_layer_2)
pool_layer_3 = pooling.MaxPooling1D(pool_length = seqlen - 4)(dropout_layer_3)
pool_layer_4 = pooling.MaxPooling1D(pool_length = seqlen - 2)(dropout_layer_4)
pool_layer_5 = pooling.MaxPooling1D(pool_length = seqlen - 3)(dropout_layer_5)
pool_layer_6 = pooling.MaxPooling1D(pool_length = seqlen - 4)(dropout_layer_6)
'''

'''
pool_layer_1 = pooling.MaxPooling1D(pool_length = 2, border_mode = 'valid')(cnn_layer_1)
pool_layer_2 = pooling.MaxPooling1D(pool_length = 2, border_mode = 'valid')(cnn_layer_2)
pool_layer_3 = pooling.MaxPooling1D(pool_length = 2, border_mode = 'valid')(cnn_layer_3)
pool_layer_4 = pooling.MaxPooling1D(pool_length = 2, border_mode = 'valid')(cnn_layer_4)
pool_layer_5 = pooling.MaxPooling1D(pool_length = seqlen - 3, border_mode = 'valid')(cnn_layer_5)
pool_layer_6 = pooling.MaxPooling1D(pool_length = seqlen - 4, border_mode = 'valid')(cnn_layer_6)
'''
merge_layer = merge([pool_layer_1, pool_layer_2, pool_layer_3, pool_layer_4, pool_layer_5, pool_layer_6], mode = 'concat', concat_axis = 2)

flatten_layer = core.Flatten()(merge_layer)
dropout_layer_flatten = Dropout(0.5)(flatten_layer)
dense_layer = Dense(2, activation = 'sigmoid', name = 'dense_layer')(dropout_layer_flatten)
activation_layer = Activation('softmax')(dense_layer)

model = Model(input = [input_layer], output = [activation_layer])
#model = Model(input = [input_layer], output = [pool_layer_1, pool_layer_2, pool_layer_3])
model.summary()
sgd = SGD()
model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

tweet_X_train = []
tweet_Y_train = []
tweet_X_dev = []
tweet_Y_dev = []
for k, line in enumerate(tweetX_train):
	ll = random.randint(0, 10)
	if ll > 9:
		tweet_X_dev.append(line)	
		tweet_Y_dev.append(tweetY_train[k])
		continue
	else:
		tweet_X_train.append(line)
		tweet_Y_train.append(tweetY_train[k])
tweet_X_train = np.array(tweet_X_train)
tweet_Y_train = np.array(tweet_Y_train)
tweet_X_dev = np.array(tweet_X_dev)
tweet_Y_dev = np.array(tweet_Y_dev)
print(len(tweet_X_train))
print(len(tweet_Y_train))
print(vocabulary_size)

freeze_embed_1 = model.get_layer('embed_1').get_weights()



#model.load_weights('baseline1.h5')
#showAcc(model, tweet_X_test, tweet_Y_test)
#showAcc(model, movie_X_test, movie_Y_test)

'''
#print(left_Y_test[1])
maxA = 0
for k in range(0, 25):
	print(k)
	model.get_layer('embed_1').set_weights(freeze_embed_1)
	model.fit(tweet_X_train, tweet_Y_train, batch_size = 50, nb_epoch = 1, validation_data = (tweet_X_dev, tweet_Y_dev))
	showAcc(model, tweet_X_test, tweet_Y_test)
	temA = showAcc(model, movie_X_test, movie_Y_test)
	
	
	if temA > maxA:
		maxA = temA
		model.save_weights('baseline1.h5', overwrite = True)
	
	#weights = model.get_layer('cnn_layer_5').get_weights()
	#print(weights)
	#showAcc(model, left_X_test, left_Y_test, 0.4)
	#showAcc(model, left_X_test, left_Y_test, 0.5)
	#showAcc(model, left_X_test, left_Y_test, 0.6)
	#showAcc(model, left_X_test, left_Y_test, 0.7)
	#showAcc(model, left_X_test, left_Y_test, 0.8)
	
'''
model.load_weights('baseline2.h5')

print('=========================')
print(len(movie_X_test))
print(len(movie_Y_test))

print(len(movie_X_test[0]))
print(movie_X_test.shape)

print('=========================')




'''
#freeze_lstm_bb = model.get_layer('lstm_bb').get_weights()

for i in range(10, 11):
    print(i)
    filePath = os.path.join('baseline1', 'baseline1_' + str(i) + '.h5')
    model.load_weights(filePath)
    showAcc(model, tweet_X_test, tweet_Y_test)
    showAcc(model, movie_X_test, movie_Y_test)

    


    freeze_embed_2 = model.get_layer('embed_2').get_weights()
    freeze_cnn_1 = model.get_layer('cnn_layer_1').get_weights()
    freeze_cnn_2 = model.get_layer('cnn_layer_2').get_weights()
    freeze_cnn_3 = model.get_layer('cnn_layer_3').get_weights()
    freeze_cnn_4 = model.get_layer('cnn_layer_4').get_weights()
    freeze_cnn_5 = model.get_layer('cnn_layer_5').get_weights()
    freeze_cnn_6 = model.get_layer('cnn_layer_6').get_weights()
    freeze_dense_layer_1 = model.get_layer('dense_layer_1').get_weights()
    freeze_dense_layer_2 = model.get_layer('dense_layer_2').get_weights()
    freeze_dense_layer_3 = model.get_layer('dense_layer_3').get_weights()
    freeze_dense_layer_4 = model.get_layer('dense_layer_4').get_weights()
    freeze_dense_layer_5 = model.get_layer('dense_layer_5').get_weights()
    freeze_dense_layer_6 = model.get_layer('dense_layer_6').get_weights()
    freeze_dense_layer = model.get_layer('dense_layer').get_weights()



    #del model
    
    from keras.models import Model
    
    input_layer = Input(shape = (seqlen,), dtype = 'int32')
    left_embed_1 = Embedding(output_dim = 300, input_dim = vocabulary_size, input_length = seqlen, name = 'left_embed_1', weights = w2v)(input_layer)
    left_embed_2 = Embedding(output_dim = 300, input_dim = vocabulary_size, input_length = seqlen, name = 'left_embed_2', weights = w2v)(input_layer)
    
    right_embed_1 = Embedding(output_dim = 300, input_dim = vocabulary_size, input_length = seqlen, name = 'right_embed_1', weights = w2v)(input_layer)
    right_embed_2 = Embedding(output_dim = 300, input_dim = vocabulary_size, input_length = seqlen, name = 'right_embed_2', weights = w2v)(input_layer)
    
    
    right_concat_embed_2 = merge([left_embed_2, right_embed_2], mode = 'concat')
    right_merge_embed_2 = MyLayer(output_dim = 300)(right_concat_embed_2)
    
    
    left_cnn_layer_1 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_1')(left_embed_1)
    left_cnn_layer_2 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_2')(left_embed_1)
    left_cnn_layer_3 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_3')(left_embed_1)
    left_cnn_layer_4 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_4')(left_embed_2)
    left_cnn_layer_5 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_5')(left_embed_2)
    left_cnn_layer_6 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'left_cnn_layer_6')(left_embed_2)
    
    right_cnn_layer_1 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_1')(right_embed_1)
    right_cnn_layer_2 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_2')(right_embed_1)
    right_cnn_layer_3 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_3')(right_embed_1)
    right_cnn_layer_4 = convolutional.Convolution1D(nb_filter = 1, filter_length = 3, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_4')(right_merge_embed_2)
    right_cnn_layer_5 = convolutional.Convolution1D(nb_filter = 1, filter_length = 4, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_5')(right_merge_embed_2)
    right_cnn_layer_6 = convolutional.Convolution1D(nb_filter = 1, filter_length = 5, border_mode = 'valid', activation = 'relu', name = 'right_cnn_layer_6')(right_merge_embed_2)
    
    
    left_dropout_layer_1 = Dropout(0.5)(left_cnn_layer_1)
    left_dropout_layer_2 = Dropout(0.5)(left_cnn_layer_2)
    left_dropout_layer_3 = Dropout(0.5)(left_cnn_layer_3)
    left_dropout_layer_4 = Dropout(0.5)(left_cnn_layer_4)
    left_dropout_layer_5 = Dropout(0.5)(left_cnn_layer_5)
    left_dropout_layer_6 = Dropout(0.5)(left_cnn_layer_6)
    
    
    right_concat_cnn_layer_1 = merge([left_cnn_layer_1, right_cnn_layer_1], mode = 'concat')
    right_concat_cnn_layer_2 = merge([left_cnn_layer_2, right_cnn_layer_2], mode = 'concat')
    right_concat_cnn_layer_3 = merge([left_cnn_layer_3, right_cnn_layer_3], mode = 'concat')
    right_concat_cnn_layer_4 = merge([left_cnn_layer_4, right_cnn_layer_4], mode = 'concat')
    right_concat_cnn_layer_5 = merge([left_cnn_layer_5, right_cnn_layer_5], mode = 'concat')
    right_concat_cnn_layer_6 = merge([left_cnn_layer_6, right_cnn_layer_6], mode = 'concat')
    
    right_merge_cnn_layer_1 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_1)
    right_merge_cnn_layer_2 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_2)
    right_merge_cnn_layer_3 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_3)
    right_merge_cnn_layer_4 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_4)
    right_merge_cnn_layer_5 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_5)
    right_merge_cnn_layer_6 = MyLayer_2(output_dim = 1)(right_concat_cnn_layer_6)
    
    
    right_dropout_layer_1 = Dropout(0.5)(right_merge_cnn_layer_1)
    right_dropout_layer_2 = Dropout(0.5)(right_merge_cnn_layer_2)
    right_dropout_layer_3 = Dropout(0.5)(right_merge_cnn_layer_3)
    right_dropout_layer_4 = Dropout(0.5)(right_merge_cnn_layer_4)
    right_dropout_layer_5 = Dropout(0.5)(right_merge_cnn_layer_5)
    right_dropout_layer_6 = Dropout(0.5)(right_merge_cnn_layer_6)
    
    
    left_dense_layer_1 = Dense(1, name = 'left_dense_layer_1')(left_dropout_layer_1)
    left_dense_layer_2 = Dense(1, name = 'left_dense_layer_2')(left_dropout_layer_2)
    left_dense_layer_3 = Dense(1, name = 'left_dense_layer_3')(left_dropout_layer_3)
    left_dense_layer_4 = Dense(1, name = 'left_dense_layer_4')(left_dropout_layer_4)
    left_dense_layer_5 = Dense(1, name = 'left_dense_layer_5')(left_dropout_layer_5)
    left_dense_layer_6 = Dense(1, name = 'left_dense_layer_6')(left_dropout_layer_6)
    
    right_dense_layer_1 = Dense(1)(right_dropout_layer_1)
    right_dense_layer_2 = Dense(1)(right_dropout_layer_2)
    right_dense_layer_3 = Dense(1)(right_dropout_layer_3)
    right_dense_layer_4 = Dense(1)(right_dropout_layer_4)
    right_dense_layer_5 = Dense(1)(right_dropout_layer_5)
    right_dense_layer_6 = Dense(1)(right_dropout_layer_6)
    
    
    left_pool_layer_1 = pooling.MaxPooling1D(pool_length = seqlen - 2)(left_dense_layer_1)
    left_pool_layer_2 = pooling.MaxPooling1D(pool_length = seqlen - 3)(left_dense_layer_2)
    left_pool_layer_3 = pooling.MaxPooling1D(pool_length = seqlen - 4)(left_dense_layer_3)
    left_pool_layer_4 = pooling.MaxPooling1D(pool_length = seqlen - 2)(left_dense_layer_4)
    left_pool_layer_5 = pooling.MaxPooling1D(pool_length = seqlen - 3)(left_dense_layer_5)
    left_pool_layer_6 = pooling.MaxPooling1D(pool_length = seqlen - 4)(left_dense_layer_6)
    
    right_pool_layer_1 = pooling.MaxPooling1D(pool_length = seqlen - 2)(right_dense_layer_1)
    right_pool_layer_2 = pooling.MaxPooling1D(pool_length = seqlen - 3)(right_dense_layer_2)
    right_pool_layer_3 = pooling.MaxPooling1D(pool_length = seqlen - 4)(right_dense_layer_3)
    right_pool_layer_4 = pooling.MaxPooling1D(pool_length = seqlen - 2)(right_dense_layer_4)
    right_pool_layer_5 = pooling.MaxPooling1D(pool_length = seqlen - 3)(right_dense_layer_5)
    right_pool_layer_6 = pooling.MaxPooling1D(pool_length = seqlen - 4)(right_dense_layer_6)
    
    
    left_merge_layer = merge([left_pool_layer_1, left_pool_layer_2, left_pool_layer_3, left_pool_layer_4, left_pool_layer_5, left_pool_layer_6], mode = 'concat', concat_axis = 2)
    
    right_merge_layer = merge([right_pool_layer_1, right_pool_layer_2, right_pool_layer_3, right_pool_layer_4, right_pool_layer_5, right_pool_layer_6], mode = 'concat', concat_axis = 2)
    
    
    right_concat_merge_layer = merge([left_merge_layer, right_merge_layer], mode = 'concat')
    right_merge_merge_layer = MyLayer_1(output_dim = 6)(right_concat_merge_layer)
    
    right_flatten_layer = core.Flatten()(right_merge_merge_layer)
    
    
    #right_concat_flatten_layer = merge([left_flatten_layer, right_flatten_layer], mode = 'concat')
    #right_merge_flatten_layer = MyLayer_1(output_dim = 6)(right_concat_flatten_layer)
    

    right_dropout_layer_flatten = Dropout(0.5)(right_flatten_layer)
    right_dense_layer = Dense(2, activation = 'sigmoid')(right_dropout_layer_flatten)
    right_activation_layer = Activation('softmax')(right_dense_layer)
    
    model_1 = Model(input = [input_layer], output = [right_activation_layer])
    model_1.summary()
    model_1.compile(loss='categorical_crossentropy', optimizer='SGD')

    #model_1.load_weights('ourMethod_' + str(i) + '.h5')
    showAcc(model_1, movie_X_test, movie_Y_test)
    maxA = 0

    for k in range(0, 1000):
        print(k)
        model_1.get_layer('left_embed_1').set_weights(freeze_embed_1)
        model_1.get_layer('left_embed_2').set_weights(freeze_embed_2)
        model_1.get_layer('right_embed_1').set_weights(freeze_embed_1)
        model_1.get_layer('left_cnn_layer_1').set_weights(freeze_cnn_1)
        model_1.get_layer('left_cnn_layer_2').set_weights(freeze_cnn_2)
        model_1.get_layer('left_cnn_layer_3').set_weights(freeze_cnn_3)
        model_1.get_layer('left_cnn_layer_4').set_weights(freeze_cnn_4)
        model_1.get_layer('left_cnn_layer_5').set_weights(freeze_cnn_5)
        model_1.get_layer('left_cnn_layer_6').set_weights(freeze_cnn_6)
        model_1.get_layer('left_dense_layer_1').set_weights(freeze_dense_layer_1)
        model_1.get_layer('left_dense_layer_2').set_weights(freeze_dense_layer_2)
        model_1.get_layer('left_dense_layer_3').set_weights(freeze_dense_layer_3)
        model_1.get_layer('left_dense_layer_4').set_weights(freeze_dense_layer_4)
        model_1.get_layer('left_dense_layer_5').set_weights(freeze_dense_layer_5)
        model_1.get_layer('left_dense_layer_6').set_weights(freeze_dense_layer_6)
        model_1.fit(movie_X_train, movie_Y_train, batch_size = 50, nb_epoch = 1, verbose = 1, validation_data = (movie_X_dev, movie_Y_dev))
        temA = showAcc(model_1, movie_X_test, movie_Y_test)
        if temA > maxA:
            model_1.save_weights('ourMethod_' + str(i) + '.h5', overwrite = True)
            maxA = temA
    
    model_1.load_weights('ourMethod_' + str(i) + '.h5')
    showAcc(model_1, movie_X_test, movie_Y_test)

    #from keras import backend as K
    #K.clear_session()
    #testX = right_X[: spix]
    #testY = right_Y[: spix]
    #testZ = model.predict(testX)
    #print(testZ.shape)
    #showAcc(model, testX, testY)
    
'''
