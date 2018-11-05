# -*- coding: utf-8 -*-
# @Time    : 18-2-7 下午6:18
# @Author  : Icy Shen
# @Email   : SAH1949@126.com
import re
from collections import Counter
import h5py
import numpy as np
import json
vocabulary_size = 5000
num = 100
data_path = "./data/"

# def save_data(X_name, Y_name, X, Y):
#     f = h5py.File(X_name + '.h5', 'w')
#     arr = np.array(X)
#     dset = f.create_dataset('data', data=arr)
#     f.close()
#
#     f = h5py.File(Y_name + '.h5', 'w')
#     arr = np.array(Y)
#     dset = f.create_dataset('data', data=arr)
#     f.close()


def get_movie_data(words, file_name):
    with open(data_path + file_name +'.txt') as f:
        train = f.readlines()
    X = []
    Y = []
    for index, line in enumerate(train):
        sentence = re.findall('\([0-9] [0-9a-zA-Z]+\)', line)
        sentence = [x[3:-1].lower() for x in sentence]
        label = int(re.findall('[0-9]+', line)[0])
        if label > 2:
            y = [1]
        elif label < 2:
            y = [0]
        else:
            continue
        words += sentence
        Y.append(y)
        X.append(sentence)

    return words, X, Y

def get_tweet_data(words, file_name):
    X, Y = [], []
    with open(data_path + 'trainingandtestdata/' + file_name +'.csv', 'r', encoding='ISO-8859-1') as fin:
        num = 0
        for k, line in enumerate(fin):
            s = line.strip()
            ss = s.split(',')
            temSen = ' , '.join(ss[5:])
            temSen = temSen.replace('  ', ' ')
            temSen = re.findall('[0-9a-zA-Z]+', temSen)
            temSen = [x.lower() for x in temSen]
            temp = re.findall('[0-9]', ss[0])[0]
            temScore = [int(temp)]
            if temScore[0] < 2:
                temScore[0] = 0
            elif temScore[0] > 2:
                temScore[0] = 1
            else:
                continue

            Y.append(temScore)
            # if eval(temScore[1]) != 2:
            #     num += 1
            # xx = temSen[1: -1].split(' ')
            X.append(temSen)#temSen[1: -1].split(' '))
            words += temSen
    return words, X, Y

def save_data(data, filename ):
    f = h5py.File(data_path + filename + '.h5', 'w')
    arr = np.array(data)
    dset = f.create_dataset('data', data=arr)
    f.close()


def Embedding_data(data, embedding_dict, filename):
    Data = []
    for item in data:
        temp = [embedding_dict[x] for x in item]

        if len(temp) > num:
            continue
        if len(temp) < num:
            temp.extend(np.zeros(num - len(temp)).tolist())
        Data.append(temp)
    save_data(Data, filename)

if __name__ == "__main__":
    all_words = []
    all_words, movie_TrainX, movie_TrainY = get_movie_data(all_words, 'train')

    all_words, movie_TestX, movie_TestY = get_movie_data(all_words, 'test')
    all_words, movie_devX, movie_devY = get_movie_data(all_words, 'dev')

    all_words, tweet_TrainX, tweet_TrainY =  get_tweet_data(all_words, 'training')
    all_words, tweet_TestX, tweet_TestY = get_tweet_data(all_words, 'testdata')

    vocabulary = Counter(all_words)
    vocabulary = sorted(vocabulary.items(), key = lambda item:item[1], reverse = True)
    # Embedding_words = list(dict(vocabulary[:vocabulary_size]).keys())
    Embedding_words = list(dict(vocabulary).keys())
    value = list(range(1, vocabulary_size + 1)) + (np.zeros(len(Embedding_words) - vocabulary_size) + vocabulary_size + 1).tolist()
    Embedding_dict = dict(zip(Embedding_words, value))

    with open(data_path + "all_words.txt", 'w') as f:
        for word in all_words:
            f.write(word)
    vocabulary_json=  json.dumps(Embedding_dict)
    with open(data_path + 'Embedding.json', 'w') as f:
        f.write(vocabulary_json)

    save_data(movie_TrainY, 'movie_TrainY')
    save_data(movie_TestY, 'movie_TestY')
    save_data(movie_devY, 'movie_devY')
    save_data(tweet_TrainY, 'tweet_TrainY')
    save_data(tweet_TestY, 'tweet_TestY')

    Embedding_data(movie_TrainX, Embedding_dict, 'movie_TrainX')
    Embedding_data(movie_TestX, Embedding_dict, 'movie_TestX')
    Embedding_data(movie_devX, Embedding_dict, 'movie_devX')
    Embedding_data(tweet_TrainX, Embedding_dict, 'tweet_TrainX')
    Embedding_data(tweet_TestX, Embedding_dict, 'tweet_TestX')



