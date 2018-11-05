import os, time
import h5py
import numpy as np
import random

def preRead():
	global scoreList, sentenceList
	scoreList = []
	sentenceList = []
	num = 0
	wordSet = set()
	for i in range(1):
		# '''
		# with open('testdata.csv', 'r', encoding = 'utf-8')as fin:
		# 	for k, line in enumerate(fin):
		# 		#print(line)
		# 		s = line.strip()
		#
		# 		ss = s.split(',')
		# 		temSen = ' , '.join(ss[5:])
		# 		temSen = temSen.replace('  ', ' ')
		# 		temScore = ss[0]
		# 		if eval(temScore[1]) != 2: num += 1
		# 		scoreList.append(eval(temScore[1]))
		# 		#xx = temSen[1: -1].split(' ')
		# 		sentenceList.append(temSen[1: -1].split(' '))
		# '''
		with open('training.csv', 'r', encoding = 'ISO-8859-1') as fin:
			for k, line in enumerate(fin):
				#if k % 1000 == 0: print(k)
				s = line.strip()
				ss = s.split(',')
				temSen = ' , '.join(ss[5:])
				temSen = temSen.replace('  ', ' ')
				temScore = ss[0]
				scoreList.append(eval(temScore[1]))
				if eval(temScore[1]) != 2: num += 1
				#xx = temSen[1: -1].split(' ')
				sentenceList.append(temSen[1: -1].split(' '))
		
	print(sentenceList[0])
	print(scoreList[0])
	#print('1')		
	#print(num)
	# '''
	# with open('sentences.txt', 'w+', encoding = 'utf-8') as fout:
	# 	for line in sentenceList:
	# 		fout.write(' '.join(line) + '\n')
	# with open('score.txt', 'w+', encoding = 'utf-8') as fout:
	# 	for line in scoreList:
	# 		fout.write('%d\n' % (line))
	# '''
	# '''
	# with open('sentences.txt', 'r', encoding = 'utf-8') as fin:
	# 	for line in fin:
	# 		s = line.strip()
	# 		ss = s.split(' ')
	# 		sentenceList.append(ss)
	# with open('score.txt', 'r', encoding = 'utf-8') as fin:
	# 	for line in fin:
	# 		s = line.strip()
	# 		scoreList.append(eval(s))
	# '''
	#print(len(sentenceList))
	#print(sentenceList[0])
	#print(len(scoreList))
	#print(scoreList[0])
	

def getVocabulary():
	global id2ch
	id2ch = []
	wordDict = {}
	temDict = []
	temSet = set()

	# '''
	# with open('wordDict.txt', 'r', encoding = 'utf-8') as fin:
	# 	for line in fin:
	# 		s = line.strip()
	# 		#print(s)
	# 		ss = s.split('\t')
	# 		wordDict[ss[0]] = eval(ss[1])
	# '''
			
	# '''
	# if os.path.exists('wordDict.txt'):
	# 	with open('wordDict.txt', 'r', encoding = 'utf-8') as fin:
	# 		for line in fin:
	# 			s = line.strip()
	# 			ss = s.split('\t')
	# 			num = eval(ss[1])
	# 			wordDict[ss[0]] = num
	# '''
	# '''
	# for sentence in sentenceList:
	# 	for word in sentence:
	# 		if word.lower() not in wordDict:
	# 			wordDict[word.lower()] = 1
	# 		else:
	# 			wordDict[word.lower()] += 1
	#
	# with open('wordDict.txt', 'w', encoding = 'utf-8') as fout:
	# 	for word in wordDict:
	# 		fout.write('%s\t%d\n' %(word, wordDict[word]))
	# '''
    #
	# '''
	# for line in temDict:
	# 	id2ch.append(line[0])
	# with open('id2ch.txt', 'w', encoding = 'utf-8') as fout:
	# 	for line in id2ch:
	# 		fout.write(line + '\n')
	# '''
	# '''
	# with open('id2ch.txt', 'r', encoding = 'utf-8') as fin:
	# 	for line in fin:
	# 		s = line.strip()
	# 		id2ch.append(s)
    #
	# for sentence in sentenceList:
	# 	for word in sentence:
	# 		if word not in wordDict:
	# 			wordDict[word] = 1
	# 		else:
	# 			wordDict[word] += 1
	# '''
	# '''
	# temDict = []
	# for word in wordDict:
	# 	temDict.append([word, wordDict[word]])
	# temDict.sort(key = lambda x: -x[1])
	# print(len(temDict))
	# print(temDict[2000])
	# print(temDict[5000])
	# print(temDict[10000])
	# id2ch = []
	# for line in temDict:
	# 	if line[1] < 20: break
	# 	id2ch.append(line[0])
	# with open('id2ch.txt', 'w', encoding = 'utf-8') as fout:
	# 	for line in id2ch:
	# 		fout.write(line + '\n')
	# print('id2ch')
	# print(len(id2ch))
	# '''
	with open('id2ch_outMethod.txt', 'r', encoding = 'utf-8') as fin:
		for line in fin:
			s = line.strip()
			id2ch.append(s)
	# '''
	# with open('id2ch_ourMethod.txt', 'w', encoding = 'utf-8') as fout:
	# 	for line in id2ch:
	# 		fout.write(line + '\n')
	# '''

def getXY():
	global rawX, rawY
	rawX = []
	rawY = []
	ch2id = {}
	ch2id = {v : k for k, v in enumerate(id2ch)}
	num0 = 0
	num1 = 0
	for kk, line in enumerate(sentenceList):
		#print(line)
		if len(line) > 100: continue
		if scoreList[kk] == 2: continue
		temList = [0] * 100
		for k, ll in enumerate(line):
			if ll.lower() in ch2id:
				temList[k] = ch2id[ll.lower()] + 1
			else:
				temList[k] = len(ch2id) + 1
		rawX.append(temList)
		temScore = [0]
		if scoreList[kk] < 2:
			temScore[0] = 0
			num0 += 1
		if scoreList[kk] > 2:
			temScore[0] = 1
			num1 += 1
		rawY.append(temScore)
	print(len(rawX))
	print(len(rawY))
	print(num0)
	print(num1)
	f = h5py.File('tweet_X_train.h5', 'w')
	arr = np.array(rawX)
	dset = f.create_dataset('data', data = arr)
	f.close()

	f = h5py.File('tweet_Y_train.h5', 'w')
	arr = np.array(rawY)
	dset = f.create_dataset('data', data = arr)
	f.close()
	


preRead()
getVocabulary()
getXY()
