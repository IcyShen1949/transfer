import os
import numpy as np
import h5py
import random
import math

def getNorm(tem):
	k = 0
	for line in tem:
		k += line * line
	k = math.sqrt(k)
	temp = []
	for line in tem:
		temp.append(line / k)
	print(temp)
	return temp


wordDict = []
wordList = set()
with open('id2ch_outMethod.txt', 'r', encoding = 'utf-8') as fin:
	for line in fin:
		s = line.strip()
		wordList.add(s)


tem = []
for l in range(300):
	tem.append(random.uniform(-1, 1))
wordDict.append(getNorm(tem))
del(tem)

with open('GoogleNews-vectors-negative300.txt', 'r', encoding = 'utf-8') as fin:
	for line in fin:
		s = line.strip()
		ss = s.split(' ')
		wordName = ss[0]
		if wordName in wordList:
			#print(wordName)
			tem  = []
			for num in ss[1:]:
				tem.append(eval(num))
			wordDict.append(tem)
			del(tem)
tem = []
for i in range(300):
	tem.append(random.uniform(-1, 1))
wordDict.append(getNorm(tem))


f = h5py.File('w2v.h5', 'w')
arr = np.array(wordDict)
dset = f.create_dataset('data', data = arr)
f.close()

'''
with open('id2ch.txt', 'r', encoding = 'utf-8') as fin:
	for line in fin:
		s = line.strip()
		if s not in wordDict: print(s)
print('2')
'''
