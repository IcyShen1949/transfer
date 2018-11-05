import os


with open('GoogleNews-vectors-negative300.txt', 'r', encoding = 'utf-8') as fin, open('id2ch_w2v.txt', 'w', encoding = 'utf-8') as fout:
	for line in fin:
		s = line.strip()
		ss = s.split(' ')
		fout.write(ss[0] + '\n')

