from collections import defaultdict


fin = open('./data/text8', 'r')
fout1 = open('./data/dictionary', 'w')
fout2 = open('./data/sparse_matrix','w')

threshold = 40
window_size = 5
num_of_voca = 20000
word_num = defaultdict(lambda: 0)
## read-in
line = fin.readline()
line = line.split()
#line = line[:10207]

for i in line:
	word_num[i] += 1

word_num_new = dict()
for i in word_num:
	if word_num[i] > threshold:
		word_num_new[i] = word_num[i]

##print(len(word_num_new))
word2index = dict()
index2word = dict()
length = 0
for i in word_num_new:
	word2index[i] = length
	index2word[length] = i
	fout1.write(i + '\t' + str(length) + '\t' + str(word_num_new[i]) + '\n')
	length += 1

sparse_matrix = defaultdict(lambda: 0)

for idx, word in enumerate(line):
	if idx % 10000 == 0:
		print('processing ' + str(idx) + ' / ' + str(len(line)))
	if word not in word_num_new:
		continue
	word_idx = word2index[word]
	for i in range(max(0,idx- window_size) ,min(len(line),idx + window_size + 1)):
		if i == idx:
			continue
		contxt = line[i]
		if contxt not in word2index:
			continue
		contxt_idx = word2index[contxt]
		pair_key = str(word_idx) + '_' + str(contxt_idx)
		sparse_matrix[pair_key] += 1

for k,v in sparse_matrix.items():
	k = k.split('_')
	fout2.write(k[0] + '\t' + k[1] + '\t' + str(v) + '\n')

fout2.close()







