###  python2 onlineIMF.py 10 20847 2
import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
from mat_compute import WC2UDV, onlineSVD
split_part = 10
num_of_word = 20847
current_num = 1
##############################################################################
import sys
split_part = int(sys.argv[1])
num_of_word = int(sys.argv[2])
current_num = int(sys.argv[3])
##############################################################################
dim = 300 
fin = open('./data/sparse_matrix', 'r')
fin2 = open('./data/dictionary', 'r')
fout = open('./result/W_matrix', 'w')
fout2 = open('./result/C_matrix', 'w')
lines = fin.readlines()
fin.close()
dict_lines = fin2.readlines()
word_list = [line.split()[0] for line in dict_lines]
##########################    split vocabulary   ########################################## 
split_results = np.random.randint(split_part, size=(num_of_word, ))
part_dict = dict()
for i in range(split_part):
	tmp = np.argwhere(split_results == i)
	tmp = list(tmp)
	part_dict[i] = [list(j)[0] for j in tmp]
	#string = [str(i) for i in part_dict[i]]
	#string = ' '.join(string)
	#fout.write(string + '\n')

#### choose current_num 
##current_num = 1 
old_word = []
for i in range(split_part):
	if i != current_num:
		old_word += part_dict[i]

new_word = part_dict[current_num]
old_word.sort()  ### list:  [0,1,2,4,6,7,...]
new_word.sort() ### list:  [3,5,12,16,...]
new_word_list = [word_list[i] for i in old_word]
new_word_list += [word_list[i] for i in new_word]
word_dict = dict()
for i in range(len(old_word)):
	word_dict[old_word[i]] = i   ## 0:0, 1:1, 2:2, 4:3, 6:4 


num_old_word = len(old_word)
num_new_word = len(new_word)
for i in range(len(new_word)):
	word_dict[new_word[i]] = i + num_old_word   ### 3:0,  5:1,  12:3,  16:4, ... , 

def transform_line(lines, word_dict):
	lines_out = []
	for line in lines:
		line = line.split()
		num1 = word_dict[int(line[0])]
		num2 = word_dict[int(line[1])]
		string = str(num1) + '\t' + str(num2) + '\t' + line[2] + '\n'
		lines_out.append(string)
	return lines_out

lines2 = transform_line(lines, word_dict)
print('split vocabulary')
##########################    split vocabulary   ########################################## 
##########################    read sparse matrix   ########################################## 
lines = lines2
row = []
col = []
value = []
for line in lines:
	line = line.strip()
	line = line.split()
	row.append(int(line[0]))
	col.append(int(line[1]))
	value.append(int(line[2]))

row = np.array(row)
col = np.array(col)
value = np.array(value)
value = value.astype(np.float32)
num_of_word = max(np.max(row),np.max(col)) + 1
co_occur = csr_matrix((value, (row, col)), shape=(num_of_word, num_of_word)).toarray()
co_occur = np.mat(co_occur)
print('read sparse matrix')
##########################    read sparse matrix   ########################################## 


### compute embedding results for old words
co_occur_small = co_occur[:num_old_word,:num_old_word]
col_sum = np.sum(co_occur_small, 1) 
row_sum = np.sum(co_occur_small, 0) 
total_sum = np.sum(co_occur_small)
PMI = co_occur_small / col_sum / row_sum * total_sum
PMI = np.log(PMI)
PPMI = np.maximum(PMI,0)
u, s, v = svds(PPMI, k=dim)
W0 = u * np.mat(np.diag(np.sqrt(s)))
C0 = v.T * np.mat(np.diag(np.sqrt(s)))
print('compute embedding results for old words')
### compute embedding results for old words

###### compute block A,B
col_sum = np.sum(co_occur, 1) 
row_sum = np.sum(co_occur, 0) 
total_sum = np.sum(co_occur)
PMI = co_occur / col_sum / row_sum * total_sum
PMI = np.log(PMI)
PPMI = np.maximum(PMI,0)
A = PPMI[:num_old_word,num_old_word:]
B = PPMI[num_old_word:,:]
print('compute block A,B')
###### compute block A,B


###### online matrix factorization
U, D, V = WC2UDV(W0, C0)
U, D, V = onlineSVD(U,D,V,A)
V, D, U = onlineSVD(V, D, U, B.T)
W = U * np.sqrt(D)
C = V * np.sqrt(D)
print('online matrix factorization')
###### online matrix factorization




for i, word in enumerate(new_word_list):
	w_vector = str(W[i,:])[2:-2]
	w_vector = ' '.join(w_vector.split())
	c_vector = str(C[i,:])[2:-2]
	c_vector = ' '.join(c_vector.split())
	fout.write(word + '\t' + w_vector + '\n')
	fout2.write(word + '\t' + c_vector + '\n')

fout.close()
fout2.close()









