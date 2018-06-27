import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

dim = 300 
fin = open('./data/sparse_matrix', 'r')
fin2 = open('./data/dictionary', 'r')
fout = open('./result/W_matrix', 'w')
fout2 = open('./result/C_matrix', 'w')
lines = fin.readlines()
fin.close()
############
#lines = lines[:500]
############


row = []
col = []
value = []
for line in lines:
	line = line.strip()
	line = line.split('\t')
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

'''
b = co_occur.transpose()
np.sum(co_occur == b)
'''

col_sum = np.sum(co_occur, 1) 
row_sum = np.sum(co_occur, 0) 
total_sum = np.sum(co_occur)  
for i in range(num_of_word):
	if col_sum[i,0] == 0:
		print('col sum ' + str(i))
	if row_sum[0,i] == 0:
		print('row sum' + str(i))
PMI = co_occur / col_sum / row_sum * total_sum
PMI = np.log(PMI)
PPMI = np.maximum(PMI,0)
u, s, v = svds(PPMI, k=dim)
W = u * np.mat(np.diag(np.sqrt(s)))
C = v.T * np.mat(np.diag(np.sqrt(s)))

lines = fin2.readlines()
fin2.close()
for i, line in enumerate(lines):
	word = line.split('\t')[0]
	w_vector = str(W[i,:])[2:-2]
	w_vector = ' '.join(w_vector.split())
	c_vector = str(C[i,:])[2:-2]
	c_vector = ' '.join(c_vector.split())
	fout.write(word + '\t' + w_vector + '\n')
	fout2.write(word + '\t' + c_vector + '\n')
fout.close()
fout2.close()



'''
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
a = np.random.random((5,6))
svd = TruncatedSVD(n_components=4)
part_1 = svd.fit(a)
'''






