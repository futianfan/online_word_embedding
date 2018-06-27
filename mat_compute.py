import numpy as np 
import numpy.linalg as LA 
from scipy.sparse.linalg import svds
np.random.seed(1)

def WC2UDV(W, C):
	##assert W.shape == C.shape
	assert type(W) == np.matrix and type(C) == np.matrix
	U, D1 = LA.qr(W, 'reduced')
	V, D2 = LA.qr(C, 'reduced')
	S = D1 * (D2.T) 
	U1, D1, V1 = LA.svd(S)
	U = U * U1
	D = np.diag(D1)
	V = V * V1.T
	return U, D, V

def onlineSVD(U,D,V,a):
	assert type(a) == np.matrix
	assert U.shape[1] == D.shape[0]
	assert V.shape[1] == D.shape[0]
	assert U.shape[0] == a.shape[0]
	dim = U.shape[1]
	b = U.T * a 
	Z = D * V.T 
	V_star = np.concatenate((Z,b), axis = 1)
	V_star = V_star.T 
	V, D, Us = LA.svd(V_star, full_matrices=False)
	D = np.diag(D)
	U = U * Us.T 
	return U, D, V 


if __name__=='__main__':
	a = np.random.random((200,100))
	a = np.mat(a)
	b = np.random.random((200,100))
	b = np.mat(b)
	U,D,V = WC2UDV(a,b)
	c = np.random.random((200,1))
	c = np.mat(c)
	#print(a * b.T - U * D * V.T)

	
	U,D,V = onlineSVD(U,D,V,c)
	true_value = np.concatenate((a * b.T, c), axis = 1)
	#print(true_value)
	#print()
	tmp = U * D * V.T
	#print(tmp)
	tt = tmp - true_value
	print(LA.norm(tt[:,:], 'fro')) 	



	pass 

