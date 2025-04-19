# You need to install cvxopt =====> conda install -c conda-forge cvxopt

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as sio
import cvxopt

data = sio.loadmat('svmdata.mat')
x = data.get('x')
y = data.get('y')

x = x.transpose()
y = y.transpose()

n = np.size(x,0)

ClassA = np.nonzero(y == 1)[0]
ClassB = np.nonzero(y == -1)[0]

x1 = x[ClassA]
x2 = x[ClassB]

plt.figure()
plt.plot(x1[:,0],x1[:,1],'ro', label='A')  # Class A
plt.plot(x2[:,0],x2[:,1],'bo', label='B')  # Class B
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


C = 10

H = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        H[i,j]=y[i]*y[j]*x[i,:]@x[j,:]

f = -np.ones([n,1])
G = np.vstack([np.eye(n,n),-np.eye(n,n)])
h = np.vstack([C*np.ones([n,1]),np.zeros([n,1])])

print('\n##### Optimization #####')

H = cvxopt.matrix(H)
f = cvxopt.matrix(f)
G = cvxopt.matrix(G)
h = cvxopt.matrix(h)
a = cvxopt.matrix(0.0)

y2 = y.astype(np.double)
Y = cvxopt.matrix(y2.T)

sol = cvxopt.solvers.qp(H, f, G, h, Y, a)

alpha = sol['x']
alpha = np.array(alpha)

almostZero = np.nonzero(alpha<=(max(alpha)/10**5))
alpha[almostZero]=0

x = x.T
y = y.T
alpha = alpha.T

S = np.nonzero((alpha>0) & (alpha<C))
print('\nsupport vectors:',S[1][:],'\n')

w = 0
for i in S[1][:]:
    w = w+alpha[0,i]*y[0,i]*x[:,i]

b=0
for i in S[1][:]:
    b = b + y[0,i]- w @ x[:,i]
b = b/np.shape(S)[1]

plt.figure()
plt.plot(x1[:,0],x1[:,1],'ro', label='A') #Class A
plt.plot(x2[:,0],x2[:,1],'bo', label='B') #Class B
t = np.linspace(-2,7,100)
plt.plot(t ,-(w[0]/w[1])*t - b/w[1])
plt.plot(t ,-(w[0]/w[1])*t - (b-1)/w[1])
plt.plot(t ,-(w[0]/w[1])*t - (b+1)/w[1])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('SVM.png', dpi=300, bbox_inches='tight')
plt.show()

print()
print('w = ',w)
print('b = ',b)
print('\nYou can check the value of \'w.T*x+b\' for each support vector:')
print('support vector 1: ',np.dot(x[:,1],w)+b, '~ +1')
print('support vector 16: ',np.dot(x[:,16],w)+b, '~ +1')
print('support vector 39: ',np.dot(x[:,39],w)+b, '~ -1')