import numpy as np

def disp(a,msg):
    print '\n### {0}'.format(msg)
    print 'nDim: {0}, shape: {1}'.format(np.ndim(a), a.shape)
    print a[:5]

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
disp(X,"X")
y = np.array([[0,1,1,0]]).T
disp(y,"y")

syn0 = 2*np.random.random((3,4)) - 1
disp(syn0,"syn0")

syn1 = 2*np.random.random((4,1)) - 1
disp(syn1,"syn1")

for j in xrange(1000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)