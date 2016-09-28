from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

############### Functions start here


def linear_kernel(x,y):
	return numpy.dot(x,y) + 1
	
def createP(X ,T, K):
	size = len(X)
	P = numpy.zeros( (size, size))
	for i in range(size):
		for j in range(size):
			P[i][j] = T[i]*T[j]* K(X[i], X[j])
	return P

############### Functions end here

x = numpy.array([1,2,3])
y = numpy.array([2,2,2])
print (linear_kernel(x,y))

P = createP(numpy.array([x,y]), numpy.array([1,-1]), linear_kernel );
print(P)

