from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

print ("This line will be printed.")


def linear_kernel(x,y):
	return numpy.dot(x,y) + 1
	
x = numpy.array([1,2,3])
y = numpy.array([2,2,2])
print (linear_kernel(x,y))


def make_p(X,Y,T,K):
	
