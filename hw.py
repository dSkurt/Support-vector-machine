from cvxopt.solvers import qp
from cvxopt.base import matrix
from pprint import pprint
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

def build_q_G_h(n ):
	q = numpy.ones(n) * -1
	h = numpy.zeros(n)
	g = (n,n)
	G = numpy.zeros(g)
	numpy.fill_diagonal(G, -1)
	return  q, h, G

def generate_data():
	classA = [(random.normalvariate(-1.5, 1), 
		random.normalvariate(0.5, 1), 
		1.0) for i in range(5)] + \
		[(random.normalvariate(1.5, 1), 
			random.normalvariate(0.5, 1), 
			1.0) 
		for i in range(5)]
	
	classB = [(random.normalvariate(0.0, 0.5), 
		random. normalvariate(-0.5, 0.5) ,
		 -1.0) 
	for i in range(10)]
	
	data = classA + classB 
	random.shuffle (data)
	return classA, classB, data

def plot_data(classA, classB):
	pylab.hold( True ) 
	pylab.plot([p[0]  for p in classA],
				[p[1] for p in classA],
				'bo') 
	pylab.plot([p[0] for p in classB],
				[p[1] for p in classB],
				'ro' )
	pylab.show()
	return


def create_X_t(data):
	(r,c) = numpy.shape(data)
	list_of_lists = numpy.array([list(elem) for elem in data])
	t = list_of_lists[ : , c-1]
	t = numpy.transpose(t)
	X = list_of_lists[:, range(c-1)]
	return X, t

############### Functions end here

x = numpy.array([1,2,3])
y = numpy.array([2,2,2])
# print (linear_kernel(x,y))

P = createP(numpy.array([x,y]), numpy.array([1,-1]), linear_kernel );
# print(P)

q, h, G = build_q_G_h( 3)

classA, classB, data = generate_data()
X, t =create_X_t(data)

print("before printing")
print (X)
print (t)
# plot_data(classA, classB)





















