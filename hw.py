from cvxopt.solvers import qp
from cvxopt.base import matrix
from pprint import pprint
import numpy, pylab, random, math
import multiprocessing

############### Functions start here

POLY_ARG = 2
RADIAL_SIGMA = 0.7

def linear_kernel(x,y):
	return numpy.dot(x,y) + 1

def poly_kernel(x,y ):
	return numpy.power((numpy.dot(x,y) + 1), POLY_ARG) 

def radial_basis_kernel(x,y):
	return numpy.exp(-(( numpy.dot( (x-y), (x-y)) / ( 2 *RADIAL_SIGMA *RADIAL_SIGMA ))))
	
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

def generate_data(meanA,varA,n):
	classA = [(random.normalvariate(meanA[0],varA[0]), 
			random.normalvariate(meanA[1], varA[1]), 
			1.0) 
			for i in range(n/4)] + \
			[(random.normalvariate(1.5, 1), 
			random.normalvariate(0.5, 1), 
			1.0) 
			for i in range(n/4)]
	
	classB = [(random.normalvariate(0.0, 0.5), 
		random. normalvariate(-2.5, 0.5) ,
		 -1.0) 
	for i in range(n/2)]
	
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
	
	return


def create_X_t(data):
	(r,c) = numpy.shape(data)
	list_of_lists = numpy.array([list(elem) for elem in data])
	t = list_of_lists[ : , c-1]
	t = numpy.transpose(t)
	X = list_of_lists[:, range(c-1)]
	return X, t

def save_non_zero_alpha(alpha):
	ret = []
	for i in range (len(alpha)):
		if numpy.absolute(alpha[i]) > 0.00001:
			ret.append([i, alpha[i]])

	return ret

def ind(est_x, xi_alpha_pair , X, T, kernel_func):
	(r,c) = numpy.shape(xi_alpha_pair);
	ret = 0;
	for i in range(r):
		dp_index = xi_alpha_pair[i][0]
		alpha = xi_alpha_pair[i][1]
		t_i = T[dp_index]
		x_i = X[dp_index]
		ret = ret + (alpha *t_i*kernel_func(est_x, x_i))
	return ret


def draw_contour(xi_alpha_pair,X,T,kernel_func):
	xrange = numpy.arange(-4,4,0.05)
	yrange = numpy.arange(-4,4,0.05)
	
	grid = matrix([[ind(numpy.array((x,y)),xi_alpha_pair,X,T,kernel_func) for y in yrange] for x in xrange])
	
	pylab.contour(xrange,yrange, grid, (-1.0, 0.0, 1.0), colors=('red','black','blue'),linewidths=(1,3,1))
	
	
	
############### Functions end here

# x = numpy.array([1,2,3])
# y = numpy.array([2,2,2])
# # print (linear_kernel(x,y))

# P = createP(numpy.array([x,y]), numpy.array([1,-1]), linear_kernel );
# print(P)





############ serious 

def run(classA,classB,data,kernel,arg):
	
	global POLY_ARG
	global RADIAL_SIGMA
		
	if kernel == poly_kernel:
		POLY_ARG = arg
	elif kernel == radial_basis_kernel:
		RADIAL_SIGMA = arg
		
	title = kernel.__name__ + " arg:" +str(arg)
		
	print (title)
	
	
	pylab.figure()
	X, t =create_X_t(data)
	P = createP(X, t, kernel)
	(N,c) = numpy.shape(data)
	q, h, G = build_q_G_h(N)

	r = qp( matrix(P), matrix(q), matrix(G), matrix(h));
	alpha = list( r['x'] )
	print("before printing")
	print ()
	pprint (alpha)
	xi_alpha_pair = save_non_zero_alpha(alpha);

	print(xi_alpha_pair)
	plot_data(classA, classB)
	est_x = [1, 3]

	retx = ind(est_x, xi_alpha_pair , X, t, kernel)
	print(retx)

	if len(xi_alpha_pair) < len(alpha):
		draw_contour(xi_alpha_pair,X,t,kernel)
		(z,c) = numpy.shape(xi_alpha_pair)
		for i in range(z):
			pylab.plot(X[xi_alpha_pair[i][0]][0],X[xi_alpha_pair[i][0]][1],'go')

	
	pylab.title(title)
	pylab.show()



classA, classB, data = generate_data((-1,0.0),(2,3),20)


proc0 = multiprocessing.Process(target=run,args=(classA,classB,data,poly_kernel,2))
proc0.start()


proc1 = multiprocessing.Process(target=run,args=(classA,classB,data,poly_kernel,3))
proc1.start()


proc2 = multiprocessing.Process(target=run,args=(classA,classB,data,radial_basis_kernel,0.7))
proc2.start()


proc2 = multiprocessing.Process(target=run,args=(classA,classB,data,radial_basis_kernel,1.5))
proc2.start()


run(classA,classB,data,linear_kernel,0)







