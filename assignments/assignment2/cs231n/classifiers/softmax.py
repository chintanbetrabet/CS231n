import numpy as np
import math
from random import shuffle

def softmax_loss_naive(W, X, y, reg,bias=None):
  assert (X.shape[0]==y.shape[0]),"ERROR"
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  sc=X.dot(W)
  
  '''
  maxv=-100000
  for i in range(X.shape[0]):
		for j in range(len(sc[i])):
			maxv=max(maxv,sc[i][j])
  for i in range(X.shape[0]):
		for j in range(len(sc[i])):
			sc[i][j]-=maxv
  '''
  L=np.exp(sc)
  #print sc

  for i in range(X.shape[0]):
	  loss-=sc[i][y[i]]
	  loss+=math.log(sum(L[i]))
  '''
  prob_sum_term=np.	sum(L,axis=1)
  for i in xrange(X.shape[0]):
	  L[i]/=prob_sum_term[i]
	  mask=[cf==y[i] for cf in xrange(W.shape[1])]
	  L[i]=L[i]-mask
	  
	  for feat in xrange(W.shape[0]):
		  dW[feat]+=(L[i].T*X[i][feat])
  '''
 	  #dW+=np.multiply(L[i].T,X[i])
   		
	  #print loss
	  #print loss
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  '''
  for i in range(X.shape[0]):
	  den=np.sum(L[i])
	  for feat in range(W.shape[0]):
		  for cf in range(W.shape[1]):
			  num=L[i][cf]*X[i][feat]
			  dW[feat][cf]+=(float(num)/den)
			  if y[i] == cf:
				   dW[feat][cf]-=(W[feat][cf]*X[i][feat])
  '''
  
  dW=np.dot(X.T,error_class(X,y,W))
  loss/=X.shape[0]
  #loss+=0.5*reg*np.sum(W*W)
  #print "Gradient in Softmax",dW/float(X.shape[0])
  return loss, (dW+reg*W)


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  sc=X.dot(W)
  dW=np.dot(X.T,error_class(X,y,W))+reg*W
  
  L=np.exp(sc)
  #print sc

  for i in range(X.shape[0]):
	  loss-=sc[i][y[i]]
	  loss+=np.log(sum(L[i]))
  loss/=X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
def error_class(X,y,W,bias=None):
	scores=X.dot(W)
	if bias is not None:
		#print "BIAS"
		scores+=bias
	L=np.exp(scores)
	prob_sum_term=np.sum(L,axis=1)
	for i in xrange(X.shape[0]):
	  L[i]/=prob_sum_term[i]
	  mask=[cf==y[i] for cf in xrange(W.shape[1])]
	  L[i]=L[i]-mask
	return L/float(X.shape[0])
