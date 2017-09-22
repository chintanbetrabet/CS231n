import numpy as np
import matplotlib.pyplot as plt
import math
from softmax import softmax_loss_naive
from softmax import softmax_loss_vectorized
from softmax import error_class
from sklearn.model_selection import train_test_split
#from NN1 import TwoLayerNet

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    
    #print "REG:",reg
    
    X1=np.insert(X,0,1,axis=1)


  
    W11=np.insert(W1,0,b1,axis=0)
    W21=np.insert(W2,0,b2,axis=0)
    
    scores = None
    z1=X1.dot(W11)
    Layer1ub=np.maximum(0,z1)
    Layer1=np.insert(np.maximum(0,z1),0,1,axis=1)
    # Compute the forward pass
    
    
    scores=Layer1.dot(W21)
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores
    loss=0.0
    # Compute the loss
    L=np.exp(scores)
	#print sc
    for i in range(X.shape[0]):
	  loss-=scores[i][y[i]]
	  loss+=math.log(sum(L[i]))
    loss/=X.shape[0]
    loss+=+0.5 * reg * np.sum(W1 * W1)+ 0.5 * reg * np.sum(W2 * W2)

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    

    l1,w2gradient=softmax_loss_vectorized(W21,Layer1,y,reg)
    derror=error_class(Layer1ub,y,W2,b2) #obtain error in outermost layer
    dhidden=np.dot(derror,W2.T) #backprop error to hidden layer
    dhidden[Layer1ub<=0]=0 #apply ReLu
    w1grads=np.dot(X.T,dhidden) #compute grad
    #print "grad:",w1grads.shape,W1.shape
    grads = {}
    

    grads['W2']=w2gradient[1:]+reg*sum(W1*W1)#np.dot(Layer1,gradient)
    grads['W1']=w1grads+reg*W1#w1Lgradient[1:]+reg*W1
    grads['b2']=w2gradient[0]
    grads['b1']=np.sum(dhidden, axis=0)#w1Lgradient[0]
    
    #print grads['W1']
    
    
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False,mu=0.1):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    #batch_size=min(batch_size-1,num_train)
    iterations_per_epoch = max(num_train / batch_size, 1)
    #num_iters=num_train/200
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
     
    '''
    tups=[]
    for i in range(X.shape[0]):
		tups.append([X[i],y[i]])
    tups=np.array(tups)
    '''
    
    #mu=0.1 #slows down the velocity factor
    
    W2_velocity=np.zeros_like(self.params['W2'])
    W1_velocity=np.zeros_like(self.params['W1'])
    b2_velocity=np.zeros_like(self.params['b2'])
    b1_velocity=np.zeros_like(self.params['b1'])
    
    for it in xrange(num_iters):
      X_batch = X
      y_batch = y
      '''
	  batches=max(1,(X.shape[0])/batch_size)
	  X_batch=X1[it%batches:it%batches+batch_size]
	  y_batch=y1[it%batches:it%batches+batch_size]
	  loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
	  #loss_history.append(loss)
	  '''
	  
      sample_indices = np.random.choice(np.arange(num_train), batch_size)
      
      
      X_batch = X[sample_indices]
      y_batch = y[sample_indices]
      
      #X_batch,X_test,y_batch,y_test=train_test_split(X,y,train_size=min(num_train-1,batch_size))
      '''
      X_batch = []
      y_batch = []
       
      for val in comb[:batch_size]:
		X_batch.append(val[0])
		y_batch.append(val[1])
      X_batch=np.array(X_batch)
      y_batch=np.array(y_batch)
      X_batch=comb[:batch_size,:1]
      y_batch=comb[:batch_size,1:]
      '''
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      
      self.params['W2']-=learning_rate*grads['W2']
      self.params['W1']-=learning_rate*grads['W1']
      self.params['b1']-=learning_rate*grads['b1']
      self.params['b2']-=learning_rate*grads['b2']
      
      W2_velocity=mu*W2_velocity-learning_rate*grads['W2']
      W1_velocity=W1_velocity*mu-learning_rate*grads['W1']
      b2_velocity=mu*b2_velocity-learning_rate*grads['b2']
      b1_velocity=mu*b1_velocity-learning_rate*grads['b1']
	  
      self.params['W2']+=W2_velocity
      self.params['W1']+=W1_velocity
      self.params['b1']+=b1_velocity
      self.params['b2']+=b2_velocity
      
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      		  

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
	  
      
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
      loss1, grads1 = self.loss(X_batch, y=y_batch, reg=reg)
      
      if verbose and it % 100 == 0:
		  print 'iteration %d / %d: loss %f learning rate=%f' % (it, num_iters, loss,learning_rate)
	  # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
		  # Check accuracy
		  train_acc = (self.predict(X_batch) == y_batch).mean()
		  val_acc = (self.predict(X_val) == y_val).mean()
		  train_acc_history.append(train_acc)
		  val_acc_history.append(val_acc)
		
          # Decay learning rate
		  learning_rate = max(5e-5,learning_rate *learning_rate_decay)

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    
    
    
    X1=np.insert(X,0,1,axis=1)
    #print "Predict"

  
    W11=np.insert(W1,0,b1,axis=0)
    W21=np.insert(W2,0,b2,axis=0)
    
    scores = None
    z1=X1.dot(W11)
    Layer1ub=np.maximum(0,z1)
    Layer1=np.insert(np.maximum(0,z1),0,1,axis=1)
    # Compute the forward pass
    scores=Layer1.dot(W21)
    y_pred = np.argmax(scores,axis=1)
    
    

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


