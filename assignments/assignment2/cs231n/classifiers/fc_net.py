import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
import math
from softmax import softmax_loss_naive
from softmax import softmax_loss_vectorized
from softmax import error_class

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-4, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.params = {}
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b2'] = np.zeros(num_classes)
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    #N, D = X.shape
    N=X.shape[0]
    reg=self.reg
    feat=1
    for num_wei in X.shape[1:]:
	  feat*=num_wei
    #out = None
    X=X.reshape([X.shape[0],feat])
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    z1 = X.dot(W1) + b1
    a1 = np.maximum(0, z1) # pass through ReLU activation function
    scores = a1.dot(W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    m1=-100000000
    for xdim in range(scores.shape[0]):
		for ydim in range(scores.shape[1]):
			m1=max(m1,scores[xdim][ydim])
    #scores-=m1
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    # compute the class probabilities
    exp_scores = np.exp(scores)
    
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    '''
    loss,dscores=softmax_loss(exp_scores,y)
    loss+=0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    '''
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # compute the gradient on scores
    
    dscores = probs
    dscores[range(N),y] -= 1
    dscores /= N
	
    # W2 and b2
    grads['W2'] = np.dot(a1.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
  def loss1(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N= X.shape[0]
    
    
    feat=1
    for num_wei in X.shape[1:]:
	  feat*=num_wei
    #out = None
    Xfeat=X.reshape([X.shape[0],feat])
    #print "X1: shape",X1.shape
    X1=np.insert(Xfeat,0,1,axis=1)
    #print "X1: shape",X1.shape

  
    W11=np.insert(W1,0,b1,axis=0)
    W21=np.insert(W2,0,b2,axis=0)
    
    scores = None
    z1=X1.dot(W11)
    Layer1ub=np.maximum(0,z1)
    scoresub=np.exp(Layer1ub.dot(W2)+b2)
    Layer1=np.insert(np.maximum(0,z1),0,1,axis=1)
    #print "ALyer1",Layer1.shape
    scores=Layer1.dot(W21)
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss=0.0
    # Compute the loss
    L=np.exp(scores)
	#print sc
    for i in range(X.shape[0]):
	  loss-=scores[i][y[i]]
	  loss+=math.log(sum(L[i]))
    loss/=X.shape[0]
    loss+=+0.5 * self.reg * np.sum(W1 * W1)+ 0.5 * self.reg * np.sum(W2 * W2)
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    
    l1,w2gradient=softmax_loss_vectorized(W21,Layer1,y,self.reg)
    derror=error_class(Layer1ub,y,W2,b2) #obtain error in outermost layer
    dhidden=np.dot(derror,W2.T) #backprop error to hidden layer
    dhidden[Layer1ub<=0]=0 #apply ReLu
    w1grads=np.dot(Xfeat.T,dhidden) #compute grad
    #print "grad:",w1grads.shape,W1.shape
    grads = {}
    

    
    #print "W1:",w1grads.shape,W1.shape
    grads['W2']=w2gradient[1:]#.reshape(#np.dot(Layer1,gradient)
    grads['W1']=w1grads#+self.reg*W1#w1Lgradient[1:]+reg*W1
    grads['b2']=w2gradient[0]#np.sum(scoresub, axis=0)#
    grads['b1']=np.sum(dhidden, axis=0)#w1Lgradient[0]

    #print "grads over" 
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    print "W1",self.params['W1'].shape
    self.params['b1'] = np.zeros(hidden_dims[0])
    for i in range(1,self.num_layers-1):
		wparam="W"+str(1+i)
		bparam="b"+str(i+1)
		#print i
		self.params[wparam]=weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
		self.params[bparam] = np.zeros(hidden_dims[i])
		print wparam,self.params[wparam].shape
    wparam="W"+str(self.num_layers)
    bparam="b"+str(self.num_layers)
    
    self.params[wparam] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params[bparam] = np.zeros(num_classes)
    print wparam,self.params[wparam].shape
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss1(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    
    scores = X
    cache={}
    Wsum=0.0
    for layer in range(1,self.num_layers):
		cache['L'+str(layer)]=scores
		Wsum+=np.sum(self.params['W'+str(layer)]*self.params['W'+str(layer)])
		scores=scores.dot(self.params['W'+str(layer)])
		scores+=self.params['b'+str(layer)]
		if self.use_batchnorm:
			pass
		scores=relu_forward(scores) #relu
		if self.use_dropout:
			dropout_forward(scores,self.dropout_param)
    cache['L'+str(self.num_layers)]=scores	
    scores=scores.dot(self.params['W'+str(self.num_layers)])
    scores+=(self.params['b'+str(self.num_layers)])
    '''
    for x in sorted(cache):
		print x,cache[x].shape
    print scores.shape
    '''

    
    #loss+=+0.5 * reg * np.sum(W1 * W1)+ 0.5 * reg * np.sum(W2 * W2)
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss,dscores=softmax_loss(scores,y)
    loss+=0.5*self.reg*Wsum
    '''
    grads['W2'] = np.dot(a1.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2r
    '''
    dout=dscores
    grads['W'+str(self.num_layers)]=np.dot(cache['L'+str(self.num_layers)].T, dout)+0*self.reg*self.params['W'+str(self.num_layers)]
    grads['b'+str(self.num_layers)]=np.sum(dout,axis=0)
    dout=np.dot(dout, self.params['W'+str(self.num_layers)].T)
    dout[cache['L'+str(self.num_layers)] <= 0] = 0
    
    for layer in range(self.num_layers-1,0,-1):
		    grads['W'+str(layer)]=np.dot(cache['L'+str(layer)].T, dout)+self.reg*self.params['W'+str(layer)]
		    grads['b'+str(layer)]=np.sum(dout,axis=0)
		    dout=np.dot(dout, self.params['W'+str(layer)].T)
		    dout[cache['L'+str(layer)] <= 0] = 0
		
		
    
    
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
    
  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    
    scores = X
    cache={}
    Wsum=0.0
    for layer in range(1,self.num_layers):
		scores,cache['L'+str(layer)]=affine_forward(scores,self.params['W'+str(layer)],self.params['b'+str(layer)])
		#print 'L'+str(layer),cache['L'+str(layer)][0].shape
		Wsum+=np.sum(self.params['W'+str(layer)]*self.params['W'+str(layer)])
		if self.use_batchnorm:
			batchnorm_forward(scores,gamma,beta,bn_param)
		scores,cache['R'+str(layer)]=leaky_relu_forward(scores)#leaky_relu
		#cache['R'+str(layer)]=scores
		if self.use_dropout:
			dropout_forward(scores,self.dropout_param)
    #cache['L'+str(self.num_layers)]=scores
    scores,cache['L'+str(self.num_layers)]=affine_forward(scores,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
    	

    '''
    for x in sorted(cache):
		print x,cache[x].shape
    print scores.shape
    '''

    
    #loss+=+0.5 * reg * np.sum(W1 * W1)+ 0.5 * reg * np.sum(W2 * W2)
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss,dscores=softmax_loss(scores,y)
    loss+=0.5*self.reg*Wsum
    '''
    grads['W2'] = np.dot(a1.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2r
    '''
    dout=dscores
    dout,dw,db=affine_backward(dout,cache['L'+str(self.num_layers)])
    grads['W'+str(self.num_layers)]=dw
    grads['b'+str(self.num_layers)]=db
    
    for layer in range(self.num_layers-1,0,-1):
		if self.use_dropout:
			dout=dropout_backward(dout,cache['L'+str(layer)]) 
		dout=leaky_relu_backward(dout,cache['R'+str(layer)])
		dout,dw,db=affine_backward(dout,cache['L'+str(layer)])
		grads['W'+str(layer)]=dw+self.reg*self.params['W'+str(layer)]
		grads['b'+str(layer)]=db
		
		
		
    
    
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
