import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast
from cs231n.fast_layers import conv_forward_fast, conv_backward_fast


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    
    conv
    - w: Filter weights of shape (F, C, HH, WW)
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.conv_param={}
    self.pool_param={}
    self.cache={}
    self.conv_param['stride']=1
    self.pool_param['stride']=2
    self.conv_param['pad']=(filter_size-1)/2
    
    HH = 1 + ( input_dim[1]+ 2 * self.conv_param['pad'] - filter_size) /  self.conv_param['stride'] #output of conv
    WW = 1 + (input_dim[2] + 2 * self.conv_param['pad'] - filter_size) / self.conv_param['stride']
    
    
    W2=(WW-2)/self.pool_param['stride']+1 #ouput of max-pool
    H2=(HH-2)/self.pool_param['stride']+1
   
    
    
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    print self.params
    self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0],filter_size,filter_size) #conv
    self.params['b1'] = np.zeros(num_filters)#conv bias
    
    self.params['W2'] = weight_scale * np.random.randn(num_filters*H2*W2,hidden_dim) #affine
    self.params['b2'] = np.zeros(hidden_dim)#conv bias
    
    print "W2:",num_filters,H2,W2
    print "W2: ", self.params['W2'].shape
    
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes) #affine2
    self.params['b3'] = np.zeros(num_classes)
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


	#conv
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax
    convout=None
    print "ipiut",X.shape
    convout,self.cache['Input']=conv_forward_fast(X,W1,b1,conv_param)
    print "conv out",convout.shape
    relu1,self.cache['relu1']=relu_forward(convout)
    print "relu out",relu1.shape
    maxpoolout,self.cache['maxpool']=max_pool_forward_fast(relu1, pool_param)
    print "maxpool out",maxpoolout.shape
    affineout1,self.cache['affine1']=affine_forward(maxpoolout,W2,b2)
    print "affine out",affineout1.shape
    
    reluout2,self.cache['relu2']=relu_forward(affineout1)
    print "relu2",reluout2.shape
    affineout2,self.cache['affine2']=affine_forward(reluout2,W3,b3)
    print "affine out2",affineout2.shape
    
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    pass
    ############################################################################s
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss,dx = softmax_loss(affineout2,y)
    grads['W3'] = np.dot(reluout2.T, dx)
    grads['b3'] = np.sum(dx, axis=0)
    
    dx[affineout2 <=0]=0
    #relu2back=relu_backward(drelu2,self.cache['relu2'])
    print "W2 shape:",self.params['W2'].shape
    affineback1,dw,db=affine_backward(dx,[reluout2,self.params['W3'],self.params['b3']])
    
    
    grads['W2'] = np.dot(maxpoolout.T,affineback1)
    grads['b2'] = np.sum(affineback1, axis=0)
    grads['W1'] = self.params['W1']
    grads['b1'] = self.params['b1']
    
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
