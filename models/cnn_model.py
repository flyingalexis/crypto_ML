import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf
import time
from datetime import timedelta

class cnn:
  def __init__(self, X , Y):
    self.X = X.reshape(X.shape[0],-1)
    self.Y = Y
    self.X_train ,self.X_test , self.Y_train, self.Y_test = sk.train_test_split(X, Y, test_size= 0.2 , shuffle=True)
    self.hyper_param =	{
      "lr": 0.2,          # learning rate
      "batch_size": 5,
      'filter_num': [4,4],
      'layers_num': 2,
      'filter_size': [2,2],
      'fc_size': [128,128],
      'input_channel' : 1,
      'concession': 0.02  # error within 2%
    }
    self.feature_size = (X.shape[1],X.shape[2])
    self.label_size = Y.shape[1]
    self.batch_num = np.where(self.X_train.shape[0] % self.hyper_param['batch_size'] == 0,
                    (self.X_train.shape[0] /self.hyper_param['batch_size']),
                    int((self.X_train.shape[0] / self.hyper_param['batch_size']) + 1))
    self.X_batches = np.array_split(self.X_train, self.batch_num, axis= 0)
    self.Y_batches = np.array_split(self.Y_train, self.batch_num, axis= 0)
    self.session = tf.Session()
    self.nn_construct()
    self.session.run(tf.global_variables_initializer())

  def nn_construct(self):
    # declare input/output to the computational graph
    self.x = tf.placeholder(tf.float32, [None, self.feature_size[0] * self.feature_size[1] ])
    x_matrix = tf.reshape(self.x, [-1, self.feature_size[0], self.feature_size[1],1])
    self.y_true = tf.placeholder(tf.float32, shape=[None, self.label_size], name='y_true')
    layer_convs = []
    weights_convs = []
    for i in range(self.hyper_param['layers_num']):
      if i == 0 :
        layer_conv, weights_conv = self.new_conv_layer( input=x_matrix,
          num_input_channels=self.hyper_param['input_channel'],
          filter_size=self.hyper_param['filter_size'][0],
          num_filters=self.hyper_param['filter_num'][0],
          use_pooling=True)
        layer_convs.append(layer_conv)
        weights_convs.append(weights_conv)
      else:
        layer_conv, weights_conv = self.new_conv_layer(
            input=layer_convs[-1],
            num_input_channels=self.hyper_param['filter_num'][i-1],     # num filter[0]
            filter_size=self.hyper_param['filter_size'][i],
            num_filters=self.hyper_param['filter_num'][i],
            use_pooling=True
        )
        layer_convs.append(layer_conv)
        weights_convs.append(weights_conv)

    layer_flat, num_features = self.__flatten_layer(layer_convs[-1])

    layer_fc1 = self.new_fc_layer(
        input=layer_flat,
        num_inputs=num_features,
        num_outputs=self.hyper_param['fc_size'][0],
        use_relu=True
    )

    self.layer_fc2 = self.new_fc_layer(
        input=layer_fc1,
        num_inputs=self.hyper_param['fc_size'][1],
        num_outputs=self.label_size,
        use_relu=False
    )

    # self.y_pred = tf.nn.softmax(self.layer_fc2) # not using the softmax for now 

    self.correct = tf.less_equal(tf.abs(tf.subtract(self.layer_fc2, self.y_true, name=None)), self.hyper_param['concession'])
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    # cost function
    cost = tf.losses.mean_squared_error(
        labels=self.y_true,
        predictions=self.layer_fc2
    )

    # optimizer
    self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

  def optimize(self, num_iterations = None):
    start_time = time.time()
    print(self.accuracy_test())
    if num_iterations == None or num_iterations > self.batch_num:
      num_iterations = self.batch_num
    for i in range(0, int(num_iterations)):
      x_batch, y_true_batch = self.X_batches[i] , self.Y_batches[i]
      feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch}
      self.session.run(self.optimizer, feed_dict=feed_dict_train)
    end_time = time.time()
    time_dif = end_time - start_time
    print(self.accuracy_test())
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif))))) 
  
  def accuracy_test(self):
    test_set = {self.x: self.X_test, self.y_true: self.Y_test}
    return self.session.run(self.accuracy, feed_dict=test_set )

  def kcv(self):
    kf = sk.KFold(n_splits=8)
    accs = []
    for idx, (train_index, test_index) in enumerate(kf.split(self.X)):
      tf.global_variables_initializer()
      f_dict = {self.x: self.X[train_index] , self.y_true: self.Y[train_index] }
      self.session.run(self.optimizer, feed_dict=f_dict )
      f_dict_test = {self.x: self.X[test_index] , self.y_true: self.Y[test_index] }
      accs.append(self.session.run(self.accuracy, feed_dict=f_dict_test))
    print(sum(accs)/len(accs))
      
  def __new_weights(self, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

  def __new_biases(self, length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

  def __flatten_layer(self, layer):
    # layer_shape = layer.get_shape()
    # num_features = layer_shape[:].num_elements()
    # layer_flat = tf.reshape(layer, [-1, num_features])
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  # because it is 2d cnn 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

  def new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels , num_filters]
    # shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = self.__new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = self.__new_biases(length=num_filters)

    layer = tf.nn.conv2d(input= input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights

  def new_fc_layer(self,input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    weights = self.__new_weights(shape=[num_inputs, num_outputs])
    biases = self.__new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
