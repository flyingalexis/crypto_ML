import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf
import time
from datetime import timedelta

# {'lr': Decimal('0.1'), 'filter_num': (144, 136, 72, 128), 'layers_num': 4}

LOGDIR = 'tmp/report/'
class cnn:
  def __init__(self, X , Y, h_params):
    self.X = X.reshape(X.shape[0],-1)
    self.Y = Y
    np.savetxt('X.out', self.X, delimiter=',')
    np.savetxt('Y.out', self.Y, delimiter=',')
    self.X_train ,self.X_test , self.Y_train, self.Y_test = sk.train_test_split(self.X, self.Y, test_size= 0.2 , shuffle=True)
    self.hyper_param =	{     # default h-params
      "lr": 0.2,          # learning rate
      "batch_size": 10,
      'filter_num': [4,4],
      'layers_num': 2,
      'filter_size': 2,
      'fc_size': [128,128],
      'input_channel' : 1,
      'concession': 0.05  # error within 2%
    }

    self.hyper_param.update(h_params)
    self.feature_size = (X.shape[1],X.shape[2])
    self.label_size = Y.shape[1]
    self.batch_num = np.where(self.X_train.shape[0] % self.hyper_param['batch_size'] == 0,
                    (self.X_train.shape[0] /self.hyper_param['batch_size']),
                    int((self.X_train.shape[0] / self.hyper_param['batch_size']) + 1))
    self.X_batches = np.array_split(self.X_train, self.batch_num, axis= 0)
    self.Y_batches = np.array_split(self.Y_train, self.batch_num, axis= 0)
    # self.session = tf.Session() # for cpu tf
    self.kf = sk.KFold(n_splits=8)
    gpu_options = tf.GPUOptions(allow_growth=True) 
    self.session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    

  def nn_construct(self):
    # declare input/output to the computational graph
    self.x = tf.placeholder(tf.float32, [None, self.feature_size[0] * self.feature_size[1] ], name="x")
    x_matrix = tf.reshape(self.x, [-1, self.feature_size[0], self.feature_size[1],1])
    self.y_true = tf.placeholder(tf.float32, shape=[None, self.label_size], name="labels")
    layer_convs = []
    weights_convs = []
    for i in range(self.hyper_param['layers_num']):
      if i == 0 :
        layer_conv, weights_conv = self.new_conv_layer( input=x_matrix,
          num_input_channels=self.hyper_param['input_channel'],
          filter_size=self.hyper_param['filter_size'],
          num_filters=self.hyper_param['filter_num'][0],
          use_pooling=True,
          name= "conv" + str(i))
        layer_convs.append(layer_conv)
        weights_convs.append(weights_conv)
      else:
        layer_conv, weights_conv = self.new_conv_layer(
            input=layer_convs[-1],
            num_input_channels=self.hyper_param['filter_num'][i-1],     # num filter[0]
            filter_size=self.hyper_param['filter_size'],
            num_filters=self.hyper_param['filter_num'][i],
            use_pooling=True,
            name= "conv" + str(i)
        )
        layer_convs.append(layer_conv)
        weights_convs.append(weights_conv)

    self.layer_flat, num_features = self.__flatten_layer(layer_convs[-1])

    # layer_fc1 = self.new_fc_layer(
    #     input=self.layer_flat,
    #     num_inputs=num_features,
    #     num_outputs=self.hyper_param['fc_size'][0],
    #     use_relu=False
    # )

    # self.layer_fc2 = self.new_fc_layer(
    #     input=self.layer_flat,
    #     num_inputs=self.hyper_param['fc_size'][1],
    #     num_outputs=self.label_size,
    #     use_relu=True
    # )
    self.layer_fc2 = self.new_fc_layer(
        input=self.layer_flat,
        num_inputs=num_features,
        num_outputs=self.label_size,
        use_relu=True
    )


    # self.y_pred = tf.nn.softmax(self.layer_fc2) # not using the softmax for now 
    with tf.name_scope("accuracy"):
      self.correct = tf.less_equal(tf.abs(tf.subtract(self.layer_fc2, self.y_true, name=None)), self.hyper_param['concession'])
      sp1,sp2 = tf.split(self.correct, num_or_size_splits=2, axis=1)
      self.accuracy = tf.reduce_mean(tf.cast(sp1, tf.float32))
      tf.summary.scalar("accuracy", self.accuracy)

    # cost function
    with tf.name_scope("xent"):
      cost = tf.losses.mean_squared_error(
          labels=self.y_true,
          predictions=self.layer_fc2
      )
      tf.summary.scalar("lost_funct", cost)

    # optimizer
    with tf.name_scope("train"):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

  def optimize(self, num_iterations = None):
    self.session = tf.Session(config=self.session_config)
    self.nn_construct()
    self.session.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR + '1')
    writer.add_graph(self.session.graph)
    start_time = time.time()
    if num_iterations == None or num_iterations > self.batch_num:
      num_iterations = self.batch_num
    for j in range(0,100):                        # Data not enough , reuse training set
      for i in range(0, int(num_iterations)):
        x_batch, y_true_batch = self.X_batches[i] , self.Y_batches[i]
        feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch}
        self.session.run(self.optimizer, feed_dict=feed_dict_train)
        s = self.session.run(merged_summary, feed_dict=feed_dict_train)
        writer.add_summary(s,i)
    end_time = time.time()
    time_dif = end_time - start_time
    feed_dict_train = {self.x: self.X_batches[0], self.y_true: self.Y_batches[0]}
    print(self.hyper_param)
    print(self.accuracy_test())
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif))))) 
  
  def accuracy_test(self):
    test_set = {self.x: self.X_test, self.y_true: self.Y_test}
    return self.session.run(self.accuracy, feed_dict=test_set )

  def kcv(self , h_params):
    self.hyper_param.update(h_params)
    tf.reset_default_graph()
    self.session = tf.Session(config=self.session_config)
    self.nn_construct()
    accs = []
    
    for train_index, test_index in self.kf.split(self.X):
      self.session.run(tf.global_variables_initializer())
      f_dict = {self.x: self.X[train_index] , self.y_true: self.Y[train_index] }
      self.session.run(self.optimizer, feed_dict=f_dict )
      f_dict_test = {self.x: self.X[test_index] , self.y_true: self.Y[test_index] }
      accs.append(self.session.run(self.accuracy, feed_dict=f_dict_test))
    return sum(accs)/len(accs)
      
  def __new_weights(self, shape):
    return tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.1), name="W")

  def __new_biases(self, length):
    return tf.Variable(tf.constant(0.1, shape=[length]), name="B")

  def __flatten_layer(self, layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  # because it is 2d cnn 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

  def new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True, name= "conv"):  # Use 2x2 max-pooling.
    with tf.name_scope(name):
      shape = [filter_size, filter_size, num_input_channels , num_filters]
      weights = self.__new_weights(shape=shape)
      biases = self.__new_biases(length=num_filters)
      layer = tf.nn.conv2d(input= input,
                          filter=weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
      layer += biases
      if use_pooling:
          layer = tf.nn.max_pool(value=layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
      layer = tf.nn.elu(layer)
      tf.summary.histogram("weight", weights)
      tf.summary.histogram("biased", biases)
      tf.summary.histogram("activation", layer)
      return layer, weights

  def new_fc_layer(self,input, num_inputs, num_outputs, use_relu=True, name= "FC"): 
    with tf.name_scope(name):
      weights = self.__new_weights(shape=[num_inputs, num_outputs])
      biases = self.__new_biases(length=num_outputs)
      layer = tf.matmul(input, weights) + biases

      if use_relu:
          layer = tf.nn.sigmoid(layer)

      return layer
    
