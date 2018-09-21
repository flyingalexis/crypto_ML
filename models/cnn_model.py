import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf
import time
from datetime import timedelta

class cnn:
  def __init__(self, X , Y):
    self.X = X
    self.Y = Y
    self.X_train ,self.X_test , self.Y_train, self.Y_test = sk.train_test_split(X, Y, test_size= 0.2 , shuffle=True)
    self.hyper_param =	{
      "lr": 0.2,          # learning rate
      "batch_size": 5,
      'filter_num': [4,4],
      'layers_num': 2,
      'filter_size': [2,2],
      'fc_size': [128,128]
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
    layer_conv1, weights_conv1 = self.new_conv_layer(
      input=x_matrix,
      num_input_channels=1,
      filter_size=self.hyper_param['filter_size'][0],
      num_filters=self.hyper_param['filter_num'][0],
      use_pooling=True
    )

    layer_conv2, weights_conv2 = self.new_conv_layer(
        input=layer_conv1,
        num_input_channels=4,     # num filter[0]
        filter_size=self.hyper_param['filter_size'][1],
        num_filters=self.hyper_param['filter_num'][1],
        use_pooling=True
    )

    layer_flat, num_features = self.__flatten_layer(layer_conv2)

    layer_fc1 = self.new_fc_layer(
        input=layer_flat,
        num_inputs=num_features,
        num_outputs=self.hyper_param['fc_size'][0],
        use_relu=True
    )

    layer_fc2 = self.new_fc_layer(
        input=layer_fc1,
        num_inputs=self.hyper_param['fc_size'][0],
        num_outputs=self.label_size,
        use_relu=False
    )

    y_pred = tf.nn.softmax(layer_fc2)

    # cost function
    cost = tf.losses.mean_squared_error(
        labels=self.y_true,
        predictions=y_pred
    )

    # optimizer
    self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

  def optimize(self, num_iterations = None):
    start_time = time.time()
    if num_iterations == None or num_iterations > self.batch_num:
      num_iterations = self.batch_num
    for i in range(0, int(num_iterations)):
        x_batch, y_true_batch = self.X_batches[i] , self.Y_batches[i]
        x_batch = x_batch.reshape(x_batch.shape[0],-1)
        feed_dict_train = {self.x: x_batch, self.y_true: y_true_batch}

        self.session.run(self.optimizer, feed_dict=feed_dict_train)

    # total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

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
    
  def __new_weights(self, shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

  def __new_biases(self, length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

  def __flatten_layer(self, layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  # because it is 2d cnn 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

#------------------ end of model -----

# # load data
# df = db.get_pastdata_by_league('Brazilian Division 1')
# df.fillna(value=0, inplace=True)

# # split dataset to x , y and lines
# # normalize
# df_y = df[df.columns[pd.Series(df.columns).str.startswith('Y_')]]
# df_x = df[df.columns[pd.Series(df.columns).str.startswith('X_')]]
# df_x_norm , standardize_dict = dp.x_normalizer(df_x)
# df_y_one_hot, df_y_cls  = dp.y_mapper(df_y)

# # split train test set
# df_x_train, df_x_test, df_Y_train, df_Y_test, X_train, X_test, y_train_cls, y_test_cls, y_train, y_test = sk.train_test_split(df_x, df_y, df_x_norm.as_matrix(), df_y_cls.as_matrix(), df_y_one_hot.as_matrix(), test_size= 0.2 ,shuffle=True)

# # hyperparameter (make a loop right here to tune the hyperparams)
# learning_rate = 0.2
# batch_size = 5
# x_size = X_train.shape[1]
# y_size = y_train.shape[1]
# batch_num = np.where(X_train.shape[0]%batch_size == 0,
#                     (X_train.shape[0]/batch_size),
#                     int((X_train.shape[0] / batch_size) + 1))

# X_batches = np.array_split(X_train, batch_num, axis= 0)
# y_batches = np.array_split(y_train, batch_num, axis= 0)

# # declare input to the computational graph
# x = tf.placeholder(tf.float32, [None, x_size])
# y_true = tf.placeholder(tf.float32, [None, y_size])
# y_true_cls = tf.placeholder(tf.int64, [None,2])

# # w and b for adaline
# weights = tf.Variable(tf.random_normal([x_size, y_size], mean=0.0,stddev=0.001))
# biases = tf.Variable(tf.random_normal([y_size], mean=0.0,stddev=0.001))

# logits = tf.matmul(x ,weights) + biases
# y_pred = tf.nn.softmax(logits)

# y_pred_full, y_pred_half = tf.split(logits,2,1)
# y_pred_full = tf.nn.softmax(y_pred_full)
# y_pred_half = tf.nn.softmax(y_pred_half)
# y_pred_full_cls = tf.argmax(y_pred_full, axis=1)
# y_pred_half_cls = tf.argmax(y_pred_half, axis=1)
# y_pred_all_cls = tf.stack([y_pred_full_cls,y_pred_half_cls],axis=1)

# # define lost function
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
# cost = tf.reduce_mean(cross_entropy)

# # define performance
# correction_prediction = tf.equal(y_pred_all_cls,y_true_cls)
# accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

# optimizer = tf.train.GradientDescentOptimizer(
#     learning_rate=learning_rate
# ).minimize(cost)


# def predict(feed_dict_test):
#     pred_half,pred_full = np.array(session.run( (y_pred_half, y_pred_full), feed_dict=feed_dict_test))
#     pred = np.concatenate( (pred_half,pred_full) ,axis= 1)
#     print(df_x_test.shape)
#     az.analyze(pred, df_x_test)
#     # y_pred_probs_full = session.run(y_pred_half, feed_dict=feed_dict_test)

# # optimize
# def optimize(num_iterations):
#     for i in range(num_iterations):
#         # feed the data to the network
#         if i + 1 < batch_num:
#           feed_dict_train = {x: X_batches[i], y_true: y_batches[i]}
#           # run the computation graph
#           session.run(optimizer, feed_dict=feed_dict_train)

# def print_accuracy(feed_dict_test):
#     # Use TensorFlow to compute the accuracy.
#     acc = session.run(accuracy, feed_dict=feed_dict_test)
#     # y_pred_result = session.run(y_pred, feed_dict=feed_dict_test)
#     # np_y_pred = np.array(y_pred_result).round(decimals=2)
#     # Print the accuracy.
#     print("Accuracy on test-set: {0:.1%}".format(acc))


# session = tf.Session()
# session.run(tf.global_variables_initializer())
# optimize(num_iterations=1000)

# feed_dict_test = {x: X_test,
#                   y_true: y_test,
#                   y_true_cls: y_test_cls}

# predict(feed_dict_test)