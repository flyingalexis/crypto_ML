import pandas as pd
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf

class cnn:
  def __init__(self, X , Y):
    self.X = X
    self.Y = Y
    self.X_train ,self.X_test , self.Y_train, self.Y_test = sk.train_test_split(X, Y, test_size= 0.2 , shuffle=True)
    self.hyper_param =	{
      "lr": 0.2,          # learning rate
      "batch_size": 5     
    }
    self.feature_size = X.shape[1]
    self.label_size = Y.shape[1]
    self.batch_num = np.where(self.X_train.shape[0] % self.hyper_param['batch_size'] == 0,
                    (self.X_train.shape[0] /self.hyper_param['batch_size']),
                    int((self.X_train.shape[0] / self.hyper_param['batch_size']) + 1))
    self.X_batches = np.array_split(X_train, batch_num, axis= 0)
    self.Y_batches = np.array_split(y_train, batch_num, axis= 0)
    self.session = tf.Session()

  def nn_construct(self):
    # declare input/output to the computational graph
    x = tf.placeholder(tf.float32, [None, self.feature_size])
    y_true = tf.placeholder(tf.float32, [None, self.label_size])
    y_true_cls = tf.placeholder(tf.int64, [None,2])

    

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