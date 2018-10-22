from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import numpy as np
import time

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
set_session(tf.Session(config=config))

def build_model(input_num , output_num):

    model = Sequential()
    layers = [input_num, 50, 100, output_num]
    model.add(LSTM(
            layers[1],
            input_shape=(None, input_num),
            return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
            layers[2],
            return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            layers[3]))
    model.add(Activation("linear"))
    ### def own metrics
    def concess_prec(y_true, y_pred):
        # K.less_equal(y_true,)
        prec = tf.subtract(y_pred, y_true)
        prec = K.abs(prec)
        prec = K.less_equal(prec, 0.001)

        return K.mean(prec)

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy', concess_prec])
    print("Compilation Time : ", time.time() - start)
    return model

def run_network(data ,model=None):
    global_start_time = time.time()
    epochs = 4
    ratio = 0.2
    data_x = data[0]
    data_y = data[1]
    x_train ,x_test , y_train, y_test = sk.train_test_split(data_x, data_y, test_size= ratio , shuffle=True)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # sequence_length = data_x.shape[1]
    # input_num = data_x.shape[2]
    # output_num = data_y.shape[1]
    print ('\nData Loaded. Compiling...\n')

    model = None
    if model is None:
        model = build_model(data_x.shape[2],data_y.shape[1])
    try:
        model.fit(
            x_train, y_train,
            batch_size=64, epochs=epochs, validation_split=0.05)
        # model.save('bike.h5')
        predicted = model.predict(x_test)
        # predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print ('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    # try:
    #     # Evaluate
    #     scores = model.evaluate(X_test, y_test, batch_size=512)
    #     print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}\nmape={:.6f}".format(scores[0], scores[1], scores[2]))

    #     # draw the figure
    #     y_test += result_mean
    #     predicted += result_mean

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot(y_test,label="Real")
    #     ax.legend(loc='upper left')
    #     plt.plot(predicted,label="Prediction")
    #     plt.legend(loc='upper left')
    #     plt.show()

    # except Exception as e:
    #     print (str(e))
    # print ('Training duration (s) : ', time.time() - global_start_time)

    # return model, y_test, predicted