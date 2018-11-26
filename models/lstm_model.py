from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import keras.models as km
import tensorflow as tf
import keras.backend as K
import pandas as pd
from keras.backend.tensorflow_backend import set_session
import traceback
import gc
# optimize
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import csv

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
set_session(tf.Session(config=config))
min_cost = None 

with open('hyper_param_output.csv', 'w', newline='') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['setting','loss', 'concess_prec', 'avg_difference', 'avg_percentage_difference'])

class lstm:
    def __init__(self, x_shape,y_shape ,decode_func, h_params = dict()):
        self.hyper_param =	{     # default h-params
            "lr": 0.1,          # learning rate
            'layers_size': [128],
            'drop_out': 0.1,
            'fc_size': [128,128]
            }
        self.hyper_param.update(h_params)
        self.batch_size = 64
        self.accuracy_concession = 0.001
        self.input_x_shape = x_shape
        self.input_y_shape = y_shape
        self.decode_func = decode_func
        self.load_model()

    def load_model(self):
        model_path = "models_data/" + self.get_model_str() + ".h5"
        my_file = Path(model_path)
        metrics = {'concess_prec': self.concess_prec,'avg_difference': self.avg_difference, 'avg_percentage_difference': self.avg_percentage_difference}# make the precision function be dynamic in the future
        if my_file.exists():
            self.model = km.load_model(model_path,custom_objects=metrics) 
            if self.model is None:
                self.model = self.build_model()
                print('create new model')
            else:
                print('load model')
        else:
            self.model = self.build_model()
            print('create new model')

    def get_model_str(self):
        st = "lstm"
        for key, value in self.hyper_param.items():
            st = st + '_' + key
            st = st + '_' + str(value)
        return st

    def generator_test(self, generator, test_generator = None):
        try:
            self.model.fit_generator(generator=generator, use_multiprocessing=False,verbose=1)
            global min_cost
            if test_generator is not None:
                metrics = self.model.evaluate_generator(test_generator)   # use test by generator
                
                with open('hyper_param_output.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(self.hyper_param), *metrics])

                if min_cost is None or metrics[0] < min_cost:
                    min_cost = metrics[0] 
                    self.model.save("models_data/best_lstm.h5")
                return metrics
        except Exception as e: print(e)


    def build_model(self):
        print('make model....')
        try:
            model = Sequential()
            for idx ,ls in enumerate(self.hyper_param['layers_size']):
                input_s = self.input_x_shape if idx == 0 else self.hyper_param['layers_size'][idx -1]
                ret_seq = True if idx != len(self.hyper_param['layers_size']) - 1 else False            # only the last lstm do not return sequence
                model.add(LSTM(
                ls,
                input_shape=(None,input_s),
                return_sequences=ret_seq))
                model.add(Dropout(self.hyper_param['drop_out']))

            for idx ,ls in enumerate(self.hyper_param['fc_size']):      # assume all fc_layers has activation function
                if idx == len(self.hyper_param['fc_size']) -1:
                    model.add(Dense(self.input_y_shape))
                else:
                    model.add(Dense(self.hyper_param['fc_size'][idx+1]))
                model.add(Activation("elu"))

            start = time.time()
            model.compile(loss="mse", optimizer="rmsprop", metrics=[self.concess_prec, self.avg_difference, self.avg_percentage_difference])
            print("Compilation Time : ", time.time() - start)
            print(model.summary())
        except Exception as e: 
            print(e)
            traceback.print_exc()
        return model

    def avg_percentage_difference(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        percentage_diff = tf.divide( diff ,true_act ,name=None)
        return K.mean(percentage_diff)
    
    def avg_difference(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        return K.mean(diff)

    ### self-defined accuracy metrics       make the concession to be managable in the future
    def concess_prec(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        percentage_diff = tf.divide( diff ,true_act ,name=None)
        prec = K.less_equal(percentage_diff, 0.002)
        return K.mean(prec)

    def delete_model(self):
        del self.model


###  data genereator  implementing

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data ,df, x_len = 1440, y_len = 60, batch_size=64, stride = 5,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = stride 
        self.x_len = x_len
        self.y_len = y_len
        self.data = data        # the data actually going to be processed
        self.df = df            # for data preprocessing
        self.__init_normalization_params()
        self.on_epoch_end()
        self.ct = 0
        print('init Daata generator {0} '.format(df.shape))

    def __init_normalization_params(self):
        df = self.df
        max = df['high'].max()
        min = df['low'].min()       # need it for decoding the prediction
        norm_fac = max - min         # need it for decoding the prediction
        vol_min = df['volume'].min()                        # need it for decoding the prediction
        vol_norm_fac = df['volume'].max() - vol_min      # need it for decoding the prediction
        p_df = (df [['high','low','open','close']] - min) / norm_fac 
        v_df = (df['volume'] - vol_min ) / vol_norm_fac
        self.norm_data = pd.concat([p_df, v_df], axis=1, sort=False)
        self.x_dim = (self.norm_data.shape[1])
        self.decode_price = lambda a: tf.add(tf.multiply(tf.cast(norm_fac, tf.float32),a), (tf.cast(min, tf.float32)))
        self.decode_vol = lambda a: a * vol_norm_fac + vol_min  # not tensor version yet 
        self.np_decode_price = lambda a: ((a * norm_fac) + min)
        self.np_decode_vol = lambda a: a * vol_norm_fac + vol_min 

    def get_decode_func(self):
        return (self.decode_price,self.decode_vol)

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_episode = self.data.shape[0] - self.x_len - self.y_len        # posible num of episode
        # print(int(np.floor(self.num_episode / (self.batch_size*self.stride))))
        return int(np.floor(self.num_episode / (self.batch_size*self.stride)))

    
    def __getitem__(self, index): # seems like the index will be random with no overlapping in the Sequence class
        'Generate one batch of data'
        indexes = self.indexes[index]       # change the name 
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):     # ok
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.x_len + self.y_len, self.data.shape[0], self.batch_size* self.stride)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.x_len ,self.x_dim))
        y = np.empty((self.batch_size, 4)) # label size is 12 now  we should make it to be dynamic in the future
        end_idx = min(idx + (self.stride * self.batch_size), self.norm_data.shape[0] )
        # Generate data
        for count,i in enumerate(range(idx ,end_idx ,self.stride)):
            x_start = i - self.x_len - self.y_len
            x_end = y_start= i - self.y_len
            y_end = i
            x_df = self.norm_data.iloc[x_start:x_end,]
            y_df = self.norm_data.iloc[y_start:y_end,]
            y_mean = y_df[['high','low','open','close']].mean().values  # only predict mean for now
            y[count-1] = np.concatenate( (y_mean,), axis = 0 )
            X[count-1] = x_df.values

        return X, y


space = [
        Integer(1, 2, name="num_layers"),
        Integer(10, 100, name="num_nodes"),
        Real(1e-5, 1e-2, "log-uniform", name="learning_rate"),
        Real(0, 0.5, name="dropout_rate"),
        Integer(1, 2, name="num_dense_layers"),
        Integer(64, 128, name="num_dense_nodes"),
    ]
run_ctr = 0
gc.enable()
df = pd.read_csv('datasets/crypto_hist/{0}_{1}.csv'.format('XRPUSD','1m'))
train_test_boundary = int(df.shape[0] * 0.8)
train_df = df.iloc[: train_test_boundary,]
test_df = df.iloc[train_test_boundary:,]
dg = DataGenerator(data = train_df, df = df , stride = 1000)
test_dg = DataGenerator(data = test_df, df = df , stride = 1000)
md = lstm(x_shape= 5, y_shape = 4,decode_func = dg.get_decode_func())

@use_named_args(dimensions=space)
def fitness(learning_rate, num_layers ,num_nodes, dropout_rate,num_dense_layers , num_dense_nodes):
    global run_ctr
    global df, dg, test_dg, md
    K.clear_session()
    del md.model
    gc.collect()
    run_ctr = run_ctr + 1
    print('Iteration {0}'.format(run_ctr))

    layers_size = [num_nodes for n in range(num_layers)]
    fc_size = [num_dense_nodes for n in range(num_dense_layers)]
    params = {
        'lr' : learning_rate,
        'layers_size' : layers_size,
        'fc_size' : fc_size,
        'dropout': dropout_rate
    }
    md.__init__(x_shape= 5, y_shape = 4,decode_func = dg.get_decode_func(), h_params = params)
    cost = md.generator_test(dg, test_dg) # x_ y _ test
    print("cost: {0}".format(cost))
    return cost[0]

try:
    search_result = gp_minimize(func=fitness,
                            dimensions=space,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40)
except Exception as e:
    print('error exist')
    print(e)
    traceback.print_exc()

print('Best params')
print(search_result.x)
md = lstm(x_shape= 5, y_shape = 4,decode_func = dg.get_decode_func())
