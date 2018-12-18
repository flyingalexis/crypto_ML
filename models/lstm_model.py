from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras, skopt, os, math, traceback, gc, time, csv
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import keras.models as km
import tensorflow as tf
import keras.backend as K
import pandas as pd
from keras.backend.tensorflow_backend import set_session
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

min_cost = None 
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
config = tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
set_session(tf.Session(config=config))

# Global params
t_crypto = None
t_tf = None
base_path = None
log_base_path = None
skopt_log_base_path = None
train_dg = None
test_dg = None
md = None

def init(t_coin, t_timeframe, train_dg ,test_dg  ,md):
    globals()["train_dg"] = train_dg
    globals()["test_dg"] = test_dg
    globals()['t_crypto'] = t_coin
    globals()['md'] = md
    globals()['t_tf'] = t_timeframe
    globals()['base_path'] = 'models_data/{0}_{1}'.format(t_crypto,t_tf)
    globals()['log_base_path'] = base_path + '/' + 'hyper_param_output.csv'
    globals()['skopt_log_base_path'] = base_path + '/' + 'skopt_lstm.pkl'
    if not os.path.isdir(base_path):
        os.mkdir(base_path)

    if not os.path.isfile(base_path + '/' + 'hyper_param_output.csv') :
        with open(log_base_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['setting','loss', 'concess_prec_0_2', 'concess_prec_0_3', 'concess_prec_0_5', 'avg_difference', 'avg_percentage_difference'])


space = [
        Integer(1, 4, name="num_layers"),
        Integer(16, 150, name="num_units"),
        Real(1e-6, 1e-2, "log-uniform", name="learning_rate"),
        Real(0, 0.3, name="dropout_rate"),
        Integer(1, 3, name="num_dense_layers"),
        Integer(64, 256, name="num_dense_nodes"),
    ]
run_ctr = 0
gc.enable()

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

    def load_model(self):
        model_path = str(base_path) + '/' + self.get_model_str() + ".h5"
        metrics = {'concess_prec_0_2': self.concess_prec_0_2,
        'concess_prec_0_3': self.concess_prec_0_3,
        'concess_prec_0_5': self.concess_prec_0_5,
        'avg_difference': self.avg_difference, 
        'avg_percentage_difference': self.avg_percentage_difference
        }# make the precision function be dynamic in the future
        if os.path.isfile(model_path):
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

    def train_and_evaluate(self, generator, test_generator = None):
        try:
            self.model.fit_generator(generator=generator, use_multiprocessing=False,verbose=1)
            global min_cost
            if test_generator is not None:
                metrics = self.model.evaluate_generator(test_generator)   # use test by generator
                
                with open(log_base_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(self.hyper_param), *metrics])

                if min_cost is None or metrics[0] < min_cost:
                    min_cost = metrics[0] 
                    self.model.save("{0}/{1}.h5".format(base_path, self.get_model_str()))
                return metrics
        except Exception as e: print(e)


    def build_model(self):
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
                # if idx == len(self.hyper_param['fc_size']) -1:
                #     model.add(Dense(self.input_y_shape))
                # else:
                model.add(Dense(self.hyper_param['fc_size'][idx]))
                model.add(Activation("elu"))
            model.add(Dense(self.input_y_shape))
            model.add(Activation("elu"))
            # model.compile(loss="mse", optimizer="rmsprop", metrics=[self.concess_prec_0_2,self.concess_prec_0_3,self.concess_prec_0_5, self.avg_difference, self.avg_percentage_difference])
            model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[self.concess_prec_0_2,self.concess_prec_0_3,self.concess_prec_0_5, self.avg_difference, self.avg_percentage_difference])
            print(self.get_model_str())
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
    def concess_prec_0_2(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        percentage_diff = tf.divide( diff ,true_act ,name=None)
        prec = K.less_equal(percentage_diff, 0.002)
        return K.mean(prec)

    def concess_prec_0_3(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        percentage_diff = tf.divide( diff ,true_act ,name=None)
        prec = K.less_equal(percentage_diff, 0.003)
        return K.mean(prec)

    def concess_prec_0_5(self, y_true, y_pred):
        pred_act = self.decode_func[0](y_pred)
        true_act = self.decode_func[0](y_true)
        diff = tf.subtract(pred_act, true_act)
        diff = K.abs(diff)
        percentage_diff = tf.divide( diff ,true_act ,name=None)
        prec = K.less_equal(percentage_diff, 0.005)
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
        self.df = df            # for normalize
        self.__init_normalization_params()
        self.on_epoch_end()
        self.ct = 0

    def __init_normalization_params(self):
        df = self.data
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
        self.encode_price = lambda a: (a - min) / norm_fac

    def get_decode_func(self):
        return (self.decode_price,self.decode_vol)

    def get_np_decode_func(self):
        return (self.np_decode_price,self.np_decode_vol)

    def __len__(self):
        'Denotes the number of batches per epoch'
        self.num_episode = self.data.shape[0] - self.x_len - self.y_len        # posible num of episode

        # print(int(np.floor(self.num_episode / (self.batch_size*self.stride))))
        return math.ceil(self.num_episode / (self.batch_size*self.stride))

    
    def __getitem__(self, index): # seems like the index will be random with no overlapping in the Sequence class
        'Generate one batch of data'
        indexes = self.indexes[index]       # change the name 
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):     # ok
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.x_len + self.y_len + (self.batch_size* self.stride), self.data.shape[0], self.batch_size* self.stride)
        if( self.data.shape[0] not in self.indexes ):
            self.indexes = np.append(self.indexes, self.data.shape[0])
            self.last_size = ( (self.data.shape[0] - self.x_len - self.y_len) % (self.batch_size * self.stride) )
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        # print(self.indexes)
        # print("indexes len " + str(len(self.indexes)))

    def __data_generation(self, idx):
        'Generates data containing batch_size samples' 
        bs = self.batch_size
        if idx == self.data.shape[0]:
            bs = int(self.last_size / (self.stride))
            start_idx = idx - self.last_size
        start_idx = max(idx - (self.stride * (bs)), self.x_len + self.y_len )  # dep on bs
        X = np.empty((bs, self.x_len ,self.x_dim))
        y = np.empty((bs, 4)) # label size is 12 now  we should make it to be dynamic in the future
        # Generate data
        for count,i in list(enumerate(range(start_idx ,idx ,self.stride))):
            x_end = i
            x_start = y_end = i - self.x_len
            y_start = i - self.x_len - self.y_len
            x_df = self.norm_data.iloc[x_start:x_end,]
            if self.y_len > 0:
                y_df = self.norm_data.iloc[y_start:y_end,]
                y_mean = y_df[['high','low','open','close']].mean().values  # only predict mean for now
                y[count] = np.concatenate( (y_mean,), axis = 0 )
            X[count] = x_df.values
        
        # print('\n y_start: {0}   y_end: {1}  x_start: {2}  x_end: {3}\n'.format(y_start,y_end,x_start, x_end))
        return X.astype('float16'), y.astype('float16')

def skparams_to_params(num_layers ,num_units,learning_rate, dropout_rate,num_dense_layers , num_dense_nodes):
    layers_size = [num_units for n in range(num_layers)]
    fc_size = [num_dense_nodes for n in range(num_dense_layers)]
    params = {
        'lr' : learning_rate,
        'layers_size' : layers_size,
        'fc_size' : fc_size,
        'drop_out': dropout_rate
    }
    return params

@use_named_args(dimensions=space)
def fitness(learning_rate, num_layers ,num_units, dropout_rate,num_dense_layers , num_dense_nodes):
    global run_ctr
    global train_dg, test_dg, md
    K.clear_session()
    gc.collect()
    run_ctr = run_ctr + 1
    print('Iteration {0}'.format(run_ctr))
    params = skparams_to_params(num_layers ,num_units,learning_rate, dropout_rate,num_dense_layers , num_dense_nodes)
    md.__init__(x_shape= 5, y_shape = 4,decode_func = train_dg.get_decode_func(), h_params = params)  #
    md.load_model()
    cost = md.train_and_evaluate(train_dg, test_dg) # x_ y _ test
    print("cost: {0}".format(cost))
    
    return cost[0]


def optimize(t_coin, t_timeframe, train_dg,test_dg,md):
    init(t_coin, t_timeframe, train_dg,test_dg,md)
    try:
        search_result = None
        if os.path.isfile(skopt_log_base_path) :
            search_result = skopt.load(skopt_log_base_path)
            search_result = gp_minimize(func=fitness,
                                dimensions=space,
                                x0=search_result.x_iters,              # already examined values for x
                                y0=search_result.func_vals,              # observed values for x0
                                acq_func='EI', # Expected Improvement.
                                callback=[CheckpointSaver(skopt_log_base_path, compress=9)],
                                n_calls=50)
        else:
            search_result = gp_minimize(func=fitness,
                                    dimensions=space,
                                    acq_func='EI', # Expected Improvement.
                                    callback=[CheckpointSaver(skopt_log_base_path, compress=9)],
                                    n_calls=50)

        skopt.dump(search_result,skopt_log_base_path)
    except Exception as e:
        print(e)
        skopt.dump(search_result,skopt_log_base_path)
        traceback.print_exc()

    # print('Best params')
    # print(search_result.x)

def get_model(t_coin, t_timeframe, train_dg,test_dg,md, h_params):
    init(t_coin, t_timeframe, train_dg,test_dg,md)
    target_md = lstm(x_shape= 5, y_shape = 4,decode_func = train_dg.get_decode_func(), h_params = h_params)
    return target_md

