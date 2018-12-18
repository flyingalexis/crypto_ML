import pandas as pd
import models.lstm_model as LSTM
import os

t_crypto = 'XRP'
t_tf = '1m'


df = pd.read_csv('datasets/crypto_hist/{0}_{1}.csv'.format(t_crypto + 'USD', t_tf ))
df = df[df['t'] >= 1420070400000]        # data start from 1420070400 the data frame that we are going to use
setup_df = df[df['t'] <= 1540118160000]    # df only for Data generator to take reference to initialize 
df = df.sort_values(by='t', ascending=False)
setup_df = setup_df.sort_values(by='t', ascending=False)
train_test_boundary = int(df.shape[0] * 0.6)
train_df = df.iloc[: train_test_boundary,]
test_df = df.iloc[train_test_boundary:,]
train_dg = LSTM.DataGenerator(data = train_df, df = setup_df , stride = 10, batch_size = 128)
test_dg = LSTM.DataGenerator(data = test_df, df = setup_df , stride = 10, batch_size = 128)
md = LSTM.lstm(x_shape= 5, y_shape = 4,decode_func = train_dg.get_decode_func())
LSTM.optimize(t_crypto, t_tf, train_dg, test_dg, md)